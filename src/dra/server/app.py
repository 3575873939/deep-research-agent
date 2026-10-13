"""FastAPI + SSE 服务。

端点：
- POST /research        同步跑完整流程，返回报告 + 验证报告 + trace 摘要
- POST /research/stream SSE 流式：plan/tool_result/token/verify/done/error
- GET  /citations/{id}  引用原文快照核查（机器可查引用的对外证明）
- GET  /traces          历史运行记录（eval 底座）
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from ..graph.builder import ResearchRuntime
from ..provider.anthropic import AnthropicProvider
from ..provider.base import LLMProvider
from ..provider.modelscope import OpenAICompatibleProvider
from .stream import event_stream


class ResearchRequest(BaseModel):
    question: str
    model: str | None = None       # 覆盖全部节点的模型名（含环境默认时省略）
    base_url: str | None = None    # 自定义供应商 base_url（OpenAI 兼容或 Anthropic）
    api_key: str | None = None     # 自定义供应商 api_key


# 前端可选的供应商预设（base_url + 默认模型）。用户也可自由填写自定义 base_url/api_key。
PROVIDER_PRESETS = [
    {"id": "deepseek", "name": "DeepSeek", "base_url": "https://api.deepseek.com",
     "protocol": "openai", "models": ["deepseek-flash", "deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner"]},
    {"id": "openai", "name": "OpenAI", "base_url": "https://api.openai.com/v1",
     "protocol": "openai", "models": ["gpt-4o", "gpt-4o-mini", "o3", "o4-mini"]},
    {"id": "anthropic", "name": "Claude (Anthropic)", "base_url": "https://api.anthropic.com",
     "protocol": "anthropic", "models": ["claude-3-5-sonnet-latest", "claude-4-5-sonnet", "claude-4-5-opus"]},
    {"id": "glm", "name": "智谱 GLM", "base_url": "https://open.bigmodel.cn/api/paas/v4",
     "protocol": "openai", "models": ["glm-4.7-flash", "glm-4.7"]},
    {"id": "qwen", "name": "通义千问", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
     "protocol": "openai", "models": ["qwen-max", "qwen-plus", "qwen-flash"]},
    {"id": "moonshot", "name": "Moonshot Kimi", "base_url": "https://api.moonshot.cn/v1",
     "protocol": "openai", "models": ["kimi-k2", "moonshot-v1-8k"]},
    {"id": "modelscope", "name": "ModelScope 魔搭", "base_url": "https://api-inference.modelscope.cn/v1",
     "protocol": "openai", "models": ["Qwen/Qwen3.8-Flash-Next", "Qwen/Qwen3.5-35B-A3B",
                                       "Qwen/Qwen3-235B-A22B-Instruct-2507", "deepseek-ai/DeepSeek-V4.1-Flash"]},
]


def _model_overrides(model: str | None) -> dict[str, str] | None:
    if not model:
        return None
    return {
        "dra_model_planner": model,
        "dra_model_researcher": model,
        "dra_model_synthesizer": model,
        "dra_model_verifier": model,
        "dra_model_light": model,
    }


def _build_provider(base_url: str | None, api_key: str | None, *, default_provider: LLMProvider,
                    thinking: bool) -> LLMProvider:
    """按 base_url 自动选协议；没提供自定义时用环境默认 provider。"""
    if not base_url and not api_key:
        return default_provider
    if not api_key:
        raise RuntimeError("自定义供应商需要提供 api_key")
    url = (base_url or "").lower()
    if "anthropic.com" in url:
        return AnthropicProvider(base_url or "https://api.anthropic.com", api_key)
    return OpenAICompatibleProvider(base_url or "https://api.openai.com/v1", api_key,
                                    thinking=thinking)


def create_app(runtime: ResearchRuntime | None = None) -> FastAPI:
    rt = runtime or ResearchRuntime()
    app = FastAPI(title="Deep Research Agent", version="0.1.0")

    @app.get("/providers")
    async def providers() -> dict:
        s = rt.settings
        return {
            "providers": PROVIDER_PRESETS,
            "default_base_url": s.dra_base_url,
            "default_model": s.dra_model_planner,
            "thinking_enabled": s.dra_thinking_enabled,
        }

    @app.post("/research")
    async def research(req: ResearchRequest) -> dict:
        provider = _build_provider(req.base_url, req.api_key,
                                   default_provider=rt._provider,
                                   thinking=rt.settings.dra_thinking_enabled)
        state, trace = await rt.run(req.question,
                                    model_overrides=_model_overrides(req.model),
                                    provider=provider)
        return {
            "run_id": trace.run_id,
            "report": state.get("draft", state.get("final_report", "")),
            "verify_report": state.get("verify_report", []),
            "budget_exhausted": trace.budget_exhausted,
            "tokens": trace.total_tokens,
            "latency_s": round(trace.latency_s, 1),
        }

    @app.post("/research/stream")
    async def research_stream(req: ResearchRequest) -> StreamingResponse:
        print(f"[stream] 收到请求: {req.question[:50]} | model={req.model or 'default'} | base_url={req.base_url or '默认'}")
        provider = _build_provider(req.base_url, req.api_key,
                                   default_provider=rt._provider,
                                   thinking=rt.settings.dra_thinking_enabled)
        return StreamingResponse(
            event_stream(rt.run_stream(req.question,
                                       model_overrides=_model_overrides(req.model),
                                       provider=provider)),
            media_type="text/event-stream",
        )

    @app.get("/citations/{citation_id}")
    async def citation(citation_id: int) -> dict:
        cite = rt.citations.get(citation_id)
        if cite is None:
            return {"ok": False, "error": "not found"}
        return {
            "ok": True,
            "id": cite.id,
            "url": cite.url,
            "title": cite.title,
            "snapshot": cite.snapshot,
            "content_hash": cite.content_hash,
        }

    @app.get("/traces")
    async def traces() -> list[dict]:
        return rt.traces.read_all()

    # 根路径返回调研页面（显式 GET /，不用 StaticFiles 挂载 / ——那会拦截 API 路由）
    static_dir = Path(__file__).parent / "static"
    static_index = static_dir / "index.html"

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(str(static_index), headers={"Cache-Control": "no-store, no-cache, must-revalidate"})

    @app.get("/test.html", include_in_schema=False)
    async def test():
        return FileResponse(str(static_dir / "test.html"), headers={"Cache-Control": "no-store, no-cache, must-revalidate"})

    return app


app = create_app()
