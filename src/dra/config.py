"""全局配置：节点级模型路由是核心设计——不同节点能力需求不同，可配不同模型。

面试讲点：verifier 与 synthesizer 默认异模（生成/验证分离），避免模型"自我认可"偏差。
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    # LLM 通道（openai_compatible）
    dra_base_url: str = "https://api-inference.modelscope.cn/v1"
    dra_api_key: str = ""

    # DeepSeek 等上游默认开思考模式（content 为空、慢、贵）。本系统 content-only，
    # 默认传 disabled；需要思考质量的节点可置 True（思考模式不支持 temperature）。
    dra_thinking_enabled: bool = False

    # 节点级模型路由
    dra_model_planner: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    dra_model_researcher: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    dra_model_synthesizer: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    dra_model_verifier: str = "deepseek-ai/DeepSeek-V3.2-Exp"
    dra_model_light: str = "Qwen/Qwen3-32B"

    # 预算与熔断
    dra_max_tokens_per_run: int = 200_000
    dra_rate_limit_cooldown_s: float = 900.0
    # 客户端最小调用间隔（秒）。0 = 不限速；免费档限速的通道（如 AIHubMix 5 请求/分钟）
    # 设为 ~12.5 可避免连续 429 触发熔断，把额度浪费在失败重试上。
    dra_min_call_interval_s: float = 0.0

    # 工具
    tavily_api_key: str = ""
    serpapi_key: str = ""  # SerpAPI Google 搜索，优先于 Tavily
    tool_timeout_s: float = 30.0
    llm_timeout_s: float = 300.0  # 推理模型单跳思考可能 >120s

    # researcher 循环纪律
    researcher_max_steps: int = 8
    researcher_max_retry_per_question: int = 2
    verifier_max_rounds: int = 2
    researcher_fetch_topk: int = 5  # 每个子问题抓取页面数（从 3 扩到 5 提升召回）
    researcher_max_tokens_per_source: int = 4000  # 单来源进 context 的截断上限（token 级）

    # 人工介入（human-in-the-loop）：verifier 检出缺乏引用支持的结论时暂停，等人确认。
    # 默认关闭——eval / 批量跑必须全自动；开启后需要 checkpointer 支持断点恢复。
    dra_human_review: bool = False
    dra_checkpoint_path: str = "./data/checkpoints.db"
    # 触发人工确认的未支持结论数下限（0 表示只要有非 supported 结论就暂停）
    human_review_min_risky: int = 1

    # 检索增强（RAG）：抓取页面 → 切分 → BM25+向量双路召回 → RRF 融合 → 可选 rerank
    # 选块进 context 而非整页灌入，降噪同时省 token；任一步不可用都自动降级
    rag_enabled: bool = True
    rag_top_k: int = 16  # 送进 LLM 的证据块数（chunk 级）
    rag_use_vector: bool = True  # 需 sentence-transformers，缺失自动退化为 BM25-only
    rag_use_reranker: bool = False  # 需 bge-reranker，缺失自动退化为 RRF 顺序
    rag_embedding_model: str = "BAAI/bge-small-zh-v1.5"
    rag_reranker_model: str = "BAAI/bge-reranker-base"
    # 只读本地模型缓存、不联网检查：避免无外网时阻塞数分钟（没缓存则直接降级 BM25-only）
    rag_local_files_only: bool = True
    # 向量召回的相关性下限：向量检索没有"零命中"概念，不设阈值会把无关块灌进证据
    rag_vector_min_score: float = 0.45

    # MCP 工具层：接入外部 MCP server（stdio 传输，server 作为子进程）。
    # 默认不配置 → 不启动任何子进程、零额外开销；配置 mcp_search_tool 后，
    # 该工具会替代内置检索源进入 researcher 主链路。
    dra_mcp_servers: str = ""       # JSON 数组，格式见 tools/mcp_client.parse_mcp_servers
    dra_mcp_search_tool: str = ""   # 充当检索源的 MCP 工具全名，如 "myserver.web_search"

    # 存储
    dra_db_path: str = "./data/research.db"
    dra_trace_path: str = "./data/traces.jsonl"

    def model_for(self, node: str) -> str:
        return getattr(self, f"dra_model_{node}")


@lru_cache
def get_settings() -> Settings:
    return Settings()
