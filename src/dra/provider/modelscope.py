"""ModelScope openai_compatible 通道实现。

任何 OpenAI 兼容端点（DeepSeek 官方、OpenAI、vLLM 自建）都可复用本类，只换 base_url/key。
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from openai import AsyncOpenAI, APIError, RateLimitError

from .base import ChatResult, LLMProvider, RateLimited, Usage


def _rate_limit_backoff(e: Exception, attempt: int, *, base: float = 2.0, cap: float = 30.0) -> float:
    """429 的退避秒数：优先用上游 Retry-After，否则指数退避。"""
    try:
        ra = e.response.headers.get("retry-after")  # type: ignore[attr-defined]
        if ra:
            return min(float(ra), cap)
    except Exception:  # noqa: BLE001 - 拿不到头就用指数退避
        pass
    return min(base * (2**attempt), cap)


class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        thinking: bool = False,
    ) -> None:
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=httpx.Timeout(timeout_s),
            max_retries=0,  # 重试自己控制，避免与熔断器叠加
        )
        self._max_retries = max_retries
        self._thinking = thinking  # DeepSeek 思考模式开关（默认关，content-only 通道）

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        thinking: bool | None = None,
    ) -> ChatResult:
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        # DeepSeek 需显式传 thinking 开关（默认开→content 为空）。未知参数对 ModelScope/Qwen 无影响。
        if thinking is None:
            thinking = self._thinking
        kwargs["extra_body"] = {"thinking": {"type": "enabled" if thinking else "disabled"}}

        resp = None
        last_transient: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = await self._client.chat.completions.create(**kwargs)
                # 部分上游（ModelScope 推理模型）偶发返回 choices=null，视为可重试的瞬态错误
                if resp.choices:
                    break
                resp = None
                last_transient = RuntimeError("empty choices from upstream")
            except RateLimitError as e:
                # 429 是"稍后再试"而非"熔断停手"：先按 Retry-After / 指数退避重试，
                # 重试耗尽才抛 RateLimited 交给熔断器。免费档下硬打会白烧额度并把
                # 偶发限流升级成 900s 熔断。
                if attempt >= self._max_retries:
                    raise RateLimited(str(e)) from e
                await asyncio.sleep(_rate_limit_backoff(e, attempt))
            except (json.JSONDecodeError, httpx.RemoteProtocolError, APIError) as e:
                # 上游返回非 JSON / 截断响应（ModelScope 偶发），视为可重试瞬态错误
                last_transient = e
                if attempt < self._max_retries:
                    continue
                raise RuntimeError(f"upstream transient error after retries: {e}") from e

        if resp is None or not resp.choices:
            raise RuntimeError(f"empty choices from upstream after retries: {last_transient}")
        choice = resp.choices[0]
        usage = Usage()
        if resp.usage:
            usage = Usage(
                prompt_tokens=resp.usage.prompt_tokens or 0,
                completion_tokens=resp.usage.completion_tokens or 0,
            )
        tool_calls = []
        if choice.message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in choice.message.tool_calls
            ]
        return ChatResult(
            content=choice.message.content or "",
            usage=usage,
            model=model,
            tool_calls=tool_calls,
        )
