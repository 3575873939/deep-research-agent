"""Anthropic 原生 Messages API 通道实现（openai_compatible 之外的第二个协议）。

协议要点：
- POST /v1/messages，Header x-api-key + anthropic-version: 2023-06-01
- body 必填 max_tokens；system 单独字段；tools 用 input_schema（非 OpenAI 的 parameters）
- 返回 content 是块数组：{type:text} 文本 / {type:tool_use, id, name, input}
"""
from __future__ import annotations

import json
from typing import Any

import httpx

from .base import ChatResult, LLMProvider, RateLimited, Usage

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(LLMProvider):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout_s: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        # 允许传入带 /v1/ 走 /messages，或裸域名直接拼 /v1/messages
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/v1"):
            self.endpoint = f"{self.base_url}/messages"
        else:
            self.endpoint = f"{self.base_url}/v1/messages"
        self.api_key = api_key
        self.timeout = timeout_s
        self.max_retries = max_retries

    @staticmethod
    def _to_anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        """OpenAI 格式 tools → Anthropic 格式（input_schema）。"""
        if not tools:
            return None
        out = []
        for t in tools:
            fn = t.get("function", {})
            out.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return out

    @staticmethod
    def _from_anthropic_content(content: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        """Anthropic content 块数组 → (text, tool_calls)。"""
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in content:
            t = block.get("type")
            if t == "text":
                text_parts.append(block.get("text", ""))
            elif t == "tool_use":
                tool_calls.append({
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                })
        return "".join(text_parts), tool_calls

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
        # 系统消息单列到 system 字段；其余按 role 映射
        system = "".join(m["content"] for m in messages if m.get("role") == "system")
        body_messages = [
            {"role": m.get("role", "user"), "content": m["content"]}
            for m in messages
            if m.get("role") in ("user", "assistant")
        ]
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": body_messages,
        }
        if system:
            payload["system"] = system
        anth_tools = self._to_anthropic_tools(tools)
        if anth_tools:
            payload["tools"] = anth_tools

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        last_transient: Exception | None = None
        for attempt in range(self.max_retries + 1):
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(self.endpoint, json=payload, headers=headers)
            if resp.status_code == 429:
                raise RateLimited(f"anthropic 429: {resp.text[:150]}")
            if resp.status_code in (401, 403):
                raise RuntimeError(f"anthropic auth failed: {resp.status_code} {resp.text[:150]}")
            if resp.status_code != 200:
                # 5xx / 截断可重试
                last_transient = RuntimeError(f"anthropic {resp.status_code}: {resp.text[:200]}")
                if attempt < self.max_retries:
                    continue
                raise last_transient
            try:
                data = resp.json()
            except json.JSONDecodeError as e:
                last_transient = e
                if attempt < self.max_retries:
                    continue
                raise
            break

        text, tool_calls = self._from_anthropic_content(data.get("content", []))
        usage = Usage()
        if data.get("usage"):
            usage = Usage(
                prompt_tokens=data["usage"].get("input_tokens", 0),
                completion_tokens=data["usage"].get("output_tokens", 0),
            )
        return ChatResult(content=text, usage=usage, model=model, tool_calls=tool_calls)