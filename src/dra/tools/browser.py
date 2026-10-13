"""网页读取工具：httpx 拉取 + HTML→正文抽取。

注：browser 非幂等（页面可能变、且重试成本高），超时 30s，不重试。
复杂 JS 页面在真实部署中交给 playwright MCP server，本实现覆盖静态页主链路。
"""
from __future__ import annotations

import re
from typing import Any

import httpx

from .base import ToolResult

MAX_CONTENT_CHARS = 12_000  # 进 context 前截断，配合注入隔离标记


class BrowserFetch:
    name = "fetch_page"
    idempotent = False

    async def run(self, params: dict[str, Any]) -> ToolResult:
        url = params["url"]
        async with httpx.AsyncClient(
            timeout=25,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DeepResearchAgent/0.1)"},
            follow_redirects=True,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text
        text = extract_text(html)
        if not text:
            from .base import ErrorType

            return ToolResult(ok=False, error="empty page text", error_type=ErrorType.EMPTY)
        truncated = text[:MAX_CONTENT_CHARS]
        return ToolResult(
            ok=True,
            data={"url": str(resp.url), "text": truncated, "truncated": len(text) > MAX_CONTENT_CHARS},
        )


def extract_text(html: str) -> str:
    """极简 HTML→文本：去 script/style/标签，压缩空白。"""
    html = re.sub(r"(?is)<(script|style|noscript|nav|footer|header)[^>]*>.*?</\1>", " ", html)
    html = re.sub(r"(?is)<!--.*?-->", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    # 实体
    for ent, ch in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&nbsp;", " "), ("&#39;", "'")):
        text = text.replace(ent, ch)
    return re.sub(r"\s+", " ", text).strip()
