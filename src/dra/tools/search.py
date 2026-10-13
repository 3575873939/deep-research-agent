"""搜索工具：Tavily 主用 + DuckDuckGo 兜底，可插拔。"""
from __future__ import annotations

import json
import urllib.parse
from typing import Any

import httpx

from .base import ErrorType, ToolResult


class SerpAPISearch:
    """SerpAPI Google 搜索：官方文档召回优于 Tavily，作为主搜索源。"""

    name = "search"
    idempotent = True

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def run(self, params: dict[str, Any]) -> ToolResult:
        query = params["query"]
        max_results = int(params.get("max_results", 8))
        async with httpx.AsyncClient(timeout=25) as client:
            resp = await client.get(
                "https://serpapi.com/search",
                params={
                    "engine": "google",
                    "q": query,
                    "api_key": self.api_key,
                    "num": max_results,
                },
            )
            if resp.status_code in (401, 403):
                return ToolResult(ok=False, error="serpapi auth failed", error_type=ErrorType.AUTH)
            if resp.status_code == 429:
                return ToolResult(ok=False, error="serpapi 429", error_type=ErrorType.RATE_LIMITED)
            resp.raise_for_status()
            payload = resp.json()
        results = [
            {"title": r.get("title", ""), "url": r.get("link", ""), "snippet": r.get("snippet", "")}
            for r in payload.get("organic_results", [])
        ]
        if not results:
            return ToolResult(ok=False, error="no results", error_type=ErrorType.EMPTY)
        return ToolResult(ok=True, data=results, meta={"engine": "serpapi"})


class TavilySearch:
    name = "search"
    idempotent = True

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def run(self, params: dict[str, Any]) -> ToolResult:
        query = params["query"]
        max_results = int(params.get("max_results", 5))
        async with httpx.AsyncClient(timeout=25) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": max_results,
                    "include_answer": False,
                },
            )
            if resp.status_code in (401, 403):
                return ToolResult(ok=False, error="tavily auth failed", error_type=ErrorType.AUTH)
            if resp.status_code == 429:
                return ToolResult(ok=False, error="tavily 429", error_type=ErrorType.RATE_LIMITED)
            resp.raise_for_status()
            payload = resp.json()
        results = [
            {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
            for r in payload.get("results", [])
        ]
        if not results:
            return ToolResult(ok=False, error="no results", error_type=ErrorType.EMPTY)
        return ToolResult(ok=True, data=results)


class DuckDuckGoSearch:
    """无 key 兜底：DDG lite HTML 端点。"""

    name = "search"
    idempotent = True

    async def run(self, params: dict[str, Any]) -> ToolResult:
        query = params["query"]
        max_results = int(params.get("max_results", 5))
        url = "https://lite.duckduckgo.com/lite/?q=" + urllib.parse.quote(query)
        async with httpx.AsyncClient(
            timeout=25, headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True
        ) as client:
            resp = await client.post(
                "https://lite.duckduckgo.com/lite/",
                data={"q": query},
            )
            resp.raise_for_status()
            html = resp.text
        results = self._parse(html, max_results)
        if not results:
            return ToolResult(ok=False, error="no results parsed", error_type=ErrorType.EMPTY)
        return ToolResult(ok=True, data=results, meta={"engine": "duckduckgo"})

    @staticmethod
    def _parse(html: str, max_results: int) -> list[dict[str, str]]:
        import re

        results: list[dict[str, str]] = []
        # DDG lite 结果链接: <a rel="nofollow" href="//duckduckgo.com/l/?uddg=ENCODED" ...>title</a>
        for m in re.finditer(
            r'<a[^>]+href="[^"]*uddg=([^"&]+)[^"]*"[^>]*>(.*?)</a>', html, re.DOTALL
        ):
            link = urllib.parse.unquote(m.group(1))
            title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
            if title and link.startswith("http"):
                results.append({"title": title, "url": link, "snippet": ""})
            if len(results) >= max_results:
                break
        return results


def make_search_tool(tavily_key: str = "", serpapi_key: str = "") -> Any:
    """优先级：SerpAPI > Tavily > DuckDuckGo。"""
    if serpapi_key:
        return SerpAPISearch(serpapi_key)
    return TavilySearch(tavily_key) if tavily_key else DuckDuckGoSearch()


def parse_results_json(text: str) -> list[dict[str, str]]:
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, TypeError):
        return []
