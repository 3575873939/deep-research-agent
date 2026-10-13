"""SSE 事件流协议：plan / tool_call / tool_result / token / verify / done / error。"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator


def sse_encode(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def event_stream(events: AsyncIterator[dict[str, Any]]) -> AsyncIterator[str]:
    async for e in events:
        yield sse_encode(e["type"], e["data"])
