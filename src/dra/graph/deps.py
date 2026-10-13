"""运行时依赖注入：节点函数不直接持有资源，全部经 ResearchDeps。

好处：测试可注入 fake provider/tools；checkpoint 只序列化 state 不序列化资源。
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

from ..config import Settings
from ..provider.base import MeteredLLM
from ..tools.base import Tool
from ..tools.citations import CitationStore
from ..trace.logger import RunTrace


@dataclass
class ResearchDeps:
    settings: Settings
    llm: MeteredLLM
    search: Tool
    fetch: Tool
    citations: CitationStore
    trace: RunTrace
    events: asyncio.Queue = field(default_factory=asyncio.Queue)  # SSE 事件流
    emit: Callable[[str, dict[str, Any]], None] | None = None

    async def emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        if self.emit:
            self.emit(event_type, data)
        await self.events.put({"type": event_type, "data": data})
