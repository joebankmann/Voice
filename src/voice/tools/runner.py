from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from voice.tools.base import Tool, ToolCall
from voice.tools.registry import ToolRegistry


class ToolRunner:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        timeout_ms: int,
        allow_online: bool,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self._registry = registry
        self._timeout_seconds = max(timeout_ms, 0) / 1000
        self._allow_online = allow_online
        self._cancel_event = cancel_event or threading.Event()

    def bind_cancel(self, cancel_event: threading.Event) -> None:
        self._cancel_event = cancel_event

    def run_all(self, calls: list[ToolCall]) -> list[tuple[ToolCall, str]]:
        return [(call, self._run_one(call)) for call in calls]

    def _cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def _run_one(self, call: ToolCall) -> str:
        if self._cancelled():
            return f"Tool {call.name} was cancelled."
        tool = self._registry.get(call.name)
        if tool is None:
            return f"Unknown tool: {call.name}."
        if not tool.offline and not self._allow_online:
            return (
                f"Tool {call.name} is unavailable while online tools are disabled."
            )
        return self._run_with_timeout(tool, call)

    def _run_with_timeout(self, tool: Tool, call: ToolCall) -> str:
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(tool.run, call.arguments)
        deadline = time.monotonic() + self._timeout_seconds
        try:
            while True:
                if self._cancelled():
                    future.cancel()
                    return f"Tool {call.name} was cancelled."
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    future.cancel()
                    return f"Tool {call.name} timed out."
                try:
                    return future.result(timeout=min(0.05, remaining))
                except TimeoutError:
                    continue
        except Exception:
            return f"Tool {call.name} failed."
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
