from __future__ import annotations

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
    ) -> None:
        self._registry = registry
        self._timeout_seconds = max(timeout_ms, 0) / 1000
        self._allow_online = allow_online

    def run_all(self, calls: list[ToolCall]) -> list[tuple[ToolCall, str]]:
        return [(call, self._run_one(call)) for call in calls]

    def _run_one(self, call: ToolCall) -> str:
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
        try:
            return future.result(timeout=self._timeout_seconds)
        except TimeoutError:
            future.cancel()
            return f"Tool {call.name} timed out."
        except Exception:
            return f"Tool {call.name} failed."
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
