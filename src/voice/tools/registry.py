from __future__ import annotations

from voice.memory.episodic import EpisodicStore
from voice.memory.preferences import PreferencesStore
from voice.tools.base import Tool
from voice.tools.builtins import LocalTimeTool, NoteAddTool, PreferenceSetTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools.values())


def build_default_registry(
    *,
    preferences: PreferencesStore | None = None,
    episodic: EpisodicStore | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(LocalTimeTool())
    registry.register(PreferenceSetTool(preferences))
    registry.register(NoteAddTool(episodic))
    return registry
