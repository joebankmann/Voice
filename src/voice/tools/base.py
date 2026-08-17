from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str
    parameters_schema: dict[str, Any]
    offline: bool

    def run(self, args: dict[str, Any]) -> str: ...


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
