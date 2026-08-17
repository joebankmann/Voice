from __future__ import annotations

import json
import re
from typing import Any

from voice.tools.base import ToolCall


_TOOL_MARKER_RE = re.compile(
    r"<<tool:(?P<name>[a-zA-Z0-9_]+)\|(?P<json>\{.*?\})>>",
    re.DOTALL,
)


def extract_tool_calls(text: str) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for match in _TOOL_MARKER_RE.finditer(text):
        try:
            arguments: Any = json.loads(match.group("json"))
        except json.JSONDecodeError:
            continue
        if isinstance(arguments, dict):
            calls.append(
                ToolCall(name=match.group("name"), arguments=arguments),
            )
    return calls


def strip_tool_markers(text: str) -> str:
    return _TOOL_MARKER_RE.sub("", text)
