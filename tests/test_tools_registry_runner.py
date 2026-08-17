from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import voice.tools as tools
from voice.memory import EpisodicStore, PreferencesStore
from voice.tools.base import Tool, ToolCall
from voice.tools.registry import ToolRegistry, build_default_registry
from voice.tools.runner import ToolRunner


@dataclass
class FakeTool:
    name: str
    result: str = "done"
    delay_seconds: float = 0.0
    offline: bool = True
    description: str = "A fake tool"
    parameters_schema: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.parameters_schema is None:
            self.parameters_schema = {}

    def run(self, args: dict[str, Any]) -> str:
        time.sleep(self.delay_seconds)
        return self.result


def test_tools_package_exports_c2_api():
    assert tools.Tool is Tool
    assert tools.ToolCall is ToolCall
    assert tools.ToolRegistry is ToolRegistry
    assert tools.ToolRunner is ToolRunner
    assert tools.build_default_registry is build_default_registry


def test_registry_registers_and_lists_tools_in_registration_order():
    first = FakeTool("first")
    second = FakeTool("second")
    registry = ToolRegistry()

    registry.register(first)
    registry.register(second)

    assert registry.get("first") is first
    assert registry.get("missing") is None
    assert registry.tools == [first, second]


def test_tool_protocol_is_runtime_checkable():
    assert isinstance(FakeTool("fake"), Tool)


def test_default_registry_local_time_returns_local_iso_datetime():
    tool = build_default_registry().get("local_time")

    assert tool is not None
    value = tool.run({})
    iso_value, separator, timezone_name = value.rpartition(" (")
    parsed = datetime.fromisoformat(iso_value)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() is not None
    assert separator
    assert timezone_name.endswith(")")


def test_default_registry_writes_preferences_and_notes(tmp_path: Path):
    preferences = PreferencesStore(tmp_path / "preferences.yaml")
    episodic = EpisodicStore(tmp_path / "episodic.jsonl")
    registry = build_default_registry(
        preferences=preferences,
        episodic=episodic,
    )

    preference = registry.get("preference_set")
    note = registry.get("note_add")
    assert preference is not None
    assert note is not None
    assert preference.run({"key": "response_length", "value": "short"}) == (
        "Saved preference response_length."
    )
    assert note.run({"text": "Dentist appointment Tuesday"}) == "Saved note."
    assert preferences.get_all() == {"response_length": "short"}
    assert episodic.retrieve("dentist Tuesday", limit=1) == [
        "Dentist appointment Tuesday"
    ]


def test_memory_builtins_report_when_store_is_unavailable():
    registry = build_default_registry()

    preference = registry.get("preference_set")
    note = registry.get("note_add")
    assert preference is not None
    assert note is not None
    assert preference.run({"key": "name", "value": "Jo"}) == (
        "I couldn't save that preference because memory is unavailable."
    )
    assert note.run({"text": "Remember this"}) == (
        "I couldn't save that note because memory is unavailable."
    )


def test_runner_returns_results_in_call_order():
    registry = ToolRegistry()
    registry.register(FakeTool("first", result="one"))
    registry.register(FakeTool("second", result="two"))
    calls = [
        ToolCall("first", {}),
        ToolCall("second", {}),
    ]

    results = ToolRunner(
        registry,
        timeout_ms=100,
        allow_online=False,
    ).run_all(calls)

    assert results == [(calls[0], "one"), (calls[1], "two")]


def test_runner_times_out_without_waiting_for_slow_tool():
    registry = ToolRegistry()
    registry.register(FakeTool("slow", delay_seconds=0.25))
    call = ToolCall("slow", {})
    runner = ToolRunner(registry, timeout_ms=20, allow_online=False)

    started = time.monotonic()
    results = runner.run_all([call])
    elapsed = time.monotonic() - started

    assert results == [(call, "Tool slow timed out.")]
    assert elapsed < 0.15


def test_runner_refuses_online_and_unknown_tools():
    registry = ToolRegistry()
    registry.register(FakeTool("online", offline=False))
    calls = [ToolCall("online", {}), ToolCall("missing", {})]

    results = ToolRunner(
        registry,
        timeout_ms=100,
        allow_online=False,
    ).run_all(calls)

    assert results == [
        (calls[0], "Tool online is unavailable while online tools are disabled."),
        (calls[1], "Unknown tool: missing."),
    ]


def test_runner_converts_tool_errors_to_safe_results():
    class BrokenTool(FakeTool):
        def run(self, args: dict[str, Any]) -> str:
            raise RuntimeError("secret failure")

    registry = ToolRegistry()
    registry.register(BrokenTool("broken"))
    call = ToolCall("broken", {})

    assert ToolRunner(
        registry,
        timeout_ms=100,
        allow_online=False,
    ).run_all([call]) == [(call, "Tool broken failed.")]
