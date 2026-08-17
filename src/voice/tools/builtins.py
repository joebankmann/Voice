from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from voice.memory.episodic import EpisodicStore
from voice.memory.preferences import PreferencesStore


@dataclass(frozen=True)
class LocalTimeTool:
    name: str = "local_time"
    description: str = "Get the current local date, time, and timezone."
    parameters_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
    )
    offline: bool = True

    def run(self, args: dict[str, Any]) -> str:
        now = datetime.now().astimezone()
        return f"{now.isoformat()} ({now.tzname() or 'local time'})"


@dataclass(frozen=True)
class PreferenceSetTool:
    preferences: PreferencesStore | None = None
    name: str = "preference_set"
    description: str = "Save a user preference."
    parameters_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["key", "value"],
            "additionalProperties": False,
        }
    )
    offline: bool = True

    def run(self, args: dict[str, Any]) -> str:
        if self.preferences is None:
            return "I couldn't save that preference because memory is unavailable."
        key = args.get("key")
        value = args.get("value")
        if not isinstance(key, str) or not isinstance(value, str):
            return "I couldn't save that preference because its arguments were invalid."
        self.preferences.set(key, value)
        return f"Saved preference {key}."


@dataclass(frozen=True)
class NoteAddTool:
    episodic: EpisodicStore | None = None
    name: str = "note_add"
    description: str = "Save a note to episodic memory."
    parameters_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        }
    )
    offline: bool = True

    def run(self, args: dict[str, Any]) -> str:
        if self.episodic is None:
            return "I couldn't save that note because memory is unavailable."
        text = args.get("text")
        if not isinstance(text, str):
            return "I couldn't save that note because its arguments were invalid."
        self.episodic.add(text)
        return "Saved note."
