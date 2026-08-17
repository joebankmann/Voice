from __future__ import annotations

from pathlib import Path

import yaml


class PreferencesStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._preferences: dict[str, str] = {}
        self._notes: list[str] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self._preferences = {}
            self._notes = []
            return

        raw = yaml.safe_load(self.path.read_text()) or {}
        preferences = raw.get("prefs", {})
        notes = raw.get("notes", [])
        self._preferences = {
            str(key): str(value) for key, value in preferences.items()
        }
        self._notes = [str(note) for note in notes]

    def get_all(self) -> dict[str, str]:
        return dict(self._preferences)

    def set(self, key: str, value: str) -> None:
        self._preferences[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            yaml.safe_dump(
                {"prefs": self._preferences, "notes": self._notes},
                sort_keys=True,
            )
        )
