from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from voice.speak_text import sanitize_for_speech

DEFAULT_LATENCY_BUDGETS_MS = {
    "stt": 1500.0,
    "llm_first_token": 800.0,
    "tts_first_audio": 1500.0,
}


@dataclass(frozen=True)
class SanitizeCase:
    source: str
    expected: str


@dataclass(frozen=True)
class SanitizeResult:
    source: str
    expected: str
    actual: str

    @property
    def ok(self) -> bool:
        return self.actual == self.expected


@dataclass(frozen=True)
class LatencyViolation:
    name: str
    duration_ms: float
    max_ms: float


class FakeClock:
    """Monotonic clock that tests can advance without waiting."""

    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def load_sanitize_cases(path: str | Path) -> list[SanitizeCase]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    cases = raw.get("sanitize", raw) if isinstance(raw, dict) else raw
    loaded: list[SanitizeCase] = []
    for item in cases:
        loaded.append(
            SanitizeCase(source=str(item["source"]), expected=str(item["expected"]))
        )
    return loaded


def run_sanitize_eval(cases: list[SanitizeCase]) -> list[SanitizeResult]:
    return [
        SanitizeResult(
            source=case.source,
            expected=case.expected,
            actual=sanitize_for_speech(case.source),
        )
        for case in cases
    ]


def check_latency_budgets(
    events: list[dict[str, Any]],
    budgets_ms: dict[str, float] | None = None,
) -> list[LatencyViolation]:
    """Fail spans (or duration_ms marks) that exceed named ceilings."""
    ceilings = budgets_ms or DEFAULT_LATENCY_BUDGETS_MS
    violations: list[LatencyViolation] = []
    for event in events:
        name = str(event.get("name", ""))
        if name not in ceilings or "duration_ms" not in event:
            continue
        duration_ms = float(event["duration_ms"])
        max_ms = ceilings[name]
        if duration_ms > max_ms:
            violations.append(
                LatencyViolation(name=name, duration_ms=duration_ms, max_ms=max_ms)
            )
    return violations
