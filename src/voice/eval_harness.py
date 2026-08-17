from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
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


def write_eval_snapshot(
    jsonl_path: str | Path,
    *,
    cases_path: str | Path,
    events: list[dict[str, Any]] | None = None,
    clock: datetime | None = None,
) -> dict[str, Any]:
    """Append one sanitize + latency snapshot to a JSONL log."""
    results = run_sanitize_eval(load_sanitize_cases(cases_path))
    violations = check_latency_budgets(events or [])
    stamp = clock or datetime.now(timezone.utc)
    snapshot = {
        "ts": stamp.isoformat(),
        "cases": str(cases_path),
        "sanitize_ok": all(result.ok for result in results),
        "sanitize_total": len(results),
        "sanitize_failed": sum(1 for result in results if not result.ok),
        "latency_violations": [
            {
                "name": item.name,
                "duration_ms": item.duration_ms,
                "max_ms": item.max_ms,
            }
            for item in violations
        ],
    }
    path = Path(jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot) + "\n")
    return snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Append a local eval snapshot (sanitize fixtures + optional latency)."
    )
    parser.add_argument("--cases", required=True, help="YAML sanitize fixture path")
    parser.add_argument("--jsonl", required=True, help="Append-only JSONL output path")
    args = parser.parse_args(argv)
    snapshot = write_eval_snapshot(args.jsonl, cases_path=args.cases)
    status = "ok" if snapshot["sanitize_ok"] else "failed"
    print(
        f"eval snapshot {status}: "
        f"{snapshot['sanitize_total'] - snapshot['sanitize_failed']}"
        f"/{snapshot['sanitize_total']} sanitize"
    )
    return 0 if snapshot["sanitize_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
