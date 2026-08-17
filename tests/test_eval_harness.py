import json
from datetime import datetime, timezone
from pathlib import Path

from voice.eval_harness import (
    FakeClock,
    check_latency_budgets,
    load_sanitize_cases,
    main,
    run_sanitize_eval,
    write_eval_snapshot,
)
from voice.metrics import MetricsSink

CASES = Path(__file__).resolve().parent / "eval" / "cases" / "sanitize.yaml"


def test_sanitize_eval_fixtures_pass():
    results = run_sanitize_eval(load_sanitize_cases(CASES))
    failures = [result for result in results if not result.ok]
    assert failures == [], failures


def test_fake_clock_latency_budgets_pass_within_ceiling():
    clock = FakeClock()
    sink = MetricsSink(enabled=True, clock=clock)
    with sink.span("stt"):
        clock.advance(0.4)
    sink.mark("llm_first_token", duration_ms=200.0)
    with sink.span("tts_first_audio"):
        clock.advance(0.9)
    assert check_latency_budgets(sink.events()) == []


def test_fake_clock_latency_budgets_fail_when_over_ceiling():
    clock = FakeClock()
    sink = MetricsSink(enabled=True, clock=clock)
    with sink.span("stt"):
        clock.advance(2.0)
    violations = check_latency_budgets(sink.events())
    assert len(violations) == 1
    assert violations[0].name == "stt"
    assert violations[0].duration_ms >= 2000


def test_latency_budgets_measure_mark_deltas_from_previous_t():
    clock = FakeClock()
    sink = MetricsSink(enabled=True, clock=clock)
    sink.mark("vad_end")
    clock.advance(0.2)
    sink.mark("llm_first_token")
    clock.advance(2.0)
    sink.mark("tts_first_audio")
    violations = check_latency_budgets(sink.events())
    assert [item.name for item in violations] == ["tts_first_audio"]
    assert violations[0].duration_ms >= 2000


def test_write_eval_snapshot_appends_jsonl(tmp_path: Path):
    path = tmp_path / "eval.jsonl"
    stamp = datetime(2026, 8, 17, tzinfo=timezone.utc)
    first = write_eval_snapshot(path, cases_path=CASES, clock=stamp)
    write_eval_snapshot(path, cases_path=CASES, clock=stamp)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    payload = json.loads(lines[0])
    assert payload["sanitize_ok"] is True
    assert payload["sanitize_total"] > 0
    assert payload["latency_violations"] == []
    assert first["ts"] == "2026-08-17T00:00:00+00:00"


def test_eval_harness_cli_appends_snapshot(tmp_path: Path):
    path = tmp_path / "eval.jsonl"
    assert main(["--cases", str(CASES), "--jsonl", str(path)]) == 0
    assert path.read_text(encoding="utf-8").strip()


def test_eval_harness_cli_includes_latency_events(tmp_path: Path):
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        json.dumps({"name": "stt", "duration_ms": 4000.0}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "eval.jsonl"
    assert main(
        ["--cases", str(CASES), "--jsonl", str(out), "--events", str(events_path)]
    ) == 0
    payload = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert payload["latency_violations"][0]["name"] == "stt"
