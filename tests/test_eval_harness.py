from pathlib import Path

from voice.eval_harness import (
    FakeClock,
    check_latency_budgets,
    load_sanitize_cases,
    run_sanitize_eval,
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
