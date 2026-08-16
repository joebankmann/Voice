import json
from pathlib import Path

from voice.metrics import MetricsSink


def test_span_records_duration_ms():
    sink = MetricsSink(enabled=True)

    with sink.span("stt"):
        pass

    events = sink.events()
    assert any(event["name"] == "stt" and "duration_ms" in event for event in events)


def test_disabled_sink_does_not_record_events():
    sink = MetricsSink(enabled=False)

    sink.mark("vad_end")
    with sink.span("stt"):
        pass

    assert sink.events() == []


def test_events_are_appended_as_jsonl(tmp_path: Path):
    log_path = tmp_path / "metrics" / "latency.jsonl"
    sink = MetricsSink(enabled=True, log_path=log_path)

    sink.mark("vad_end", turn_id=7)

    records = [
        json.loads(line) for line in log_path.read_text().splitlines()
    ]
    assert records[0]["name"] == "vad_end"
    assert records[0]["turn_id"] == 7
