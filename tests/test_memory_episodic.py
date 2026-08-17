import json
from pathlib import Path

from voice.memory.episodic import EpisodicStore


def test_add_appends_jsonl_records(tmp_path: Path):
    path = tmp_path / "memory" / "episodic.jsonl"
    store = EpisodicStore(path)

    store.add("Joseph's dentist is Tuesday", tags=["appointment"])
    store.add("The car is parked on Oak Street")

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["text"] for record in records] == [
        "Joseph's dentist is Tuesday",
        "The car is parked on Oak Street",
    ]
    assert records[0]["tags"] == ["appointment"]
    assert records[1]["tags"] == []
    assert all(record["ts"] for record in records)


def test_retrieve_ranks_notes_by_case_insensitive_keyword_overlap(tmp_path: Path):
    path = tmp_path / "episodic.jsonl"
    store = EpisodicStore(path)
    store.add("Dentist appointment on Tuesday")
    store.add("Dentist appointment moved to Tuesday afternoon")
    store.add("Pick up groceries")

    assert store.retrieve("TUESDAY afternoon dentist", limit=2) == [
        "Dentist appointment moved to Tuesday afternoon",
        "Dentist appointment on Tuesday",
    ]


def test_retrieve_empty_query_returns_no_notes(tmp_path: Path):
    store = EpisodicStore(tmp_path / "episodic.jsonl")
    store.add("Remember this note")

    assert store.retrieve("", limit=3) == []
    assert store.retrieve("an of", limit=3) == []
