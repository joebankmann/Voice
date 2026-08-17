from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"\w+", text.casefold()) if len(token) >= 3}


class EpisodicStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def add(self, text: str, tags: list[str] | None = None) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "text": text,
            "tags": list(tags or []),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as episodic_file:
            episodic_file.write(json.dumps(record) + "\n")

    def retrieve(self, query: str, *, limit: int) -> list[str]:
        query_tokens = _tokens(query)
        if not query_tokens or limit <= 0 or not self.path.exists():
            return []

        scored_notes: list[tuple[int, str]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            text = str(json.loads(line)["text"])
            score = len(query_tokens & _tokens(text))
            if score:
                scored_notes.append((score, text))

        scored_notes.sort(key=lambda note: note[0], reverse=True)
        return [text for _, text in scored_notes[:limit]]
