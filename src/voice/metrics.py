from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class MetricsSink:
    """Record local pipeline telemetry in memory and optionally as JSONL."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        log_path: str | Path | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.enabled = enabled
        self.log_path = Path(log_path) if log_path else None
        self._clock = clock
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)

    def mark(self, name: str, **fields: Any) -> None:
        if not self.enabled:
            return
        self._record({"name": name, "t": self._clock(), **fields})

    @contextmanager
    def span(self, name: str, **fields: Any) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        started_at = self._clock()
        try:
            yield
        finally:
            self._record(
                {
                    "name": name,
                    "duration_ms": (self._clock() - started_at) * 1000.0,
                    **fields,
                }
            )

    def _record(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(event)
            if self.log_path is None:
                return
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(event) + "\n")
