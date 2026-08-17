from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any

SUMMARIZE_SYSTEM_PROMPT = (
    "Summarize the dropped conversation turns as one short factual note "
    "the assistant should remember. No markdown, lists, or quotes. "
    "If nothing durable is worth remembering, reply with NONE."
)


class HelperAgent:
    """Off-path specialist: summarize evicted history. Never on the TTS hot path."""

    def __init__(
        self,
        llm: Any,
        *,
        timeout_ms: int = 4000,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self._llm = llm
        self._timeout_s = max(0.001, timeout_ms / 1000.0)
        self._cancel_event = cancel_event or threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()
        cancel = getattr(self._llm, "cancel", None)
        if callable(cancel):
            cancel()

    def summarize_dropped(self, messages: list[dict[str, str]]) -> str | None:
        if not messages or self._cancel_event.is_set():
            return None
        transcript = "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '').strip()}"
            for item in messages
            if str(item.get("content", "")).strip()
        )
        if not transcript:
            return None
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                self._llm.complete,
                [{"role": "user", "content": transcript}],
                SUMMARIZE_SYSTEM_PROMPT,
            )
            try:
                raw = future.result(timeout=self._timeout_s)
            except FuturesTimeout:
                self.cancel()
                return None
            except Exception:
                return None
        if self._cancel_event.is_set():
            return None
        note = str(raw).strip()
        if not note or note.upper() == "NONE":
            return None
        return note.splitlines()[0].strip()

    def begin_turn(self) -> None:
        self._cancel_event.clear()
