from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any

SUMMARIZE_SYSTEM_PROMPT = (
    "Summarize the dropped conversation turns as one short factual note "
    "the assistant should remember. No markdown, lists, or quotes. "
    "If nothing durable is worth remembering, reply with NONE."
)


ACTIONS_SYSTEM_PROMPT = (
    "From the dropped conversation turns, extract at most one short action item "
    "the assistant should remember. No markdown. If none, reply with NONE."
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
        return self._complete_note(messages, SUMMARIZE_SYSTEM_PROMPT)

    def extract_actions(self, messages: list[dict[str, str]]) -> str | None:
        return self._complete_note(messages, ACTIONS_SYSTEM_PROMPT)

    def begin_turn(self) -> None:
        self._cancel_event.clear()

    def _complete_note(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
    ) -> str | None:
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
                system_prompt,
            )
            deadline = time.monotonic() + self._timeout_s
            raw: Any = None
            try:
                while True:
                    if self._cancel_event.is_set():
                        self.cancel()
                        return None
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self.cancel()
                        return None
                    try:
                        raw = future.result(timeout=min(0.05, remaining))
                        break
                    except FuturesTimeout:
                        continue
            except Exception:
                return None
        if self._cancel_event.is_set():
            return None
        note = str(raw).strip()
        if not note or note.upper() == "NONE":
            return None
        return note.splitlines()[0].strip()
