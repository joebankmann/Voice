from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class SessionState(Enum):
    IDLE = auto()
    LISTENING = auto()
    THINKING_SPEAKING = auto()


class SessionEventType(Enum):
    START_LISTENING = auto()
    STOP_PLAYBACK = auto()
    CANCEL_GENERATION = auto()
    REQUEST_REPLY = auto()


@dataclass(frozen=True)
class SessionEvent:
    type: SessionEventType
    transcript: str | None = None


class ConversationSession:
    def __init__(self, max_history_messages: int | None = None) -> None:
        self.state = SessionState.IDLE
        self.history: list[dict[str, str]] = []
        self.max_history_messages = max_history_messages
        self._evicted: list[dict[str, str]] = []

    def force_state(self, state: SessionState) -> None:
        self.state = state

    def _trim_history(self) -> list[dict[str, str]]:
        dropped: list[dict[str, str]] = []
        if self.max_history_messages is None or self.max_history_messages <= 0:
            return dropped
        while len(self.history) > self.max_history_messages:
            if (
                len(self.history) >= 2
                and self.history[0].get("role") == "user"
                and self.history[1].get("role") == "assistant"
            ):
                dropped.append(self.history.pop(0))
                dropped.append(self.history.pop(0))
            else:
                dropped.append(self.history.pop(0))
        self._evicted.extend(dropped)
        return dropped

    def take_evicted(self) -> list[dict[str, str]]:
        dropped = self._evicted
        self._evicted = []
        return dropped

    def restore_evicted(self, messages: list[dict[str, str]]) -> None:
        if messages:
            self._evicted = list(messages) + self._evicted

    def on_user_speech_start(self) -> list[SessionEvent]:
        events: list[SessionEvent] = [
            SessionEvent(SessionEventType.STOP_PLAYBACK),
            SessionEvent(SessionEventType.CANCEL_GENERATION),
        ]
        self.state = SessionState.LISTENING
        events.append(SessionEvent(SessionEventType.START_LISTENING))
        return events

    def on_user_speech_end(self, transcript: str) -> list[SessionEvent]:
        text = transcript.strip()
        if not text:
            self.state = SessionState.LISTENING
            return [SessionEvent(SessionEventType.START_LISTENING)]
        self.history.append({"role": "user", "content": text})
        self._trim_history()
        self.state = SessionState.THINKING_SPEAKING
        return [SessionEvent(SessionEventType.REQUEST_REPLY, transcript=text)]

    def on_assistant_audio_done(self) -> list[SessionEvent]:
        self.state = SessionState.LISTENING
        return [SessionEvent(SessionEventType.START_LISTENING)]

    def append_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})
        self._trim_history()
