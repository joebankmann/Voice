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
    def __init__(self) -> None:
        self.state = SessionState.IDLE
        self.history: list[dict[str, str]] = []

    def force_state(self, state: SessionState) -> None:
        self.state = state

    def on_user_speech_start(self) -> list[SessionEvent]:
        events: list[SessionEvent] = []
        if self.state == SessionState.THINKING_SPEAKING:
            events.append(SessionEvent(SessionEventType.STOP_PLAYBACK))
            events.append(SessionEvent(SessionEventType.CANCEL_GENERATION))
        self.state = SessionState.LISTENING
        events.append(SessionEvent(SessionEventType.START_LISTENING))
        return events

    def on_user_speech_end(self, transcript: str) -> list[SessionEvent]:
        text = transcript.strip()
        if not text:
            self.state = SessionState.LISTENING
            return [SessionEvent(SessionEventType.START_LISTENING)]
        self.history.append({"role": "user", "content": text})
        self.state = SessionState.THINKING_SPEAKING
        return [SessionEvent(SessionEventType.REQUEST_REPLY, transcript=text)]

    def on_assistant_audio_done(self) -> list[SessionEvent]:
        self.state = SessionState.LISTENING
        return [SessionEvent(SessionEventType.START_LISTENING)]

    def append_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})
