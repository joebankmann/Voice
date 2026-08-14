from __future__ import annotations

from collections.abc import Callable
from typing import Any

from voice.chunker import PhraseChunker
from voice.session import (
    ConversationSession,
    SessionEvent,
    SessionEventType,
    SessionState,
)

PipelineEvent = dict[str, Any]
PipelineListener = Callable[[PipelineEvent], None]


class VoicePipeline:
    """Coordinate conversation state, streamed generation, and speech output."""

    def __init__(
        self,
        *,
        session: ConversationSession,
        llm: Any,
        tts: Any,
        chunker: PhraseChunker,
        audio: Any | None = None,
        system_prompt: str = "",
    ) -> None:
        self.session = session
        self.llm = llm
        self.tts = tts
        self.chunker = chunker
        self.audio = audio
        self.system_prompt = system_prompt
        self._listeners: list[PipelineListener] = []
        self._interrupted = False
        self._started = False

    def add_listener(self, listener: PipelineListener) -> None:
        self._listeners.append(listener)

    def on_state_change(self, listener: Callable[[str], None]) -> None:
        self._add_typed_listener("state", "state", listener)

    def on_user_transcript(self, listener: Callable[[str], None]) -> None:
        self._add_typed_listener("user_transcript", "text", listener)

    def on_assistant_partial(self, listener: Callable[[str], None]) -> None:
        self._add_typed_listener("assistant_partial", "text", listener)

    def on_assistant_final(self, listener: Callable[[str], None]) -> None:
        self._add_typed_listener("assistant_final", "text", listener)

    def start(self) -> None:
        if self._started:
            return
        if self.audio is not None:
            self.audio.start()
        self._started = True
        self.session.force_state(SessionState.LISTENING)
        self._emit_state()

    def stop(self) -> None:
        self._interrupted = True
        self.llm.cancel()
        self._stop_speech()
        if self.audio is not None and self._started:
            self.audio.close()
        self._started = False
        self.session.force_state(SessionState.IDLE)
        self._emit_state()

    def interrupt(self) -> None:
        self.handle_speech_start()

    def handle_speech_start(self) -> None:
        self._interrupted = True
        for event in self.session.on_user_speech_start():
            self._apply_session_event(event)
        self._emit_state()

    def run_turn(self, transcript: str) -> str:
        events = self.session.on_user_speech_end(transcript)
        cleaned_transcript = transcript.strip()
        if cleaned_transcript:
            self._emit({"type": "user_transcript", "text": cleaned_transcript})
        self._emit_state()

        for event in events:
            if event.type == SessionEventType.REQUEST_REPLY:
                return self._stream_reply()
            self._apply_session_event(event)
        return ""

    def _stream_reply(self) -> str:
        self._interrupted = False
        self.chunker.flush()
        tokens: list[str] = []

        try:
            for token in self.llm.stream_chat(
                self.session.history,
                self.system_prompt,
            ):
                if self._interrupted:
                    break
                tokens.append(token)
                partial = "".join(tokens)
                self._emit({"type": "assistant_partial", "text": partial})
                for phrase in self.chunker.push(token):
                    self._speak(phrase)
            if not self._interrupted:
                for phrase in self.chunker.flush():
                    self._speak(phrase)
        except Exception:
            if not self._interrupted:
                raise

        assistant_text = "".join(tokens).strip()
        if assistant_text:
            self.session.append_assistant(assistant_text)
            self._emit({"type": "assistant_final", "text": assistant_text})
        return assistant_text

    def _apply_session_event(self, event: SessionEvent) -> None:
        if event.type == SessionEventType.STOP_PLAYBACK:
            self._stop_speech()
        elif event.type == SessionEventType.CANCEL_GENERATION:
            self.llm.cancel()

    def _speak(self, text: str) -> None:
        speak = getattr(self.tts, "speak", None)
        if callable(speak):
            speak(text)
            return

        audio = self.tts.synthesize(text)
        if self.audio is None:
            raise RuntimeError("AudioHub is required for synthesized TTS playback")
        self.audio.play(audio, sample_rate=self.tts.sample_rate)

    def _stop_speech(self) -> None:
        stop = getattr(self.tts, "stop", None)
        if callable(stop):
            stop()
        if self.audio is not None:
            self.audio.stop_playback()

    def _add_typed_listener(
        self,
        event_type: str,
        value_key: str,
        listener: Callable[[str], None],
    ) -> None:
        def dispatch(event: PipelineEvent) -> None:
            if event.get("type") == event_type:
                listener(str(event[value_key]))

        self.add_listener(dispatch)

    def _emit_state(self) -> None:
        self._emit({"type": "state", "state": self.session.state.name})

    def _emit(self, event: PipelineEvent) -> None:
        for listener in tuple(self._listeners):
            listener(event)
