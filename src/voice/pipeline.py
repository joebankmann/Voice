from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from voice.chunker import PhraseChunker
from voice.config import MemoryConfig
from voice.memory import EpisodicStore, PreferencesStore
from voice.memory.intent import extract_remember_intent
from voice.metrics import MetricsSink
from voice.session import (
    ConversationSession,
    SessionEvent,
    SessionEventType,
    SessionState,
)

PipelineEvent = dict[str, Any]
PipelineListener = Callable[[PipelineEvent], None]
MemoryPromptComposer = Callable[..., str]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineMemory:
    config: MemoryConfig
    preferences: PreferencesStore
    episodic: EpisodicStore
    compose_prompt: MemoryPromptComposer


class _ReportedPipelineError(Exception):
    """Mark an exception whose user-facing error event was already emitted."""


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
        metrics: MetricsSink | None = None,
        warmup_tts: bool = False,
        memory: PipelineMemory | None = None,
    ) -> None:
        self.session = session
        self.llm = llm
        self.tts = tts
        self.chunker = chunker
        self.audio = audio
        self.system_prompt = system_prompt
        self.base_system_prompt = system_prompt
        self.metrics = metrics or MetricsSink()
        self.warmup_tts = warmup_tts
        self.memory = memory
        self._listeners: list[PipelineListener] = []
        self._interrupted = threading.Event()
        self._started = False
        self._tts_audio_marked = False

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

    def emit_error(self, stage: str, error: BaseException | str) -> None:
        detail = str(error).strip()
        if not detail and isinstance(error, BaseException):
            detail = type(error).__name__
        self._emit({"type": "error", "text": f"{stage} error: {detail}"})

    def start(self) -> None:
        if self._started:
            return
        self._interrupted.clear()
        if self.warmup_tts:
            try:
                warmup = getattr(self.tts, "warmup", None)
                if callable(warmup):
                    warmup()
                else:
                    self.tts.synthesize("Ready.")
            except Exception as error:
                logger.exception("TTS warmup failed; continuing startup")
                self.emit_error("TTS", error)
        if self.audio is not None:
            self.audio.on_first_playback = self._mark_tts_first_audio
            self.audio.start()
        self._started = True
        self.session.force_state(SessionState.LISTENING)
        self._emit_state()

    def stop(self) -> None:
        self._interrupted.set()
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
        self._interrupted.set()
        for event in self.session.on_user_speech_start():
            self._apply_session_event(event)
        self._emit_state()

    def run_turn(self, transcript: str) -> str:
        events = self.session.on_user_speech_end(transcript)
        cleaned_transcript = transcript.strip()
        if cleaned_transcript:
            self._emit({"type": "user_transcript", "text": cleaned_transcript})
            self._prepare_memory(cleaned_transcript)
        self._emit_state()

        for event in events:
            if event.type == SessionEventType.REQUEST_REPLY:
                return self._stream_reply()
            self._apply_session_event(event)
        return ""

    def _prepare_memory(self, transcript: str) -> None:
        memory = self.memory
        if memory is None or not memory.config.enabled:
            return

        memory_write = extract_remember_intent(transcript)
        if memory_write is not None:
            if memory_write.kind == "preference" and memory_write.key is not None:
                memory.preferences.set(memory_write.key, memory_write.value)
            elif memory_write.kind == "episodic":
                memory.episodic.add(memory_write.value)

        preferences = memory.preferences.get_all()
        episodic_notes = memory.episodic.retrieve(
            transcript,
            limit=memory.config.max_episodic_hits,
        )
        self.system_prompt = memory.compose_prompt(
            preferences=preferences,
            episodic_notes=episodic_notes,
            max_inject_chars=memory.config.max_inject_chars,
        )
        separator = self.base_system_prompt + "\n\n"
        injected_chars = (
            len(self.system_prompt) - len(separator)
            if self.system_prompt.startswith(separator)
            else 0
        )
        self.metrics.mark(
            "memory_inject",
            prefs=len(preferences),
            episodic=len(episodic_notes),
            chars=injected_chars,
        )

    def _stream_reply(self) -> str:
        self._interrupted.clear()
        self.chunker.flush()
        tokens: list[str] = []
        llm_token_marked = False
        self._tts_audio_marked = False
        if self.audio is not None:
            arm = getattr(self.audio, "arm_first_playback", None)
            if callable(arm):
                arm()

        try:
            for token in self.llm.stream_chat(
                self.session.history,
                self.system_prompt,
            ):
                if self._interrupted.is_set():
                    break
                if not llm_token_marked:
                    self.metrics.mark("llm_first_token")
                    llm_token_marked = True
                tokens.append(token)
                partial = "".join(tokens)
                self._emit({"type": "assistant_partial", "text": partial})
                for phrase in self.chunker.push(token):
                    self._speak(phrase)
            if not self._interrupted.is_set():
                for phrase in self.chunker.flush():
                    self._speak(phrase)
        except _ReportedPipelineError:
            return ""
        except Exception as error:
            if not self._interrupted.is_set():
                self._recover_from_error("LLM", error)
                return ""

        assistant_text = "".join(tokens).strip()
        if assistant_text:
            self.session.append_assistant(assistant_text)
            self._emit({"type": "assistant_final", "text": assistant_text})
        if not self._interrupted.is_set():
            for event in self.session.on_assistant_audio_done():
                self._apply_session_event(event)
            self._emit_state()
        return assistant_text

    def _apply_session_event(self, event: SessionEvent) -> None:
        if event.type == SessionEventType.STOP_PLAYBACK:
            self._stop_speech()
        elif event.type == SessionEventType.CANCEL_GENERATION:
            self.llm.cancel()

    def _speak(self, text: str) -> None:
        try:
            speak = getattr(self.tts, "speak", None)
            if callable(speak):
                speak(text)
                # Speak-owned playback has no AudioHub callback; mark handoff.
                if self.audio is None:
                    self._mark_tts_first_audio()
                return

            audio = self.tts.synthesize(text)
            if self.audio is None:
                raise RuntimeError("AudioHub is required for synthesized TTS playback")
            self.audio.play(audio, sample_rate=self.tts.sample_rate)
            # If AudioHub cannot callback (tests / no stream), mark enqueue.
            if not getattr(self.audio, "on_first_playback", None):
                self._mark_tts_first_audio()
        except Exception as error:
            if self._interrupted.is_set():
                raise
            self._recover_from_error("TTS", error)
            raise _ReportedPipelineError from error

    def _recover_from_error(self, stage: str, error: BaseException) -> None:
        self.session.force_state(SessionState.LISTENING)
        self.emit_error(stage, error)

    def _mark_tts_first_audio(self) -> None:
        if self._tts_audio_marked:
            return
        self.metrics.mark("tts_first_audio")
        self._tts_audio_marked = True

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
