from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from voice.affect import estimate_affect
from voice.chunker import PhraseChunker
from voice.config import AgentsConfig, FutureConfig, MemoryConfig, ToolsConfig
from voice.inbox import consume_inbox
from voice.memory import EpisodicStore, PreferencesStore
from voice.memory.intent import extract_remember_intent
from voice.metrics import MetricsSink
from voice.prompting import adapt_personality, append_memory_inject
from voice.session import (
    ConversationSession,
    SessionEvent,
    SessionEventType,
    SessionState,
)
from voice.tools.base import ToolCall
from voice.tools.markers import extract_tool_calls, strip_tool_markers
from voice.tools.registry import ToolRegistry
from voice.tools.runner import ToolRunner

PipelineEvent = dict[str, Any]
PipelineListener = Callable[[PipelineEvent], None]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineMemory:
    config: MemoryConfig
    preferences: PreferencesStore
    episodic: EpisodicStore


@dataclass(frozen=True)
class PipelineTools:
    config: ToolsConfig
    registry: ToolRegistry
    runner: ToolRunner


@dataclass(frozen=True)
class PipelineAgents:
    config: AgentsConfig
    helper: Any


@dataclass(frozen=True)
class PipelineFuture:
    config: FutureConfig
    inbox_dir: str


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
        tools: PipelineTools | None = None,
        agents: PipelineAgents | None = None,
        future: PipelineFuture | None = None,
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
        self.tools = tools
        self.agents = agents
        self.future = future
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
        self._cancel_helper()
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
        self._cancel_helper()
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

        # Barge-in during memory I/O must not start a new reply.
        if self._interrupted.is_set():
            return ""

        for event in events:
            if event.type == SessionEventType.REQUEST_REPLY:
                return self._stream_reply()
            self._apply_session_event(event)
        return ""

    def _prepare_memory(self, transcript: str) -> None:
        memory = self.memory
        if memory is not None and memory.config.enabled:
            memory_write = extract_remember_intent(transcript)
            if memory_write is not None:
                if memory_write.kind == "preference" and memory_write.key is not None:
                    memory.preferences.set(memory_write.key, memory_write.value)
                elif memory_write.kind == "episodic":
                    memory.episodic.add(memory_write.value)

        preferences: dict[str, str] = {}
        episodic_notes: list[str] = []
        max_inject = 1200
        if memory is not None and memory.config.enabled:
            preferences = memory.preferences.get_all()
            episodic_notes = memory.episodic.retrieve(
                transcript,
                limit=memory.config.max_episodic_hits,
            )
            max_inject = memory.config.max_inject_chars

        extra_sections: list[str] = []
        future = self.future
        if future is not None and future.config.affect:
            label = estimate_affect(transcript)
            if label:
                extra_sections.append(f"User affect hint: {label}.")
                self.metrics.mark("affect", label=label)
        if future is not None and future.config.inbox:
            for note in consume_inbox(future.inbox_dir):
                extra_sections.append(f"Shared note: {note}")

        prompt = self.base_system_prompt
        if future is not None and future.config.adaptive_personality:
            prompt = adapt_personality(prompt, preferences)

        if (
            memory is None or not memory.config.enabled
        ) and not extra_sections and prompt == self.base_system_prompt:
            return

        self.system_prompt, injected_chars = append_memory_inject(
            prompt,
            preferences=preferences or None,
            episodic_notes=episodic_notes or None,
            extra_sections=extra_sections or None,
            max_inject_chars=max_inject,
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
            pending_speech = ""
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
                visible, pending_speech = self._filter_tool_marker_stream(
                    pending_speech + token
                )
                for phrase in self.chunker.push(visible):
                    self._speak(phrase)
            if not self._interrupted.is_set():
                for phrase in self.chunker.push(strip_tool_markers(pending_speech)):
                    self._speak(phrase)
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
            self._emit(
                {
                    "type": "assistant_final",
                    "text": strip_tool_markers(assistant_text).strip(),
                }
            )
        continuation_text = ""
        if not self._interrupted.is_set():
            continuation_text = self._run_tool_continuation(assistant_text)
        if not self._interrupted.is_set():
            self._summarize_evicted_history()
        if not self._interrupted.is_set():
            for event in self.session.on_assistant_audio_done():
                self._apply_session_event(event)
            self._emit_state()
        spoken_text = " ".join(
            part
            for part in (
                strip_tool_markers(assistant_text).strip(),
                strip_tool_markers(continuation_text).strip(),
            )
            if part
        )
        return spoken_text

    def _summarize_evicted_history(self) -> None:
        dropped = self.session.take_evicted()
        agents = self.agents
        memory = self.memory
        if (
            agents is None
            or not agents.config.enabled
            or memory is None
            or not memory.config.enabled
            or not dropped
        ):
            return
        reset = getattr(agents.helper, "begin_turn", None)
        if callable(reset):
            reset()
        started_at = time.perf_counter()
        note: str | None = None
        try:
            note = agents.helper.summarize_dropped(dropped)
        except Exception as error:
            self.emit_error("Agent", error)
        duration_ms = (time.perf_counter() - started_at) * 1000.0
        if not self._interrupted.is_set() and note:
            memory.episodic.add(note)
            self.metrics.mark(
                "agent_summarize",
                ok=True,
                duration_ms=duration_ms,
                chars=len(note),
            )
        else:
            self.metrics.mark(
                "agent_summarize",
                ok=False,
                duration_ms=duration_ms,
                chars=0,
            )
        if self._interrupted.is_set() or not agents.config.collaborative:
            return
        reset = getattr(agents.helper, "begin_turn", None)
        if callable(reset):
            reset()
        extract = getattr(agents.helper, "extract_actions", None)
        if not callable(extract):
            return
        started_at = time.perf_counter()
        action: str | None = None
        try:
            action = extract(dropped)
        except Exception as error:
            self.emit_error("Agent", error)
        action_ms = (time.perf_counter() - started_at) * 1000.0
        if self._interrupted.is_set() or not action:
            self.metrics.mark(
                "agent_actions",
                ok=False,
                duration_ms=action_ms,
                chars=0,
            )
            return
        memory.episodic.add(action)
        self.metrics.mark(
            "agent_actions",
            ok=True,
            duration_ms=action_ms,
            chars=len(action),
        )

    def _cancel_helper(self) -> None:
        if self.agents is None:
            return
        cancel = getattr(self.agents.helper, "cancel", None)
        if callable(cancel):
            cancel()

    def _run_tool_continuation(self, assistant_text: str) -> str:
        tools = self.tools
        if tools is None or not tools.config.enabled:
            return ""
        calls = extract_tool_calls(assistant_text)
        if not calls or self._interrupted.is_set():
            return ""

        results: list[tuple[ToolCall, str]] = []
        for call in calls:
            if self._interrupted.is_set():
                return ""
            started_at = time.perf_counter()
            result = tools.runner.run_all([call])[0][1]
            duration_ms = (time.perf_counter() - started_at) * 1000.0
            ok = not (
                result.startswith("Unknown tool:")
                or (result.startswith("Tool ") and (
                    result.endswith(" timed out.")
                    or result.endswith(" failed.")
                    or " is unavailable " in result
                ))
            )
            self.metrics.mark(
                "tool_call",
                tool=call.name,
                ok=ok,
                duration_ms=duration_ms,
            )
            results.append((call, result))

        if self._interrupted.is_set():
            return ""
        result_message = {
            "role": "user",
            "content": "Tool results:\n"
            + "\n".join(f"{call.name}: {result}" for call, result in results),
        }
        self.session.history.append(result_message)
        self.session._trim_history()
        if self._interrupted.is_set():
            return ""
        return self._stream_continuation()

    def _stream_continuation(self) -> str:
        self.chunker.flush()
        tokens: list[str] = []
        pending_speech = ""
        try:
            for token in self.llm.stream_chat(
                self.session.history,
                self.system_prompt,
            ):
                if self._interrupted.is_set():
                    break
                tokens.append(token)
                visible, pending_speech = self._filter_tool_marker_stream(
                    pending_speech + token
                )
                for phrase in self.chunker.push(visible):
                    self._speak(phrase)
            if not self._interrupted.is_set():
                for phrase in self.chunker.push(strip_tool_markers(pending_speech)):
                    self._speak(phrase)
                for phrase in self.chunker.flush():
                    self._speak(phrase)
        except _ReportedPipelineError:
            return ""
        except Exception as error:
            if not self._interrupted.is_set():
                self._recover_from_error("LLM", error)
            return ""

        continuation_text = "".join(tokens).strip()
        if continuation_text:
            self.session.append_assistant(continuation_text)
            self._emit(
                {
                    "type": "assistant_final",
                    "text": strip_tool_markers(continuation_text).strip(),
                }
            )
        return continuation_text

    @staticmethod
    def _filter_tool_marker_stream(text: str) -> tuple[str, str]:
        visible: list[str] = []
        marker_prefix = "<<tool:"
        while text:
            marker_start = text.find(marker_prefix)
            if marker_start >= 0:
                visible.append(text[:marker_start])
                marker_end = text.find(">>", marker_start + len(marker_prefix))
                if marker_end < 0:
                    return "".join(visible), text[marker_start:]
                text = text[marker_end + 2 :]
                continue

            protected_chars = 0
            for length in range(1, min(len(text), len(marker_prefix) - 1) + 1):
                if text.endswith(marker_prefix[:length]):
                    protected_chars = length
            if protected_chars:
                visible.append(text[:-protected_chars])
                return "".join(visible), text[-protected_chars:]
            visible.append(text)
            return "".join(visible), ""
        return "".join(visible), ""

    def _apply_session_event(self, event: SessionEvent) -> None:
        if event.type == SessionEventType.STOP_PLAYBACK:
            self._stop_speech()
        elif event.type == SessionEventType.CANCEL_GENERATION:
            self.llm.cancel()

    def _speak(self, text: str) -> None:
        if self._interrupted.is_set():
            return
        text = strip_tool_markers(text).strip()
        if not text:
            return
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
