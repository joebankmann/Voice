from __future__ import annotations

import argparse
import queue
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from voice.agents import HelperAgent
from voice.audio_io import AudioHub
from voice.chunker import PhraseChunker
from voice.config import AppConfig, load_config
from voice.llm import LlmClient
from voice.memory import EpisodicStore, PreferencesStore
from voice.metrics import MetricsSink
from voice.pipeline import PipelineAgents, PipelineMemory, PipelineTools, VoicePipeline
from voice.prompting import append_personality, append_tools_section, build_system_prompt
from voice.profiles import discover_profile_packs, resolve_profile_pack
from voice.session import ConversationSession
from voice.stt import SttBackend, build_stt
from voice.tools import ToolRunner, build_default_registry
from voice.tts_factory import create_tts_engine
from voice.ui import run_app
from voice.vad import VadEngine, create_default_vad
from voice.voices import discover_voices, resolve_voice


def _resolve_path(config_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_dir / path


def build_pipeline(config: AppConfig, config_dir: Path) -> VoicePipeline:
    audio = AudioHub(sample_rate=config.audio.sample_rate)
    voices = discover_voices(
        _resolve_path(config_dir, config.tts.voices_dir),
        clones_dir=_resolve_path(config_dir, config.tts.clones_dir),
        default_sample_rate=config.tts.sample_rate,
    )
    selected = resolve_voice(voices, config.tts.voice_path)
    if selected is None and config.tts.clone_ref_wav:
        selected = resolve_voice(voices, Path(config.tts.clone_ref_wav).parent.name)
    system_prompt_path = _resolve_path(config_dir, config.llm.system_prompt_path)
    world_context_path = _resolve_path(config_dir, config.llm.world_context_path)
    system_prompt = build_system_prompt(
        system_prompt_path,
        world_context_path=world_context_path,
    )
    packs = discover_profile_packs(
        _resolve_path(config_dir, config.profiles.packs_dir)
    )
    active_pack = resolve_profile_pack(packs, config.profiles.active)
    if active_pack is not None:
        system_prompt = append_personality(system_prompt, active_pack.personality_text)
        if active_pack.voice:
            packed_voice = resolve_voice(voices, active_pack.voice)
            if packed_voice is not None:
                selected = packed_voice
    tts = create_tts_engine(config, selected=selected, config_dir=config_dir)
    memory = None
    max_history_messages = None
    if config.memory.enabled:
        memory = PipelineMemory(
            config=config.memory,
            preferences=PreferencesStore(
                _resolve_path(config_dir, config.memory.preferences_path)
            ),
            episodic=EpisodicStore(
                _resolve_path(config_dir, config.memory.episodic_path)
            ),
        )
        max_history_messages = config.memory.max_history_messages
    tools = None
    if config.tools.enabled:
        registry = build_default_registry(
            preferences=memory.preferences if memory is not None else None,
            episodic=memory.episodic if memory is not None else None,
        )
        system_prompt = append_tools_section(system_prompt, registry.tools)
        tools = PipelineTools(
            config=config.tools,
            registry=registry,
            runner=ToolRunner(
                registry,
                timeout_ms=config.tools.timeout_ms,
                allow_online=config.tools.allow_online,
            ),
        )
    telemetry_log_path = (
        _resolve_path(config_dir, config.telemetry.log_path)
        if config.telemetry.log_path
        else None
    )
    llm = LlmClient(
        config.llm.base_url,
        config.llm.model,
        config.llm.temperature,
    )
    agents = None
    if config.agents.enabled:
        helper_llm = llm
        if config.agents.helper_base_url:
            helper_llm = LlmClient(
                config.agents.helper_base_url,
                config.agents.helper_model or config.llm.model,
                config.llm.temperature,
            )
        agents = PipelineAgents(
            config=config.agents,
            helper=HelperAgent(
                helper_llm,
                timeout_ms=config.agents.timeout_ms,
            ),
        )
    return VoicePipeline(
        session=ConversationSession(max_history_messages=max_history_messages),
        llm=llm,
        tts=tts,
        chunker=PhraseChunker(),
        audio=audio,
        system_prompt=system_prompt,
        metrics=MetricsSink(
            enabled=config.telemetry.enabled,
            log_path=telemetry_log_path,
        ),
        warmup_tts=config.tts.warmup_on_start,
        memory=memory,
        tools=tools,
        agents=agents,
    )


def run_listen_loop(
    pipeline: VoicePipeline,
    *,
    config: AppConfig,
    stt: SttBackend,
    vad: VadEngine,
    stop_event: threading.Event,
) -> None:
    audio = pipeline.audio
    if audio is None:
        raise RuntimeError("Listening requires an AudioHub")

    silence_frames_needed = max(
        1,
        round(
            config.audio.end_of_turn_silence_ms
            * config.audio.sample_rate
            / (1000 * audio.frame_samples)
        ),
    )
    speech_frames_needed = max(
        1,
        round(
            config.vad.min_speech_ms
            * config.audio.sample_rate
            / (1000 * audio.frame_samples)
        ),
    )
    utterance: list[np.ndarray] = []
    silence_frames = 0
    pending_turns: deque[Future[str]] = deque()
    turn_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice-turn")

    try:
        while not stop_event.is_set():
            while pending_turns and pending_turns[0].done():
                try:
                    reply = pending_turns.popleft().result()
                except Exception as error:
                    pipeline.emit_error("Pipeline", error)
                    continue
                if reply:
                    print(f"Assistant: {reply}")

            try:
                frame = audio.read_frame(timeout=0.2)
            except queue.Empty:
                continue
            if vad.is_speech(frame):
                if not utterance:
                    pipeline.handle_speech_start()
                utterance.append(frame)
                silence_frames = 0
                continue

            if not utterance:
                continue
            silence_frames += 1
            if silence_frames < silence_frames_needed:
                utterance.append(frame)
                continue

            if len(utterance) >= speech_frames_needed:
                pipeline.metrics.mark("vad_end")
                samples = np.concatenate(utterance)
                pcm16 = (
                    np.clip(samples, -1.0, 1.0) * np.iinfo(np.int16).max
                ).astype("<i2").tobytes()
                try:
                    with pipeline.metrics.span("stt"):
                        transcript = stt.transcribe(pcm16, config.audio.sample_rate)
                    if transcript:
                        print(f"You: {transcript}")
                        pending_turns.append(
                            turn_executor.submit(pipeline.run_turn, transcript)
                        )
                except Exception as error:
                    pipeline.emit_error("STT", error)
            utterance = []
            silence_frames = 0
    finally:
        for pending_turn in pending_turns:
            pending_turn.cancel()
        turn_executor.shutdown(wait=True, cancel_futures=True)


def run_cli(
    pipeline: VoicePipeline,
    *,
    config: AppConfig,
    stt: SttBackend,
    vad: VadEngine,
) -> None:
    stop_event = threading.Event()
    stt.start()
    pipeline.start()
    print("Listening. Press Ctrl-C to stop.")
    try:
        run_listen_loop(
            pipeline,
            config=config,
            stt=stt,
            vad=vad,
            stop_event=stop_event,
        )
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        stop_event.set()
        pipeline.stop()
        stt.stop()


class ListenLoopThread:
    def __init__(
        self,
        pipeline: VoicePipeline,
        *,
        config: AppConfig,
        stt: SttBackend,
        vad: VadEngine,
    ) -> None:
        self.pipeline = pipeline
        self.config = config
        self.stt = stt
        self.vad = vad
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event = threading.Event()
        try:
            self.stt.start()
        except Exception as error:
            self.pipeline.emit_error("STT", error)
            try:
                self.stt.stop()
            except Exception:
                pass
            return
        try:
            self.pipeline.start()
        except Exception as error:
            self.pipeline.emit_error("Pipeline", error)
            try:
                self.stt.stop()
            except Exception:
                pass
            return
        self.thread = threading.Thread(
            target=run_listen_loop,
            kwargs={
                "pipeline": self.pipeline,
                "config": self.config,
                "stt": self.stt,
                "vad": self.vad,
                "stop_event": self.stop_event,
            },
            name="voice-listen",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
        self.pipeline.stop()
        self.stt.stop()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local streaming voice chat")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="path to the YAML configuration file",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="run the headless microphone loop",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    pipeline = build_pipeline(config, config_path.parent)
    stt = build_stt(config.stt, config_dir=config_path.parent)
    vad = create_default_vad(
        threshold=config.vad.threshold,
        sample_rate=config.audio.sample_rate,
    )

    if not args.cli:
        listen_loop = ListenLoopThread(
            pipeline,
            config=config,
            stt=stt,
            vad=vad,
        )
        voices = discover_voices(
            _resolve_path(config_path.parent, config.tts.voices_dir),
            clones_dir=_resolve_path(config_path.parent, config.tts.clones_dir),
            default_sample_rate=config.tts.sample_rate,
        )
        selected_name = ""
        matched = resolve_voice(voices, Path(pipeline.tts.voice_path))
        if matched is not None:
            selected_name = matched.name
        elif voices:
            selected_name = voices[0].name
        run_app(
            pipeline,
            on_start_listening=listen_loop.start,
            on_stop_listening=listen_loop.stop,
            voices=voices,
            selected_voice=selected_name,
            on_voice_selected=lambda voice: pipeline.tts.apply_voice_info(
                voice,
                length_scale=config.tts.length_scale,
            ),
        )
        return

    run_cli(
        pipeline,
        config=config,
        stt=stt,
        vad=vad,
    )


if __name__ == "__main__":
    main()
