from __future__ import annotations

import argparse
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from voice.audio_io import AudioHub
from voice.chunker import PhraseChunker
from voice.config import AppConfig, load_config
from voice.llm import LlmClient
from voice.pipeline import VoicePipeline
from voice.session import ConversationSession
from voice.stt import SttEngine
from voice.tts import TtsEngine
from voice.ui import run_app
from voice.vad import VadEngine, create_default_vad


def build_pipeline(config: AppConfig, config_dir: Path) -> VoicePipeline:
    audio = AudioHub(sample_rate=config.audio.sample_rate)
    tts = TtsEngine(
        config.tts.piper_bin,
        config.tts.voice_path,
        sample_rate=config.tts.sample_rate,
    )
    system_prompt_path = config_dir / config.llm.system_prompt_path
    return VoicePipeline(
        session=ConversationSession(),
        llm=LlmClient(
            config.llm.base_url,
            config.llm.model,
            config.llm.temperature,
        ),
        tts=tts,
        chunker=PhraseChunker(),
        audio=audio,
        system_prompt=system_prompt_path.read_text().strip(),
    )


def run_listen_loop(
    pipeline: VoicePipeline,
    *,
    config: AppConfig,
    stt: SttEngine,
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
                reply = pending_turns.popleft().result()
                if reply:
                    print(f"Assistant: {reply}")

            frame = audio.read_frame()
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
                samples = np.concatenate(utterance)
                pcm16 = (
                    np.clip(samples, -1.0, 1.0) * np.iinfo(np.int16).max
                ).astype("<i2").tobytes()
                transcript = stt.transcribe(pcm16, config.audio.sample_rate)
                if transcript:
                    print(f"You: {transcript}")
                    pending_turns.append(
                        turn_executor.submit(pipeline.run_turn, transcript)
                    )
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
    stt: SttEngine,
    vad: VadEngine,
) -> None:
    stop_event = threading.Event()
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


class ListenLoopThread:
    def __init__(
        self,
        pipeline: VoicePipeline,
        *,
        config: AppConfig,
        stt: SttEngine,
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
        self.stop_event.clear()
        self.pipeline.start()
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
            self.thread.join(timeout=0.5)
        self.pipeline.stop()


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
    stt = SttEngine(config.stt.whisper_bin, config.stt.model_path)
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
        run_app(
            pipeline,
            on_start_listening=listen_loop.start,
            on_stop_listening=listen_loop.stop,
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
