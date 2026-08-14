from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

import numpy as np

from voice.audio_io import AudioHub
from voice.chunker import PhraseChunker
from voice.config import AppConfig, load_config
from voice.llm import LlmClient
from voice.pipeline import VoicePipeline
from voice.session import ConversationSession
from voice.stt import SttEngine
from voice.tts import TtsEngine
from voice.vad import VadEngine, create_default_vad


def build_pipeline(config: AppConfig, config_dir: Path) -> VoicePipeline:
    audio = AudioHub(sample_rate=config.audio.sample_rate)
    tts = TtsEngine(
        config.tts.piper_bin,
        config.tts.voice_path,
        sample_rate=config.audio.sample_rate,
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


def run_cli(
    pipeline: VoicePipeline,
    *,
    config: AppConfig,
    stt: SttEngine,
    vad: VadEngine,
) -> None:
    audio = pipeline.audio
    if audio is None:
        raise RuntimeError("CLI mode requires an AudioHub")

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

    pipeline.start()
    print("Listening. Press Ctrl-C to stop.")
    try:
        while True:
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
                    reply = pipeline.run_turn(transcript)
                    if reply:
                        print(f"Assistant: {reply}")
            utterance = []
            silence_frames = 0
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        pipeline.stop()


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

    if not args.cli:
        try:
            ui: Any = importlib.import_module("voice.ui")
        except ModuleNotFoundError as exc:
            if exc.name != "voice.ui":
                raise
            print("Desktop UI arrives in Task 7; falling back to --cli mode.")
        else:
            ui.run_app(pipeline)
            return

    run_cli(
        pipeline,
        config=config,
        stt=SttEngine(config.stt.whisper_bin, config.stt.model_path),
        vad=create_default_vad(
            threshold=config.vad.threshold,
            sample_rate=config.audio.sample_rate,
        ),
    )


if __name__ == "__main__":
    main()
