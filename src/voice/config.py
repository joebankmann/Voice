from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int
    end_of_turn_silence_ms: int


@dataclass(frozen=True)
class LlmConfig:
    base_url: str
    model: str
    temperature: float
    system_prompt_path: str


@dataclass(frozen=True)
class SttConfig:
    whisper_bin: str
    model_path: str


@dataclass(frozen=True)
class TtsConfig:
    piper_bin: str
    voice_path: str
    sample_rate: int = 22_050
    voices_dir: str = "models"
    length_scale: float = 1.0


@dataclass(frozen=True)
class VadConfig:
    threshold: float
    min_speech_ms: int


@dataclass(frozen=True)
class TelemetryConfig:
    enabled: bool = False
    log_path: str = ""


@dataclass(frozen=True)
class AppConfig:
    audio: AudioConfig
    llm: LlmConfig
    stt: SttConfig
    tts: TtsConfig
    vad: VadConfig
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)


def load_config(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return AppConfig(
        audio=AudioConfig(**raw["audio"]),
        llm=LlmConfig(**raw["llm"]),
        stt=SttConfig(**raw["stt"]),
        tts=TtsConfig(**raw["tts"]),
        vad=VadConfig(**raw["vad"]),
        telemetry=TelemetryConfig(**raw.get("telemetry", {})),
    )
