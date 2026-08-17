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
    mode: str = "resident"
    whisper_server_bin: str = "whisper-server"
    server_host: str = "127.0.0.1"
    server_port: int = 8178
    inference_path: str = "/inference"
    manage_server: bool = True


@dataclass(frozen=True)
class TtsConfig:
    piper_bin: str
    voice_path: str
    sample_rate: int = 22_050
    voices_dir: str = "models"
    warmup_on_start: bool = True
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
class MemoryConfig:
    enabled: bool = False
    preferences_path: str = "data/preferences.yaml"
    episodic_path: str = "data/episodic.jsonl"
    max_history_messages: int = 24
    max_episodic_hits: int = 3
    max_inject_chars: int = 1200


@dataclass(frozen=True)
class AppConfig:
    audio: AudioConfig
    llm: LlmConfig
    stt: SttConfig
    tts: TtsConfig
    vad: VadConfig
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)


def load_config(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return AppConfig(
        audio=AudioConfig(**raw["audio"]),
        llm=LlmConfig(**raw["llm"]),
        stt=SttConfig(**raw["stt"]),
        tts=TtsConfig(**raw["tts"]),
        vad=VadConfig(**raw["vad"]),
        telemetry=TelemetryConfig(**raw.get("telemetry", {})),
        memory=MemoryConfig(**raw.get("memory", {})),
    )
