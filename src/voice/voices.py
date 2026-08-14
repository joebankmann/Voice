from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VoiceInfo:
    """A local Piper voice model and its optional metadata sidecar."""

    name: str
    model_path: Path
    config_path: Path | None
    sample_rate: int


def _read_sample_rate(config_path: Path | None, default: int) -> int:
    if config_path is None or not config_path.is_file():
        return default
    try:
        payload = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return default
    audio = payload.get("audio") if isinstance(payload, dict) else None
    if isinstance(audio, dict):
        rate = audio.get("sample_rate")
        if isinstance(rate, int) and rate > 0:
            return rate
    return default


def discover_voices(
    voices_dir: str | Path,
    *,
    default_sample_rate: int = 22_050,
) -> list[VoiceInfo]:
    """List Piper `.onnx` voices under ``voices_dir`` (non-recursive)."""
    root = Path(voices_dir)
    if not root.is_dir():
        return []

    voices: list[VoiceInfo] = []
    for model_path in sorted(root.glob("*.onnx")):
        config_path = Path(f"{model_path}.json")
        if not config_path.is_file():
            config_path_opt: Path | None = None
        else:
            config_path_opt = config_path
        voices.append(
            VoiceInfo(
                name=model_path.stem,
                model_path=model_path,
                config_path=config_path_opt,
                sample_rate=_read_sample_rate(config_path_opt, default_sample_rate),
            )
        )
    return voices


def resolve_voice(
    voices: list[VoiceInfo],
    voice_path: str | Path,
) -> VoiceInfo | None:
    """Match a configured path or stem against discovered voices."""
    target = Path(voice_path)
    for voice in voices:
        if voice.model_path.resolve() == target.resolve():
            return voice
        if voice.model_path.name == target.name:
            return voice
        if voice.name == target.stem or voice.name == target.name:
            return voice
    return None
