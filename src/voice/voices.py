from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VoiceInfo:
    """A selectable TTS voice: Piper ONNX or an F5 clone profile."""

    name: str
    model_path: Path
    config_path: Path | None
    sample_rate: int
    engine: str = "piper"
    ref_text: str | None = None


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


def discover_piper_voices(
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
        config_path_opt = config_path if config_path.is_file() else None
        voices.append(
            VoiceInfo(
                name=model_path.stem,
                model_path=model_path,
                config_path=config_path_opt,
                sample_rate=_read_sample_rate(config_path_opt, default_sample_rate),
                engine="piper",
            )
        )
    return voices


def discover_clone_voices(
    clones_dir: str | Path,
    *,
    default_sample_rate: int = 24_000,
) -> list[VoiceInfo]:
    """List F5 clone profiles: ``clones_dir/<name>/{ref.wav,ref.txt}``."""
    root = Path(clones_dir)
    if not root.is_dir():
        return []

    voices: list[VoiceInfo] = []
    for profile_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        ref_wav = profile_dir / "ref.wav"
        ref_txt = profile_dir / "ref.txt"
        if not ref_wav.is_file() or not ref_txt.is_file():
            continue
        name = profile_dir.name
        sample_rate = default_sample_rate
        profile_json = profile_dir / "profile.json"
        if profile_json.is_file():
            try:
                payload = json.loads(profile_json.read_text())
                if isinstance(payload, dict):
                    name = str(payload.get("name", name))
                    rate = payload.get("sample_rate")
                    if isinstance(rate, int) and rate > 0:
                        sample_rate = rate
            except (OSError, json.JSONDecodeError):
                pass
        voices.append(
            VoiceInfo(
                name=name,
                model_path=ref_wav,
                config_path=ref_txt,
                sample_rate=sample_rate,
                engine="f5",
                ref_text=ref_txt.read_text().strip(),
            )
        )
    return voices


def discover_voices(
    voices_dir: str | Path,
    *,
    clones_dir: str | Path | None = None,
    default_sample_rate: int = 22_050,
) -> list[VoiceInfo]:
    """Piper voices plus optional F5 clone profiles."""
    voices = discover_piper_voices(
        voices_dir,
        default_sample_rate=default_sample_rate,
    )
    if clones_dir is not None:
        voices.extend(discover_clone_voices(clones_dir))
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
        if voice.model_path.parent.name == target.name:
            return voice
    return None


# Back-compat alias used by older imports/tests.
discover_piper_only = discover_piper_voices
