from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from voice.speak_text import sanitize_for_speech
from voice.voices import VoiceInfo, resolve_voice


ProcessRunner = Callable[..., subprocess.CompletedProcess[Any]]


class SupportsSynthesize(Protocol):
    sample_rate: int

    def synthesize(self, text: str) -> bytes: ...


class PiperTtsEngine:
    """Thin Piper CLI adapter returning mono int16 PCM bytes."""

    def __init__(
        self,
        piper_bin: str,
        voice_path: str,
        *,
        sample_rate: int,
        length_scale: float = 1.05,
        sentence_silence: float = 0.25,
        noise_scale: float = 0.667,
        noise_w_scale: float = 0.8,
        runner: ProcessRunner = subprocess.run,
    ) -> None:
        self._piper_bin = piper_bin
        self._voice_path = voice_path
        self.sample_rate = sample_rate
        self.length_scale = length_scale
        self.sentence_silence = sentence_silence
        self.noise_scale = noise_scale
        self.noise_w_scale = noise_w_scale
        self._runner = runner

    @property
    def voice_path(self) -> str:
        return self._voice_path

    def set_voice(
        self,
        voice_path: str | Path,
        *,
        sample_rate: int | None = None,
        length_scale: float | None = None,
    ) -> None:
        self._voice_path = str(voice_path)
        if sample_rate is not None:
            self.sample_rate = sample_rate
        if length_scale is not None:
            self.length_scale = length_scale

    def apply_voice_info(self, voice: VoiceInfo, *, length_scale: float | None = None) -> None:
        if voice.engine != "piper":
            raise ValueError(f"Piper engine cannot use voice profile engine={voice.engine}")
        self.set_voice(
            voice.model_path,
            sample_rate=voice.sample_rate,
            length_scale=length_scale,
        )

    def select_voice(
        self,
        voices: list[VoiceInfo],
        voice_path: str | Path,
        *,
        length_scale: float | None = None,
    ) -> VoiceInfo:
        matched = resolve_voice(voices, voice_path)
        if matched is None:
            raise ValueError(f"Unknown Piper voice: {voice_path}")
        self.apply_voice_info(matched, length_scale=length_scale)
        return matched

    def warmup(self) -> None:
        """Piper is process-based, so startup warmup has nothing to retain."""

    def synthesize(self, text: str) -> bytes:
        spoken = sanitize_for_speech(text)
        if not spoken:
            return b""
        command = [
            self._piper_bin,
            "--model",
            self._voice_path,
            "--output_raw",
            "--length_scale",
            str(self.length_scale),
            "--sentence_silence",
            str(self.sentence_silence),
            "--noise_scale",
            str(self.noise_scale),
            "--noise_w_scale",
            str(self.noise_w_scale),
        ]
        result = self._runner(
            command,
            input=spoken.encode(),
            check=True,
            capture_output=True,
        )
        return bytes(result.stdout)


# Back-compat name used throughout the codebase.
TtsEngine = PiperTtsEngine
