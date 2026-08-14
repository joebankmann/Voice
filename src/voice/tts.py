from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from voice.voices import VoiceInfo, resolve_voice


ProcessRunner = Callable[..., subprocess.CompletedProcess[Any]]


class TtsEngine:
    """Thin Piper CLI adapter returning mono int16 PCM bytes."""

    def __init__(
        self,
        piper_bin: str,
        voice_path: str,
        *,
        sample_rate: int,
        length_scale: float = 1.0,
        runner: ProcessRunner = subprocess.run,
    ) -> None:
        self._piper_bin = piper_bin
        self._voice_path = voice_path
        self.sample_rate = sample_rate
        self.length_scale = length_scale
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
        """Switch the active Piper model; optional rate/tempo overrides."""
        self._voice_path = str(voice_path)
        if sample_rate is not None:
            self.sample_rate = sample_rate
        if length_scale is not None:
            self.length_scale = length_scale

    def apply_voice_info(self, voice: VoiceInfo, *, length_scale: float | None = None) -> None:
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

    def synthesize(self, text: str) -> bytes:
        command = [
            self._piper_bin,
            "--model",
            self._voice_path,
            "--output_raw",
            "--length_scale",
            str(self.length_scale),
        ]
        result = self._runner(
            command,
            input=text.encode(),
            check=True,
            capture_output=True,
        )
        return bytes(result.stdout)
