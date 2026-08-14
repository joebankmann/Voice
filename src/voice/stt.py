from __future__ import annotations

import subprocess
import tempfile
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any


ProcessRunner = Callable[..., subprocess.CompletedProcess[Any]]


class SttEngine:
    """Thin whisper.cpp CLI adapter.

    The temporary-WAV boundary is suitable for the MVP. A streaming pipeline
    can later replace this adapter without changing its callers.
    """

    def __init__(
        self,
        whisper_bin: str,
        model_path: str,
        *,
        runner: ProcessRunner = subprocess.run,
    ) -> None:
        self._whisper_bin = whisper_bin
        self._model_path = model_path
        self._runner = runner

    def transcribe(self, pcm16: bytes, sample_rate: int) -> str:
        with tempfile.TemporaryDirectory(prefix="voice-stt-") as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "input.wav"
            output_prefix = temp_path / "transcript"
            self._write_wav(wav_path, pcm16, sample_rate)
            self._runner(
                [
                    self._whisper_bin,
                    "-m",
                    self._model_path,
                    "-f",
                    str(wav_path),
                    "-otxt",
                    "-of",
                    str(output_prefix),
                ],
                check=True,
                capture_output=True,
            )
            return output_prefix.with_suffix(".txt").read_text().strip()

    @staticmethod
    def _write_wav(path: Path, pcm16: bytes, sample_rate: int) -> None:
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm16)
