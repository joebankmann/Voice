from __future__ import annotations

import subprocess
from collections.abc import Callable
from typing import Any


ProcessRunner = Callable[..., subprocess.CompletedProcess[Any]]


class TtsEngine:
    """Thin Piper CLI adapter returning mono int16 PCM bytes."""

    def __init__(
        self,
        piper_bin: str,
        voice_path: str,
        *,
        runner: ProcessRunner = subprocess.run,
    ) -> None:
        self._piper_bin = piper_bin
        self._voice_path = voice_path
        self._runner = runner

    def synthesize(self, text: str) -> bytes:
        result = self._runner(
            [
                self._piper_bin,
                "--model",
                self._voice_path,
                "--output_raw",
            ],
            input=text.encode(),
            check=True,
            capture_output=True,
        )
        return bytes(result.stdout)
