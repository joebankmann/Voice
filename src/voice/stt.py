from __future__ import annotations

import io
import socket
import subprocess
import tempfile
import time
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import httpx

from voice.config import SttConfig


ProcessRunner = Callable[..., subprocess.CompletedProcess[Any]]
ProcessFactory = Callable[..., subprocess.Popen[Any]]
HttpPost = Callable[..., Any]
ReadyProbe = Callable[[str, int], bool]


class SttBackend(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def transcribe(self, pcm16: bytes, sample_rate: int) -> str: ...


def _default_ready_probe(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


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

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

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


class ResidentSttEngine:
    """HTTP adapter for a long-lived whisper.cpp server."""

    def __init__(
        self,
        whisper_server_bin: str,
        model_path: str,
        *,
        host: str = "127.0.0.1",
        port: int = 8178,
        inference_path: str = "/inference",
        manage_server: bool = True,
        startup_timeout_s: float = 30.0,
        request_timeout_s: float = 120.0,
        popen: ProcessFactory = subprocess.Popen,
        request_post: HttpPost = httpx.post,
        ready_probe: ReadyProbe = _default_ready_probe,
    ) -> None:
        self._whisper_server_bin = whisper_server_bin
        self._model_path = model_path
        self._host = host
        self._port = port
        self._inference_path = f"/{inference_path.lstrip('/')}"
        self._manage_server = manage_server
        self._startup_timeout_s = startup_timeout_s
        self._request_timeout_s = request_timeout_s
        self._popen = popen
        self._request_post = request_post
        self._ready_probe = ready_probe
        self._process: subprocess.Popen[Any] | None = None

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        if self._manage_server:
            self._process = self._popen(
                [
                    self._whisper_server_bin,
                    "-m",
                    self._model_path,
                    "--host",
                    self._host,
                    "--port",
                    str(self._port),
                    "--inference-path",
                    self._inference_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        self._wait_until_ready()

    def stop(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)

    def transcribe(self, pcm16: bytes, sample_rate: int) -> str:
        self.start()
        response = self._request_post(
            f"http://{self._host}:{self._port}{self._inference_path}",
            files={
                "file": (
                    "input.wav",
                    self._wav_bytes(pcm16, sample_rate),
                    "audio/wav",
                )
            },
            data={"response_format": "json"},
            timeout=self._request_timeout_s,
        )
        response.raise_for_status()
        try:
            payload = response.json()
        except (TypeError, ValueError):
            return response.text.strip()
        if isinstance(payload, dict):
            transcript = payload.get("text") or payload.get("transcription")
            if isinstance(transcript, str):
                return transcript.strip()
        if isinstance(payload, str):
            return payload.strip()
        return response.text.strip()

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self._startup_timeout_s
        while not self._ready_probe(self._host, self._port):
            if self._process is not None:
                return_code = self._process.poll()
                if return_code is not None:
                    raise subprocess.CalledProcessError(
                        return_code,
                        self._whisper_server_bin,
                    )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"whisper-server did not become ready on "
                    f"{self._host}:{self._port}"
                )
            time.sleep(0.05)

    @staticmethod
    def _wav_bytes(pcm16: bytes, sample_rate: int) -> bytes:
        output = io.BytesIO()
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm16)
        return output.getvalue()


def build_stt(
    config: SttConfig,
    *,
    config_dir: Path | None = None,
) -> SttBackend:
    base_dir = config_dir or Path.cwd()
    model_path = Path(config.model_path)
    if not model_path.is_absolute():
        model_path = base_dir / model_path
    if config.mode == "cli":
        return SttEngine(config.whisper_bin, str(model_path))
    if config.mode == "resident":
        return ResidentSttEngine(
            config.whisper_server_bin,
            str(model_path),
            host=config.server_host,
            port=config.server_port,
            inference_path=config.inference_path,
            manage_server=config.manage_server,
        )
    raise ValueError(f"Unsupported STT mode: {config.mode}")
