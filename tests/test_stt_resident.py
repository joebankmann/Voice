from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from voice.__main__ import ListenLoopThread
from voice.config import (
    AppConfig,
    AudioConfig,
    LlmConfig,
    SttConfig,
    TtsConfig,
    VadConfig,
)
from voice.stt import ResidentSttEngine, SttEngine, build_stt


class FakeResponse:
    text = '{"text": "fallback"}'

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, str]:
        return {"text": "hello resident whisper"}


class FakeProcess:
    def __init__(self) -> None:
        self.terminated = False

    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        pass


def test_resident_engine_loads_model_once_for_multiple_transcriptions() -> None:
    popen_calls: list[list[str]] = []
    post_calls: list[dict[str, Any]] = []
    process = FakeProcess()

    def popen(command: list[str], **kwargs: Any) -> FakeProcess:
        popen_calls.append(command)
        return process

    def post(url: str, **kwargs: Any) -> FakeResponse:
        post_calls.append({"url": url, **kwargs})
        return FakeResponse()

    engine = ResidentSttEngine(
        whisper_server_bin="whisper-server",
        model_path="models/whisper.bin",
        host="127.0.0.1",
        port=8178,
        popen=popen,
        request_post=post,
        ready_probe=lambda host, port: True,
    )

    engine.start()
    assert engine.transcribe(b"\x00\x00", 16_000) == "hello resident whisper"
    assert engine.transcribe(b"\x01\x00", 16_000) == "hello resident whisper"
    engine.stop()

    assert len(popen_calls) == 1
    assert popen_calls[0].count("-m") == 1
    assert popen_calls[0][popen_calls[0].index("-m") + 1] == "models/whisper.bin"
    assert len(post_calls) == 2
    assert all("-m" not in str(call) for call in post_calls)
    assert all(call["url"] == "http://127.0.0.1:8178/inference" for call in post_calls)
    assert all(call["files"]["file"][1].startswith(b"RIFF") for call in post_calls)
    assert process.terminated is True


def test_build_stt_selects_cli_or_resident_and_resolves_model_path(
    tmp_path: Path,
) -> None:
    cli = build_stt(
        SttConfig(
            mode="cli",
            whisper_bin="whisper-cli",
            model_path="models/whisper.bin",
        ),
        config_dir=tmp_path,
    )
    resident = build_stt(
        SttConfig(
            mode="resident",
            whisper_bin="whisper-cli",
            model_path="models/whisper.bin",
            whisper_server_bin="whisper-server",
            server_port=8178,
        ),
        config_dir=tmp_path,
    )

    assert isinstance(cli, SttEngine)
    assert isinstance(resident, ResidentSttEngine)
    assert cli._model_path == str(tmp_path / "models/whisper.bin")
    assert resident._model_path == str(tmp_path / "models/whisper.bin")


def test_resident_engine_reports_server_exit_during_startup() -> None:
    class ExitedProcess(FakeProcess):
        def poll(self) -> int | None:
            return 2

    engine = ResidentSttEngine(
        whisper_server_bin="whisper-server",
        model_path="whisper.bin",
        popen=lambda *args, **kwargs: ExitedProcess(),
        ready_probe=lambda host, port: False,
    )

    try:
        engine.start()
    except subprocess.CalledProcessError as error:
        assert error.returncode == 2
    else:
        raise AssertionError("expected exited whisper-server to fail startup")


def test_listen_loop_thread_manages_stt_lifecycle(monkeypatch: Any) -> None:
    events: list[str] = []

    class FakePipeline:
        def start(self) -> None:
            events.append("pipeline-start")

        def stop(self) -> None:
            events.append("pipeline-stop")

    class FakeStt:
        def start(self) -> None:
            events.append("stt-start")

        def stop(self) -> None:
            events.append("stt-stop")

    config = AppConfig(
        audio=AudioConfig(sample_rate=16_000, end_of_turn_silence_ms=600),
        llm=LlmConfig("", "", 0.0, ""),
        stt=SttConfig("", ""),
        tts=TtsConfig("", ""),
        vad=VadConfig(threshold=0.5, min_speech_ms=250),
    )
    monkeypatch.setattr("voice.__main__.run_listen_loop", lambda *args, **kwargs: None)
    listen_loop = ListenLoopThread(
        FakePipeline(),
        config=config,
        stt=FakeStt(),
        vad=object(),
    )

    listen_loop.start()
    listen_loop.stop()

    assert events == [
        "stt-start",
        "pipeline-start",
        "pipeline-stop",
        "stt-stop",
    ]
