import threading

import numpy as np

from voice.__main__ import ListenLoopThread, run_cli
from voice.config import (
    AppConfig,
    AudioConfig,
    LlmConfig,
    SttConfig,
    TtsConfig,
    VadConfig,
)
from voice.metrics import MetricsSink


class FakeAudio:
    frame_samples = 1

    def __init__(self):
        self._frames = iter(
            [
                np.array([1.0], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
                np.array([1.0], dtype=np.float32),
            ]
        )

    def read_frame(self, timeout=None):
        try:
            return next(self._frames)
        except StopIteration:
            raise KeyboardInterrupt


class FakePipeline:
    def __init__(self):
        self.audio = FakeAudio()
        self.metrics = MetricsSink(enabled=True)
        self._interrupt = threading.Event()
        self.interrupted_during_turn = False

    def start(self):
        pass

    def stop(self):
        self._interrupt.set()

    def handle_speech_start(self):
        self._interrupt.set()

    def interrupt(self):
        self.handle_speech_start()

    def run_turn(self, transcript):
        self._interrupt.clear()
        self.interrupted_during_turn = self._interrupt.wait(timeout=0.25)
        return ""


class FakeStt:
    def start(self):
        pass

    def stop(self):
        pass

    def transcribe(self, pcm16, sample_rate):
        return "hello"


class FakeVad:
    def is_speech(self, frame):
        return bool(frame[0])


def test_cli_keeps_listening_while_turn_runs():
    pipeline = FakePipeline()
    config = AppConfig(
        audio=AudioConfig(sample_rate=1, end_of_turn_silence_ms=1000),
        llm=LlmConfig("", "", 0.0, ""),
        stt=SttConfig("", ""),
        tts=TtsConfig("", ""),
        vad=VadConfig(threshold=0.5, min_speech_ms=1000),
    )

    run_cli(pipeline, config=config, stt=FakeStt(), vad=FakeVad())

    assert pipeline.interrupted_during_turn is True
    event_names = [event["name"] for event in pipeline.metrics.events()]
    assert "vad_end" in event_names
    assert "stt" in event_names


def test_cli_reports_stt_error_and_keeps_listening():
    class RecoveringAudio:
        frame_samples = 1

        def __init__(self):
            self._frames = iter(
                [
                    np.array([1.0], dtype=np.float32),
                    np.array([0.0], dtype=np.float32),
                    np.array([1.0], dtype=np.float32),
                    np.array([0.0], dtype=np.float32),
                ]
            )

        def read_frame(self, timeout=None):
            try:
                return next(self._frames)
            except StopIteration:
                raise KeyboardInterrupt

    class RecoveringPipeline(FakePipeline):
        def __init__(self):
            super().__init__()
            self.audio = RecoveringAudio()
            self.errors = []

        def emit_error(self, stage, error):
            self.errors.append(f"{stage} error: {error}")

    class RecoveringStt(FakeStt):
        def __init__(self):
            self.calls = 0

        def transcribe(self, pcm16, sample_rate):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("decoder unavailable")
            return "recovered"

    pipeline = RecoveringPipeline()
    stt = RecoveringStt()
    config = AppConfig(
        audio=AudioConfig(sample_rate=1, end_of_turn_silence_ms=1000),
        llm=LlmConfig("", "", 0.0, ""),
        stt=SttConfig("", ""),
        tts=TtsConfig("", ""),
        vad=VadConfig(threshold=0.5, min_speech_ms=1000),
    )

    run_cli(pipeline, config=config, stt=stt, vad=FakeVad())

    assert stt.calls == 2
    assert pipeline.errors == ["STT error: decoder unavailable"]


def test_listen_loop_thread_reports_stt_start_failure_without_starting_listen():
    class ErrorPipeline(FakePipeline):
        def __init__(self):
            super().__init__()
            self.errors = []
            self.started = False

        def emit_error(self, stage, error):
            self.errors.append(f"{stage} error: {error}")

        def start(self):
            self.started = True

    class BrokenStartStt(FakeStt):
        def start(self):
            raise RuntimeError("whisper-server missing")

        def stop(self):
            self.stopped = True

    pipeline = ErrorPipeline()
    stt = BrokenStartStt()
    config = AppConfig(
        audio=AudioConfig(sample_rate=1, end_of_turn_silence_ms=1000),
        llm=LlmConfig("", "", 0.0, ""),
        stt=SttConfig("", ""),
        tts=TtsConfig("", ""),
        vad=VadConfig(threshold=0.5, min_speech_ms=1000),
    )
    loop = ListenLoopThread(pipeline, config=config, stt=stt, vad=FakeVad())

    loop.start()

    assert pipeline.errors == ["STT error: whisper-server missing"]
    assert pipeline.started is False
    assert loop.thread is None
    assert getattr(stt, "stopped", False) is True


def test_listen_loop_thread_stop_interrupts_before_join():
    calls: list[str] = []

    class OrderPipeline(FakePipeline):
        def interrupt(self):
            calls.append("interrupt")
            self.handle_speech_start()

        def stop(self):
            calls.append("stop")
            super().stop()

    class SlowThread:
        def is_alive(self):
            return True

        def join(self, timeout=None):
            calls.append("join")

    pipeline = OrderPipeline()
    config = AppConfig(
        audio=AudioConfig(sample_rate=1, end_of_turn_silence_ms=1000),
        llm=LlmConfig("", "", 0.0, ""),
        stt=SttConfig("", ""),
        tts=TtsConfig("", ""),
        vad=VadConfig(threshold=0.5, min_speech_ms=1000),
    )
    loop = ListenLoopThread(pipeline, config=config, stt=FakeStt(), vad=FakeVad())
    loop.thread = SlowThread()
    loop.stop()

    assert calls == ["interrupt", "join", "stop"]
