import threading

import numpy as np

from voice.__main__ import run_cli
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

    def read_frame(self):
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

    def run_turn(self, transcript):
        self._interrupt.clear()
        self.interrupted_during_turn = self._interrupt.wait(timeout=0.25)
        return ""


class FakeStt:
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
