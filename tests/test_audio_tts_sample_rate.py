import subprocess

import numpy as np

from voice.audio_io import AudioHub
from voice.tts import TtsEngine


class FakeStream:
    def __init__(self, **kwargs):
        self.callback = kwargs["callback"]

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


def test_tts_exposes_model_sample_rate():
    def runner(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=b"\x00\x00")

    engine = TtsEngine(
        "piper",
        "voice.onnx",
        sample_rate=22_050,
        runner=runner,
    )

    assert engine.sample_rate == 22_050


def test_audio_hub_resamples_pcm_to_output_rate():
    hub = AudioHub(
        sample_rate=4,
        frame_samples=4,
        input_stream_factory=FakeStream,
        output_stream_factory=FakeStream,
    )
    pcm = np.array([0, 32767], dtype="<i2").tobytes()

    hub.play(pcm, sample_rate=2)

    output = np.empty((4, 1), dtype=np.float32)
    hub._on_output(output, 4, None, None)
    np.testing.assert_allclose(
        output[:, 0],
        [0.0, 0.5, 32767 / 32768, 32767 / 32768],
        atol=1e-4,
    )
