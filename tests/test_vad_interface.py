import numpy as np

from voice.vad import VadEngine


class FakeVad:
    def __call__(self, frame: np.ndarray) -> float:
        return float(np.mean(np.abs(frame)) > 0.01)


def test_vad_engine_threshold():
    eng = VadEngine(backend=FakeVad(), threshold=0.5)
    silence = np.zeros(512, dtype=np.float32)
    speech = np.ones(512, dtype=np.float32) * 0.2
    assert eng.is_speech(silence) is False
    assert eng.is_speech(speech) is True
