import json
from pathlib import Path

from voice.tts import TtsEngine
from voice.voices import discover_voices, resolve_voice


def test_discover_voices_reads_onnx_and_sample_rate(tmp_path: Path):
    model = tmp_path / "en_US-lessac-medium.onnx"
    model.write_bytes(b"fake")
    sidecar = Path(f"{model}.json")
    sidecar.write_text(json.dumps({"audio": {"sample_rate": 22050}}))

    voices = discover_voices(tmp_path)
    assert len(voices) == 1
    assert voices[0].name == "en_US-lessac-medium"
    assert voices[0].sample_rate == 22050
    assert resolve_voice(voices, "en_US-lessac-medium") is voices[0]


def test_tts_set_voice_updates_model_and_rate():
    calls: list[list[str]] = []

    def runner(command, **kwargs):
        calls.append(command)

        class Result:
            stdout = b"pcm"

        return Result()

    tts = TtsEngine("piper", "a.onnx", sample_rate=16000, length_scale=1.0, runner=runner)
    tts.set_voice("b.onnx", sample_rate=22050, length_scale=0.9)
    assert tts.voice_path == "b.onnx"
    assert tts.sample_rate == 22050
    assert tts.length_scale == 0.9
    assert tts.synthesize("hi") == b"pcm"
    assert calls[0][:5] == ["piper", "--model", "b.onnx", "--output_raw", "--length_scale"]
    assert calls[0][5] == "0.9"
