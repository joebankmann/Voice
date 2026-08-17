import json
from pathlib import Path

from voice.tts import TtsEngine
from voice.tts_f5 import F5CloneEngine
from voice.tts_factory import HybridTtsEngine
from voice.voices import VoiceInfo, discover_voices, resolve_voice


def test_discover_voices_reads_onnx_and_sample_rate(tmp_path: Path):
    model = tmp_path / "en_US-lessac-medium.onnx"
    model.write_bytes(b"fake")
    sidecar = Path(f"{model}.json")
    sidecar.write_text(json.dumps({"audio": {"sample_rate": 22050}}))

    voices = discover_voices(tmp_path)
    assert len(voices) == 1
    assert voices[0].name == "en_US-lessac-medium"
    assert voices[0].sample_rate == 22050
    assert voices[0].engine == "piper"
    assert resolve_voice(voices, "en_US-lessac-medium") is voices[0]


def test_discover_clone_profiles(tmp_path: Path):
    piper_dir = tmp_path / "piper"
    clones = tmp_path / "clones" / "morgan"
    piper_dir.mkdir()
    clones.mkdir(parents=True)
    (piper_dir / "a.onnx").write_bytes(b"x")
    (clones / "ref.wav").write_bytes(b"RIFF")
    (clones / "ref.txt").write_text("Hello from morgan.")
    voices = discover_voices(piper_dir, clones_dir=tmp_path / "clones")
    assert {v.engine for v in voices} == {"piper", "f5"}
    clone = resolve_voice(voices, "morgan")
    assert clone is not None
    assert clone.ref_text == "Hello from morgan."
    assert clone.sample_rate == 24000


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
    assert calls[0][0:5] == ["piper", "--model", "b.onnx", "--output_raw", "--length_scale"]
    assert calls[0][5] == "0.9"
    assert "--sentence_silence" in calls[0]


def test_tts_sanitize_before_piper():
    seen: dict[str, bytes] = {}

    def runner(command, **kwargs):
        seen["input"] = kwargs["input"]

        class Result:
            stdout = b"x"

        return Result()

    engine = TtsEngine("piper", "v.onnx", sample_rate=22050, runner=runner)
    engine.synthesize("Hello **world** 🙂")
    assert seen["input"] == b"Hello world"


def test_f5_engine_uses_injectable_generate(tmp_path: Path):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"fake")
    engine = F5CloneEngine(
        ref_audio_path=ref,
        ref_text="reference line",
        generate_fn=lambda text, **kwargs: b"\x00\x10",
    )
    assert engine.synthesize("Hi there") == b"\x00\x10"


def test_hybrid_falls_back_to_piper_on_f5_failure(tmp_path: Path):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"fake")

    def boom(*args, **kwargs):
        raise RuntimeError("f5 down")

    primary = F5CloneEngine(
        ref_audio_path=ref,
        ref_text="reference line",
        generate_fn=boom,
    )

    def runner(command, **kwargs):
        class Result:
            stdout = b"piper"

        return Result()

    fallback = TtsEngine("piper", "v.onnx", sample_rate=22050, runner=runner)
    hybrid = HybridTtsEngine(primary=primary, fallback=fallback)
    assert hybrid.synthesize("hello") == b"piper"
    assert hybrid.sample_rate == 22050
