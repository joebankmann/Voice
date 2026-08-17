from pathlib import Path

from voice.config import load_config
from voice.profiles import discover_profile_packs, resolve_profile_pack


def test_discover_and_resolve_profile_packs(tmp_path: Path):
    pack = tmp_path / "profiles" / "casual"
    pack.mkdir(parents=True)
    (pack / "manifest.yaml").write_text(
        "\n".join(
            [
                "schema_version: 1",
                "id: casual",
                "name: Casual companion",
                "version: 1.0.0",
                "personality: personality.txt",
                "voice: default",
            ]
        )
    )
    (pack / "personality.txt").write_text("Be warm and brief.\n")
    (tmp_path / "profiles" / "ignored").mkdir()
    (tmp_path / "profiles" / "old").mkdir()
    (tmp_path / "profiles" / "old" / "manifest.yaml").write_text(
        "schema_version: 2\nid: old\n"
    )

    packs = discover_profile_packs(tmp_path / "profiles")
    assert [item.id for item in packs] == ["casual"]
    casual = resolve_profile_pack(packs, "casual")
    assert casual is not None
    assert casual.name == "Casual companion"
    assert casual.version == "1.0.0"
    assert casual.personality_text == "Be warm and brief."
    assert casual.voice == "default"
    assert resolve_profile_pack(packs, "") is None
    assert resolve_profile_pack(packs, "missing") is None


def test_load_config_defaults_profiles_inactive(tmp_path: Path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
audio:
  sample_rate: 16000
  end_of_turn_silence_ms: 600
llm:
  base_url: http://127.0.0.1:8080/v1
  model: qwen3-8b
  temperature: 0.7
  system_prompt_path: prompts/system.txt
stt:
  whisper_bin: whisper-cli
  model_path: models/ggml-large-v3-turbo.bin
tts:
  piper_bin: piper
  voice_path: models/en_US-lessac-medium.onnx
vad:
  threshold: 0.5
  min_speech_ms: 250
""".strip()
    )
    profiles = load_config(cfg_path).profiles
    assert profiles.active == ""
    assert profiles.packs_dir == "profiles"


def test_load_config_reads_profiles_fields(tmp_path: Path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
audio:
  sample_rate: 16000
  end_of_turn_silence_ms: 600
llm:
  base_url: http://127.0.0.1:8080/v1
  model: qwen3-8b
  temperature: 0.7
  system_prompt_path: prompts/system.txt
stt:
  whisper_bin: whisper-cli
  model_path: models/ggml-large-v3-turbo.bin
tts:
  piper_bin: piper
  voice_path: models/en_US-lessac-medium.onnx
vad:
  threshold: 0.5
  min_speech_ms: 250
profiles:
  packs_dir: packs
  active: casual
""".strip()
    )
    profiles = load_config(cfg_path).profiles
    assert profiles.packs_dir == "packs"
    assert profiles.active == "casual"


def test_build_pipeline_applies_active_profile_pack(tmp_path: Path, monkeypatch):
    import voice.__main__ as voice_main
    from voice.config import (
        AppConfig,
        AudioConfig,
        LlmConfig,
        ProfilesConfig,
        SttConfig,
        TtsConfig,
        VadConfig,
    )
    from voice.voices import VoiceInfo

    pack = tmp_path / "profiles" / "casual"
    pack.mkdir(parents=True)
    (pack / "manifest.yaml").write_text(
        "schema_version: 1\nid: casual\nname: Casual\nversion: 1.0.0\n"
        "personality: personality.txt\nvoice: default\n"
    )
    (pack / "personality.txt").write_text("Be warm and brief.")
    (tmp_path / "system.txt").write_text("Base instructions.")
    default_voice = VoiceInfo(
        name="default",
        model_path=tmp_path / "ref.wav",
        config_path=None,
        sample_rate=24000,
        engine="f5",
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(voice_main, "AudioHub", lambda **kwargs: object())
    monkeypatch.setattr(
        voice_main, "discover_voices", lambda *args, **kwargs: [default_voice]
    )
    monkeypatch.setattr(
        voice_main,
        "create_tts_engine",
        lambda config, *, selected, config_dir: captured.update(selected=selected)
        or object(),
    )
    monkeypatch.setattr(
        voice_main,
        "LlmClient",
        lambda *args, **kwargs: type(
            "L",
            (),
            {"stream_chat": lambda *a, **k: iter(()), "cancel": lambda self: None},
        )(),
    )
    config = AppConfig(
        audio=AudioConfig(sample_rate=16_000, end_of_turn_silence_ms=600),
        llm=LlmConfig("", "", 0.0, "system.txt"),
        stt=SttConfig("", ""),
        tts=TtsConfig("", "", warmup_on_start=False),
        vad=VadConfig(0.5, 100),
        profiles=ProfilesConfig(packs_dir="profiles", active="casual"),
    )
    pipeline = voice_main.build_pipeline(config, tmp_path)
    assert "Personality:\nBe warm and brief." in pipeline.base_system_prompt
    assert captured["selected"] is default_voice


def test_build_pipeline_skips_pack_when_inactive(tmp_path: Path, monkeypatch):
    import voice.__main__ as voice_main
    from voice.config import (
        AppConfig,
        AudioConfig,
        LlmConfig,
        SttConfig,
        TtsConfig,
        VadConfig,
    )

    (tmp_path / "system.txt").write_text("Base instructions.")
    monkeypatch.setattr(voice_main, "AudioHub", lambda **kwargs: object())
    monkeypatch.setattr(voice_main, "discover_voices", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        voice_main, "create_tts_engine", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        voice_main,
        "LlmClient",
        lambda *args, **kwargs: type(
            "L",
            (),
            {"cancel": lambda self: None},
        )(),
    )
    config = AppConfig(
        audio=AudioConfig(sample_rate=16_000, end_of_turn_silence_ms=600),
        llm=LlmConfig("", "", 0.0, "system.txt"),
        stt=SttConfig("", ""),
        tts=TtsConfig("", "", warmup_on_start=False),
        vad=VadConfig(0.5, 100),
    )
    pipeline = voice_main.build_pipeline(config, tmp_path)
    assert "Personality:" not in pipeline.base_system_prompt
