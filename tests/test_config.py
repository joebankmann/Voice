from pathlib import Path

from voice.config import load_config


def test_load_config_reads_core_fields(tmp_path: Path):
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
  voices_dir: models
  sample_rate: 22050
  length_scale: 1.0
vad:
  threshold: 0.5
  min_speech_ms: 250
telemetry:
  enabled: true
  log_path: logs/latency.jsonl
""".strip()
    )
    cfg = load_config(cfg_path)
    assert cfg.audio.sample_rate == 16000
    assert cfg.audio.end_of_turn_silence_ms == 600
    assert cfg.llm.base_url == "http://127.0.0.1:8080/v1"
    assert cfg.llm.model == "qwen3-8b"
    assert cfg.stt.model_path.endswith("ggml-large-v3-turbo.bin")
    assert cfg.tts.sample_rate == 22050
    assert cfg.tts.voices_dir == "models"
    assert cfg.tts.length_scale == 1.0
    assert cfg.tts.warmup_on_start is True
    assert cfg.telemetry.enabled is True
    assert cfg.telemetry.log_path == "logs/latency.jsonl"
    assert cfg.memory.enabled is False


def test_load_config_reads_memory_fields(tmp_path: Path):
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
memory:
  enabled: true
  preferences_path: custom/preferences.yaml
  episodic_path: custom/episodic.jsonl
  max_history_messages: 12
  max_episodic_hits: 2
  max_inject_chars: 800
""".strip()
    )

    memory = load_config(cfg_path).memory
    assert memory.enabled is True
    assert memory.preferences_path == "custom/preferences.yaml"
    assert memory.episodic_path == "custom/episodic.jsonl"
    assert memory.max_history_messages == 12
    assert memory.max_episodic_hits == 2
    assert memory.max_inject_chars == 800
