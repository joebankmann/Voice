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
  world_context_path: prompts/world_context.txt
stt:
  whisper_bin: whisper-cli
  model_path: models/ggml-large-v3-turbo.bin
tts:
  piper_bin: piper
  voice_path: models/en_US-lessac-medium.onnx
  voices_dir: models
  sample_rate: 22050
  length_scale: 1.08
  sentence_silence: 0.28
  noise_scale: 0.7
  noise_w_scale: 0.85
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
    assert cfg.llm.world_context_path == "prompts/world_context.txt"
    assert cfg.stt.mode == "resident"
    assert cfg.stt.model_path.endswith("ggml-large-v3-turbo.bin")
    assert cfg.stt.server_port == 8178
    assert cfg.tts.sample_rate == 22050
    assert cfg.tts.voices_dir == "models"
    assert cfg.tts.length_scale == 1.08
    assert cfg.tts.sentence_silence == 0.28
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


def test_load_config_defaults_tools_to_disabled(tmp_path: Path):
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

    tools = load_config(cfg_path).tools

    assert tools.enabled is False
    assert tools.timeout_ms == 2000
    assert tools.allow_online is False


def test_load_config_reads_tools_fields(tmp_path: Path):
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
tools:
  enabled: true
  timeout_ms: 750
  allow_online: true
""".strip()
    )

    tools = load_config(cfg_path).tools

    assert tools.enabled is True
    assert tools.timeout_ms == 750
    assert tools.allow_online is True


def test_load_config_defaults_agents_to_disabled(tmp_path: Path):
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

    agents = load_config(cfg_path).agents
    assert agents.enabled is False
    assert agents.helper_base_url == ""
    assert agents.helper_model == ""
    assert agents.timeout_ms == 4000
    assert agents.collaborative is False


def test_load_config_reads_agents_fields(tmp_path: Path):
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
agents:
  enabled: true
  helper_base_url: http://127.0.0.1:8082/v1
  helper_model: qwen3-0.8b
  timeout_ms: 1500
""".strip()
    )

    agents = load_config(cfg_path).agents
    assert agents.enabled is True
    assert agents.helper_base_url == "http://127.0.0.1:8082/v1"
    assert agents.helper_model == "qwen3-0.8b"
    assert agents.timeout_ms == 1500
    assert agents.collaborative is False


def test_load_config_defaults_future_to_off(tmp_path: Path):
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

    future = load_config(cfg_path).future
    assert future.affect is False
    assert future.inbox is False
    assert future.inbox_dir == "data/inbox"
    assert future.adaptive_personality is False


def test_load_config_reads_future_and_collaborative(tmp_path: Path):
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
agents:
  enabled: true
  collaborative: true
future:
  affect: true
  inbox: true
  inbox_dir: notes/drop
  adaptive_personality: true
""".strip()
    )

    cfg = load_config(cfg_path)
    assert cfg.agents.collaborative is True
    assert cfg.future.affect is True
    assert cfg.future.inbox is True
    assert cfg.future.inbox_dir == "notes/drop"
    assert cfg.future.adaptive_personality is True
