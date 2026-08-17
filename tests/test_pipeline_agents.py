import threading
from datetime import date
from pathlib import Path

from voice.agents.helper import HelperAgent
from voice.chunker import PhraseChunker
from voice.config import AgentsConfig, MemoryConfig
from voice.memory import EpisodicStore, PreferencesStore
from voice.metrics import MetricsSink
from voice.pipeline import PipelineAgents, PipelineMemory, VoicePipeline
from voice.prompting import build_system_prompt
from voice.session import ConversationSession, SessionState


class StreamLlm:
    def stream_chat(self, messages, system_prompt):
        yield "Okay."

    def complete(self, messages, system_prompt):
        raise AssertionError("main LLM complete should not run when helper is separate")

    def cancel(self):
        pass


class SilentTts:
    def speak(self, text: str):
        pass

    def stop(self):
        pass


class RecordingHelper:
    def __init__(self, note: str = "Dropped talk about cedar."):
        self.note = note
        self.calls: list[list[dict[str, str]]] = []
        self.cancelled = False
        self._block = threading.Event()
        self._started = threading.Event()
        self.block = False

    def begin_turn(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True
        self._block.set()

    def summarize_dropped(self, messages):
        self.calls.append(list(messages))
        if self.block:
            self._started.set()
            self._block.wait(timeout=1)
            if self.cancelled:
                return None
        return self.note


def _pipeline(
    tmp_path: Path,
    *,
    agents_enabled: bool = True,
    helper: RecordingHelper | HelperAgent | None = None,
    max_history: int = 2,
) -> tuple[VoicePipeline, EpisodicStore, object]:
    system_path = tmp_path / "system.txt"
    system_path.write_text("Base.")
    preferences = PreferencesStore(tmp_path / "preferences.yaml")
    episodic = EpisodicStore(tmp_path / "episodic.jsonl")
    helper = helper or RecordingHelper()
    memory = PipelineMemory(
        config=MemoryConfig(enabled=True, max_history_messages=max_history),
        preferences=preferences,
        episodic=episodic,
    )
    agents = PipelineAgents(
        config=AgentsConfig(enabled=agents_enabled, timeout_ms=500),
        helper=helper,
    )
    pipeline = VoicePipeline(
        session=ConversationSession(max_history_messages=max_history),
        llm=StreamLlm(),
        tts=SilentTts(),
        chunker=PhraseChunker(),
        system_prompt=build_system_prompt(system_path, today=date(2026, 8, 16)),
        metrics=MetricsSink(enabled=True),
        memory=memory,
        agents=agents,
    )
    return pipeline, episodic, helper


def test_disabled_agents_do_not_summarize_evicted_history(tmp_path: Path):
    helper = RecordingHelper()
    pipeline, episodic, _ = _pipeline(tmp_path, agents_enabled=False, helper=helper)
    pipeline.run_turn("first fact about cedar")
    pipeline.run_turn("second fact about oak")
    assert helper.calls == []
    assert episodic.retrieve("cedar", limit=3) == []


def test_enabled_agents_write_episodic_note_for_evicted_turns(tmp_path: Path):
    pipeline, episodic, helper = _pipeline(tmp_path)
    pipeline.run_turn("first fact about cedar")
    pipeline.run_turn("second fact about oak")
    assert helper.calls
    assert any("cedar" in item["content"] for call in helper.calls for item in call)
    assert "Dropped talk about cedar." in episodic.retrieve("cedar", limit=3)
    event = next(e for e in pipeline.metrics.events() if e["name"] == "agent_summarize")
    assert event["ok"] is True
    assert event["chars"] > 0


def test_interrupt_during_helper_skips_episodic_write(tmp_path: Path):
    helper = RecordingHelper()
    helper.block = True
    pipeline, episodic, _ = _pipeline(tmp_path, helper=helper)
    pipeline.run_turn("first fact about cedar")
    worker = threading.Thread(target=pipeline.run_turn, args=("second fact about oak",))
    worker.start()
    assert helper._started.wait(timeout=1)
    pipeline.handle_speech_start()
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert helper.cancelled is True
    assert episodic.retrieve("cedar", limit=3) == []
    assert pipeline.session.state == SessionState.LISTENING
    restored = pipeline.session.take_evicted()
    assert any("cedar" in item["content"] for item in restored)


def test_helper_agent_timeout_path_does_not_write(tmp_path: Path):
    class SlowLlm:
        def complete(self, messages, system_prompt):
            threading.Event().wait(1)
            return "too late"

        def cancel(self):
            pass

    helper = HelperAgent(SlowLlm(), timeout_ms=40)
    pipeline, episodic, _ = _pipeline(tmp_path, helper=helper)
    pipeline.run_turn("first fact about cedar")
    pipeline.run_turn("second fact about oak")
    assert episodic.retrieve("cedar", limit=3) == []
    event = next(e for e in pipeline.metrics.events() if e["name"] == "agent_summarize")
    assert event["ok"] is False


def test_build_pipeline_wires_helper_when_agents_enabled(tmp_path: Path, monkeypatch):
    import voice.__main__ as voice_main
    from voice.config import (
        AppConfig,
        AgentsConfig,
        AudioConfig,
        LlmConfig,
        MemoryConfig,
        SttConfig,
        ToolsConfig,
        TtsConfig,
        VadConfig,
    )

    (tmp_path / "system.txt").write_text("Base.")
    monkeypatch.setattr(voice_main, "AudioHub", lambda **kwargs: object())
    monkeypatch.setattr(voice_main, "discover_voices", lambda *args, **kwargs: [])
    monkeypatch.setattr(voice_main, "resolve_voice", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        voice_main,
        "create_tts_engine",
        lambda *args, **kwargs: SilentTts(),
    )
    monkeypatch.setattr(
        voice_main,
        "LlmClient",
        lambda *args, **kwargs: StreamLlm(),
    )
    config = AppConfig(
        audio=AudioConfig(sample_rate=16_000, end_of_turn_silence_ms=600),
        llm=LlmConfig("", "", 0.0, "system.txt", "world.txt"),
        stt=SttConfig("", ""),
        tts=TtsConfig("", "", warmup_on_start=False),
        vad=VadConfig(0.5, 100),
        memory=MemoryConfig(enabled=True),
        tools=ToolsConfig(enabled=False),
        agents=AgentsConfig(enabled=True, timeout_ms=1234),
    )
    pipeline = voice_main.build_pipeline(config, tmp_path)
    assert pipeline.agents is not None
    assert pipeline.agents.config.enabled is True
    assert pipeline.agents.helper._timeout_s == 1.234
    assert pipeline.agents.helper._llm is not pipeline.llm


def test_build_pipeline_omits_agents_when_disabled(tmp_path: Path, monkeypatch):
    import voice.__main__ as voice_main
    from voice.config import (
        AppConfig,
        AudioConfig,
        LlmConfig,
        SttConfig,
        TtsConfig,
        VadConfig,
    )

    (tmp_path / "system.txt").write_text("Base.")
    monkeypatch.setattr(voice_main, "AudioHub", lambda **kwargs: object())
    monkeypatch.setattr(voice_main, "discover_voices", lambda *args, **kwargs: [])
    monkeypatch.setattr(voice_main, "resolve_voice", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        voice_main,
        "create_tts_engine",
        lambda *args, **kwargs: SilentTts(),
    )
    monkeypatch.setattr(
        voice_main,
        "LlmClient",
        lambda *args, **kwargs: StreamLlm(),
    )
    config = AppConfig(
        audio=AudioConfig(sample_rate=16_000, end_of_turn_silence_ms=600),
        llm=LlmConfig("", "", 0.0, "system.txt"),
        stt=SttConfig("", ""),
        tts=TtsConfig("", "", warmup_on_start=False),
        vad=VadConfig(0.5, 100),
    )
    pipeline = voice_main.build_pipeline(config, tmp_path)
    assert pipeline.agents is None
    assert pipeline.future is not None
    assert pipeline.future.config.affect is False
    assert pipeline.future.config.inbox is False
    assert pipeline.future.config.adaptive_personality is False
    assert Path(pipeline.future.inbox_dir) == tmp_path / "data" / "inbox"
