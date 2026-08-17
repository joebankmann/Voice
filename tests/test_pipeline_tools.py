import threading
from pathlib import Path

import voice.__main__ as voice_main
from voice.chunker import PhraseChunker
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
from voice.metrics import MetricsSink
from voice.pipeline import PipelineTools, VoicePipeline
from voice.session import ConversationSession, SessionState
from voice.tools import ToolRegistry, ToolRunner


class RecordingLlm:
    def __init__(self, responses: list[list[str]]) -> None:
        self.responses = responses
        self.calls: list[list[dict[str, str]]] = []
        self.cancelled = False

    def stream_chat(self, messages, system_prompt):
        self.calls.append([dict(message) for message in messages])
        yield from self.responses[len(self.calls) - 1]

    def cancel(self) -> None:
        self.cancelled = True


class RecordingTts:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)

    def stop(self) -> None:
        pass


class EchoTool:
    name = "echo"
    description = "Echo text."
    parameters_schema = {"text": "string"}
    offline = True

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def run(self, args: dict[str, object]) -> str:
        self.calls.append(args)
        return f"Echo result: {args.get('text', '')}"


def _pipeline_tools(tool: EchoTool) -> PipelineTools:
    config = ToolsConfig(enabled=True, timeout_ms=100, allow_online=False)
    registry = ToolRegistry()
    registry.register(tool)
    return PipelineTools(
        config=config,
        registry=registry,
        runner=ToolRunner(
            registry,
            timeout_ms=config.timeout_ms,
            allow_online=config.allow_online,
        ),
    )


def test_enabled_tools_execute_once_and_speak_one_marker_free_continuation():
    tool = EchoTool()
    llm = RecordingLlm(
        [
            ["I will check. ", '<<tool:echo|{"text":"secret.marker"}>>'],
            ["The result is ready. ", "<<tool:echo|{}>>"],
        ]
    )
    tts = RecordingTts()
    metrics = MetricsSink(enabled=True)
    pipeline = VoicePipeline(
        session=ConversationSession(),
        llm=llm,
        tts=tts,
        chunker=PhraseChunker(),
        metrics=metrics,
        tools=_pipeline_tools(tool),
    )

    pipeline.run_turn("Check it")

    assert tool.calls == [{"text": "secret.marker"}]
    assert len(llm.calls) == 2
    assert llm.calls[1][-1] == {
        "role": "user",
        "content": 'Tool results:\necho: Echo result: secret.marker',
    }
    assert "I will check." in tts.spoken
    assert "The result is ready." in tts.spoken
    assert all("<<tool:" not in phrase for phrase in tts.spoken)
    tool_events = [
        event for event in metrics.events() if event["name"] == "tool_call"
    ]
    assert len(tool_events) == 1
    assert tool_events[0]["tool"] == "echo"
    assert tool_events[0]["ok"] is True


def test_disabled_tools_strip_markers_without_executing_or_continuing():
    llm = RecordingLlm(
        [["Before. ", '<<tool:echo|{"text":"do not run.marker"}>>', " After."]]
    )
    tts = RecordingTts()
    pipeline = VoicePipeline(
        session=ConversationSession(),
        llm=llm,
        tts=tts,
        chunker=PhraseChunker(),
    )

    pipeline.run_turn("Check it")

    assert len(llm.calls) == 1
    assert tts.spoken == ["Before.", "After."]


def test_interrupt_during_tool_execution_skips_continuation():
    started = threading.Event()
    release = threading.Event()

    class BlockingTool(EchoTool):
        def run(self, args: dict[str, object]) -> str:
            started.set()
            release.wait(timeout=1)
            return "Finished"

    tool = BlockingTool()
    llm = RecordingLlm([["<<tool:echo|{}>>"], ["Must not continue."]])
    tts = RecordingTts()
    pipeline = VoicePipeline(
        session=ConversationSession(),
        llm=llm,
        tts=tts,
        chunker=PhraseChunker(),
        tools=_pipeline_tools(tool),
    )
    worker = threading.Thread(target=pipeline.run_turn, args=("Check it",))

    worker.start()
    assert started.wait(timeout=1)
    pipeline.handle_speech_start()
    release.set()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert len(llm.calls) == 1
    assert tts.spoken == []
    assert pipeline.session.state == SessionState.LISTENING


def test_interrupt_before_continuation_speak_drops_queued_phrase():
    llm = RecordingLlm([["Should not speak."]])
    tts = RecordingTts()
    pipeline = VoicePipeline(
        session=ConversationSession(),
        llm=llm,
        tts=tts,
        chunker=PhraseChunker(),
        tools=_pipeline_tools(EchoTool()),
    )
    pipeline._interrupted.set()

    continuation = pipeline._stream_continuation()

    assert continuation == ""
    assert tts.spoken == []


def _app_config(*, tools_enabled: bool, memory_enabled: bool = False) -> AppConfig:
    return AppConfig(
        audio=AudioConfig(sample_rate=16_000, end_of_turn_silence_ms=600),
        llm=LlmConfig(
            base_url="",
            model="",
            temperature=0.0,
            system_prompt_path="system.txt",
            world_context_path="world.txt",
        ),
        stt=SttConfig(whisper_bin="", model_path=""),
        tts=TtsConfig(piper_bin="", voice_path="", warmup_on_start=False),
        vad=VadConfig(threshold=0.5, min_speech_ms=100),
        memory=MemoryConfig(enabled=memory_enabled),
        tools=ToolsConfig(enabled=tools_enabled),
        agents=AgentsConfig(enabled=False),
    )


def _stub_pipeline_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(voice_main, "AudioHub", lambda **kwargs: object())
    monkeypatch.setattr(voice_main, "discover_voices", lambda *args, **kwargs: [])
    monkeypatch.setattr(voice_main, "resolve_voice", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        voice_main,
        "create_tts_engine",
        lambda *args, **kwargs: RecordingTts(),
    )
    monkeypatch.setattr(
        voice_main,
        "LlmClient",
        lambda *args, **kwargs: RecordingLlm([["unused"]]),
    )


def test_build_pipeline_wires_enabled_tools_prompt_and_memory_stores(
    tmp_path: Path,
    monkeypatch,
):
    (tmp_path / "system.txt").write_text("Base instructions.")
    _stub_pipeline_dependencies(monkeypatch)

    pipeline = voice_main.build_pipeline(
        _app_config(tools_enabled=True, memory_enabled=True),
        tmp_path,
    )

    assert pipeline.tools is not None
    assert "\n\nTools\n" in pipeline.base_system_prompt
    preference_tool = pipeline.tools.registry.get("preference_set")
    assert preference_tool is not None
    assert preference_tool.run({"key": "tone", "value": "brief"}) == (
        "Saved preference tone."
    )
    assert pipeline.memory is not None
    assert pipeline.memory.preferences.get_all() == {"tone": "brief"}


def test_build_pipeline_omits_tools_when_disabled(tmp_path: Path, monkeypatch):
    (tmp_path / "system.txt").write_text("Base instructions.")
    _stub_pipeline_dependencies(monkeypatch)

    pipeline = voice_main.build_pipeline(
        _app_config(tools_enabled=False),
        tmp_path,
    )

    assert pipeline.tools is None
    assert "\n\nTools\n" not in pipeline.base_system_prompt
