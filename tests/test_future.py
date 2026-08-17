from datetime import date
from pathlib import Path
import threading

from voice.affect import estimate_affect
from voice.chunker import PhraseChunker
from voice.config import AgentsConfig, FutureConfig, MemoryConfig
from voice.inbox import consume_inbox
from voice.memory import EpisodicStore, PreferencesStore
from voice.metrics import MetricsSink
from voice.pipeline import PipelineAgents, PipelineFuture, PipelineMemory, VoicePipeline
from voice.prompting import build_system_prompt
from voice.session import ConversationSession


class RecordingLlm:
    def __init__(self):
        self.system_prompts: list[str] = []

    def stream_chat(self, messages, system_prompt):
        self.system_prompts.append(system_prompt)
        yield "Okay."

    def cancel(self):
        pass


class SilentTts:
    def speak(self, text: str):
        pass

    def stop(self):
        pass


class RecordingHelper:
    def __init__(
        self,
        note: str = "Dropped talk about cedar.",
        action: str | None = "Follow up on cedar.",
    ):
        self.note = note
        self.action = action
        self.summarize_calls: list[list[dict[str, str]]] = []
        self.action_calls: list[list[dict[str, str]]] = []
        self.cancelled = False

    def begin_turn(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def summarize_dropped(self, messages):
        self.summarize_calls.append(list(messages))
        return self.note

    def extract_actions(self, messages):
        self.action_calls.append(list(messages))
        return self.action


def _pipeline(
    tmp_path: Path,
    *,
    future: PipelineFuture | None = None,
    memory_enabled: bool = False,
    collaborative: bool = False,
    helper: RecordingHelper | None = None,
    max_history: int = 2,
) -> tuple[VoicePipeline, RecordingLlm, PreferencesStore, EpisodicStore, RecordingHelper]:
    system_path = tmp_path / "system.txt"
    system_path.write_text("Base.")
    preferences = PreferencesStore(tmp_path / "preferences.yaml")
    episodic = EpisodicStore(tmp_path / "episodic.jsonl")
    llm = RecordingLlm()
    helper = helper or RecordingHelper()
    memory = PipelineMemory(
        config=MemoryConfig(enabled=memory_enabled, max_history_messages=max_history),
        preferences=preferences,
        episodic=episodic,
    )
    agents = None
    if collaborative or memory_enabled:
        agents = PipelineAgents(
            config=AgentsConfig(
                enabled=True,
                collaborative=collaborative,
                timeout_ms=500,
            ),
            helper=helper,
        )
    pipeline = VoicePipeline(
        session=ConversationSession(max_history_messages=max_history),
        llm=llm,
        tts=SilentTts(),
        chunker=PhraseChunker(),
        system_prompt=build_system_prompt(system_path, today=date(2026, 8, 16)),
        metrics=MetricsSink(enabled=True),
        memory=memory if memory_enabled else None,
        agents=agents,
        future=future,
    )
    return pipeline, llm, preferences, episodic, helper


def test_estimate_affect_lexicon_labels():
    assert estimate_affect("I'm so frustrated with this") == "frustrated"
    assert estimate_affect("I feel sad today") == "sad"
    assert estimate_affect("thanks, that was awesome") == "warm"
    assert estimate_affect("the cedar plank is dry") is None


def test_consume_inbox_reads_text_once_and_reports_images(tmp_path: Path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "note.txt").write_text("Cedar inventory is low.")
    (inbox / "shot.png").write_bytes(b"not-an-image")
    (inbox / "skip.bin").write_bytes(b"x")

    notes = consume_inbox(inbox)
    assert "Cedar inventory is low." in notes
    assert any("Unsupported image shot.png" in note for note in notes)
    assert consume_inbox(inbox) == []
    assert not (inbox / "note.txt").exists()
    assert (inbox / "processed" / "note.txt").is_file()
    assert (inbox / "processed" / "shot.png").is_file()
    assert (inbox / "skip.bin").is_file()


def test_future_flags_off_do_not_inject_affect(tmp_path: Path):
    pipeline, llm, _, _, _ = _pipeline(tmp_path)
    pipeline.run_turn("I'm so frustrated with this")
    assert llm.system_prompts
    assert "affect" not in llm.system_prompts[-1].casefold()


def test_affect_injects_hint_without_memory(tmp_path: Path):
    pipeline, llm, _, _, _ = _pipeline(
        tmp_path,
        future=PipelineFuture(
            config=FutureConfig(affect=True),
            inbox_dir=str(tmp_path / "inbox"),
        ),
    )
    pipeline.run_turn("I'm so frustrated with this")
    assert "User affect hint: frustrated." in llm.system_prompts[-1]
    event = next(item for item in pipeline.metrics.events() if item["name"] == "affect")
    assert event["label"] == "frustrated"


def test_inbox_injects_note_and_moves_file(tmp_path: Path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "note.md").write_text("Meet at the marina.")
    pipeline, llm, _, _, _ = _pipeline(
        tmp_path,
        future=PipelineFuture(
            config=FutureConfig(inbox=True),
            inbox_dir=str(inbox),
        ),
    )
    pipeline.run_turn("hello")
    assert "Shared note: Meet at the marina." in llm.system_prompts[-1]
    assert not (inbox / "note.md").exists()
    assert (inbox / "processed" / "note.md").is_file()


def test_adaptive_personality_mentions_prefer(tmp_path: Path):
    pipeline, llm, preferences, _, _ = _pipeline(
        tmp_path,
        memory_enabled=True,
        future=PipelineFuture(
            config=FutureConfig(adaptive_personality=True),
            inbox_dir=str(tmp_path / "inbox"),
        ),
    )
    preferences.set("prefer", "short answers")
    pipeline.run_turn("How should you respond?")
    assert "Adapt to the user's stored preference: short answers." in llm.system_prompts[-1]


def test_collaborative_writes_second_episodic_note(tmp_path: Path):
    helper = RecordingHelper()
    pipeline, _, _, episodic, _ = _pipeline(
        tmp_path,
        memory_enabled=True,
        collaborative=True,
        helper=helper,
    )
    pipeline.run_turn("first fact about cedar")
    pipeline.run_turn("second fact about oak")
    assert helper.summarize_calls
    assert helper.action_calls
    notes = episodic.retrieve("cedar", limit=5)
    assert "Dropped talk about cedar." in notes
    assert "Follow up on cedar." in notes
    event = next(item for item in pipeline.metrics.events() if item["name"] == "agent_actions")
    assert event["ok"] is True


def test_inbox_survives_barge_in_during_prep(tmp_path: Path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "note.txt").write_text("Meet at the marina.")
    pipeline, llm, _, _, _ = _pipeline(
        tmp_path,
        future=PipelineFuture(
            config=FutureConfig(inbox=True),
            inbox_dir=str(inbox),
        ),
    )
    original_prepare = pipeline._prepare_memory

    def slow_prepare(transcript: str) -> None:
        original_prepare(transcript)
        pipeline.handle_speech_start()

    pipeline._prepare_memory = slow_prepare  # type: ignore[method-assign]
    assert pipeline.run_turn("hello") == ""
    assert (inbox / "note.txt").is_file()

    pipeline._prepare_memory = original_prepare  # type: ignore[method-assign]
    pipeline.run_turn("hello again")
    assert "Shared note: Meet at the marina." in llm.system_prompts[-1]
    assert not (inbox / "note.txt").exists()


def test_interrupt_during_extract_actions_skips_write(tmp_path: Path):
    class BlockingHelper(RecordingHelper):
        def __init__(self):
            super().__init__()
            self._block = threading.Event()
            self._started = threading.Event()
            self.block_extract = True

        def extract_actions(self, messages):
            self.action_calls.append(list(messages))
            if self.block_extract:
                self._started.set()
                self._block.wait(timeout=1)
                if self.cancelled:
                    return None
            return self.action

        def cancel(self):
            self.cancelled = True
            self._block.set()

    helper = BlockingHelper()
    pipeline, _, _, episodic, _ = _pipeline(
        tmp_path,
        memory_enabled=True,
        collaborative=True,
        helper=helper,
    )
    pipeline.run_turn("first fact about cedar")
    worker = threading.Thread(target=pipeline.run_turn, args=("second fact about oak",))
    worker.start()
    assert helper._started.wait(timeout=1)
    pipeline.handle_speech_start()
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert helper.cancelled is True
    assert "Follow up on cedar." not in episodic.retrieve("cedar", limit=5)


def test_collaborative_off_does_not_extract_actions(tmp_path: Path):
    helper = RecordingHelper()
    pipeline, _, _, episodic, _ = _pipeline(
        tmp_path,
        memory_enabled=True,
        collaborative=False,
        helper=helper,
    )
    pipeline.run_turn("first fact about cedar")
    pipeline.run_turn("second fact about oak")
    assert helper.summarize_calls
    assert helper.action_calls == []
    assert "Follow up on cedar." not in episodic.retrieve("cedar", limit=5)
