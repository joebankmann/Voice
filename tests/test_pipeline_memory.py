from datetime import date
from functools import partial
from pathlib import Path

from voice.chunker import PhraseChunker
from voice.config import MemoryConfig
from voice.memory import EpisodicStore, PreferencesStore
from voice.metrics import MetricsSink
from voice.pipeline import PipelineMemory, VoicePipeline
from voice.prompting import build_system_prompt, compose_system_prompt
from voice.session import ConversationSession


class RecordingLlm:
    def __init__(self):
        self.system_prompts: list[str] = []

    def stream_chat(self, messages, system_prompt):
        self.system_prompts.append(system_prompt)
        yield "Acknowledged."

    def cancel(self):
        pass


class SilentTts:
    def speak(self, text: str):
        pass

    def stop(self):
        pass


def _memory_pipeline(
    tmp_path: Path,
    *,
    enabled: bool = True,
) -> tuple[VoicePipeline, PreferencesStore, EpisodicStore, RecordingLlm]:
    system_path = tmp_path / "system.txt"
    system_path.write_text("Base instructions.")
    preferences = PreferencesStore(tmp_path / "preferences.yaml")
    episodic = EpisodicStore(tmp_path / "episodic.jsonl")
    llm = RecordingLlm()
    config = MemoryConfig(
        enabled=enabled,
        max_history_messages=4,
        max_episodic_hits=2,
        max_inject_chars=200,
    )
    memory = PipelineMemory(
        config=config,
        preferences=preferences,
        episodic=episodic,
        compose_prompt=partial(
            compose_system_prompt,
            system_path,
            today=date(2026, 8, 16),
        ),
    )
    pipeline = VoicePipeline(
        session=ConversationSession(),
        llm=llm,
        tts=SilentTts(),
        chunker=PhraseChunker(),
        system_prompt=build_system_prompt(
            system_path,
            today=date(2026, 8, 16),
        ),
        metrics=MetricsSink(enabled=True),
        memory=memory,
    )
    return pipeline, preferences, episodic, llm


def test_memory_turn_persists_preference_and_injects_it(tmp_path: Path):
    pipeline, preferences, _, llm = _memory_pipeline(tmp_path)

    pipeline.run_turn("Remember that I prefer short answers")
    pipeline.run_turn("How should you respond?")

    assert preferences.get_all() == {"prefer": "short answers"}
    assert "User preferences:\n- prefer: short answers" in llm.system_prompts[-1]


def test_memory_turn_retrieves_notes_and_marks_injection(tmp_path: Path):
    pipeline, _, episodic, llm = _memory_pipeline(tmp_path)
    episodic.add("The sailboat restoration uses cedar planks.")

    pipeline.run_turn("What wood is the sailboat restoration using?")

    assert "The sailboat restoration uses cedar planks." in llm.system_prompts[-1]
    memory_event = next(
        event
        for event in pipeline.metrics.events()
        if event["name"] == "memory_inject"
    )
    assert memory_event["prefs"] == 0
    assert memory_event["episodic"] == 1
    assert 0 < memory_event["chars"] <= 200


def test_disabled_memory_skips_write_injection_and_metrics(tmp_path: Path):
    pipeline, preferences, _, llm = _memory_pipeline(tmp_path, enabled=False)
    base_prompt = pipeline.system_prompt

    pipeline.run_turn("Remember that I prefer short answers")

    assert preferences.get_all() == {}
    assert llm.system_prompts == [base_prompt]
    assert all(
        event["name"] != "memory_inject" for event in pipeline.metrics.events()
    )
