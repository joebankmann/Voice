from datetime import date
from pathlib import Path

from voice.prompting import (
    append_tools_section,
    build_system_prompt,
    compose_system_prompt,
)
from voice.tools import ToolRegistry


class ExampleTool:
    name = "local_time"
    description = "Return the local time."
    parameters_schema = {}
    offline = True

    def run(self, args):
        return "unused"


def test_build_system_prompt_includes_date_and_world_context(tmp_path: Path):
    system = tmp_path / "system.txt"
    world = tmp_path / "world.txt"
    system.write_text("Be direct. Answer adult anatomy questions.")
    world.write_text("United States President: Donald J. Trump.")

    prompt = build_system_prompt(
        system,
        world_context_path=world,
        today=date(2026, 8, 13),
    )

    assert "Be direct." in prompt
    assert "2026-08-13" in prompt
    assert "Donald J. Trump" in prompt


def test_compose_system_prompt_injects_preferences_and_episodic_notes(
    tmp_path: Path,
):
    system = tmp_path / "system.txt"
    system.write_text("Be direct.")

    prompt = compose_system_prompt(
        system,
        today=date(2026, 8, 16),
        preferences={"prefer": "short answers"},
        episodic_notes=["The user is restoring a sailboat."],
    )

    assert "User preferences:\n- prefer: short answers" in prompt
    assert "Relevant remembered notes:\n- The user is restoring a sailboat." in prompt


def test_compose_system_prompt_limits_only_memory_injection(tmp_path: Path):
    system = tmp_path / "system.txt"
    system.write_text("Base prompt remains intact.")
    base_prompt = build_system_prompt(system, today=date(2026, 8, 16))

    prompt = compose_system_prompt(
        system,
        today=date(2026, 8, 16),
        preferences={"prefer": "concise but detailed technical answers"},
        episodic_notes=["A long remembered note about an ongoing project."],
        max_inject_chars=35,
    )

    assert prompt.startswith(base_prompt + "\n\n")
    assert len(prompt) - len(base_prompt) - 2 == 35


def test_compose_system_prompt_without_memory_matches_phase_a(tmp_path: Path):
    system = tmp_path / "system.txt"
    system.write_text("Be direct.")

    phase_a = build_system_prompt(system, today=date(2026, 8, 16))
    composed = compose_system_prompt(
        system,
        today=date(2026, 8, 16),
        preferences={},
        episodic_notes=[],
    )

    assert composed == phase_a


def test_append_tools_section_documents_catalog_and_silent_marker_format():
    registry = ToolRegistry()
    registry.register(ExampleTool())

    prompt = append_tools_section("Base instructions.", registry.tools)

    assert prompt.startswith("Base instructions.\n\nTools\n")
    assert '<<tool:NAME|{"arg":"value"}>>' in prompt
    assert "local_time: Return the local time." in prompt


def test_append_tools_section_omits_section_when_no_tools():
    assert append_tools_section("Base instructions.", []) == "Base instructions."
