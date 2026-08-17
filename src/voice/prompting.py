from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from pathlib import Path

from voice.tools.base import Tool


def build_system_prompt(
    system_prompt_path: str | Path,
    *,
    world_context_path: str | Path | None = None,
    today: date | None = None,
) -> str:
    """Compose the spoken-system prompt with date and optional world facts."""
    parts: list[str] = [Path(system_prompt_path).read_text().strip()]
    day = today or date.today()
    parts.append(f"Today's date is {day.isoformat()}.")
    if world_context_path is not None:
        context_file = Path(world_context_path)
        if context_file.is_file():
            context = context_file.read_text().strip()
            if context:
                parts.append("Current world context for factual questions:\n" + context)
    return "\n\n".join(parts)


def compose_system_prompt(
    system_prompt_path: str | Path,
    *,
    world_context_path: str | Path | None = None,
    today: date | None = None,
    preferences: dict[str, str] | None = None,
    episodic_notes: list[str] | None = None,
    max_inject_chars: int = 1200,
) -> str:
    """Compose the base prompt with a bounded local-memory block."""
    prompt = build_system_prompt(
        system_prompt_path,
        world_context_path=world_context_path,
        today=today,
    )
    composed, _chars = append_memory_inject(
        prompt,
        preferences=preferences,
        episodic_notes=episodic_notes,
        max_inject_chars=max_inject_chars,
    )
    return composed


def format_memory_inject_block(
    *,
    preferences: dict[str, str] | None = None,
    episodic_notes: list[str] | None = None,
    extra_sections: list[str] | None = None,
    max_inject_chars: int = 1200,
) -> str:
    """Build the truncated preference/episodic block (may be empty)."""
    if max_inject_chars <= 0:
        return ""

    memory_sections: list[str] = []
    if preferences:
        preference_lines = [
            f"- {key}: {value}" for key, value in sorted(preferences.items())
        ]
        memory_sections.append("User preferences:\n" + "\n".join(preference_lines))
    if episodic_notes:
        note_lines = [f"- {note}" for note in episodic_notes]
        memory_sections.append(
            "Relevant remembered notes:\n" + "\n".join(note_lines)
        )
    if extra_sections:
        memory_sections.extend(section.strip() for section in extra_sections if section.strip())
    if not memory_sections:
        return ""
    return "\n\n".join(memory_sections)[:max_inject_chars]


def append_memory_inject(
    base_prompt: str,
    *,
    preferences: dict[str, str] | None = None,
    episodic_notes: list[str] | None = None,
    extra_sections: list[str] | None = None,
    max_inject_chars: int = 1200,
) -> tuple[str, int]:
    """Append a budgeted memory block to an immutable base prompt.

    Returns ``(full_prompt, injected_char_count)``.
    """
    block = format_memory_inject_block(
        preferences=preferences,
        episodic_notes=episodic_notes,
        extra_sections=extra_sections,
        max_inject_chars=max_inject_chars,
    )
    if not block:
        return base_prompt, 0
    return base_prompt + "\n\n" + block, len(block)


def append_tools_section(base_prompt: str, tools: Iterable[Tool]) -> str:
    """Append silent tool-call instructions and the available tool catalog."""
    catalog = [
        f"- {tool.name}: {tool.description} Arguments: {tool.parameters_schema}"
        for tool in tools
    ]
    if not catalog:
        return base_prompt
    section = (
        "Tools\n"
        "Tool calls are silent. Never read their marker or JSON aloud.\n"
        'Use this exact format: <<tool:NAME|{"arg":"value"}>>\n'
        "Available tools:\n"
        + "\n".join(catalog)
    )
    return base_prompt + "\n\n" + section


def append_personality(base_prompt: str, personality_text: str) -> str:
    """Append a profile-pack personality block when text is non-empty."""
    text = personality_text.strip()
    if not text:
        return base_prompt
    return base_prompt + "\n\nPersonality:\n" + text


def adapt_personality(base_prompt: str, preferences: dict[str, str]) -> str:
    """Nudge tone from stored prefs without a trait engine."""
    prefer = (preferences.get("prefer") or preferences.get("tone") or "").strip()
    if not prefer:
        return base_prompt
    return (
        base_prompt
        + f"\n\nAdapt to the user's stored preference: {prefer}."
    )
