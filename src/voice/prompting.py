from __future__ import annotations

from datetime import date
from pathlib import Path


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
    if not memory_sections:
        return ""
    return "\n\n".join(memory_sections)[:max_inject_chars]


def append_memory_inject(
    base_prompt: str,
    *,
    preferences: dict[str, str] | None = None,
    episodic_notes: list[str] | None = None,
    max_inject_chars: int = 1200,
) -> tuple[str, int]:
    """Append a budgeted memory block to an immutable base prompt.

    Returns ``(full_prompt, injected_char_count)``.
    """
    block = format_memory_inject_block(
        preferences=preferences,
        episodic_notes=episodic_notes,
        max_inject_chars=max_inject_chars,
    )
    if not block:
        return base_prompt, 0
    return base_prompt + "\n\n" + block, len(block)
