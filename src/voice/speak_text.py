from __future__ import annotations

import re
import unicodedata

_MARKDOWN_CHARS = re.compile(r"[*_`#~>|]+")
_MARKDOWN_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_BULLET_PREFIX = re.compile(r"^\s*[-•]\s+", re.MULTILINE)
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_NL = re.compile(r"\n{3,}")
_TABLE_SEPARATOR = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)
_SPACED_ELLIPSIS = re.compile(r"[ \t]*\.(?:[ \t]+\.){2,}")
_MULTI_DOTS = re.compile(r"\.{3,}")
_UNICODE_ELLIPSIS = re.compile(r"…+")
_GESTURE_TAGS = re.compile(
    r"\[(laugh|chuckle|giggle|sigh|gasp|groan|clears throat|pause|soft|whisper)\]",
    re.IGNORECASE,
)

_GESTURE_MAP = {
    "laugh": " ha ha. ",
    "chuckle": " heh. ",
    "giggle": " heh heh. ",
    "sigh": " ... ",
    "gasp": " oh. ",
    "groan": " ugh. ",
    "clears throat": " ahem. ",
    "pause": "... ",
    "soft": " ",
    "whisper": " ",
}


def _flatten_markdown_tables(text: str) -> str:
    source_lines = text.splitlines()
    table_rows: set[int] = set()
    for index, line in enumerate(source_lines):
        if not _TABLE_SEPARATOR.fullmatch(line):
            continue
        if index > 0 and "|" in source_lines[index - 1]:
            table_rows.add(index - 1)
        row_index = index + 1
        while row_index < len(source_lines) and "|" in source_lines[row_index]:
            table_rows.add(row_index)
            row_index += 1

    lines: list[str] = []
    for index, line in enumerate(source_lines):
        if _TABLE_SEPARATOR.fullmatch(line):
            continue

        stripped = line.strip()
        if index in table_rows or (
            stripped.startswith("|") and stripped.endswith("|")
        ):
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            line = ", ".join(cell for cell in cells if cell)
        lines.append(line)
    return "\n".join(lines)


def sanitize_for_speech(text: str) -> str:
    """Strip markup/emoji and expand light vocal gesture tags for TTS."""

    def _replace_gesture(match: re.Match[str]) -> str:
        key = match.group(1).lower()
        return _GESTURE_MAP.get(key, " ")

    cleaned = _flatten_markdown_tables(text)
    cleaned = _MARKDOWN_LINK.sub(r"\1", cleaned)
    cleaned = _GESTURE_TAGS.sub(_replace_gesture, cleaned)
    cleaned_chars: list[str] = []
    for char in cleaned:
        category = unicodedata.category(char)
        if category.startswith("So") or category in {"Cn"}:
            continue
        cleaned_chars.append(char)
    cleaned = "".join(cleaned_chars)
    cleaned = _MARKDOWN_CHARS.sub("", cleaned)
    cleaned = _BULLET_PREFIX.sub("", cleaned)
    cleaned = _UNICODE_ELLIPSIS.sub("...", cleaned)
    cleaned = _SPACED_ELLIPSIS.sub("...", cleaned)
    cleaned = _MULTI_DOTS.sub("...", cleaned)
    cleaned = _MULTI_SPACE.sub(" ", cleaned)
    cleaned = _MULTI_NL.sub("\n\n", cleaned)
    return cleaned.strip()
