import re
from dataclasses import dataclass
from typing import Literal


_REMEMBER_CUE = re.compile(
    r"^\s*(?:remember that|please remember(?: that)?|"
    r"don['’]t forget that|remind yourself that)\s+(?P<payload>.+?)\s*$",
    re.IGNORECASE,
)
_PREFERENCE_PATTERNS: tuple[
    tuple[Literal["prefer", "name", "address_as"], re.Pattern[str]], ...
] = (
    ("prefer", re.compile(r"^i prefer\s+(.+)$", re.IGNORECASE)),
    ("name", re.compile(r"^my name is\s+(.+)$", re.IGNORECASE)),
    ("address_as", re.compile(r"^(?:call me|address me as)\s+(.+)$", re.IGNORECASE)),
)


@dataclass(frozen=True)
class MemoryWrite:
    kind: Literal["preference", "episodic"]
    key: str | None
    value: str


def extract_remember_intent(text: str) -> MemoryWrite | None:
    cue_match = _REMEMBER_CUE.match(text)
    if cue_match is None:
        return None

    payload = cue_match.group("payload").strip()
    if not payload:
        return None

    for key, pattern in _PREFERENCE_PATTERNS:
        preference_match = pattern.match(payload)
        if preference_match is not None:
            return MemoryWrite(
                kind="preference",
                key=key,
                value=preference_match.group(1).strip(),
            )

    return MemoryWrite(kind="episodic", key=None, value=payload)
