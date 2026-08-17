from __future__ import annotations

import re

_FRUSTRATED = frozenset(
    {"frustrated", "angry", "annoyed", "furious", "mad", "pissed"}
)
_SAD = frozenset({"sad", "depressed", "miserable", "heartbroken", "upset"})
_WARM = frozenset(
    {"thanks", "thankful", "grateful", "happy", "excited", "love", "awesome"}
)


def estimate_affect(text: str) -> str | None:
    """Cheap lexicon affect from a transcript. No extra model."""
    tokens = set(re.findall(r"[a-z']+", text.casefold()))
    if tokens & _FRUSTRATED:
        return "frustrated"
    if tokens & _SAD:
        return "sad"
    if tokens & _WARM:
        return "warm"
    return None
