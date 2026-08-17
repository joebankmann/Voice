import pytest

from voice.memory.intent import MemoryWrite, extract_remember_intent


@pytest.mark.parametrize(
    ("text", "key", "value"),
    [
        ("remember that I prefer short answers", "prefer", "short answers"),
        ("Please remember call me Jo", "address_as", "Jo"),
        ("DON'T FORGET THAT my name is Joseph", "name", "Joseph"),
        ("remind yourself that address me as Captain", "address_as", "Captain"),
    ],
)
def test_extracts_preference_from_remember_cue(
    text: str,
    key: str,
    value: str,
):
    assert extract_remember_intent(text) == MemoryWrite(
        kind="preference",
        key=key,
        value=value,
    )


def test_extracts_unstructured_payload_as_episodic_note():
    assert extract_remember_intent(
        "remember that the spare key is under the blue planter"
    ) == MemoryWrite(
        kind="episodic",
        key=None,
        value="the spare key is under the blue planter",
    )


@pytest.mark.parametrize(
    "text",
    [
        "I prefer short answers",
        "remember that",
        "please remember",
        "don't forget that   ",
        "",
    ],
)
def test_ignores_text_without_nonempty_remember_payload(text: str):
    assert extract_remember_intent(text) is None
