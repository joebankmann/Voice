from voice.speak_text import sanitize_for_speech


def test_sanitize_strips_markdown_and_emoji():
    text = "You're **right** 🙂 — use *care*, not asterisks."
    cleaned = sanitize_for_speech(text)
    assert "*" not in cleaned
    assert "🙂" not in cleaned
    assert "right" in cleaned
    assert "care" in cleaned


def test_sanitize_removes_bullet_markers():
    text = "- first point\n- second point"
    cleaned = sanitize_for_speech(text)
    assert cleaned.startswith("first")
    assert "-" not in cleaned.split("\n")[0]


def test_sanitize_expands_gesture_tags():
    cleaned = sanitize_for_speech("That is funny [laugh] okay")
    assert "[laugh]" not in cleaned
    assert "ha ha" in cleaned.lower()


def test_sanitize_flattens_markdown_tables_for_speech():
    text = (
        "| Option | Benefit |\n"
        "| :--- | ---: |\n"
        "| **Fast** | Low delay |\n"
        "| Local | More privacy |"
    )

    cleaned = sanitize_for_speech(text)

    assert cleaned == "Option, Benefit\nFast, Low delay\nLocal, More privacy"


def test_sanitize_flattens_markdown_tables_without_outer_pipes():
    text = "Option | Benefit\n--- | ---\nFast | Low delay"

    cleaned = sanitize_for_speech(text)

    assert cleaned == "Option, Benefit\nFast, Low delay"


def test_sanitize_preserves_markdown_link_text_without_destination():
    cleaned = sanitize_for_speech(
        "Read [the local guide](https://example.com/guide) before speaking."
    )

    assert cleaned == "Read the local guide before speaking."


def test_sanitize_normalizes_ellipsis_variants():
    cleaned = sanitize_for_speech("Wait…… really..... okay . . .")

    assert cleaned == "Wait... really... okay..."


def test_sanitize_expands_additional_gesture_tags():
    cleaned = sanitize_for_speech(
        "[giggle] Fine. [gasp] Oh. [groan] Again? [clears throat] Ready."
    )

    assert cleaned == "heh heh. Fine. oh. Oh. ugh. Again? ahem. Ready."
