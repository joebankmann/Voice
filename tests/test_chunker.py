from voice.chunker import PhraseChunker


def test_emits_on_sentence_end():
    c = PhraseChunker(max_chars=120)
    assert c.push("Hello") == []
    assert c.push(" there") == []
    assert c.push("!") == ["Hello there!"]


def test_emits_on_soft_length_break():
    c = PhraseChunker(max_chars=20)
    out = []
    out += c.push("This is a fairly long")
    out += c.push(" clause without end")
    assert out, "expected a soft flush before waiting forever"
    assert all(len(x) <= 40 for x in out)


def test_flush_returns_remainder():
    c = PhraseChunker(max_chars=120)
    assert c.push("Partial") == []
    assert c.flush() == ["Partial"]
