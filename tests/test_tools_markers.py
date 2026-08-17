from voice.tools.base import ToolCall
from voice.tools.markers import extract_tool_calls, strip_tool_markers


def test_extract_tool_calls_parses_valid_markers_in_order():
    text = (
        'First <<tool:local_time|{}>> then '
        '<<tool:preference_set|{"key":"name","value":"Jo"}>>.'
    )

    assert extract_tool_calls(text) == [
        ToolCall(name="local_time", arguments={}),
        ToolCall(
            name="preference_set",
            arguments={"key": "name", "value": "Jo"},
        ),
    ]


def test_extract_tool_calls_skips_invalid_json():
    assert extract_tool_calls("<<tool:note_add|{not-json}>>") == []


def test_strip_tool_markers_removes_valid_and_invalid_markers():
    text = (
        "Before <<tool:local_time|{}>> middle "
        "<<tool:note_add|{not-json}>> after"
    )

    assert strip_tool_markers(text) == "Before  middle  after"


def test_non_marker_text_is_unchanged():
    text = "Use <<tool:bad-name|{}>> literally."

    assert extract_tool_calls(text) == []
    assert strip_tool_markers(text) == text
