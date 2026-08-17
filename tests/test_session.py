from voice.session import ConversationSession, SessionEventType, SessionState


def test_barge_in_cancels_playback_and_generation():
    s = ConversationSession()
    s.force_state(SessionState.THINKING_SPEAKING)
    events = s.on_user_speech_start()
    types = [e.type for e in events]
    assert SessionEventType.STOP_PLAYBACK in types
    assert SessionEventType.CANCEL_GENERATION in types
    assert s.state == SessionState.LISTENING


def test_speech_end_requests_reply():
    s = ConversationSession()
    s.force_state(SessionState.LISTENING)
    events = s.on_user_speech_end("hello there")
    assert any(e.type == SessionEventType.REQUEST_REPLY and e.transcript == "hello there" for e in events)
    assert s.state == SessionState.THINKING_SPEAKING


def test_history_trim_drops_oldest_when_over_limit():
    s = ConversationSession(max_history_messages=4)
    s.history.extend(
        [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "3"},
            {"role": "assistant", "content": "4"},
        ]
    )
    s.append_assistant("5")
    assert len(s.history) == 4
    assert s.history[0]["content"] == "2"
    assert s.history[-1]["content"] == "5"


def test_history_trim_on_user_speech_end():
    s = ConversationSession(max_history_messages=2)
    s.force_state(SessionState.LISTENING)
    s.on_user_speech_end("first")
    s.append_assistant("reply")
    s.on_user_speech_end("second")
    assert len(s.history) == 2
    assert s.history[0]["content"] == "reply"
    assert s.history[1]["content"] == "second"


def test_max_history_none_means_no_trim():
    s = ConversationSession(max_history_messages=None)
    s.force_state(SessionState.LISTENING)
    for i in range(10):
        s.on_user_speech_end(f"msg{i}")
        s.append_assistant(f"reply{i}")
    assert len(s.history) == 20


def test_max_history_zero_or_negative_means_no_trim():
    for limit in (0, -1):
        s = ConversationSession(max_history_messages=limit)
        s.force_state(SessionState.LISTENING)
        for i in range(5):
            s.on_user_speech_end(f"msg{i}")
        assert len(s.history) == 5
