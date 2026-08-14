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
