from voice.chunker import PhraseChunker
from voice.pipeline import VoicePipeline
from voice.session import ConversationSession, SessionState


class FakeLlm:
    def __init__(self):
        self.cancelled = False

    def stream_chat(self, messages, system_prompt):
        yield "Hello "
        yield "world."
        if not self.cancelled:
            yield " more"

    def cancel(self):
        self.cancelled = True


class FakeTts:
    def __init__(self):
        self.spoken = []
        self.stopped = False

    def speak(self, text: str):
        if not self.stopped:
            self.spoken.append(text)

    def stop(self):
        self.stopped = True


def test_barge_in_stops_playback_and_cancels_llm():
    session = ConversationSession()
    session.force_state(SessionState.THINKING_SPEAKING)
    llm = FakeLlm()
    tts = FakeTts()
    pipeline = VoicePipeline(
        session=session,
        llm=llm,
        tts=tts,
        chunker=PhraseChunker(),
    )

    pipeline.handle_speech_start()

    assert llm.cancelled is True
    assert tts.stopped is True
    assert session.state.name == "LISTENING"
