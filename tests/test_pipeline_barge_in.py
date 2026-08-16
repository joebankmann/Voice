import threading

from voice.chunker import PhraseChunker
from voice.metrics import MetricsSink
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


def test_completed_turn_returns_to_listening_and_emits_state():
    session = ConversationSession()
    session.force_state(SessionState.LISTENING)
    pipeline = VoicePipeline(
        session=session,
        llm=FakeLlm(),
        tts=FakeTts(),
        chunker=PhraseChunker(),
    )
    states = []
    pipeline.on_state_change(states.append)

    reply = pipeline.run_turn("Hello")

    assert reply
    assert session.state == SessionState.LISTENING
    assert states[-1] == "LISTENING"


def test_completed_turn_marks_first_llm_token_and_tts_audio():
    session = ConversationSession()
    session.force_state(SessionState.LISTENING)
    metrics = MetricsSink(enabled=True)
    pipeline = VoicePipeline(
        session=session,
        llm=FakeLlm(),
        tts=FakeTts(),
        chunker=PhraseChunker(),
        metrics=metrics,
    )

    pipeline.run_turn("Hello")

    event_names = [event["name"] for event in metrics.events()]
    assert event_names.count("llm_first_token") == 1
    assert event_names.count("tts_first_audio") == 1


def test_barge_in_interrupts_slow_stream_running_on_worker_thread():
    stream_blocked = threading.Event()
    release_stream = threading.Event()

    class SlowLlm(FakeLlm):
        def stream_chat(self, messages, system_prompt):
            yield "Starting."
            stream_blocked.set()
            release_stream.wait(timeout=1)
            yield " This should be interrupted."

        def cancel(self):
            super().cancel()
            release_stream.set()

    session = ConversationSession()
    session.force_state(SessionState.LISTENING)
    llm = SlowLlm()
    pipeline = VoicePipeline(
        session=session,
        llm=llm,
        tts=FakeTts(),
        chunker=PhraseChunker(),
    )
    worker = threading.Thread(target=pipeline.run_turn, args=("Hello",))

    worker.start()
    assert stream_blocked.wait(timeout=1)
    pipeline.handle_speech_start()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert llm.cancelled is True
    assert session.state == SessionState.LISTENING
