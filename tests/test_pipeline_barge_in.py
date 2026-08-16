import threading

from voice.chunker import PhraseChunker
from voice.metrics import MetricsSink
from voice.pipeline import VoicePipeline
from voice.session import ConversationSession, SessionState
from voice.tts import TtsEngine


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


class WarmupTts(FakeTts):
    def __init__(self):
        super().__init__()
        self.synthesized = []

    def synthesize(self, text: str):
        self.synthesized.append(text)
        return b"discarded"


def test_start_warms_tts_once_without_playback():
    tts = WarmupTts()
    pipeline = VoicePipeline(
        session=ConversationSession(),
        llm=FakeLlm(),
        tts=tts,
        chunker=PhraseChunker(),
        warmup_tts=True,
    )

    pipeline.start()
    pipeline.start()

    assert tts.synthesized == ["Ready."]


def test_start_warms_mocked_f5_once():
    class MockF5Tts:
        def __init__(self):
            self.synthesized = []

        def warmup(self):
            self.synthesize("Ready.")

        def synthesize(self, text):
            self.synthesized.append(text)
            return b"pcm"

    tts = MockF5Tts()
    pipeline = VoicePipeline(
        session=ConversationSession(),
        llm=FakeLlm(),
        tts=tts,
        chunker=PhraseChunker(),
        warmup_tts=True,
    )

    pipeline.start()
    pipeline.start()

    assert tts.synthesized == ["Ready."]


def test_start_skips_piper_synthesis_during_warmup():
    calls = []
    tts = TtsEngine(
        "piper",
        "voice.onnx",
        sample_rate=22050,
        runner=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    pipeline = VoicePipeline(
        session=ConversationSession(),
        llm=FakeLlm(),
        tts=tts,
        chunker=PhraseChunker(),
        warmup_tts=True,
    )

    pipeline.start()

    assert calls == []
    assert pipeline.session.state == SessionState.LISTENING


def test_start_hybrid_warmup_falls_back_to_piper():
    class MockHybridTts:
        def __init__(self):
            self.f5_calls = 0
            self.piper_calls = 0

        def warmup(self):
            try:
                self._synthesize_f5()
            except RuntimeError:
                self._synthesize_piper()

        def _synthesize_f5(self):
            self.f5_calls += 1
            raise RuntimeError("f5 down")

        def _synthesize_piper(self):
            self.piper_calls += 1
            return b"piper"

    tts = MockHybridTts()
    pipeline = VoicePipeline(
        session=ConversationSession(),
        llm=FakeLlm(),
        tts=tts,
        chunker=PhraseChunker(),
        warmup_tts=True,
    )

    pipeline.start()

    assert tts.f5_calls == 1
    assert tts.piper_calls == 1
    assert pipeline.session.state == SessionState.LISTENING


def test_start_continues_when_tts_warmup_fails():
    class BrokenWarmupTts(FakeTts):
        def warmup(self):
            raise RuntimeError("all tts backends down")

    pipeline = VoicePipeline(
        session=ConversationSession(),
        llm=FakeLlm(),
        tts=BrokenWarmupTts(),
        chunker=PhraseChunker(),
        warmup_tts=True,
    )

    pipeline.start()

    assert pipeline.session.state == SessionState.LISTENING


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


def test_llm_failure_emits_error_and_recovers_listening_state():
    class BrokenLlm(FakeLlm):
        def stream_chat(self, messages, system_prompt):
            raise RuntimeError("model unavailable")
            yield

    pipeline = VoicePipeline(
        session=ConversationSession(),
        llm=BrokenLlm(),
        tts=FakeTts(),
        chunker=PhraseChunker(),
    )
    events = []
    pipeline.add_listener(events.append)
    pipeline.start()

    assert pipeline.run_turn("Hello") == ""
    assert pipeline.session.state == SessionState.LISTENING
    assert events[-1] == {"type": "error", "text": "LLM error: model unavailable"}


def test_tts_failure_emits_error_and_recovers_listening_state():
    class BrokenTts(FakeTts):
        def speak(self, text: str):
            raise RuntimeError("speaker unavailable")

    pipeline = VoicePipeline(
        session=ConversationSession(),
        llm=FakeLlm(),
        tts=BrokenTts(),
        chunker=PhraseChunker(),
    )
    events = []
    pipeline.add_listener(events.append)
    pipeline.start()

    assert pipeline.run_turn("Hello") == ""
    assert pipeline.session.state == SessionState.LISTENING
    assert events[-1] == {"type": "error", "text": "TTS error: speaker unavailable"}


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
