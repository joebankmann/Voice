from pathlib import Path

from voice.session import SessionState
from voice.ui import UiController
from voice.voices import VoiceInfo


class FakePipeline:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.interrupted = False
        self.listeners = []

    def add_listener(self, listener):
        self.listeners.append(listener)

    def start(self):
        self.started = True
        self._emit_state(SessionState.LISTENING)

    def stop(self):
        self.stopped = True
        self._emit_state(SessionState.IDLE)

    def interrupt(self):
        self.interrupted = True
        self._emit_state(SessionState.LISTENING)

    def _emit_state(self, state):
        for listener in self.listeners:
            listener({"type": "state", "state": state.name})


def test_controller_start_stop_interrupt_update_status():
    pipeline = FakePipeline()
    listening_actions = []
    controller = UiController(
        pipeline=pipeline,
        on_start_listening=lambda: listening_actions.append("start"),
        on_stop_listening=lambda: listening_actions.append("stop"),
    )

    controller.on_start()
    assert listening_actions == ["start"]

    controller.on_interrupt()
    assert pipeline.interrupted is True

    controller.on_stop()
    assert listening_actions == ["start", "stop"]


def test_controller_appends_transcript_events():
    controller = UiController(pipeline=FakePipeline())

    controller.handle_event({"type": "user_transcript", "text": "hi"})
    controller.handle_event({"type": "assistant_final", "text": "hello"})

    assert controller.transcript_lines == ["You: hi", "Assistant: hello"]


def test_controller_shows_error_without_disabling_stop_start():
    listening_actions = []
    controller = UiController(
        pipeline=FakePipeline(),
        on_start_listening=lambda: listening_actions.append("start"),
        on_stop_listening=lambda: listening_actions.append("stop"),
    )

    controller.handle_event({"type": "error", "text": "STT error: microphone lost"})
    controller.on_stop()
    controller.on_start()

    assert controller.status == "STT error: microphone lost"
    assert listening_actions == ["stop", "start"]


def test_controller_voice_change_invokes_callback():
    selected: list[str] = []
    voices = [
        VoiceInfo("alpha", Path("alpha.onnx"), None, 22050),
        VoiceInfo("beta", Path("beta.onnx"), None, 22050),
    ]
    controller = UiController(
        pipeline=FakePipeline(),
        voices=voices,
        selected_voice="alpha",
        on_voice_selected=lambda voice: selected.append(voice.name),
    )
    controller.on_voice_change("beta")
    assert selected == ["beta"]
    assert controller.selected_voice == "beta"


def test_pending_transcript_lines_are_append_only():
    controller = UiController(pipeline=FakePipeline())
    controller.handle_event({"type": "user_transcript", "text": "hi"})
    assert controller.pending_transcript_lines() == ["You: hi"]
    assert controller.pending_transcript_lines() == []
    controller.handle_event({"type": "assistant_final", "text": "hello"})
    assert controller.pending_transcript_lines() == ["Assistant: hello"]
