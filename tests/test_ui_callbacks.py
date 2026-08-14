from voice.session import SessionState
from voice.ui import UiController


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
