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
    controller = UiController(pipeline=pipeline)

    controller.on_start()
    assert pipeline.started is True
    assert controller.status == "LISTENING"

    controller.on_interrupt()
    assert pipeline.interrupted is True

    controller.on_stop()
    assert pipeline.stopped is True
    assert controller.status == "IDLE"


def test_controller_appends_transcript_events():
    controller = UiController(pipeline=FakePipeline())

    controller.handle_event({"type": "user_transcript", "text": "hi"})
    controller.handle_event({"type": "assistant_final", "text": "hello"})

    assert controller.transcript_lines == ["You: hi", "Assistant: hello"]
