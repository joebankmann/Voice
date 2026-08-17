import threading

from voice.agents.helper import ACTIONS_SYSTEM_PROMPT, HelperAgent


class FakeHelperLlm:
    def __init__(self, reply: str = "User prefers cedar planks.", *, delay: float = 0):
        self.reply = reply
        self.delay = delay
        self.prompts: list[str] = []
        self.cancelled = False

    def complete(self, messages, system_prompt):
        if self.delay:
            threading.Event().wait(self.delay)
        self.prompts.append(system_prompt)
        return self.reply

    def cancel(self):
        self.cancelled = True


def test_summarize_dropped_returns_first_line():
    agent = HelperAgent(FakeHelperLlm("Sailboat uses cedar.\nExtra."), timeout_ms=1000)
    note = agent.summarize_dropped(
        [{"role": "user", "content": "remember cedar"}, {"role": "assistant", "content": "ok"}]
    )
    assert note == "Sailboat uses cedar."


def test_summarize_dropped_none_and_empty():
    assert HelperAgent(FakeHelperLlm("NONE")).summarize_dropped(
        [{"role": "user", "content": "hi"}]
    ) is None
    assert HelperAgent(FakeHelperLlm("ok")).summarize_dropped([]) is None


def test_summarize_dropped_timeout_returns_none_and_cancels():
    llm = FakeHelperLlm("late", delay=1.0)
    agent = HelperAgent(llm, timeout_ms=50)
    assert agent.summarize_dropped([{"role": "user", "content": "x"}]) is None
    assert llm.cancelled is True


def test_summarize_skipped_when_already_cancelled():
    event = threading.Event()
    event.set()
    llm = FakeHelperLlm("should not run")
    agent = HelperAgent(llm, timeout_ms=1000, cancel_event=event)
    assert agent.summarize_dropped([{"role": "user", "content": "x"}]) is None
    assert llm.prompts == []


def test_extract_actions_uses_action_prompt_and_first_line():
    llm = FakeHelperLlm("Call the marina.\nExtra.")
    agent = HelperAgent(llm, timeout_ms=1000)
    note = agent.extract_actions([{"role": "user", "content": "need to call the marina"}])
    assert note == "Call the marina."
    assert llm.prompts == [ACTIONS_SYSTEM_PROMPT]


def test_extract_actions_none_is_skipped():
    assert HelperAgent(FakeHelperLlm("NONE")).extract_actions(
        [{"role": "user", "content": "hi"}]
    ) is None
