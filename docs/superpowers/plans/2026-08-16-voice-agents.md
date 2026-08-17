# Voice Specialist Agents (Phase D) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Design:** `docs/superpowers/specs/2026-08-16-voice-agents-design.md`

**Goal:** Add a config-gated off-path helper that summarizes evicted conversation history into episodic memory without touching time-to-first-audio.

**Architecture:** `AgentsConfig` + `LlmClient.complete()` + `HelperAgent` + session trim that returns dropped messages; pipeline runs helper only after an uninterrupted spoken turn.

**Tech Stack:** Existing httpx `LlmClient`, memory stores, pytest

## Global Constraints

- Helper never runs before first audio / not on the token hot path
- Missing `agents` key → `enabled: false`
- Barge-in and Stop cancel the helper
- No second Metal-resident model required (reuse main server by default)
- Imports at file top; commit only the current task's files

---

### Task D1: AgentsConfig + LlmClient.complete + session evict

**Files:** `src/voice/config.py`, `config.example.yaml`, `src/voice/llm.py`, `src/voice/session.py`, tests

**Interfaces:**
- `AgentsConfig(enabled=False, helper_base_url="", helper_model="", timeout_ms=4000)`
- `LlmClient.complete(messages, system_prompt) -> str` (non-stream; `cancel()` closes active response)
- `ConversationSession._trim_history() -> list[dict]` returning popped messages (oldest first)

- [ ] TDD + commit `feat: add agents config, LLM complete, and history eviction`

---

### Task D2: HelperAgent summarizer

**Files:** `src/voice/agents/helper.py`, `src/voice/agents/__init__.py`, `tests/test_helper_agent.py`

**Interfaces:**
```python
class HelperAgent:
    def __init__(self, llm, *, timeout_ms: int, cancel_event: threading.Event | None = None): ...
    def cancel(self) -> None: ...
    def summarize_dropped(self, messages: list[dict[str, str]]) -> str | None: ...
```

Empty/whitespace summary → None. Prompt: one short factual note, no markdown.

- [ ] TDD + commit `feat: add off-path helper agent for memory summarization`

---

### Task D3: Pipeline wiring + cancel

**Files:** `pipeline.py`, `__main__.py`, `tests/test_pipeline_agents.py`

After uninterrupted reply (including tool continuation), if agents+memory enabled and last trim dropped messages, summarize and `episodic.add`. Check interrupt before/after helper. `stop()` / `handle_speech_start()` call `helper.cancel()`. Telemetry `agent_summarize`.

`build_pipeline`: if agents.enabled, construct HelperAgent (optional second LlmClient when helper_base_url set).

- [ ] TDD: disabled = no complete(); overflow writes note; interrupt skips write
- [ ] Commit `feat: wire specialist helper after spoken turns`

---

### Task D4: Docs

README, roadmap status, `docs/superpowers/checklists/2026-08-16-phase-d-acceptance.md`

- [ ] Commit `docs: Phase D specialist agents usage and acceptance checklist`
