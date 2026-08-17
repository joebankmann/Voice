# Phase D — Specialist Agents Design

**Date:** 2026-08-16  
**Status:** Approved for implementation (user: execute phase D)  
**Depends on:** Phase C tools  

## Goals

1. Optional helper LLM for **off-path** work only — never before first audio.
2. Default conversation path stays the single main model.
3. `agents.enabled: false` (missing key) restores Phase C behavior exactly.
4. Barge-in / Stop cancel the helper as well as the main stream.

## Non-goals

- Dual-model routing on every token (hurts TTFA on 36 GB)
- Multi-agent collaboration / specialist personas
- A second always-resident GGUF by default

## Judgment

A resident helper that shares Metal with Qwen-8B + Whisper + F5 can worsen phone-feel. Phase D therefore:

- Defaults **off**
- Runs **after** an uninterrupted spoken turn
- Uses the **same** `llama-server` via a non-streaming `complete()` unless `helper_base_url` points elsewhere
- Times out and no-ops on failure (never blocks the next listen)

## Capability (v1): memory summarizer

When memory is enabled and working history would drop messages, the helper summarizes the **evicted** slice into one short episodic note.

Hook: `ConversationSession` returns dropped messages from trim; after a successful (not interrupted) reply, `HelperAgent.summarize(dropped)` → `EpisodicStore.add`.

If memory is disabled, the helper is a no-op even when `agents.enabled` is true.

## Interfaces

```python
class HelperAgent:
    def complete(self, prompt: str) -> str: ...
    def cancel(self) -> None: ...
    def summarize_dropped(self, messages: list[dict[str, str]]) -> str | None: ...
```

`LlmClient.complete(messages, system_prompt) -> str` — non-streaming; respects `cancel()`.

## Config

```yaml
agents:
  enabled: false
  helper_base_url: ""    # empty → reuse main LLM
  helper_model: ""       # empty → reuse main model id
  timeout_ms: 4000
```

## Telemetry

`metrics.mark("agent_summarize", ok=bool, duration_ms=..., chars=N)`

## Acceptance

1. Flag off: no helper calls, no extra prompt, history trim unchanged besides existing Phase B.
2. Flag on + memory on: overflowing history produces an episodic note from dropped turns.
3. Interrupt during helper: no note written (or partial ignored); listening resumes.
4. Helper timeout/error: turn still succeeds; error may surface in status without crashing.
