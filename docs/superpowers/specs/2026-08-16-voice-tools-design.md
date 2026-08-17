# Phase C — Tool Framework Design

**Date:** 2026-08-16  
**Status:** Approved for implementation (user: execute phase C)  
**Depends on:** Phase B memory stores  

## Goals

1. Pluggable local tools behind a shared interface (`name`, schema, `run`, timeout, offline flag).
2. At least one end-to-end offline tool round-trip that affects spoken output.
3. `tools.enabled: false` → zero tool instructions in the prompt (Phase B behavior).
4. Hard timeouts; barge-in still cancels generation/playback.

## Non-goals

- Shell execution, browser automation, web search
- Native OpenAI `tools=` API (llama-server support varies; defer)
- Multi-hop tool chains beyond one tool round + one spoken continuation

## Approach: constrained action markers (not native tool API)

LLMs speak; they must not read tool JSON aloud. When tools are enabled, the system prompt documents a **silent** marker format:

```text
<<tool:NAME|{"arg":"value"}>>
```

Pipeline behavior:

1. Stream tokens as today through the phrase chunker.
2. Before TTS, strip any complete `<<tool:...>>` markers from the phrase (do not speak them).
3. After the assistant stream finishes (or when a marker is complete mid-turn), execute queued tool calls with a wall-clock timeout.
4. If any tools ran, append a short `tool`/`user` result message to history and run **one** continuation `stream_chat` whose output is spoken (no nested tools in v1, or strip nested markers without executing).
5. On timeout/error: spoken-safe fallback string from the tool layer (e.g. “I couldn’t save that note.”) injected as the tool result.

### Default offline tools

| Name | Args | Effect |
|---|---|---|
| `local_time` | `{}` | Return local ISO date/time + timezone name |
| `preference_set` | `{key, value}` | Write `PreferencesStore` (requires memory enabled / store present) |
| `note_add` | `{text}` | Append episodic note |

If memory stores are unavailable, preference/note tools return a clear error string (still spoken via continuation).

## Config

```yaml
tools:
  enabled: false          # missing key → false
  timeout_ms: 2000
  allow_online: false     # reserved; online tools refuse when false
```

## Prompt injection

When enabled, append a short **Tools** section to the base system prompt (before memory inject, or as part of base at startup). Document marker format + available tool names/arg summaries. When disabled, omit entirely.

## Telemetry

`metrics.mark("tool_call", name=..., ok=bool, duration_ms=...)`

## Acceptance

1. Enabled: model emits `<<tool:local_time|{}>>` (or test injects it); user hears a continuation that includes the time; marker never spoken.
2. Disabled: no Tools section in prompt; markers (if any) are not executed (strip as unknown speech garbage OR leave — prefer strip silently without execute when disabled).
3. Timeout path covered by unit test with a slow fake tool.
4. Barge-in during tool wait or continuation cancels further speech.
