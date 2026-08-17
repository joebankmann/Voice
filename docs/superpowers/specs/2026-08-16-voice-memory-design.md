# Phase B — Memory & Context Orchestration Design

**Date:** 2026-08-16  
**Status:** Approved for implementation (user: execute phase B)  
**Depends on:** Phase A (telemetry, resident STT, warmup)

## Goals

1. Persist user preferences across sessions (e.g. “remember that I prefer short answers”).
2. Keep a small local episodic note store with **keyword** retrieval (no embed model).
3. Budget working-memory (session history) and memory injection into the system prompt.
4. `memory.enabled: false` restores Phase A prompt behavior exactly.

## Non-goals

- Cloud vector DBs, embeddings, semantic graphs
- Auto-memorizing every turn
- Dual/helper LLM for summarization
- Tools (Phase C)

## Architecture

```
User transcript
  → RememberIntent.extract(text)  # optional write
  → PreferencesStore / EpisodicStore
  → PromptComposer.build(...)     # system + world + prefs + episodic (budgeted)
  → ConversationSession.history   # trimmed working window
  → LlmClient.stream_chat
```

### Stores (local files)

| Store | Path (default) | Format |
|---|---|---|
| Preferences | `data/preferences.yaml` | YAML map: `prefs: {key: value}`, plus freeform `notes: []` |
| Episodic | `data/episodic.jsonl` | One JSON object per line: `{ts, text, tags[]}` |

### Remember intents (v1 heuristics)

Match (case-insensitive) prefixes like:

- `remember that …`
- `please remember …`
- `don't forget that …`
- `remind yourself that …`

Payload after the cue:

- Preference if it matches `i prefer …`, `call me …`, `my name is …`, `address me as …`
- Otherwise append an episodic note

Spoken ack is left to the LLM (system prompt tells it memory was saved); no forced canned TTS.

### Context injection

`PromptComposer` builds:

1. Base system prompt + date + world context (existing)
2. If memory enabled and prefs non-empty: `User preferences:\n- …`
3. If episodic hits: `Relevant remembered notes:\n- …`
4. Truncate preference+episodic block to `memory.max_inject_chars`

Retrieval: lowercase keyword overlap of current user utterance tokens (len≥3) against note text; top `memory.max_episodic_hits` by score.

### Working memory

`ConversationSession` keeps at most `memory.max_history_messages` messages (drop oldest pairs preferentially, always keep system out of history — history is user/assistant only today).

### Telemetry

When telemetry enabled: `metrics.mark("memory_inject", prefs=N, episodic=M, chars=C)` once per turn before LLM stream.

### Config

```yaml
memory:
  enabled: true
  preferences_path: data/preferences.yaml
  episodic_path: data/episodic.jsonl
  max_history_messages: 24
  max_episodic_hits: 3
  max_inject_chars: 1200
```

## Acceptance

1. With memory on, “remember that I prefer short answers” persists and is present in the next process’s system prompt.
2. Token/char budget enforced; telemetry shows inject sizes when enabled.
3. `memory.enabled: false` → identical prompt composition to Phase A (no prefs/episodic sections; history unbounded or only existing behavior — use a high default when disabled and skip trim).
4. Unit tests cover stores, intent, composer, session trim without mic/models.
