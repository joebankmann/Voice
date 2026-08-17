# Voice Memory (Phase B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Design:** `docs/superpowers/specs/2026-08-16-voice-memory-design.md`

**Goal:** Add local preferences + episodic memory with budgeted prompt injection and working-history trim, toggleable via `memory.enabled`.

**Architecture:** File-backed `PreferencesStore` and `EpisodicStore`, heuristic `RememberIntent`, `PromptComposer` extending `build_system_prompt`, session history trim, wired in `__main__` / pipeline turn path.

**Tech Stack:** Python 3.11+, PyYAML, pytest (no new heavy deps; no embeddings)

## Global Constraints

- Fully local; no cloud vector DBs or embed models
- `memory.enabled: false` restores Phase A prompt behavior
- No second LLM for summarization
- Keep barge-in / pipeline cancel behavior unchanged
- Imports at top of modules only
- Commit only files for the current task

---

## File map

| File | Responsibility |
|---|---|
| `src/voice/config.py` | `MemoryConfig` + load |
| `config.example.yaml` | memory section |
| `src/voice/memory/preferences.py` | YAML preferences CRUD |
| `src/voice/memory/episodic.py` | JSONL append + keyword retrieve |
| `src/voice/memory/intent.py` | Remember-intent extraction |
| `src/voice/memory/__init__.py` | Exports |
| `src/voice/prompting.py` | `PromptComposer` / extend `build_system_prompt` |
| `src/voice/session.py` | History message budget |
| `src/voice/pipeline.py` | Apply remember + rebuild inject per turn; metrics |
| `src/voice/__main__.py` | Construct stores/composer; pass into pipeline |
| Tests under `tests/test_memory_*.py`, extend existing |

---

### Task B1: MemoryConfig + PreferencesStore

**Files:**
- Create: `src/voice/memory/__init__.py`, `src/voice/memory/preferences.py`
- Modify: `src/voice/config.py`, `config.example.yaml`
- Test: `tests/test_memory_preferences.py`, `tests/test_config.py`

**Interfaces:**
- `MemoryConfig(enabled, preferences_path, episodic_path, max_history_messages, max_episodic_hits, max_inject_chars)`
- `PreferencesStore(path).get_all() -> dict[str, str]`
- `PreferencesStore.set(key, value) -> None`
- `PreferencesStore.load()` / persist on set

- [ ] **Step 1: Write failing tests** for set/get round-trip on a temp YAML path and config defaults

- [ ] **Step 2: Implement PreferencesStore + MemoryConfig (defaults: enabled=False for safe Phase A parity until wired; example yaml may show enabled=true with comment — prefer default enabled=True in example, False in dataclass if missing key for backward compat)**

Decision: missing `memory` key → `enabled=False` (Phase A identical). `config.example.yaml` sets `enabled: true`.

- [ ] **Step 3: pytest focused + commit**

```bash
git commit -m "$(cat <<'EOF'
feat: add memory config and preferences store

EOF
)"
```

---

### Task B2: EpisodicStore + keyword retrieve

**Files:**
- Create: `src/voice/memory/episodic.py`
- Test: `tests/test_memory_episodic.py`

**Interfaces:**
- `EpisodicStore(path).add(text: str, tags: list[str] | None = None) -> None`
- `EpisodicStore.retrieve(query: str, *, limit: int) -> list[str]` ranked by token overlap

- [ ] **Step 1: Failing tests** — add notes, retrieve by keyword, empty query returns []

- [ ] **Step 2: Implement JSONL store**

- [ ] **Step 3: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat: add local episodic memory with keyword retrieval

EOF
)"
```

---

### Task B3: Remember intent extractor

**Files:**
- Create: `src/voice/memory/intent.py`
- Test: `tests/test_memory_intent.py`

**Interfaces:**
```python
@dataclass(frozen=True)
class MemoryWrite:
    kind: Literal["preference", "episodic"]
    key: str | None  # preference key
    value: str

def extract_remember_intent(text: str) -> MemoryWrite | None: ...
```

Preference keys: `prefer` / `name` / `address_as` from patterns; episodic value = remaining text.

- [ ] **Step 1–3: TDD + commit**

```bash
git commit -m "$(cat <<'EOF'
feat: detect remember intents for preferences and notes

EOF
)"
```

---

### Task B4: Working-memory history trim

**Files:**
- Modify: `src/voice/session.py`
- Test: `tests/test_session.py`

**Interfaces:**
- `ConversationSession(max_history_messages: int | None = None)`
- After append user/assistant, trim oldest messages if over limit (drop from front; if odd count keep consistent — drop oldest until `len(history) <= max`)

When `max_history_messages` is None or ≤0, no trim (Phase A when memory disabled).

- [ ] **Step 1–3: TDD + commit**

```bash
git commit -m "$(cat <<'EOF'
feat: budget conversation working-memory history

EOF
)"
```

---

### Task B5: PromptComposer + pipeline wiring

**Files:**
- Modify: `src/voice/prompting.py`, `src/voice/pipeline.py`, `src/voice/__main__.py`
- Test: `tests/test_prompting.py`, `tests/test_pipeline_memory.py` (new)

**Interfaces:**
```python
def compose_system_prompt(
    system_prompt_path,
    *,
    world_context_path=None,
    today=None,
    preferences: dict[str, str] | None = None,
    episodic_notes: list[str] | None = None,
    max_inject_chars: int = 1200,
) -> str: ...
```

Pipeline responsibilities when memory enabled:
1. On `run_turn(transcript)` before reply: `extract_remember_intent`; write to stores; mark metrics
2. Retrieve episodic for transcript; rebuild `self.system_prompt` via composer (or pass dynamic suffix — prefer updating `self.system_prompt` each turn from a base + inject)
3. Keep `base_system_prompt` immutable from startup; inject prefs+episodic each turn

`VoicePipeline.__init__` gains optional `memory: MemoryFacade | None` where facade holds stores + config helpers.

Or pass `preferences_store`, `episodic_store`, `memory_config` optionally — if None, Phase A path.

- [ ] **Step 1: Tests** — composer truncates; pipeline persists preference and injects on next `run_turn`; disabled memory skips extract

- [ ] **Step 2: Implement + wire `__main__.build_pipeline`**

- [ ] **Step 3: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat: inject budgeted memory into voice system prompts

EOF
)"
```

---

### Task B6: Docs + acceptance notes

**Files:**
- Modify: `README.md` (short memory section)
- Modify: `docs/superpowers/plans/2026-08-16-voice-architecture-roadmap.md` (Phase B status)
- Create: `docs/superpowers/checklists/2026-08-16-phase-b-acceptance.md`

- [ ] Checklist: remember preference across restart; memory off parity; telemetry inject mark; barge-in still works
- [ ] Commit

```bash
git commit -m "$(cat <<'EOF'
docs: Phase B memory usage and acceptance checklist

EOF
)"
```

---

## Self-review

| Exit criterion | Task |
|---|---|
| Remember preference across sessions | B1+B3+B5 |
| Budget + telemetry | B5 |
| Memory off = Phase A | B1 default + B5 skip |
| No embeddings | B2 keywords only |
