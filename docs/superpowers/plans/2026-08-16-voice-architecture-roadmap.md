# Voice Architecture Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Spec digest:** `docs/superpowers/specs/2026-08-16-voice-architecture-from-chatgpt-plan.md`  
> **PDF source:** `docs/VOICE_ChatGPT-Plan.pdf` (concept outline only; do not treat filler pages as requirements)

**Goal:** Evolve the existing local Voice app from a working conversational loop into the PDF’s architecture (core → memory → tools → agents → ecosystem), without regressing latency or local-first constraints.

**Architecture:** One Python orchestrator (`src/voice/`) plus local `llama-server`. New capabilities appear as focused modules behind stable interfaces. Prefer config/prompt swaps over new processes until a subsystem proves it needs isolation.

**Tech Stack:** Python 3.11+, sounddevice, Silero VAD, whisper.cpp, llama.cpp Metal, F5-TTS MLX + Piper hybrid, Tkinter, PyYAML, pytest

## Global Constraints

- Platform: macOS Apple Silicon (M3 Pro / 36 GB) primary target
- No cloud model APIs for the conversational core
- One conversational LLM until Phase D (agents)
- Reasoning / thinking mode off for ordinary voice turns
- Do not add a separate speech-rewrite LLM unless Phase A quality still fails after prompt + sanitizer work
- Prefer resident whisper.cpp over a full STT rewrite to MLX Whisper
- Keep barge-in cancel correct whenever pipeline timing changes
- Large model weights stay untracked; scripts document downloads
- Detailed TDD task lists below cover **Phase A only**; later phases get a short design + task list when that phase starts (YAGNI)

---

## Phase map (PDF → judged roadmap)

| Phase | PDF idea | Status | Ships when |
|---|---|---|---|
| **0** | Conversational core | **Done** (MVP + F5 hybrid TTS) | Offline loop, UI, barge-in, clone voices |
| **A** | Low latency + hardening | **Implemented** (human checklist pending) | Warm TTFA ≤ ~1.5 s; STT not CLI-per-turn; telemetry; human E2E pass |
| **B** | Memory + context orchestration | **Implemented** (human checklist pending) | Preferences + episodic recall without blowing context |
| **C** | Tools | **Implemented** (human checklist pending) | Opt-in local tools; core stays tool-free by default |
| **D** | Specialist agents | Planned | Optional helpers; default path still one LLM |
| **E** | Ecosystem + eval | Planned | Profiles marketplace-style packaging, regression harness |
| **F** | Future directions | Backlog | Emotion, multimodal, adaptive personality |

```mermaid
flowchart LR
  P0[Phase 0 Core done] --> PA[Phase A Latency]
  PA --> PB[Phase B Memory]
  PB --> PC[Phase C Tools]
  PC --> PD[Phase D Agents]
  PD --> PE[Phase E Ecosystem]
  PE --> PF[Phase F Future]
```

---

## File map (current + planned)

### Already own these responsibilities

| File | Responsibility |
|---|---|
| `src/voice/audio_io.py` | Mic/speaker streams, barge-in-aware playback |
| `src/voice/vad.py` | Silero end-of-turn |
| `src/voice/stt.py` | Whisper adapter (today: CLI-per-turn) |
| `src/voice/llm.py` | Streaming OpenAI-compatible client + cancel |
| `src/voice/chunker.py` | Phrase boundaries for TTS |
| `src/voice/speak_text.py` | Speech sanitizer / gesture map |
| `src/voice/tts.py` / `tts_f5.py` / `tts_factory.py` | Piper / F5 / hybrid |
| `src/voice/voices.py` | Voice + clone discovery |
| `src/voice/prompting.py` | System + world context assembly |
| `src/voice/session.py` | Turn / barge-in session state |
| `src/voice/pipeline.py` | Orchestration |
| `src/voice/ui.py` | Tkinter control surface |
| `src/voice/config.py` | YAML config |

### Planned new modules (by phase)

| Phase | Create | Responsibility |
|---|---|---|
| A | `src/voice/metrics.py` | Local latency / stage timers (JSONL or in-memory) |
| A | `src/voice/stt_resident.py` (or extend `stt.py`) | Resident whisper process / server client |
| B | `src/voice/memory/` | Working + episodic store + retrieval |
| B | `src/voice/context.py` | Selective prompt injection |
| C | `src/voice/tools/` | Tool registry + local executors |
| D | `src/voice/agents/` | Optional specialist routing (config-gated) |
| E | `tests/eval/` | Scripted conversation + latency fixtures |

---

## Phase 0 — Conversational core (COMPLETE)

**Outcome:** Local Mic → VAD → STT → LLM → chunked TTS → speakers with barge-in and Tkinter UI.

**Done means:** Package runs with `llama-server` + `python3.11 -m voice`; Piper and hybrid F5 paths exist; clone profiles under `voices/clones/`.

**Do not rebuild.** Only touch Phase 0 modules when Phase A requires it.

---

## Phase A — Latency & quality hardening (NEXT)

**PDF coverage:** §§5 Low Latency, §6 STT (endpointing/partials), §9 Speech rewrite (light), §15 Interruptions, §19 Observability, §21 Failure handling  
**Why before memory:** Memory adds tokens and retrieval latency; prove phone-feel first.

### Acceptance

1. After warm start, measured **time from end-of-turn → first PCM out** ≤ **1.5 s** on typical short replies (instrumented, not guessed)
2. STT does **not** reload model weights every turn
3. Barge-in still cancels LLM stream and playback within one VAD speech onset
4. F5 cold start is documented; warm path meets the budget or falls back to Piper without hanging
5. `pytest` green; one documented manual mic checklist

### Task A1: Stage latency telemetry

**Files:**
- Create: `src/voice/metrics.py`
- Create: `tests/test_metrics.py`
- Modify: `src/voice/pipeline.py` (emit stage timers)
- Modify: `src/voice/config.py` / `config.example.yaml` (`telemetry.enabled`, `telemetry.log_path`)

**Interfaces:**
- Consumes: pipeline stage boundaries (vad_end, stt_done, llm_first_token, tts_first_audio)
- Produces: `MetricsSink.mark(name: str, **fields) -> None` and `MetricsSink.span(name: str) -> context manager`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
from voice.metrics import MetricsSink

def test_span_records_duration_ms():
    sink = MetricsSink(enabled=True)
    with sink.span("stt"):
        pass
    events = sink.events()
    assert any(e["name"] == "stt" and "duration_ms" in e for e in events)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py::test_span_records_duration_ms -v`  
Expected: FAIL (module missing)

- [ ] **Step 3: Implement minimal `MetricsSink`**

```python
# src/voice/metrics.py
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator


class MetricsSink:
    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = enabled
        self._events: list[dict[str, Any]] = []

    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    def mark(self, name: str, **fields: Any) -> None:
        if not self.enabled:
            return
        self._events.append({"name": name, "t": time.perf_counter(), **fields})

    @contextmanager
    def span(self, name: str, **fields: Any) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._events.append(
                {
                    "name": name,
                    "duration_ms": (time.perf_counter() - t0) * 1000.0,
                    **fields,
                }
            )
```

- [ ] **Step 4: Wire spans in `pipeline.py` around STT, LLM first-token, TTS first-chunk**

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_metrics.py -v`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/voice/metrics.py tests/test_metrics.py src/voice/pipeline.py src/voice/config.py config.example.yaml
git commit -m "$(cat <<'EOF'
feat: add stage latency telemetry for voice pipeline

EOF
)"
```

### Task A2: Resident STT (kill per-turn whisper CLI reload)

**Files:**
- Modify: `src/voice/stt.py` (or create `src/voice/stt_resident.py` + factory)
- Modify: `src/voice/config.py` / `config.example.yaml` (`stt.mode: cli | resident`)
- Modify: `scripts/` if a small whisper server helper is needed
- Test: `tests/test_stt_resident.py` (mock subprocess / socket; no model download in CI)

**Interfaces:**
- Consumes: same `transcribe(pcm16: bytes, sample_rate: int) -> str`
- Produces: identical caller API so `pipeline.py` does not branch on transport

**Approach (preferred order):**
1. Keep `whisper.cpp` binary; run a **long-lived** process or whisper-server style endpoint if available in the installed brew formula
2. If server mode is unavailable, keep one warm subprocess with stdin/file protocol — still avoid reloading `-m` every turn
3. Do **not** migrate to MLX Whisper in this task unless resident whisper.cpp proves impossible

- [ ] **Step 1: Write failing test that `SttEngine` (or factory) can be constructed in `resident` mode and `transcribe` does not invoke a full cold CLI with `-m` every call when a fake runner tracks invocations**

- [ ] **Step 2: Implement resident adapter behind existing interface**

- [ ] **Step 3: Manual warm-path timing with telemetry from A1; record STT span before/after**

- [ ] **Step 4: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat: keep whisper model resident across turns

EOF
)"
```

### Task A3: Warm-path TTS readiness

**Files:**
- Modify: `src/voice/tts_f5.py` / `tts_factory.py` / `pipeline.py` or `__main__.py`
- Modify: `README.md` (warmup note)
- Test: extend factory tests; mock F5 load

**Behavior:**
- Optional `tts.warmup_on_start: true` synthesizes a 1–3 word phrase at Start so first user turn is warm
- Hybrid still falls back to Piper on F5 failure without crashing the session

- [ ] **Step 1: Test that enabling warmup calls synthesize once at start (mocked engine)**

- [ ] **Step 2: Implement warmup hook on UI Start / pipeline start**

- [ ] **Step 3: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat: optional F5 TTS warmup on session start

EOF
)"
```

### Task A4: Speech-rewrite quality without a second LLM

**Files:**
- Modify: `src/voice/speak_text.py`, `prompts/system.txt`
- Test: `tests/test_speak_text.py`

**Behavior:**
- Expand sanitizer: strip remaining markdown/tables, normalize ellipses, map more gesture tags
- Tighten system prompt for spoken brevity (already partly done — verify + extend)
- Explicit non-goal: dedicated rewrite model

- [ ] **Step 1: Add failing tests for new sanitize cases**

- [ ] **Step 2: Implement + run `pytest tests/test_speak_text.py -v`**

- [ ] **Step 3: Commit**

```bash
git commit -m "$(cat <<'EOF'
improve: broaden speak_text sanitizer for spoken delivery

EOF
)"
```

### Task A5: Human acceptance checklist + failure UX

**Files:**
- Create: `docs/superpowers/checklists/2026-08-16-phase-a-acceptance.md`
- Modify: `src/voice/ui.py` (surface STT/LLM/TTS errors in status without killing restart)
- Test: `tests/test_ui_callbacks.py` as needed

**Checklist must include:** warm TTFA measurement, barge-in, Stop→Start, clone voice A/B, Piper fallback, offline (wifi off) smoke

- [ ] **Step 1: Write checklist doc with pass/fail lines**

- [ ] **Step 2: Fix any failure-handling gaps found while running the checklist**

- [ ] **Step 3: Commit checklist + fixes**

---

## Phase B — Memory & context orchestration

**PDF coverage:** §§13–14  
**Depends on:** Phase A acceptance  

### Scope

- **Working memory:** current session turns (already partly in `session.py`) — formalize budgeted window
- **Preferences:** small durable YAML/JSON (`~/.voice/preferences.yaml` or `data/preferences.yaml`) — name, address style, voice defaults
- **Episodic:** append-only local store of summarized turns; retrieve top-k by simple keyword / embedding **only if** a local embed model fits without starving LLM+TTS RAM
- **Context composer:** `PromptComposer` injects only selected snippets + personality + world_context

### Explicit non-goals

- Cloud vector DBs
- Full semantic graph in v1 of memory
- Auto-memorizing everything (selective + user-visible “remember this”)

### Exit criteria

- User can say “remember that I prefer short answers” and hear it on a later session
- Token budget for memory injection is configurable and visible in telemetry
- Turning memory off restores Phase A behavior exactly

**When starting Phase B:** write a dedicated design + TDD plan (`docs/superpowers/plans/YYYY-MM-DD-voice-memory.md`) before coding.

---

## Phase C — Tool framework

**PDF coverage:** §18  
**Depends on:** Phase B (so tools can write preferences/notes cleanly)

### Scope

- Tool interface: `name`, JSON schema, `run(args) -> str`, timeout, online/offline flag
- Default **offline** tools only (e.g. local clock/timezone already in world_context, note store, config get/set)
- LLM tool-calling via llama-server if stable; otherwise constrained “action lines” parsed by the app
- Never block TTS forever: tool calls have hard timeouts and spoken error fallbacks

### Explicit non-goals

- Arbitrary shell execution
- Browser automation as a default tool
- Online search unless user opts in and Phase E privacy flags are clear

### Exit criteria

- One local tool round-trip spoken end-to-end with barge-in still working
- Tools disabled by config → zero tool prompts in context

**When starting Phase C:** dedicated plan `...-voice-tools.md`.

---

## Phase D — Specialist agents

**PDF coverage:** PDF Phase 4  
**Depends on:** Phase C + proven latency budget headroom  

### Scope

- Config-gated optional **helper** models (small GGUF) for routing, summarization, or memory extract — not on the hot path for every token
- Default conversation path remains single Qwen-class model
- Clear cancellation: barge-in aborts helpers too

### Judgment

Defer dual-model routing until Phase A metrics show headroom. A resident 1B helper that steals Metal bandwidth can **worsen** TTFA on 36 GB if STT+TTS+8B already contend.

### Exit criteria

- Feature flag off = identical to Phase C
- Feature flag on = measurable benefit on at least one scenario (e.g. memory summarization) without breaking TTFA budget on short turns

---

## Phase E — Ecosystem, eval, packaging

**PDF coverage:** §§19–20, PDF Phase 5  
**Depends on:** stable A–C  

### Scope

- Voice + personality **profile packs** (directory + manifest), versioned
- Eval harness: fixed transcripts → expected sanitize / latency ceilings (CI-safe mocks)
- Optional golden WAV smoke outside CI
- Clear offline vs online capability matrix in README

### Exit criteria

- `pytest` includes latency-budget unit tests with fake clocks
- A profile pack can be added without code changes

---

## Phase F — Future directions (backlog only)

**PDF coverage:** §24  

- Emotion recognition, multimodal input, adaptive personalities, collaborative agents, continuous benchmarking  
- Revisit only after E; each item needs its own design gate

---

## Personality & voice profiles (cross-cutting, light touches)

**PDF §§11–12** — do **not** build a trait engine early.

| When | What |
|---|---|
| Now / A | Clone dirs + Piper voices + system prompt; optional `prompts/personalities/*.txt` selectable in UI |
| B | Preferences link default personality + voice |
| E | Packaged profile manifests |

---

## Self-review (plan vs PDF)

| PDF section | Covered by |
|---|---|
| Vision / principles | Global constraints + Phase 0 done |
| Architecture services | File map; libraries not microservices |
| Voice pipeline / latency / STT | Phase 0 + A |
| Conversation / prompts / rewrite / TTS / profiles | Phase 0 + A4 + cross-cutting |
| Personality | Cross-cutting (prompt profiles) |
| Memory / context | Phase B |
| Interruptions | Phase 0 + A acceptance |
| Config / swappable models | Existing + each phase |
| Tools | Phase C |
| Observability / eval / failure | A1, A5, E |
| Security | Global constraints |
| Roadmap | Phase map (A inserted before memory) |
| Future | Phase F |

**Corrections applied:** MLX Whisper not required; speech-rewrite LLM deferred; dual agents deferred; Phase A latency inserted before PDF “Phase 2 memory.”

---

## Execution order

1. Complete **Phase A tasks A1→A5** on `feat/local-voice-chat` (or a branch from it)  
2. Stop and re-plan **Phase B** with a full TDD plan  
3. Do not start C/D/E until the prior phase’s exit criteria pass
