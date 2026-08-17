# Voice Ecosystem & Eval (Phase E) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Design:** `docs/superpowers/specs/2026-08-16-voice-eval-packaging-design.md`

**Goal:** Add drop-in profile packs and a CI-safe eval harness (sanitize + fake-clock latency budgets) without changing the default offline loop.

**Architecture:** Discover `profiles/<id>/manifest.yaml`; apply personality/voice in `build_pipeline`. Injectable metrics clock + `eval_harness` over committed fixtures.

**Tech Stack:** Existing PyYAML, pytest; no new runtime deps

## Global Constraints

- No model/GPU work in default pytest
- Missing `profiles` key → no pack applied
- Adding a pack requires only files under `profiles/`
- Imports at file top; commit only the current task’s files

---

### Task E1: Profile pack discovery + config

**Files:** `src/voice/profiles.py`, `config.py`, `config.example.yaml`, `tests/test_profiles.py`, sample `profiles/casual/`

**Interfaces:**
```python
@dataclass(frozen=True)
class ProfilePack:
    id: str
    name: str
    version: str
    path: Path
    personality_text: str
    voice: str  # may be ""

def discover_profile_packs(packs_dir: Path) -> list[ProfilePack]: ...
def resolve_profile_pack(packs: list[ProfilePack], pack_id: str) -> ProfilePack | None: ...
```

- [ ] TDD + commit `feat: discover versioned voice profile packs`

---

### Task E2: Apply active pack in build_pipeline

**Files:** `__main__.py`, `prompting.py` (append personality), tests

When `config.profiles.active` resolves, append personality to system prompt; select matching voice if present.

- [ ] TDD + commit `feat: apply active profile pack at pipeline build`

---

### Task E3: Fake-clock metrics + eval harness

**Files:** `metrics.py`, `eval_harness.py`, `tests/eval/`, `tests/test_eval_harness.py`, `tests/test_metrics.py`

- [ ] TDD + commit `feat: add CI-safe eval harness and latency budgets`

---

### Task E4: Docs, matrix, golden-WAV opt-in script

README capability matrix, roadmap status, checklist, `scripts/eval_golden_wav.sh` (exits 0 with skip message unless `VOICE_GOLDEN=1`).

- [ ] Commit `docs: Phase E profile packs, eval harness, and capability matrix`
