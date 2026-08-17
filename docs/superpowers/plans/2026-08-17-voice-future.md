# Voice Future Directions (Phase F) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Design:** `docs/superpowers/specs/2026-08-17-voice-future-design.md`

**Goal:** Ship gated, local slices of the PDF’s future directions without putting new models on the spoken hot path.

**Architecture:** `FutureConfig` + small modules (`affect`, `inbox`, personality adapt) wired in prompt prep; optional second off-path helper; eval JSONL CLI.

**Tech Stack:** Existing pipeline, pytest, PyYAML

## Global Constraints

- Missing `future` / `agents.collaborative` → off (Phase E identical)
- No neural emotion, vision, or extra Metal-resident models
- Barge-in still cancels helpers
- Imports at file top

---

### Task F1: FutureConfig + affect + inbox + adaptive personality

Modules + config + unit tests; pipeline inject extras when enabled.

- [x] Implemented

### Task F2: Collaborative off-path helper pass

`HelperAgent.extract_actions`; pipeline after summarize when `agents.collaborative`.

- [x] Implemented

### Task F3: Continuous eval JSONL CLI + docs

`python -m voice.eval_harness`; README; roadmap; checklist.

- [x] Implemented
