# Voice Tools (Phase C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Design:** `docs/superpowers/specs/2026-08-16-voice-tools-design.md`

**Goal:** Add a config-gated offline tool framework with silent `<<tool:name|{json}>>` markers, timed execution, and one spoken continuation turn.

**Architecture:** `Tool` protocol + registry; marker parse/strip; `ToolRunner` with timeout; pipeline hooks after assistant stream; optional Tools section in system prompt.

**Tech Stack:** Python 3.11+, existing pipeline/memory, pytest (no new deps)

## Global Constraints

- Offline tools only by default; no shell/browser/search
- `tools.enabled: false` (default when missing) → no tool prompt section; no execution
- Hard `timeout_ms` per call; never block forever
- Preserve barge-in cancel semantics
- Imports at file top only; commit only current task files

---

## File map

| File | Role |
|---|---|
| `src/voice/config.py` | `ToolsConfig` |
| `config.example.yaml` | tools section |
| `src/voice/tools/base.py` | `Tool` protocol / dataclass |
| `src/voice/tools/registry.py` | Registry + builtin registration |
| `src/voice/tools/builtins.py` | `local_time`, `preference_set`, `note_add` |
| `src/voice/tools/markers.py` | extract + strip `<<tool:...>>` |
| `src/voice/tools/runner.py` | run with timeout |
| `src/voice/tools/__init__.py` | exports |
| `src/voice/prompting.py` | tools section helper |
| `src/voice/pipeline.py` | wire tools into turn |
| `src/voice/__main__.py` | build registry when enabled |
| tests | `tests/test_tools_*.py`, pipeline tools tests |

---

### Task C1: ToolsConfig + Tool protocol + marker parse/strip

**Files:** config, `tools/base.py`, `tools/markers.py`, tests

**Interfaces:**
```python
@dataclass(frozen=True)
class ToolsConfig:
    enabled: bool = False
    timeout_ms: int = 2000
    allow_online: bool = False

@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]

def extract_tool_calls(text: str) -> list[ToolCall]: ...
def strip_tool_markers(text: str) -> str: ...
```

Marker regex: `<<tool:(?P<name>[a-zA-Z0-9_]+)\|(?P<json>\{.*?\})>>` with careful non-greedy JSON; invalid JSON → skip call but still strip marker.

- [ ] TDD + commit `feat: add tools config and tool marker parsing`

---

### Task C2: Registry, builtins, runner timeout

**Files:** `registry.py`, `builtins.py`, `runner.py`, tests

**Interfaces:**
```python
class Tool(Protocol):
    name: str
    description: str
    parameters_schema: dict[str, Any]
    offline: bool
    def run(self, args: dict[str, Any]) -> str: ...

def build_default_registry(*, preferences=None, episodic=None) -> ToolRegistry: ...

class ToolRunner:
    def __init__(self, registry, *, timeout_ms: int, allow_online: bool): ...
    def run_all(self, calls: list[ToolCall]) -> list[tuple[ToolCall, str]]: ...
```

Timeout via `concurrent.futures.ThreadPoolExecutor` + `future.result(timeout=...)`.

- [ ] TDD + commit `feat: add offline tool registry and timed runner`

---

### Task C3: Prompt tools section + pipeline wiring

**Files:** prompting, pipeline, __main__, tests

When tools enabled:
1. Append tools instructions + catalog to `base_system_prompt` at build time (or each turn before memory inject — prefer once at startup into base).
2. In `_speak` / before synthesize: `strip_tool_markers` on phrase text.
3. After main stream: collect calls from full assistant text (and from stripped pieces); `runner.run_all`; if results: append to history, one continuation stream (strip tools, no second execute), speak continuation.
4. Check `_interrupted` before/after tools and before continuation.
5. `metrics.mark("tool_call", ...)`

`PipelineTools` dataclass similar to `PipelineMemory`.

- [ ] TDD: disabled = no execute; enabled local_time continuation; barge-in during tool skips continuation
- [ ] Commit `feat: wire offline tools into the voice pipeline`

---

### Task C4: Docs + acceptance checklist

**Files:** README, roadmap status, `docs/superpowers/checklists/2026-08-16-phase-c-acceptance.md`

- [ ] Commit `docs: Phase C tools usage and acceptance checklist`

---

## Self-review vs exit criteria

| Criterion | Task |
|---|---|
| One local tool round-trip spoken | C2+C3 |
| Tools disabled → zero tool prompts | C1+C3 |
| Timeout + barge-in | C2+C3 |
