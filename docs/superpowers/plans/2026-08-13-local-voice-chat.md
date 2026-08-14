# Local Real-Time Voice Chat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a fully local, streaming voice conversation loop on an M3 Pro / 36 GB Mac with barge-in and sub‑~1.5s time-to-first-audio under normal conditions.

**Architecture:** A Python orchestrator captures mic audio, runs Silero VAD, sends speech to whisper.cpp, streams completions from local `llama-server` (Qwen3-class ~8B GGUF, non-thinking), phrase-chunks tokens into Piper TTS, and cancels playback/generation on barge-in. A simple Tkinter window is the default control surface; audio remains in-process (not browser/WebRTC).

**Tech Stack:** Python 3.11+, Tkinter (stdlib), `sounddevice`, Silero VAD (PyTorch/`torch` or ONNX path as implemented), whisper.cpp, llama.cpp `llama-server` + Metal, Piper, PyYAML, pytest

## Global Constraints

- Platform: macOS Apple Silicon (M3 Pro, 36 GB) only for MVP verification
- No cloud model APIs; all inference local
- One conversational LLM only (no helper/router model in MVP)
- Thinking/reasoning mode off for ordinary turns
- Main model class: ~7–9B Q4_K_M or Q5_K_M GGUF
- TTS for MVP: Piper (not a high-fidelity upgrade)
- NSFW / LoRA work is out of scope for this plan
- UI is a simple Tkinter control surface (status, start/stop, interrupt, transcripts) — not a design-system polish pass
- Prefer small focused modules under `src/voice/`; no Docker requirement
- Spec: `docs/superpowers/specs/2026-08-13-local-voice-chat-design.md`

---

## Milestones

| Milestone | Outcome | Tasks | Done when |
|---|---|---|---|
| **M0 — Spec locked** | Design + plan approved; offline MVP scope clear | — | You say go |
| **M1 — Foundation** | Installable Python package, config, pure logic tested without mic/models | 1–3 | `pytest` green for config, chunker, session/barge-in events |
| **M2 — Local brain** | Streaming LLM client talks to `llama-server`; cancel works | 4 | Unit test with mock SSE passes; manual curl/stream to local server optional |
| **M3 — Ears & mouth** | Audio, VAD, whisper.cpp, Piper adapters exist | 5 | Adapters callable; VAD interface tests pass |
| **M4 — Conversation loop** | End-to-end pipeline with barge-in (CLI path) | 6 | Pipeline tests pass; `--cli` can listen → reply → interrupt |
| **M5 — Simple UI** | Tkinter Start/Stop/Interrupt + status + transcripts | 7 | UI controller tests pass; default `voice` opens window |
| **M6 — Runnable offline kit** | Scripts + README; models downloadable; documented smoke path | 8 | Fresh machine can follow README with network only for downloads |
| **M7 — Hardware acceptance** | Real M3 Pro mic/speaker verification | 9 | Offline chat works; barge-in feels immediate; TTFA ~0.5–1.5s target noted |

**Suggested build order:** M1 → M2 → M3 → M4 → M5 → M6 → M7 (no parallel milestones that share the pipeline).

**Phase 2 (after M7, not in this plan):** NSFW/model swap, better TTS, optional helper model, semantic turn detection.

---

## File Structure

```
Voice/
  README.md
  pyproject.toml
  config.example.yaml
  prompts/system.txt
  scripts/download_models.sh
  scripts/run_llama_server.sh
  src/voice/
    __init__.py
    __main__.py
    config.py
    audio_io.py
    vad.py
    stt.py
    llm.py
    chunker.py
    tts.py
    session.py
    pipeline.py
    ui.py
  tests/
    test_config.py
    test_chunker.py
    test_session.py
    test_llm_stream.py
    test_pipeline_barge_in.py
    test_ui_callbacks.py
```

---

### Task 1: Project scaffold and config

**Files:**
- Create: `pyproject.toml`
- Create: `config.example.yaml`
- Create: `prompts/system.txt`
- Create: `src/voice/__init__.py`
- Create: `src/voice/config.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: none
- Produces: `load_config(path: str | Path) -> AppConfig` and typed settings used by later tasks

- [ ] **Step 1: Write the failing config test**

```python
# tests/test_config.py
from pathlib import Path

from voice.config import load_config


def test_load_config_reads_core_fields(tmp_path: Path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
audio:
  sample_rate: 16000
  end_of_turn_silence_ms: 600
llm:
  base_url: http://127.0.0.1:8080/v1
  model: qwen3-8b
  temperature: 0.7
  system_prompt_path: prompts/system.txt
stt:
  whisper_bin: whisper-cli
  model_path: models/ggml-large-v3-turbo.bin
tts:
  piper_bin: piper
  voice_path: models/en_US-lessac-medium.onnx
vad:
  threshold: 0.5
  min_speech_ms: 250
""".strip()
    )
    cfg = load_config(cfg_path)
    assert cfg.audio.sample_rate == 16000
    assert cfg.audio.end_of_turn_silence_ms == 600
    assert cfg.llm.base_url == "http://127.0.0.1:8080/v1"
    assert cfg.llm.model == "qwen3-8b"
    assert cfg.stt.model_path.endswith("ggml-large-v3-turbo.bin")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Volumes/ComfyUI/Voice && python -m pytest tests/test_config.py -v`  
Expected: FAIL with `ModuleNotFoundError` or import error for `voice.config`

- [ ] **Step 3: Add project metadata and minimal implementation**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "voice"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "numpy",
  "pyyaml",
  "sounddevice",
  "httpx",
  "torch",
]

[project.optional-dependencies]
dev = ["pytest"]

[project.scripts]
voice = "voice.__main__:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

```python
# src/voice/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int
    end_of_turn_silence_ms: int


@dataclass(frozen=True)
class LlmConfig:
    base_url: str
    model: str
    temperature: float
    system_prompt_path: str


@dataclass(frozen=True)
class SttConfig:
    whisper_bin: str
    model_path: str


@dataclass(frozen=True)
class TtsConfig:
    piper_bin: str
    voice_path: str


@dataclass(frozen=True)
class VadConfig:
    threshold: float
    min_speech_ms: int


@dataclass(frozen=True)
class AppConfig:
    audio: AudioConfig
    llm: LlmConfig
    stt: SttConfig
    tts: TtsConfig
    vad: VadConfig


def load_config(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return AppConfig(
        audio=AudioConfig(**raw["audio"]),
        llm=LlmConfig(**raw["llm"]),
        stt=SttConfig(**raw["stt"]),
        tts=TtsConfig(**raw["tts"]),
        vad=VadConfig(**raw["vad"]),
    )
```

Also create `config.example.yaml` matching the test shape, `prompts/system.txt` with a short conversational system prompt, and empty `src/voice/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Volumes/ComfyUI/Voice && python -m pip install -e ".[dev]" && python -m pytest tests/test_config.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml config.example.yaml prompts/system.txt src/voice/__init__.py src/voice/config.py tests/test_config.py
git commit -m "$(cat <<'EOF'
chore: scaffold voice project and config loader

EOF
)"
```

---

### Task 2: Phrase chunker

**Files:**
- Create: `src/voice/chunker.py`
- Create: `tests/test_chunker.py`

**Interfaces:**
- Consumes: none
- Produces: `PhraseChunker` with `push(token: str) -> list[str]` and `flush() -> list[str]`

- [ ] **Step 1: Write failing chunker tests**

```python
# tests/test_chunker.py
from voice.chunker import PhraseChunker


def test_emits_on_sentence_end():
    c = PhraseChunker(max_chars=120)
    assert c.push("Hello") == []
    assert c.push(" there") == []
    assert c.push("!") == ["Hello there!"]


def test_emits_on_soft_length_break():
    c = PhraseChunker(max_chars=20)
    out = []
    out += c.push("This is a fairly long")
    out += c.push(" clause without end")
    assert out, "expected a soft flush before waiting forever"
    assert all(len(x) <= 40 for x in out)


def test_flush_returns_remainder():
    c = PhraseChunker(max_chars=120)
    assert c.push("Partial") == []
    assert c.flush() == ["Partial"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_chunker.py -v`  
Expected: FAIL with import error for `voice.chunker`

- [ ] **Step 3: Implement chunker**

```python
# src/voice/chunker.py
from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"[.!?…][\"')\]]*$")


class PhraseChunker:
    def __init__(self, max_chars: int = 120) -> None:
        self.max_chars = max_chars
        self._buf = ""

    def push(self, token: str) -> list[str]:
        self._buf += token
        return self._drain(force=False)

    def flush(self) -> list[str]:
        return self._drain(force=True)

    def _drain(self, force: bool) -> list[str]:
        emitted: list[str] = []
        while True:
            stripped = self._buf.strip()
            if not stripped:
                self._buf = ""
                break

            # Prefer sentence boundary
            for i, ch in enumerate(self._buf):
                if ch in ".!?…" and _SENTENCE_END.search(self._buf[: i + 1].rstrip()):
                    # include trailing quotes/brackets after punct if present
                    end = i + 1
                    while end < len(self._buf) and self._buf[end] in "\"')]\n ":
                        if self._buf[end] == "\n":
                            end += 1
                            break
                        end += 1
                    piece = self._buf[:end].strip()
                    self._buf = self._buf[end:]
                    if piece:
                        emitted.append(piece)
                    break
            else:
                if force and self._buf.strip():
                    emitted.append(self._buf.strip())
                    self._buf = ""
                elif len(self._buf) >= self.max_chars:
                    # soft break at last space
                    cut = self._buf.rfind(" ", 0, self.max_chars)
                    if cut <= 0:
                        cut = self.max_chars
                    piece = self._buf[:cut].strip()
                    self._buf = self._buf[cut:]
                    if piece:
                        emitted.append(piece)
                else:
                    break
                continue
        return emitted
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_chunker.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voice/chunker.py tests/test_chunker.py
git commit -m "$(cat <<'EOF'
feat: add streaming phrase chunker for TTS handoff

EOF
)"
```

---

### Task 3: Session state machine and barge-in signals

**Files:**
- Create: `src/voice/session.py`
- Create: `tests/test_session.py`

**Interfaces:**
- Consumes: none
- Produces: `SessionState` enum and `ConversationSession` with methods:
  - `on_user_speech_start() -> list[SessionEvent]`
  - `on_user_speech_end(transcript: str) -> list[SessionEvent]`
  - `on_assistant_audio_done() -> list[SessionEvent]`
  - Events include `CancelGeneration`, `StopPlayback`, `StartListening`, `RequestReply`

- [ ] **Step 1: Write failing session tests**

```python
# tests/test_session.py
from voice.session import ConversationSession, SessionEventType, SessionState


def test_barge_in_cancels_playback_and_generation():
    s = ConversationSession()
    s.force_state(SessionState.THINKING_SPEAKING)
    events = s.on_user_speech_start()
    types = [e.type for e in events]
    assert SessionEventType.STOP_PLAYBACK in types
    assert SessionEventType.CANCEL_GENERATION in types
    assert s.state == SessionState.LISTENING


def test_speech_end_requests_reply():
    s = ConversationSession()
    s.force_state(SessionState.LISTENING)
    events = s.on_user_speech_end("hello there")
    assert any(e.type == SessionEventType.REQUEST_REPLY and e.transcript == "hello there" for e in events)
    assert s.state == SessionState.THINKING_SPEAKING
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_session.py -v`  
Expected: FAIL with import error for `voice.session`

- [ ] **Step 3: Implement session module**

```python
# src/voice/session.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class SessionState(Enum):
    IDLE = auto()
    LISTENING = auto()
    THINKING_SPEAKING = auto()


class SessionEventType(Enum):
    START_LISTENING = auto()
    STOP_PLAYBACK = auto()
    CANCEL_GENERATION = auto()
    REQUEST_REPLY = auto()


@dataclass(frozen=True)
class SessionEvent:
    type: SessionEventType
    transcript: str | None = None


class ConversationSession:
    def __init__(self) -> None:
        self.state = SessionState.IDLE
        self.history: list[dict[str, str]] = []

    def force_state(self, state: SessionState) -> None:
        self.state = state

    def on_user_speech_start(self) -> list[SessionEvent]:
        events: list[SessionEvent] = []
        if self.state == SessionState.THINKING_SPEAKING:
            events.append(SessionEvent(SessionEventType.STOP_PLAYBACK))
            events.append(SessionEvent(SessionEventType.CANCEL_GENERATION))
        self.state = SessionState.LISTENING
        events.append(SessionEvent(SessionEventType.START_LISTENING))
        return events

    def on_user_speech_end(self, transcript: str) -> list[SessionEvent]:
        text = transcript.strip()
        if not text:
            self.state = SessionState.LISTENING
            return [SessionEvent(SessionEventType.START_LISTENING)]
        self.history.append({"role": "user", "content": text})
        self.state = SessionState.THINKING_SPEAKING
        return [SessionEvent(SessionEventType.REQUEST_REPLY, transcript=text)]

    def on_assistant_audio_done(self) -> list[SessionEvent]:
        self.state = SessionState.LISTENING
        return [SessionEvent(SessionEventType.START_LISTENING)]

    def append_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_session.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voice/session.py tests/test_session.py
git commit -m "$(cat <<'EOF'
feat: add conversation session state with barge-in events

EOF
)"
```

---

### Task 4: LLM streaming client with cancellation

**Files:**
- Create: `src/voice/llm.py`
- Create: `tests/test_llm_stream.py`

**Interfaces:**
- Consumes: `AppConfig.llm`, session `history`
- Produces: `LlmClient.stream_chat(messages, system_prompt) -> Iterator[str]` and `LlmClient.cancel()`

- [ ] **Step 1: Write failing LLM client tests using httpx MockTransport**

```python
# tests/test_llm_stream.py
import json

import httpx

from voice.llm import LlmClient


def test_stream_chat_yields_content_tokens():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        chunks = [
            {"choices": [{"delta": {"content": "Hi"}}]},
            {"choices": [{"delta": {"content": "!"}}]},
        ]
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    client = LlmClient(
        base_url="http://test/v1",
        model="qwen3-8b",
        temperature=0.7,
        transport=transport,
    )
    tokens = list(client.stream_chat([{"role": "user", "content": "hey"}], system_prompt="Be brief."))
    assert tokens == ["Hi", "!"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_llm_stream.py -v`  
Expected: FAIL with import error for `voice.llm`

- [ ] **Step 3: Implement LLM client**

```python
# src/voice/llm.py
from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx


class LlmClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self._client = httpx.Client(transport=transport, timeout=None)
        self._active: httpx.Response | None = None

    def cancel(self) -> None:
        if self._active is not None:
            self._active.close()
            self._active = None

    def stream_chat(self, messages: list[dict[str, str]], system_prompt: str) -> Iterator[str]:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "stream": True,
            "messages": [{"role": "system", "content": system_prompt}, *messages],
        }
        with self._client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Accept": "text/event-stream"},
        ) as resp:
            self._active = resp
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    obj = json.loads(data)
                    delta = obj["choices"][0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
            self._active = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_llm_stream.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voice/llm.py tests/test_llm_stream.py
git commit -m "$(cat <<'EOF'
feat: add cancellable streaming LLM client for llama-server

EOF
)"
```

---

### Task 5: Audio I/O, VAD, STT, and TTS adapters

**Files:**
- Create: `src/voice/audio_io.py`
- Create: `src/voice/vad.py`
- Create: `src/voice/stt.py`
- Create: `src/voice/tts.py`
- Create: `tests/test_vad_interface.py`

**Interfaces:**
- Consumes: `AppConfig` audio/vad/stt/tts fields
- Produces:
  - `AudioHub` for mic frames + speaker playback with `stop_playback()`
  - `VadEngine.is_speech(frame: np.ndarray) -> bool`
  - `SttEngine.transcribe(pcm16: bytes, sample_rate: int) -> str`
  - `TtsEngine.synthesize(text: str) -> bytes` (PCM/WAV bytes)

- [ ] **Step 1: Write a narrow VAD interface test with a fake backend**

```python
# tests/test_vad_interface.py
import numpy as np

from voice.vad import VadEngine


class FakeVad:
    def __call__(self, frame: np.ndarray) -> float:
        return float(np.mean(np.abs(frame)) > 0.01)


def test_vad_engine_threshold():
    eng = VadEngine(backend=FakeVad(), threshold=0.5)
    silence = np.zeros(512, dtype=np.float32)
    speech = np.ones(512, dtype=np.float32) * 0.2
    assert eng.is_speech(silence) is False
    assert eng.is_speech(speech) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_vad_interface.py -v`  
Expected: FAIL with import error for `voice.vad`

- [ ] **Step 3: Implement adapters**

Implement:

- `vad.py`: `VadEngine` wrapping injectable backend; default loader documents Silero model load in README and lazy-loads in `create_default_vad()`
- `audio_io.py`: `AudioHub` using `sounddevice` InputStream/OutputStream; maintain a playback queue; `stop_playback()` clears queue and stops current buffer
- `stt.py`: `SttEngine` that shells out to whisper.cpp binary with temp WAV for MVP (document streaming upgrade path); method `transcribe(wav_path or pcm)`
- `tts.py`: `TtsEngine` that shells out to Piper and returns raw PCM/WAV bytes

Keep subprocess wrappers thin and deterministic enough to unit-test with fakes in Task 6.

- [ ] **Step 4: Run VAD interface test**

Run: `python -m pytest tests/test_vad_interface.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voice/audio_io.py src/voice/vad.py src/voice/stt.py src/voice/tts.py tests/test_vad_interface.py
git commit -m "$(cat <<'EOF'
feat: add audio, VAD, STT, and TTS adapters

EOF
)"
```

---

### Task 6: Pipeline orchestration with barge-in

**Files:**
- Create: `src/voice/pipeline.py`
- Create: `src/voice/__main__.py`
- Create: `tests/test_pipeline_barge_in.py`

**Interfaces:**
- Consumes: `ConversationSession`, `PhraseChunker`, `LlmClient`, STT/TTS/VAD/Audio fakes
- Produces: `VoicePipeline.run_turn(transcript: str)` and `VoicePipeline.handle_speech_start()` that cancel LLM + stop playback

- [ ] **Step 1: Write failing barge-in pipeline test with fakes**

```python
# tests/test_pipeline_barge_in.py
from voice.chunker import PhraseChunker
from voice.pipeline import VoicePipeline
from voice.session import ConversationSession, SessionState


class FakeLlm:
    def __init__(self):
        self.cancelled = False

    def stream_chat(self, messages, system_prompt):
        yield "Hello "
        yield "world."
        if not self.cancelled:
            yield " more"

    def cancel(self):
        self.cancelled = True


class FakeTts:
    def __init__(self):
        self.spoken = []
        self.stopped = False

    def speak(self, text: str):
        if not self.stopped:
            self.spoken.append(text)

    def stop(self):
        self.stopped = True


def test_barge_in_stops_playback_and_cancels_llm():
    session = ConversationSession()
    session.force_state(SessionState.THINKING_SPEAKING)
    llm = FakeLlm()
    tts = FakeTts()
    pipeline = VoicePipeline(session=session, llm=llm, tts=tts, chunker=PhraseChunker())
    pipeline.handle_speech_start()
    assert llm.cancelled is True
    assert tts.stopped is True
    assert session.state.name == "LISTENING"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline_barge_in.py -v`  
Expected: FAIL with import error for `voice.pipeline`

- [ ] **Step 3: Implement pipeline + CLI entrypoint**

`VoicePipeline` should:

1. On `REQUEST_REPLY`, stream LLM tokens through `PhraseChunker`
2. Send each phrase to TTS/`speak`
3. Append full assistant text to session history
4. On `handle_speech_start()`, apply session events: cancel LLM, stop TTS, enter listening
5. Expose observer hooks / callbacks the UI can subscribe to: `on_state_change`, `on_user_transcript`, `on_assistant_partial`, `on_assistant_final`

`__main__.py` loads config, constructs real adapters, and:
- default: launch Tkinter UI (Task 7)
- `--cli`: run a headless listen → transcribe → reply loop for debugging

- [ ] **Step 4: Run unit tests**

Run: `python -m pytest tests/ -v`  
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/voice/pipeline.py src/voice/__main__.py tests/test_pipeline_barge_in.py
git commit -m "$(cat <<'EOF'
feat: wire streaming voice pipeline with barge-in

EOF
)"
```

---

### Task 7: Simple Tkinter UI

**Files:**
- Create: `src/voice/ui.py`
- Create: `tests/test_ui_callbacks.py`
- Modify: `src/voice/__main__.py`

**Interfaces:**
- Consumes: `VoicePipeline` callbacks / control methods (`start()`, `stop()`, `interrupt()`)
- Produces: `VoiceAppUI` window with Start/Stop, Interrupt, status label, transcript text widget

- [ ] **Step 1: Write failing UI callback/controller tests (no display required)**

```python
# tests/test_ui_callbacks.py
from voice.session import SessionState
from voice.ui import UiController


class FakePipeline:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.interrupted = False
        self.listeners = []

    def add_listener(self, fn):
        self.listeners.append(fn)

    def start(self):
        self.started = True
        for fn in self.listeners:
            fn({"type": "state", "state": SessionState.LISTENING.name})

    def stop(self):
        self.stopped = True
        for fn in self.listeners:
            fn({"type": "state", "state": SessionState.IDLE.name})

    def interrupt(self):
        self.interrupted = True
        for fn in self.listeners:
            fn({"type": "state", "state": SessionState.LISTENING.name})


def test_controller_start_stop_interrupt_update_status():
    pipe = FakePipeline()
    ui = UiController(pipeline=pipe)
    ui.on_start()
    assert pipe.started is True
    assert ui.status == "LISTENING"
    ui.on_interrupt()
    assert pipe.interrupted is True
    ui.on_stop()
    assert pipe.stopped is True
    assert ui.status == "IDLE"


def test_controller_appends_transcript_events():
    pipe = FakePipeline()
    ui = UiController(pipeline=pipe)
    ui.handle_event({"type": "user_transcript", "text": "hi"})
    ui.handle_event({"type": "assistant_final", "text": "hello"})
    assert ui.transcript_lines == ["You: hi", "Assistant: hello"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ui_callbacks.py -v`  
Expected: FAIL with import error for `voice.ui`

- [ ] **Step 3: Implement `UiController` + Tkinter shell**

```python
# src/voice/ui.py (controller portion required; Tk bindings wrap it)
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class UiController:
    pipeline: object
    status: str = "IDLE"
    transcript_lines: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        add_listener = getattr(self.pipeline, "add_listener", None)
        if callable(add_listener):
            add_listener(self.handle_event)

    def on_start(self) -> None:
        self.pipeline.start()

    def on_stop(self) -> None:
        self.pipeline.stop()

    def on_interrupt(self) -> None:
        self.pipeline.interrupt()

    def handle_event(self, event: dict) -> None:
        etype = event.get("type")
        if etype == "state":
            self.status = str(event.get("state", self.status))
        elif etype == "user_transcript":
            self.transcript_lines.append(f"You: {event['text']}")
        elif etype == "assistant_final":
            self.transcript_lines.append(f"Assistant: {event['text']}")
        elif etype == "assistant_partial":
            # UI may replace last assistant partial line; controller stores finals primarily
            pass


def run_app(pipeline: object) -> None:
    import tkinter as tk
    from tkinter import scrolledtext

    controller = UiController(pipeline=pipeline)
    root = tk.Tk()
    root.title("Voice")
    root.geometry("520x640")

    status_var = tk.StringVar(value=controller.status)
    tk.Label(root, textvariable=status_var, font=("Helvetica", 16)).pack(pady=8)

    transcript = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=28)
    transcript.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

    def refresh() -> None:
        status_var.set(controller.status)
        transcript.delete("1.0", tk.END)
        transcript.insert(tk.END, "\n".join(controller.transcript_lines))
        root.after(100, refresh)

    btn_row = tk.Frame(root)
    btn_row.pack(pady=8)
    tk.Button(btn_row, text="Start", command=controller.on_start).pack(side=tk.LEFT, padx=4)
    tk.Button(btn_row, text="Stop", command=controller.on_stop).pack(side=tk.LEFT, padx=4)
    tk.Button(btn_row, text="Interrupt", command=controller.on_interrupt).pack(side=tk.LEFT, padx=4)

    refresh()
    root.mainloop()
```

Wire `__main__.py` so default launches `run_app(pipeline)` and `--cli` keeps the headless loop.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_ui_callbacks.py tests/test_pipeline_barge_in.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voice/ui.py src/voice/__main__.py tests/test_ui_callbacks.py
git commit -m "$(cat <<'EOF'
feat: add simple Tkinter control UI for voice sessions

EOF
)"
```

---

### Task 8: Model/runtime scripts and README smoke path

**Files:**
- Create: `scripts/download_models.sh`
- Create: `scripts/run_llama_server.sh`
- Create: `README.md`

**Interfaces:**
- Consumes: local paths from `config.example.yaml`
- Produces: documented commands to install deps, download Whisper/Piper/LLM GGUF, start `llama-server`, run `voice` UI (and `--cli`)

- [ ] **Step 1: Write `scripts/run_llama_server.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH=${1:-models/qwen3-8b-q4_k_m.gguf}
PORT=${PORT:-8080}
exec llama-server -m "$MODEL_PATH" --port "$PORT" -ngl 99 --ctx-size 8192
```

- [ ] **Step 2: Write `scripts/download_models.sh`**

Script should:

- create `models/`
- print exact download commands/URLs placeholders for:
  - Whisper large-v3-turbo ggml for whisper.cpp
  - Qwen3-class ~8B Q4_K_M GGUF
  - Piper English voice ONNX + JSON
- refuse to silently fetch huge blobs without echoing sizes/paths

- [ ] **Step 3: Write README with smoke checklist**

README must include:

1. Install whisper.cpp, llama.cpp (Metal), Piper
2. `python -m pip install -e ".[dev]"`
3. Copy `config.example.yaml` → `config.yaml` and edit paths
4. Start `scripts/run_llama_server.sh`
5. Run `voice` / `python -m voice` (opens UI); `python -m voice --cli` for headless
6. Manual smoke:
   - Start in UI → speak a short sentence → hear reply start before full generation ends
   - status flips Listening/Speaking; transcripts appear
   - Interrupt button and/or speak during reply → audio stops
7. Explicit Phase 2 note: NSFW adapter / better TTS / helper model deferred

- [ ] **Step 4: Make scripts executable and verify pytest still passes**

Run:

```bash
chmod +x scripts/*.sh
python -m pytest tests/ -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/download_models.sh scripts/run_llama_server.sh README.md
git commit -m "$(cat <<'EOF'
docs: add model scripts and local smoke-run instructions

EOF
)"
```

---

### Task 9: Manual hardware verification on the M3 Pro

**Files:**
- Modify: `README.md` (add measured notes section only if numbers are captured)

**Interfaces:**
- Consumes: running pipeline + UI from Tasks 7–8
- Produces: pass/fail against acceptance criteria in the design spec

- [ ] **Step 1: Cold-start services**

Start `llama-server` with the chosen ~8B Q4/Q5 GGUF; confirm Metal is active in server logs.

- [ ] **Step 2: Run three scripted utterances from the UI**

1. “Hey, tell me a two-sentence story.”
2. Interrupt mid-reply with the Interrupt button, then ask “What’s two plus two?”
3. Short back-and-forth of 4 turns

Record approximate time-to-first-audio for (1).

- [ ] **Step 3: Check acceptance criteria**

- Offline only: pass/fail
- Barge-in subjectively immediate: pass/fail
- First audio before full completion: pass/fail
- System prompt from file: pass/fail
- UI start/stop/status/transcript/interrupt: pass/fail

- [ ] **Step 4: If TTFA > ~1.5s, apply only these MVP knobs**

- smaller Whisper model
- lower first-chunk `PhraseChunker.max_chars`
- ensure non-thinking mode
- keep Piper; do not jump to 14B+ yet

- [ ] **Step 5: Commit measurement notes only if README updated**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs: record local smoke-test latency notes

EOF
)"
```

---

## Self-review (plan vs spec)

| Spec requirement | Task |
|---|---|
| Streaming pipeline Mic→VAD→STT→LLM→chunker→TTS | Tasks 5–6 |
| Barge-in cancels TTS + LLM | Tasks 3, 4, 6, 9 |
| ~8B Q4/Q5 via llama-server Metal | Tasks 8–9 |
| whisper.cpp + Silero + Piper | Tasks 5, 8 |
| Non-thinking ordinary turns | Task 8 README + LLM config/prompt |
| Simple Tkinter UI (status/start/stop/interrupt/transcripts) | Task 7 |
| No dual helper model / no NSFW in MVP | Global constraints + Task 8 Phase 2 note |
| Automated tests for chunker, session, barge-in, UI controller | Tasks 2, 3, 6, 7 |
| README install/run | Task 8 |
| Acceptance / TTFA smoke | Task 9 |

No intentional placeholders left in task steps. Phase 2 items remain outside this plan by design.
