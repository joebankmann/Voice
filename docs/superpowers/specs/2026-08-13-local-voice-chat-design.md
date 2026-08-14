# Local Real-Time Voice Chat — Design Spec

**Date:** 2026-08-13  
**Hardware target:** Apple MacBook Pro, M3 Pro, 36 GB unified memory  
**Status:** Draft for review  

## Goal

Build a fully local voice conversation app that feels like a phone call: low time-to-first-audio, barge-in, and continuous turn-taking — not “speak, wait, hear a full answer.”

Primary success metric for MVP: after the user stops speaking, the system starts speaking within roughly **0.5–1.5 seconds** under normal conditions, with playback interruptible the moment the user speaks again.

## Non-goals (MVP)

- Dual / helper LLM for routing, memory, or tools
- Semantic end-of-turn beyond VAD + silence heuristics
- NSFW / roleplay fine-tunes or LoRAs (Phase 2: swap model/adapter only)
- Highest-fidelity human TTS (Piper is intentional MVP; upgrade later)
- Cloud APIs, accounts, or network model calls
- Polished visual design systems, theming, animations, or browser/WebRTC audio
- Multi-user, telephony, or mobile clients

## Product decisions

| Decision | Choice | Why |
|---|---|---|
| Architecture | Streaming pipeline, one conversational LLM | Latency > peak model quality on this Mac |
| Main model class | ~7–9B instruct, Q4_K_M or Q5_K_M GGUF | Fits comfortably; leaves headroom for STT/TTS/macOS |
| Thinking / reasoning | Off for ordinary turns | Avoids burning first-token latency |
| Inference runtime | `llama.cpp` (`llama-server`) + Metal, native macOS | Best practical Apple Silicon path; OpenAI-compatible HTTP |
| STT | `whisper.cpp`, turbo / small-medium class | Local, Apple-friendly; avoid CUDA-oriented stacks for MVP |
| VAD | Silero VAD | Lightweight enough to be negligible vs LLM |
| TTS | Piper | Fast local neural TTS for MVP responsiveness |
| App language | Python 3.11+ orchestration process | Fastest path to glue audio + subprocess/HTTP services |
| UI (MVP) | Simple desktop window (Tkinter) over the same pipeline | Control surface without browser mic/permissions complexity; audio stays in Python |

## Pipeline

```
Mic
 → capture (sounddevice)
 → Silero VAD (speech vs silence; barge-in signal)
 → whisper.cpp (streaming / segment transcription)
 → turn assembler (silence threshold + min utterance length)
 → llama-server (Qwen3-class ~8B, non-thinking, streaming tokens)
 → phrase chunker (emit on sentence boundary / length / punctuation)
 → Piper (stream PCM to speakers)
 → barge-in cancels TTS + LLM generation and returns to listen
```

### Latency contract

| Stage | Target behavior |
|---|---|
| VAD speech start | Stop TTS immediately; cancel in-flight LLM request |
| End of user turn | Detect via VAD silence (configurable ms) after speech |
| STT | Prefer partial/final segments; do not wait for a “perfect” transcript forever |
| LLM | Stream tokens; never wait for full completion before TTS |
| Chunker | Emit first speakable phrase ASAP (short sentence / clause) |
| TTS | Start audio on first chunk; keep queue only slightly ahead of speech |

### Barge-in

1. While assistant audio is playing (or LLM is generating), VAD detects user speech above threshold.
2. Stop audio output immediately.
3. Abort LLM stream.
4. Clear pending TTS chunks.
5. Enter listening state and accumulate the new user utterance.

## Components

### 1. `voice` orchestration app (Python)

Owns session state machine:

- `idle` / `listening` / `transcribing` / `thinking_speaking` / `interrupted`

Responsibilities:

- Audio I/O
- VAD decisions
- Calling STT / LLM / TTS
- Phrase chunking
- Cancellation / barge-in
- Config loading (model paths, silence ms, voice id, system prompt)

### 2. STT worker

- Wrap `whisper.cpp` CLI or binding
- Input: PCM segments from VAD speech regions
- Output: text segments with timestamps / finals
- Model default: large-v3-turbo or equivalently fast Whisper build available for whisper.cpp on Mac

### 3. LLM worker

- Talk to local `llama-server` OpenAI-compatible `/v1/chat/completions` with `stream=true`
- System prompt owned by app config
- Force non-thinking / no-reasoning mode via model-appropriate flags or prompt conventions for the chosen Qwen3-class GGUF
- Support abort of the active stream on barge-in

### 4. Phrase chunker

- Buffer streamed tokens
- Flush when: sentence-ending punctuation, or soft length limit with a safe break, or stream end
- Never hold the first chunk waiting for a long paragraph

### 5. TTS worker

- Piper CLI or library
- Input: text chunks
- Output: PCM/WAV to playback queue
- Configurable voice

### 6. Simple desktop UI (Tkinter)

A single local window that controls and observes the pipeline. Mic/speaker stay in the Python audio path (not browser capture).

Must show:

- Connection / run status: Idle, Listening, Speaking, Error
- Start / Stop conversation toggle
- Manual Interrupt button (same cancel path as barge-in)
- Scrolling transcript: user and assistant turns as they finalize / stream in
- Optional one-line latency hint (time-to-first-audio for last turn) when measured

CLI mode remains available (`python -m voice --cli`) for headless debugging; default entrypoint launches the UI.

## Configuration (MVP)

`config.yaml` (or equivalent) includes:

- Paths: whisper model, llama-server URL, Piper model/voice
- VAD: speech threshold, min speech ms, end-of-turn silence ms
- LLM: model name/id, temperature, max tokens, system prompt path
- Audio: input/output device names or defaults, sample rate
- Feature flags: none for dual-model / NSFW in MVP

## Memory budget (planning)

Approximate resident use on 36 GB:

| Piece | Approx |
|---|---|
| 8B Q4_K_M weights | 5–6 GB |
| KV + llama runtime | 2–6 GB |
| Whisper turbo/small-medium | 1–3 GB |
| Piper + VAD + Python + macOS | 4–8 GB |

Leave headroom; do not start with 27B/30B+/35B MoE for MVP.

## Phase 2 (explicitly out of this plan’s implementation tasks)

After MVP feels conversational:

1. Swap main GGUF / attach NSFW-oriented LoRA without redesigning the pipeline
2. Upgrade TTS to a more natural streaming voice
3. Optional ~1B helper for memory extraction / command routing
4. Semantic turn detection beyond silence

## Acceptance criteria (MVP)

1. Fully offline conversation loop works on the target Mac
2. Barge-in stops assistant audio within one VAD frame window of user speech (subjectively immediate)
3. First assistant audio begins before the full LLM answer is finished
4. Configurable system prompt; no cloud LLM dependency
5. README documents install, model download, and how to run `llama-server` + the app (UI default + `--cli`)
6. Automated tests cover phrase chunker, session state transitions, and barge-in cancellation logic (audio hardware may remain manual smoke-tested)
7. Simple UI can start/stop the loop, show live status, display transcripts, and trigger interrupt

## Risks

| Risk | Mitigation |
|---|---|
| Whisper or Piper startup latency | Keep processes warm; avoid cold-start per turn |
| llama.cpp Metal throttling | Stay at ~8B Q4/Q5; measure tokens/sec before polish |
| Silence-based EOT cuts users off | Tunable silence ms; later semantic EOT in Phase 2 |
| Piper sounds robotic | Accept for MVP; TTS upgrade is Phase 2 |
| Model refusals for adult content | Out of MVP; Phase 2 model/adapter swap |

## Open items resolved by this spec

- First build = Approach A (latency-first), not dual-model, not NSFW-first
- Python orchestration + external native binaries for STT/LLM/TTS
- Simple Tkinter control UI in MVP (not terminal-only; not a polished design system)
