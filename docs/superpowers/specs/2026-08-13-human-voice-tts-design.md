# Phase 2 Human Voice TTS — Design Spec

**Date:** 2026-08-13  
**Status:** Approved for implementation (user: execute phase 2 TTS)  
**Hardware:** Apple M3 Pro / 36 GB, fully local  

## Goals

1. **Natural** — clearly less robotic than Piper for conversational English  
2. **Fast** — warm-path first audio competitive with phone chat (chunked synthesis; resident model; low step count)  
3. **Clone** — zero-shot voice profiles from a short reference WAV + transcript  

## Architecture

```
LLM tokens → phrase chunker → speech sanitizer (+ gesture map)
  → TTS backend (f5 | piper | hybrid)
  → AudioHub (resample to device rate) → speakers
```

### Backends

| Backend | Engine | Role |
|---|---|---|
| `f5` | `f5-tts-mlx` (Apple Silicon / MLX) | Default Phase 2 mouth: natural + zero-shot clone |
| `piper` | Existing Piper CLI | Legacy / ultra-light fallback |
| `hybrid` | F5 primary, Piper if F5 cannot load | Boot resilience |

### Voice profiles

- **Piper profiles:** `*.onnx` (+ `.onnx.json`) under `tts.voices_dir` (unchanged)  
- **Clone profiles:** directories under `tts.clones_dir` (default `voices/clones/<name>/`):
  - `ref.wav` — mono 24 kHz, ~5–15 s  
  - `ref.txt` — exact transcript of the reference audio  
  - optional `profile.json` — `{ "name", "engine": "f5" }`  

UI voice dropdown lists Piper models and clone profile names.

### Latency tactics

- Keep F5 model resident after first load (no per-phrase reload)  
- Use short phrases from existing chunker  
- F5 sample steps default **8**, method **euler**, optional 4-bit quant  
- Hybrid does **not** dual-speak different voices mid-turn in v1 (avoids identity flip); Piper is error fallback only  

### Gestures

LLM may emit light tags such as `[laugh]`, `[sigh]`, `[pause]`. Sanitizer maps them to short spoken fillers or brief silence markers before synthesis (not read as the word “bracket”).

## Non-goals (this slice)

- Cloud TTS  
- Perfect Hollywood acting  
- Separate fast “filler voice” that talks in a different identity before the clone (deferred if F5 alone is too slow)  

## Acceptance

1. With a clone profile selected, synthesized speech resembles the reference in a casual A/B listen  
2. Default F5 path sounds clearly more natural than Piper on the same lines  
3. Barge-in still stops playback  
4. Piper backend still works when selected  
5. Automated tests cover profile discovery, factory selection, and gesture sanitization without requiring GPU model download in CI (F5 calls mocked)  
