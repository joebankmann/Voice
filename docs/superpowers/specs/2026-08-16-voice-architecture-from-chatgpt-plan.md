# Voice Architecture — Spec Digest (from ChatGPT Plan PDF)

**Date:** 2026-08-16  
**Source:** `docs/VOICE_ChatGPT-Plan.pdf` (49 pages; mostly template filler)  
**Status:** Digested + judged against the live Voice codebase  

## What the PDF actually says

The useful content is a **section outline**, not an implementation spec. Nearly every page repeats the same “Design Consideration …” boilerplate. Treat the PDF as a **concept checklist**, not as authority on models, APIs, or phase ordering.

### Concepts worth keeping

| # | PDF concept | Intent |
|---|---|---|
| 1 | Vision | Private, local-first, natural conversation; modular |
| 2 | Guiding principles | Local-first, interfaces, streaming, graceful degradation |
| 3 | Cooperating services | Audio, STT, conversation, speech rewrite, TTS, memory, context, personality, tools, config, telemetry |
| 4 | Voice pipeline | Capture → VAD → STT → LLM → rewrite → TTS → playback |
| 5 | Low latency | Overlap stages; sentence-level TTS; avoid serialization |
| 6 | STT | Endpointing, partials, punctuation, confidence |
| 7 | Conversation engine | Swappable LLM behind one interface |
| 8 | Prompt architecture | Split system / personality / task / tool / rewrite prompts |
| 9 | Speech rewrite | Written → spoken; strip markdown/lists/verbosity |
| 10 | TTS | Swappable; multi-voice; prosody |
| 11 | Voice profiles | Timbre/pacing/etc. separate from personality |
| 12 | Personality | Configurable traits (humor, directness, …) |
| 13 | Memory | Working / episodic / semantic / preferences |
| 14 | Context orchestration | Selective injection per turn |
| 15 | Interruptions | Barge-in, cancel, recover, resume |
| 16 | Configuration | Versioned YAML/JSON profiles |
| 17 | Swappable models | Config-only model swaps |
| 18 | Tool framework | Optional tools; core stays independent |
| 19 | Observability | Latency, tokens, failures, quality |
| 20 | Evaluation | Benchmarks, latency, voice, regressions |
| 21 | Failure handling | Independent subsystem failure + degrade |
| 22 | Security & privacy | Local default; explicit online vs offline |
| 23 | Roadmap | Core → memory → tools → agents → ecosystem |
| 24 | Future | Emotion, multimodal, adaptive personality, collab agents |

### PDF roadmap (raw)

1. Conversational core  
2. Memory  
3. Tools  
4. Specialist agents  
5. Ecosystem  

## Judgment: what to accept vs change

### Accept (aligns with product)

- Local-first modular pipeline with barge-in  
- Streaming LLM + chunked TTS  
- Separate voice profile vs personality  
- Layered memory **after** the core feels phone-like  
- Tools behind interfaces; no cloud by default  
- Observability and eval as first-class later phases  

### Correct / reject (PDF specifics are wrong or premature)

| PDF claim | Judgment | What we do instead |
|---|---|---|
| “Use MLX Whisper” | Optional, not mandatory | Keep **whisper.cpp** (already integrated). Prefer **resident / streaming whisper.cpp** before rewriting STT in MLX |
| “Streaming STT” as day-one | Correct goal, wrong current state | Today: CLI-per-turn WAV. Next latency work: resident process or streaming decoder |
| Separate **Speech Rewrite** LLM stage | Hurts TTFA; overkill | Keep **prompt style + `speak_text` sanitizer** (+ optional gesture map). Only add a rewrite model if quality still fails |
| Cooperating microservices | Too heavy for one Mac app | Keep **one Python process** + `llama-server`; modules as libraries, not services |
| Dual / specialist agents early | Premature | One conversational LLM until TTFA and turn-taking are solid |
| Full personality trait engine | Nice later | Start with **prompt profiles** (YAML + text), not a trait DSL |
| Resumable interrupted answers | Nice-to-have | Hard; ship **cancel + clean next turn** first; resume later |
| Phase 2 = memory immediately | Wrong for *this* repo | Insert a **Latency & quality hardening** phase before memory |

### Already implemented (map to PDF)

| PDF concept | Current Voice status |
|---|---|
| Voice pipeline | Done: Silero → whisper.cpp → Qwen3-8B (`llama-server`) → chunker → F5/Piper hybrid → AudioHub |
| Low latency (partial) | Phrase chunking + streaming LLM + barge-in; **STT reload** and **F5 cold start** still dominate TTFA |
| Conversation engine | OpenAI-compatible client to local server |
| Speech rewrite (light) | `prompts/system.txt` + `speak_text.py` sanitizer |
| TTS + voice profiles | Piper voices + F5 clone dirs (`voices/clones/`) |
| Interruptions | Barge-in cancel of LLM + playback |
| Configuration | `config.yaml` |
| Privacy | Offline after model download |
| UI | Tkinter Start/Stop/Interrupt + transcripts |

### Still missing (relative to PDF)

- Resident / streaming STT; partial transcripts; confidence  
- Telemetry / latency budgets as product metrics  
- Layered memory + selective context injection  
- Tool framework  
- Personality/voice profile depth beyond prompts + clone files  
- Formal eval harness  
- Specialist agents / ecosystem  

## Target architecture (judged)

```
Mic → AudioHub → Silero VAD → SttEngine (resident whisper)
  → Session (history + optional memory retrieval)
  → PromptComposer (system + world + personality + retrieved)
  → LlmClient (llama-server Metal, reasoning off)
  → PhraseChunker → SpeakText sanitizer
  → TtsFactory (f5 | piper | hybrid)
  → AudioHub playback
  ↺ barge-in cancels LLM + TTS queue
```

**Non-goals until core latency is verified:** multi-agent routing, cloud tools, emotion models, multimodal.

## Success metric (unchanged)

After user end-of-turn, **time-to-first-audio ≤ ~1.5 s** on M3 Pro / 36 GB under warm models, with barge-in feeling immediate.
