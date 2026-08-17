# Local Voice Chat

Offline-first streaming voice chat using whisper.cpp for speech recognition,
llama.cpp for local generation, Piper for speech synthesis, and Silero for
voice activity detection.

**Operator manual** (setup, companion processes, UI, and every config key):
[docs/VOICE_MANUAL.md](docs/VOICE_MANUAL.md).

## Prerequisites

The project requires Python 3.11 or newer. On macOS, install the native
runtimes with Homebrew:

```bash
brew install python@3.11 python-tk@3.11 whisper-cpp llama.cpp
python3.11 -m pip install piper-tts
```

The `whisper-cli`, `whisper-server`, `llama-server`, and `piper` executables
must be available on `PATH`. Default `stt.mode` is `resident` (managed
`whisper-server` on port 8178). Set `stt.mode: cli` to fall back to a
cold `whisper-cli` process per turn. If a package-manager binary name or path
differs, update `config.yaml`.

Tkinter is required for the default desktop UI. A Homebrew Python installation
can add it with `brew install python-tk@3.11`. The macOS `/usr/bin/python3`
usually includes `_tkinter` and can be used to launch the UI if it is Python
3.11 or newer and the project dependencies are installed for that interpreter.
Confirm support with:

```bash
python3.11 -c "import tkinter"
```

## Install

From the repository root:

```bash
python3.11 -m pip install -e ".[dev]"
python3.11 -m pip install silero-vad
cp config.example.yaml config.yaml
chmod +x scripts/*.sh
```

Edit `config.yaml` so `stt.whisper_bin`, `stt.model_path`, `tts.piper_bin`, and
`tts.voice_path` match the local executables and model files. Keep
`llm.base_url` aligned with the llama.cpp server port.

## Download models

The helper creates `models/` and prints reviewed `curl` commands, expected
sizes, and destinations for Whisper large-v3-turbo, Qwen3 8B Q4_K_M, and the
Piper English voice:

```bash
scripts/download_models.sh
```

It deliberately does not download multi-gigabyte files silently. Review and
run the printed commands. Their destination names match
`config.example.yaml` and the llama server's default model path.

## Run

Start the Metal-enabled llama.cpp server in one terminal:

```bash
scripts/run_llama_server.sh
```

Pass a different GGUF as the first argument or set another port with
`PORT=8081 scripts/run_llama_server.sh path/to/model.gguf`. The script uses
GPU offload and an 8192-token context.

In a second terminal, start the desktop UI (the default):

```bash
voice
# Equivalent:
python3.11 -m voice
```

For a headless microphone loop:

```bash
python3.11 -m voice --cli
```

Both entry points read `config.yaml` by default. Use `--config path/to.yaml`
to select another configuration.

With `tts.warmup_on_start: true`, Start synthesizes and discards a short phrase
so F5 is ready for the first reply. Set it to `false` for Piper-only setups
where warmup is unnecessary.

## Customize behavior

- `prompts/system.txt` — personality and “no refusal” spoken rules.
- `prompts/world_context.txt` — editable current facts (president, etc.). The app
  also injects today’s date. Update this file when world facts change; the model
  has no live internet.
- If the base Qwen instruct weights still refuse after a restart, that refusal is
  inside the model weights, not app filters—swap to a less-aligned local GGUF.

## Customize voices

### F5 clone (Phase 2 default via `tts.backend: hybrid`)

Natural zero-shot cloning uses **F5-TTS on MLX** (Apple Silicon), with Piper as
fallback if F5 cannot load.

```bash
python3.11 -m pip install -e ".[tts]"
```

Add a clone profile:

```bash
mkdir -p voices/clones/myvoice
ffmpeg -i your_sample.wav -ac 1 -ar 24000 -sample_fmt s16 -t 10 \
  voices/clones/myvoice/ref.wav
echo 'Exact words spoken in that clip.' > voices/clones/myvoice/ref.txt
```

Restart the app and select the profile in the Voice dropdown. A starter
`voices/clones/default` profile is included.

Config: `tts.backend` (`hybrid` | `f5` | `piper`), `tts.clones_dir`,
`tts.f5_steps`, `tts.f5_quantization_bits`.

Gesture tags: `[laugh] [chuckle] [sigh] [pause]`.

### Piper (legacy)

Piper `.onnx` voices under `tts.voices_dir` still appear in the dropdown when
present. See `scripts/download_models.sh` and Piper’s VOICES.md for more.

## Memory (Phase B)

Local preferences and notes persist under `data/` (paths configurable).

In `config.yaml`:

```yaml
memory:
  enabled: true
  preferences_path: data/preferences.yaml
  episodic_path: data/episodic.jsonl
  max_history_messages: 24
  max_episodic_hits: 3
  max_inject_chars: 1200
```

Say phrases like **“Remember that I prefer short answers”** or **“Remember that
the sailboat uses cedar”**. Preferences survive app restarts and are injected
into the system prompt within `max_inject_chars`. With `memory.enabled: false`,
behavior matches the pre-memory pipeline.

Enable `telemetry.enabled` to log `memory_inject` events (`prefs`, `episodic`,
`chars`).

## Tools (Phase C)

Opt-in offline tools use silent markers the model may emit (never spoken):

```text
<<tool:local_time|{}>>
<<tool:preference_set|{"key":"prefer","value":"short answers"}>>
<<tool:note_add|{"text":"Sailboat uses cedar"}>>
```

In `config.yaml`:

```yaml
tools:
  enabled: true
  timeout_ms: 2000
  allow_online: false
```

When enabled, a Tools section is added to the system prompt. After a tool runs,
the pipeline does one spoken continuation with the results. With
`tools.enabled: false` (default when omitted), there is no Tools prompt and no
execution (markers are still stripped from speech).

## Specialist agents (Phase D)

An optional **off-path** helper summarizes conversation turns that fall out of
the working-memory window into an episodic note. It never runs before first
audio.

```yaml
agents:
  enabled: true
  helper_base_url: ""    # empty = reuse the main llama-server
  helper_model: ""
  timeout_ms: 4000
```

Requires `memory.enabled: true`. Leave `enabled: false` (the default) for
Phase C behavior. A second GGUF is optional; pointing `helper_base_url` at
another `llama-server` is only for when you have RAM/GPU headroom.

## Profile packs (Phase E)

Drop a directory under `profiles/<id>/` with `manifest.yaml` (no code change):

```yaml
schema_version: 1
id: casual
name: Casual companion
version: "1.0.0"
personality: personality.txt
voice: default
```

```yaml
profiles:
  packs_dir: profiles
  active: casual    # empty = no pack
```

A `casual` pack ships in the repo. Eval (sanitize fixtures + fake-clock latency
budgets) runs in default `pytest`. Optional WAV smoke:

```bash
VOICE_GOLDEN=1 scripts/eval_golden_wav.sh
```

## Future directions (Phase F)

All of these default **off**. They never add a second model to the spoken path.

```yaml
future:
  affect: false              # lexicon hint from the user transcript
  inbox: false               # consume .txt/.md from inbox_dir once
  inbox_dir: data/inbox
  adaptive_personality: false  # blend stored prefer/tone into the prompt
agents:
  collaborative: false       # extra off-path action-item pass after summarize
```

Drop a markdown or text file into `data/inbox/` when inbox is on; it is moved
to `processed/` after one turn. Images are recorded as unsupported (no vision
model). Continuous eval snapshots:

```bash
python3.11 -m voice.eval_harness --cases tests/eval/cases/sanitize.yaml --jsonl data/eval.jsonl
```

## Capability matrix

| Capability | Runtime | Network after models are local |
|---|---|---|
| VAD, STT, LLM, TTS, UI | Local | Offline |
| Memory / preferences | Local files | Offline |
| Tools (`local_time`, notes, prefs) | Local | Offline (`allow_online: false`) |
| Specialist helper | Local llama-server | Offline |
| Profile packs | Local directories | Offline |
| Affect / text inbox / adaptive tone | Local lexicon + files | Offline |
| Model/weight **download** | curl / pip / Hugging Face | Online once, then cache |
| Online search / browser / shell | Not included | — |

## Manual smoke checklist

1. Start the llama server, then launch the UI and click **Start**.
2. Speak a short sentence and confirm its transcript appears.
3. Confirm the status changes between **Listening** and **Speaking**.
4. Confirm reply audio begins before the full generated response finishes.
5. Click **Interrupt**, and/or speak while the reply is playing, and confirm
   playback stops.

Run the automated suite with:

```bash
python3.11 -m pytest tests/ -v
```

## Silero VAD

`create_default_vad()` imports the optional `silero-vad` package and calls its
`load_silero_vad()` loader when the VAD is created. Install the package and
provision its model assets/cache before taking the machine offline; after that
initial local provisioning, VAD inference does not require a network
connection or cloud API.

## Phase 2

An NSFW adapter and a helper model remain deferred. **Human TTS** (F5-TTS MLX
clone + Piper fallback) is implemented — see Customize voices.

## Local smoke-test notes

Measured on an Apple M3 Pro on 2026-08-13 with Homebrew `llama.cpp` b10360,
Qwen3 8B Q4_K_M, Whisper large-v3-turbo, and Piper en_US-lessac-medium.

- Initial model load reached the listening state in about 11.46 s. A warm traced
  restart loaded in 0.81 s. The server log identified `Apple M3 Pro` and
  reported all 37/37 model layers offloaded to the GPU.
- A non-streaming localhost completion returned HTTP 200 in 0.319 s at 16.64
  generated tokens/s (two generated tokens). A streaming completion delivered
  its first HTTP bytes in 0.176 s, completed in 0.525 s, and generated at 30.48
  tokens/s (seven generated tokens).
- With the server default reasoning mode, the configured `LlmClient` took 4.844
  s to yield its first visible token because Qwen generated hidden reasoning
  tokens. `scripts/run_llama_server.sh` now defaults to
  `LLAMA_ARG_REASONING=off`, which reduced first visible token latency to
  0.323 s and total completion latency to 0.731 s in the same smoke. Override
  with `LLAMA_ARG_REASONING=on` only when you want reasoning.
- The configured Piper adapter produced 3.448 s of non-silent mono PCM (152,064
  bytes, RMS 3,969) in 0.837 s. Feeding that audio to the configured Whisper
  adapter returned the exact phrase in 3.530 s.
- Silero VAD loaded locally in 0.032 s.
- Microphone barge-in, physical playback, UI interaction, and end-to-end
  time-to-first-audio remain **inconclusive — requires human mic test**.
