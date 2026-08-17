# Voice operator manual

Fully local spoken chat on this Mac: microphone → VAD → Whisper → Qwen → streaming TTS → speakers. After models are on disk, nothing leaves the machine.

Work from the repo root: `/Volumes/ComfyUI/Voice` (or wherever you cloned it).

---

## Companion programs

Voice is a Python orchestrator. It does **not** replace these processes:

| Program | Role | Who starts it | Default |
|---|---|---|---|
| **llama-server** (llama.cpp) | Conversational LLM (Qwen3-8B GGUF, Metal) | **You**, every session | `127.0.0.1:8080` |
| **whisper-server** (whisper.cpp) | Resident STT | Voice, on **Start** (`stt.mode: resident`) | `127.0.0.1:8178` |
| **whisper-cli** | Cold STT fallback | Voice, per turn if `stt.mode: cli` | PATH |
| **piper** | Fallback / Piper-only TTS | Voice, as a subprocess | PATH |
| **F5-TTS MLX** | Clone TTS (hybrid/f5) | Loaded in-process on Start | Hugging Face cache after first download |
| **Silero VAD** | Speech vs silence | Loaded in-process | Local torch cache after first install |
| **Tkinter** | Desktop UI | In-process | Python 3.11 + `python-tk` |

Optional second **llama-server** on another port only if you set `agents.helper_base_url` to a different URL.

---

## Daily use

### One-time setup

```bash
brew install python@3.11 python-tk@3.11 whisper-cpp llama.cpp
python3.11 -c "import tkinter"
python3.11 -m pip install -e ".[dev]"
python3.11 -m pip install silero-vad
python3.11 -m pip install -e ".[tts]"          # F5 clone voices
cp config.example.yaml config.yaml
chmod +x scripts/*.sh
scripts/download_models.sh                     # prints curl commands; run the ones you need
```

Place at least: Whisper `models/ggml-large-v3-turbo.bin`, Qwen `models/qwen3-8b-q4_k_m.gguf`, Piper `models/en_US-lessac-medium.onnx` (+ `.onnx.json`).

### Every session

**Terminal 1 — LLM (required):**

```bash
scripts/run_llama_server.sh
```

Leave it running. Override model or port: `PORT=8081 scripts/run_llama_server.sh path/to/model.gguf`. Reasoning off by default; `LLAMA_ARG_REASONING=on` only if you want hidden thinking (much slower first token).

**Terminal 2 — Voice:**

```bash
python3.11 -m voice                 # desktop UI
python3.11 -m voice --cli           # headless mic loop
python3.11 -m voice --config path/to.yaml
```

Whisper is started by Voice when `stt.manage_server: true`. To run it yourself instead: `scripts/run_whisper_server.sh` and set `stt.manage_server: false`.

### Desktop UI

| Control | Action |
|---|---|
| **Start** | Open mic, start VAD/STT, warm TTS if configured |
| **Stop** | Cancel the current turn, close audio, stop managed whisper-server |
| **Interrupt** | Barge-in: stop playback and generation; keep listening |
| **Voice** | Pick a Piper `.onnx` or F5 clone folder |
| Transcript | You / Assistant lines; selectable text |
| Status | `IDLE` / `LISTENING` / `THINKING_SPEAKING` (or an error string) |
| Window close | Same as Stop |

Speak, pause (~`end_of_turn_silence_ms`), wait for the transcript, then the reply. Speaking during a reply interrupts it.

`--cli` prints `You:` / `Assistant:` to the terminal. Ctrl-C stops.

---

## How to change behavior

Almost everything is **`config.yaml`** (gitignored; copy from `config.example.yaml`). Paths in that file are relative to the config file’s directory. Restart Voice after edits. Restart **llama-server** if you change the GGUF, port, or `LLAMA_ARG_REASONING`.

### Required core

| Key | What it does |
|---|---|
| `audio.sample_rate` | Mic capture rate (keep **16000** unless you change VAD/STT together) |
| `audio.end_of_turn_silence_ms` | Pause length that ends your utterance (higher = wait longer) |
| `vad.threshold` | Silero speech threshold (higher = less likely to hear you) |
| `vad.min_speech_ms` | Ignore shorter blips |
| `llm.base_url` | Must match llama-server (`http://127.0.0.1:8080/v1`) |
| `llm.model` | Label sent to llama-server (does not select the GGUF file) |
| `llm.temperature` | Higher = more variation |
| `llm.system_prompt_path` | Spoken persona (`prompts/system.txt`) |
| `llm.world_context_path` | Editable facts (`prompts/world_context.txt`); date is injected automatically |

**LLM weights** are chosen by `scripts/run_llama_server.sh`’s model path, not by `llm.model`.

### Speech recognition

| Key | What it does |
|---|---|
| `stt.mode` | `resident` (whisper-server, default) or `cli` (whisper-cli each turn) |
| `stt.model_path` | Whisper GGUF/bin |
| `stt.whisper_server_bin` / `whisper_bin` | Executable names on PATH |
| `stt.server_host` / `server_port` / `inference_path` | Resident HTTP |
| `stt.manage_server` | `true`: Voice starts/stops whisper-server |

### Speech synthesis

| Key | What it does |
|---|---|
| `tts.backend` | `hybrid` (F5 + Piper fallback), `f5`, or `piper` |
| `tts.piper_bin` / `voice_path` / `voices_dir` | Piper binary and `.onnx` library |
| `tts.clones_dir` | F5 clone folders (`ref.wav` + `ref.txt`) |
| `tts.clone_ref_wav` / `clone_ref_text` | Default clone if none selected |
| `tts.warmup_on_start` | Synthesize and discard “Ready.” on Start |
| `tts.sample_rate` | Piper output rate |
| `tts.length_scale` / `sentence_silence` / `noise_scale` / `noise_w_scale` | Piper prosody |
| `tts.f5_model` / `f5_steps` / `f5_speed` / `f5_quantization_bits` | F5 quality vs latency (steps **8**, 4-bit is the local default) |

Add a clone: `voices/clones/<name>/ref.wav` (mono 24 kHz, ~5–10 s) and `ref.txt` with the exact words. Restart and pick it in **Voice**. Gesture tags the model may speak: `[laugh] [chuckle] [sigh] [pause]` (and similar; see `prompts/system.txt`).

### Prompts (not YAML)

| File | Effect |
|---|---|
| `prompts/system.txt` | Persona, spoken style, no-refusal rules |
| `prompts/world_context.txt` | Facts the model should treat as current |
| `profiles/<id>/personality.txt` | Extra personality when that pack is active |

Edit and restart Voice. No llama-server restart needed.

### Memory

`memory.enabled: true` in the example config.

| Key | What it does |
|---|---|
| `memory.enabled` | Off = no prefs, notes, or history trim |
| `preferences_path` | YAML map of stored prefs |
| `episodic_path` | JSONL notes |
| `max_history_messages` | Working window (drops oldest **pairs**) |
| `max_episodic_hits` / `max_inject_chars` | How much is stuffed into the prompt |

Say **“Remember that I prefer short answers”** or **“Remember that the sailboat uses cedar.”** Files live under `data/` and survive restarts.

### Tools (off by default)

```yaml
tools:
  enabled: true
  timeout_ms: 2000
  allow_online: false    # keep false; no web/shell tools ship
```

Available: `local_time`, `preference_set`, `note_add`. The model emits silent `<<tool:name|{json}>>` markers; they are never spoken. Interrupt cancels leftover tools and skips preference/note writes.

### Specialist helper (off by default)

Needs `memory.enabled: true`. Summarizes turns that fall out of the history window into episodic notes (after first audio, not on the TTS hot path).

| Key | What it does |
|---|---|
| `agents.enabled` | Master switch |
| `helper_base_url` / `helper_model` | Empty = same llama-server, separate HTTP client |
| `timeout_ms` | Helper budget |
| `collaborative` | Second pass: extract one action item from evicted turns |

A second GGUF is optional. Point `helper_base_url` at another `llama-server` only if you have RAM/GPU left.

### Profile packs

Drop `profiles/<id>/manifest.yaml` (+ optional `personality.txt`). Set `profiles.active` to that `id`, or `""` for none. `voice` in the manifest must match a discovered Voice name (clone folder or Piper stem).

Shipped example: `profiles/casual` → `profiles.active: casual`.

### Future slices (all off by default)

| Key | What it does |
|---|---|
| `future.affect` | Lexicon mood hint from your transcript (no extra model) |
| `future.inbox` | Consume `.txt`/`.md` from `inbox_dir` once per committed turn |
| `future.inbox_dir` | Default `data/inbox` (processed files go to `processed/`) |
| `future.adaptive_personality` | Fold stored `prefer` / `tone` into the Personality block |

Images in the inbox are ignored (no vision). Inbox notes are not used if you barge-in before the turn streams.

### Telemetry and eval

| Key / command | What it does |
|---|---|
| `telemetry.enabled` | Record marks/spans in memory |
| `telemetry.log_path` | Optional JSONL (relative to config) |
| `python3.11 -m voice.eval_harness --cases tests/eval/cases/sanitize.yaml --jsonl data/eval.jsonl` | Append sanitize snapshot |
| `--events path.jsonl` | Also score latency budgets from a metrics log |
| `python3.11 -m pytest -q` | Full unit suite (no GPU) |
| `VOICE_GOLDEN=1 scripts/eval_golden_wav.sh` | Optional WAV smoke (stub unless you add a clip) |

---

## Data files Voice writes

| Path | Contents |
|---|---|
| `data/preferences.yaml` | Remembered prefs |
| `data/episodic.jsonl` | Notes + helper summaries |
| `data/inbox/` → `processed/` | Dropped text notes |
| `data/eval.jsonl` | Eval snapshots if you point the harness there |

`config.yaml` and those data files are gitignored.

---

## Troubleshooting

| Symptom | Check |
|---|---|
| UI never answers | llama-server running? `llm.base_url` port match? |
| Start fails on STT | `whisper-server` on PATH, model file exists, port **8178** free |
| Slow first word (~5 s) | `LLAMA_ARG_REASONING` must be `off` on llama-server |
| No clone / robotic voice | `pip install -e ".[tts]"`, valid `ref.wav`/`ref.txt`, or set `tts.backend: piper` |
| Tkinter error | `brew install python-tk@3.11` |
| Refusals / lectures | Prompt is in `prompts/system.txt`; remaining refusals are inside the GGUF — swap the model |
| Barge-in does not stop audio | Click **Interrupt** or speak; confirm status leaves `THINKING_SPEAKING` |

Hardware target: Apple Silicon, ~36 GB (M3 Pro class). Do not put a second resident GGUF on the spoken path unless metrics show headroom.
