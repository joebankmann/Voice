# Local Voice Chat

Offline-first streaming voice chat using whisper.cpp for speech recognition,
llama.cpp for local generation, Piper for speech synthesis, and Silero for
voice activity detection.

## Prerequisites

The project requires Python 3.11 or newer. On macOS, install the native
runtimes with Homebrew:

```bash
brew install python@3.11 python-tk@3.11 whisper-cpp llama.cpp
python3.11 -m pip install piper-tts
```

The `whisper-cli`, `llama-server`, and `piper` executables must be available on
`PATH`. If a package-manager binary name or path differs, update
`config.yaml`.

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

## Customize voices

Assistant speech uses local **Piper** voices. You can change the voice in three
ways:

1. **Config (default voice)** — in `config.yaml` set:
   - `tts.voice_path` — path to the active `.onnx` model
   - `tts.voices_dir` — folder scanned for available voices (default `models`)
   - `tts.length_scale` — speaking rate (`1.0` normal; lower is faster, higher is slower)
   - `tts.sample_rate` — fallback rate if a voice has no `.onnx.json` sidecar
2. **Desktop UI** — the **Voice** dropdown lists every `*.onnx` file in
   `tts.voices_dir`. Pick one to switch immediately for the next spoken reply
   (sample rate is read from the matching `.onnx.json` when present).
3. **Add more voices** — download another Piper `.onnx` + `.onnx.json` pair into
   `voices_dir` (see `scripts/download_models.sh` and
   https://github.com/rhasspy/piper/blob/master/VOICES.md). Restart the app (or
   re-open the UI) so newly added files appear in the dropdown.

Voice cloning / non-Piper TTS engines are Phase 2.

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

An NSFW adapter, improved TTS, and a helper model are intentionally deferred to
Phase 2.

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
- Silero VAD loaded locally in 0.032 s. The automated suite passed 19/19 tests.
- Microphone barge-in, physical playback, UI interaction, and end-to-end
  time-to-first-audio remain **inconclusive — requires human mic test**.
