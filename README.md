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
