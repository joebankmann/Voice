#!/usr/bin/env bash
set -euo pipefail

mkdir -p models

cat <<'EOF'
Model downloads are large, so this script does not fetch them automatically.
Review the expected sizes and destinations below, then run the commands you need.

Whisper large-v3-turbo for whisper.cpp (~1.6 GB)
Destination: models/ggml-large-v3-turbo.bin
curl -L --fail --output models/ggml-large-v3-turbo.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin

Qwen3 8B Q4_K_M GGUF (~5 GB)
Destination: models/qwen3-8b-q4_k_m.gguf
curl -L --fail --output models/qwen3-8b-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf

Piper en_US-lessac-medium ONNX (~61 MB)
Destination: models/en_US-lessac-medium.onnx
curl -L --fail --output models/en_US-lessac-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx

Piper en_US-lessac-medium JSON (<1 MB)
Destination: models/en_US-lessac-medium.onnx.json
curl -L --fail --output models/en_US-lessac-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json

Additional Piper voices
Browse https://github.com/rhasspy/piper/blob/master/VOICES.md
Download any matching .onnx + .onnx.json pair into the configured voices_dir
(default: models/). The desktop UI lists every *.onnx file found there.
EOF
