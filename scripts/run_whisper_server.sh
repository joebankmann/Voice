#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:-models/ggml-large-v3-turbo.bin}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8178}
WHISPER_SERVER_BIN=${WHISPER_SERVER_BIN:-whisper-server}

exec "$WHISPER_SERVER_BIN" \
  -m "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --inference-path /inference
