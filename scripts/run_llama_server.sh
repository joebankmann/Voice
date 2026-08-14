#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:-models/qwen3-8b-q4_k_m.gguf}
PORT=${PORT:-8080}

exec llama-server -m "$MODEL_PATH" --port "$PORT" -ngl 99 --ctx-size 8192
