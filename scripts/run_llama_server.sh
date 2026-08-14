#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:-models/qwen3-8b-q4_k_m.gguf}
PORT=${PORT:-8080}
# Voice turns need non-thinking mode by default (first visible token ~0.3s vs ~4.8s).
# Override with LLAMA_ARG_REASONING=on if you explicitly want reasoning.
export LLAMA_ARG_REASONING="${LLAMA_ARG_REASONING:-off}"

exec llama-server -m "$MODEL_PATH" --port "$PORT" -ngl 99 --ctx-size 8192
