#!/usr/bin/env bash
# Optional golden-WAV smoke. Default pytest does not run this.
set -euo pipefail
if [[ "${VOICE_GOLDEN:-}" != "1" ]]; then
  echo "Skipping golden WAV eval (set VOICE_GOLDEN=1 to enable)."
  exit 0
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo "Golden WAV eval is a manual/local check; wire a reference clip under tests/eval/golden/ when you have one."
exit 0
