#!/usr/bin/env bash
# Standalone single-host Gufo server with Q4_K_XL Flash-Next + MTP.
# Usage: serve-q4.sh [PORT]   (default 8000)
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
MODEL_DIR="${MODEL_DIR:-$HOME/.cache/models/Qwen3.8-Flash-Next}"
GUFO="${GUFO:-$(cd "$HERE/../.." && pwd)/build/release/gufo}"
PORT="${1:-8000}"

exec "$GUFO" serve --host 127.0.0.1 --port "$PORT" --max-request-bytes 268435456 llm \
  --served-model-name "${SERVED_MODEL_NAME:-Qwen3.8 Flash Next}" \
  --model "$MODEL_DIR/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf" \
  --speculative mtp \
  --mtp-model "$MODEL_DIR/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" \
  --mmproj "$MODEL_DIR/mmproj-BF16.gguf" \
  --context "${CONTEXT:-8192}" \
  --sessions "${SESSIONS:-1}"
