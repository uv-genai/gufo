#!/usr/bin/env bash
# llama-swap: single-host Gufo Qwen3.8 Flash-Next Q4_K_XL + MTP.
#
#   run PORT : listen on 127.0.0.1:PORT (cmdStart)
#   stop     : stop the local gufo process (cmdStop)
#
# Example llama-swap config:
#   cmd: /path/to/tools/llama-swap/single-q4.sh run ${PORT}
#   cmdStop: /path/to/tools/llama-swap/single-q4.sh stop
set -euo pipefail

exec > >(tee -a "$HOME/gufo-single-q4-swap.log") 2>&1
echo "=== single-q4 $(date '+%F %T') args: $*"

HERE=$(cd "$(dirname "$0")" && pwd)
GUFO="${GUFO:-$(cd "$HERE/../.." && pwd)/build/release/gufo}"
MODEL_DIR="${MODEL_DIR:-$HOME/.cache/models/Qwen3.8-Flash-Next}"

stop_one() {
  echo "single-q4: stopping local gufo"
  pkill -x gufo 2>/dev/null || true
  for _ in $(seq 1 15); do
    pgrep -x gufo >/dev/null 2>&1 || break
    sleep 1
  done
}

case "${1:?usage: single-q4.sh run PORT | single-q4.sh stop}" in
  stop)
    stop_one
    ;;
  run)
    PORT="${2:?single-q4.sh run PORT: PORT missing}"
    stop_one
    exec "$GUFO" serve --host 127.0.0.1 --port "$PORT" \
      --max-request-bytes 268435456 llm \
      --served-model-name "${SERVED_MODEL_NAME:-gufo-qwen3.8-flash-next-q4}" \
      --model "$MODEL_DIR/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf" \
      --speculative mtp \
      --mtp-model "$MODEL_DIR/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" \
      --mmproj "$MODEL_DIR/mmproj-BF16.gguf" \
      --context "${CONTEXT:-8192}" \
      --sessions "${SESSIONS:-4}"
    ;;
  *)
    echo "usage: single-q4.sh run PORT | single-q4.sh stop" >&2
    exit 2
    ;;
esac
