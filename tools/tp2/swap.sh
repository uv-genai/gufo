#!/usr/bin/env bash
# llama-swap launcher for the two-host Gufo TP2 pair.
#
#   run PORT [q4|q8]  : start rank 1 on strix-halo-2 (ssh), then exec rank 0 on PORT
#   stop              : stop both ranks (used as cmdStop by llama-swap)
#
# Only one TP2 pair can run at a time: both ranks share bootstrap/control
# ports 18515/18516. Rank 0 logs its events to stdout (llama-swap logs) and
# llama-swap polls http://127.0.0.1:$PORT/health until the pair is ready.
set -euo pipefail

exec > >(tee -a "$HOME/tp2-swap.log") 2>&1
echo "=== tp2-swap $(date '+%F %T') args: $*"

RANK1_HOST="${RANK1_HOST:-strix-halo-2.local}"
HERE=$(cd "$(dirname "$0")" && pwd)

stop_all() {
  echo "tp2-swap: stopping rank 0 (local) and rank 1 ($RANK1_HOST)"
  pkill -x gufo 2>/dev/null || true
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$RANK1_HOST" 'pkill -x gufo' 2>/dev/null || true
  # Wait for both ranks to exit so the next variant can bind the bootstrap
  # ports; otherwise swaps race and the new rank 0 exits with "upstream
  # command exited prematurely".
  for _ in $(seq 1 30); do
    if ! pgrep -x gufo >/dev/null 2>&1 \
       && ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$RANK1_HOST" 'pgrep -x gufo >/dev/null' 2>/dev/null; then
      break
    fi
    sleep 1
  done
  sleep 1
}

case "${1:?usage: tp2-swap.sh run PORT [q4|q8] | tp2-swap.sh stop}" in
  stop)
    stop_all
    ;;
  run)
    PORT="${2:?tp2-swap.sh run PORT: PORT missing}"
    VARIANT="${3:-q4}"
    case "$VARIANT" in
      q4)
        RANK0=$HERE/rank0-q4.sh
        RANK1=$HERE/rank1-q4.sh
        export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gufo-qwen3.8-flash-next-tp2-q4}"
        ;;
      q8)
        RANK0=$HERE/rank0-q8.sh
        RANK1=$HERE/rank1-q8.sh
        export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gufo-qwen3.8-flash-next-tp2-q8}"
        ;;
      *)
        echo "tp2-swap.sh: unknown variant '$VARIANT' (use q4 or q8)" >&2
        exit 2
        ;;
    esac
    echo "tp2-swap: starting rank 1 ($VARIANT) on $RANK1_HOST"
    # Don't start while a previous pair still holds the bootstrap ports.
    for _ in $(seq 1 15); do
      ss -tln | grep -qE ':(18515|18516) ' || break
      sleep 1
    done
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$RANK1_HOST" \
      "nohup $RANK1 >/home/ugo/tp2-rank1-swap.log 2>&1 &"
    sleep 2   # both ranks pair within 30 s of each other
    echo "tp2-swap: starting rank 0 ($VARIANT) on port $PORT"
    exec "$RANK0" "$PORT"
    ;;
  *)
    echo "usage: tp2-swap.sh run PORT [q4|q8] | tp2-swap.sh stop" >&2
    exit 2
    ;;
esac
