#!/usr/bin/env bash
# llama-swap: two-host Gufo TP2 (RDMA) Qwen3.8 Flash-Next Q4 or Q8.
#
#   run PORT [q4|q8] : rank 1 on $RANK1_HOST, then exec rank 0 on PORT (cmdStart)
#   stop             : stop both ranks (cmdStop)
#
# Only one TP2 pair at a time (shared bootstrap/control 18515/18516).
# Copy the same build/release/gufo to the peer first. Start rank 0 first in
# production; this launcher starts rank 1 then rank 0 so llama-swap keeps
# rank 0 in the foreground (both still pair within 30 s).
#
# Example llama-swap config:
#   cmd: /path/to/tools/llama-swap/tp2.sh run ${PORT} q4
#   cmdStop: /path/to/tools/llama-swap/tp2.sh stop
set -euo pipefail

exec > >(tee -a "$HOME/gufo-tp2-swap.log") 2>&1
echo "=== tp2-swap $(date '+%F %T') args: $*"

RANK1_HOST="${RANK1_HOST:-strix-halo-2.local}"
HERE=$(cd "$(dirname "$0")" && pwd)
TP2="$HERE/../tp2"

stop_all() {
  echo "tp2-swap: stopping rank 0 (local) and rank 1 ($RANK1_HOST)"
  pkill -x gufo 2>/dev/null || true
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$RANK1_HOST" 'pkill -x gufo' 2>/dev/null || true
  for _ in $(seq 1 30); do
    if ! pgrep -x gufo >/dev/null 2>&1 \
       && ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$RANK1_HOST" \
            'pgrep -x gufo >/dev/null' 2>/dev/null; then
      break
    fi
    sleep 1
  done
  sleep 1
}

case "${1:?usage: tp2.sh run PORT [q4|q8] | tp2.sh stop}" in
  stop)
    stop_all
    ;;
  run)
    PORT="${2:?tp2.sh run PORT: PORT missing}"
    VARIANT="${3:-q4}"
    case "$VARIANT" in
      q4)
        RANK0="$TP2/rank0-q4.sh"
        RANK1="$TP2/rank1-q4.sh"
        export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gufo-qwen3.8-flash-next-tp2-q4}"
        ;;
      q8)
        RANK0="$TP2/rank0-q8.sh"
        RANK1="$TP2/rank1-q8.sh"
        export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gufo-qwen3.8-flash-next-tp2-q8}"
        ;;
      *)
        echo "tp2-swap: unknown variant '$VARIANT' (use q4 or q8)" >&2
        exit 2
        ;;
    esac
    stop_all
    echo "tp2-swap: starting rank 1 ($VARIANT) on $RANK1_HOST"
    for _ in $(seq 1 15); do
      ss -tln | grep -qE ':(18515|18516) ' || break
      sleep 1
    done
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$RANK1_HOST" \
      "nohup $RANK1 >/home/ugo/tp2-rank1-swap.log 2>&1 </dev/null &"
    sleep 2
    echo "tp2-swap: starting rank 0 ($VARIANT) on port $PORT"
    exec "$RANK0" "$PORT"
    ;;
  *)
    echo "usage: tp2.sh run PORT [q4|q8] | tp2.sh stop" >&2
    exit 2
    ;;
esac
