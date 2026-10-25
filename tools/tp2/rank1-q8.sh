#!/usr/bin/env bash
# Gufo TP2 rank 1 (worker) — runs on strix-halo-2.
# Full Q8_0 weights. Peer is rank 0 on strix-halo-1 (192.168.1.22).
# TCP bootstrap over LAN; RDMA data over usb4_rdma* (thunderbolt-ibverbs,
# write striping across both USB4 cables). Start rank 0 first, then this
# within 30 s.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
MODEL_DIR="${MODEL_DIR:-$HOME/.cache/models/Qwen3.8-Flash-Next}"
GUFO="${GUFO:-$(cd "$HERE/../.." && pwd)/build/release/gufo}"
# Device names shift after re-plugs; striping covers every rail anyway.
RDEV=$(ibv_devices | awk '/usb4_rdma/{print $2, $1}' | sort | head -n1 | awk '{print $2}')
[ -n "$RDEV" ] || { echo "no usb4_rdma device found" >&2; exit 1; }

exec "$GUFO" serve llm \
  --model "$MODEL_DIR/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf" \
  --speculative mtp \
  --mtp-model "$MODEL_DIR/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" \
  --mmproj "$MODEL_DIR/mmproj-BF16.gguf" \
  --tp-world-size 2 --tp-rank 1 \
  --tp-bootstrap-host 192.168.1.22 \
  --tp-bootstrap-port 18515 \
  --tp-control-port 18516 \
  --tp-control-token 92ffa404804d336980d95f224d6bd678 \
  --tp-rdma-device "$RDEV" \
  --sessions "${SESSIONS:-2}" \
  --context "${CONTEXT:-524288}" \
  --enable-yarn
