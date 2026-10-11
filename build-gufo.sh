#!/usr/bin/env bash
# Build gufo with TP2 RDMA on either twin Strix Halo host (identical environments).
# See BUILD-LOCAL.md for why the workarounds below are needed on CachyOS.
#
# Usage:
#   ./build-gufo.sh          # configure + build + gufo diagnose
#   ./build-gufo.sh --clean  # wipe build/release first (fresh configure)
#   JOBS=8 ./build-gufo.sh   # override build parallelism (default 16)
set -euo pipefail

cd "$(dirname "$0")"

if [[ "${1:-}" == "--clean" ]]; then
  rm -rf build/release
fi

# ROCm ships under /opt/rocm as symlinks into /opt/rocm/core.
# Do NOT set ROCM_PATH: it breaks clang's ROCm device-library lookup through
# the symlinked layout ("cannot find ROCm device library"). See BUILD-LOCAL.md.
unset ROCM_PATH
export PATH="/opt/rocm/bin:/opt/rocm/core/lib/llvm/bin:$PATH"

cmake --preset release -DGUFO_ENABLE_TP2_RDMA=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/core/lib/llvm/bin/clang++ \
  -DCMAKE_CXX_FLAGS="-include format" \
  -DCMAKE_HIP_FLAGS="-include format"

cmake --build --preset release --parallel "${JOBS:-16}"

./build/release/gufo diagnose
