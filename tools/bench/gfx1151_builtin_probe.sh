#!/usr/bin/env bash
# Compile-probe for amdgcn matrix builtins on gfx1151.
#
# __has_builtin is unreliable for __builtin_amdgcn_mfma_* (it reports names the
# tree demonstrably compiles as absent), so the only ground truth is compiling.
# For each candidate we emit a device function that calls the builtin with dummy
# args and check clang's error:
#   - "use of undeclared identifier"  -> builtin does not exist
#   - anything else (type errors, ok)  -> builtin exists
#
# Usage: bash tools/bench/gfx1151_builtin_probe.sh
set -u
export PATH=/opt/rocm/bin:/opt/rocm/core/lib/llvm/bin:${PATH:-}

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

CANDIDATES=(
  # tree-verified names (controls)
  __builtin_amdgcn_mfma_f32_16x16x16f16
  __builtin_amdgcn_mfma_f32_16x16x16bf16_1k
  __builtin_amdgcn_mfma_f32_16x16x8bf16
  __builtin_amdgcn_mfma_f32_16x16x4f32
  __builtin_amdgcn_mfma_f32_16x16x8_xf32
  __builtin_amdgcn_mfma_i32_16x16x16i8
  __builtin_amdgcn_mfma_i32_16x16x32_i8
  __builtin_amdgcn_mfma_i32_32x32x16_i8
  __builtin_amdgcn_mfma_i32_32x32x8i8
  __builtin_amdgcn_wmma_f32_16x16x16_f16_w32
  __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32

  # FP8 - mfma naming
  __builtin_amdgcn_mfma_f32_16x16x32f8
  __builtin_amdgcn_mfma_f32_16x16x32f8e4m3
  __builtin_amdgcn_mfma_f32_16x16x32f8e4m3fn
  __builtin_amdgcn_mfma_f32_16x16x32f8e5m2
  __builtin_amdgcn_mfma_f32_32x32x16f8e4m3
  __builtin_amdgcn_mfma_f32_32x32x16f8e4m3fn
  __builtin_amdgcn_mfma_f32_16x16x128f8f6f4
  __builtin_amdgcn_mfma_f32_32x32x128f8f6f4
  __builtin_amdgcn_mfma_f32_16x16x16f8e4m3
  __builtin_amdgcn_mfma_f32_16x16x16f8e5m2
  __builtin_amdgcn_mfma_f32_16x16x32f8fn

  # FP8 - wmma naming
  __builtin_amdgcn_wmma_f32_16x16x32_f8e4m3_fn_w32
  __builtin_amdgcn_wmma_f32_16x16x32_f8e5m2_w32
  __builtin_amdgcn_wmma_f32_16x16x32_f8e4m3_w32
  __builtin_amdgcn_wmma_f32_16x16x32_f8e5m2_fn_w32

  # FP4
  __builtin_amdgcn_mfma_f32_16x16x32f4
  __builtin_amdgcn_mfma_f32_32x32x32f4
  __builtin_amdgcn_mfma_f32_16x16x128f4
  __builtin_amdgcn_mfma_f32_32x32x128f4
  __builtin_amdgcn_mfma_f32_16x16x64f4e2m1
  __builtin_amdgcn_smfmac_f32_16x16x32f8e4m3
  __builtin_amdgcn_smfmac_f32_16x16x128f4
  __builtin_amdgcn_wmma_f32_16x16x64_f4e2m1_w32
)

present=0
absent=0
printf '%-52s %s\n' "BUILTIN" "STATUS"
printf '%-52s %s\n' "-------" "------"
for b in "${CANDIDATES[@]}"; do
  f="$WORK/probe.hip"
  cat >"$f" <<EOF
#include <hip/hip_runtime.h>
__device__ int probe_fn(int x) {
  (void)x;
  return (int)(long)$b;
}
EOF
  err=$(hipcc -fsyntax-only --offload-arch=gfx1151 -std=c++20 "$f" 2>&1)
  if printf '%s' "$err" | grep -qi 'use of undeclared identifier'; then
    status="absent"
    absent=$((absent + 1))
  else
    status="PRESENT"
    present=$((present + 1))
  fi
  printf '%-52s %s\n' "$b" "$status"
done

printf '\n%d present, %d absent\n' "$present" "$absent"
