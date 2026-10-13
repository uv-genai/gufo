#!/usr/bin/env bash
# Find the working operand signature of gfx1151 WMMA builtins, serially.
#
# __has_builtin is unreliable for these names and the earlier xargs harness was
# flaky, so this compiles one candidate per call and reads clang's error
# directly. Vector operands cannot be broadcast from a scalar, so each vector
# type is brace-initialised.
#
# Usage: bash tools/bench/wmma_sig_probe.sh [builtin...]
set -u
export PATH=/opt/rocm/bin:/opt/rocm/core/lib/llvm/bin:${PATH:-}

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

init_of() {
  case "$1" in
    vint)  echo "1" ;;
    v2) echo "{1, 1}" ;;
    v4) echo "{1, 1, 1, 1}" ;;
    v8) echo "{1, 1, 1, 1, 1, 1, 1, 1}" ;;
    v16) echo "{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}" ;;
    v32) echo "{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}" ;;
    *) echo "1" ;;
  esac
}

try() {  # try builtin A B C  -> prints "" on success, else first error line
  local b="$1" A="$2" B="$3" C="$4"
  local f="$WORK/t.hip"
  cat >"$f" <<EOF
#include <hip/hip_runtime.h>
using v2 = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using v4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using v8 = __attribute__((__vector_size__(8 * sizeof(int)))) int;
using v16 = __attribute__((__vector_size__(16 * sizeof(int)))) int;
using v32 = __attribute__((__vector_size__(32 * sizeof(int)))) int;
__device__ void probe() {
  const bool s = true;
  const $A a = $(init_of "$A");
  const $B b = $(init_of "$B");
  const $C c = $(init_of "$C");
  auto r = $b(s, a, s, b, c, s);
  (void)r;
}
EOF
  hipcc -fsyntax-only --offload-arch=gfx1151 -std=c++20 "$f" 2>&1 |
    grep -E "error:" | head -1
}

if [ "$#" -gt 0 ]; then
  BUILTINS=("$@")
else
  BUILTINS=(
    __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32
    __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64
    __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32
    __builtin_amdgcn_wmma_i32_16x16x16_iu4_w64
    __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12
    __builtin_amdgcn_wmma_i32_16x16x16_iu4_w64_gfx12
    __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12
    __builtin_amdgcn_wmma_i32_16x16x32_iu4_w64_gfx12
  )
fi

# Operands are normally the same width; the accumulator is wider. Probe the
# plausible widths rather than every Cartesian product.
OPERANDS=(v2 v4 v8 v16)
ACCS=(v4 v8 v16 v32)

for b in "${BUILTINS[@]}"; do
  found=""
  for A in "${OPERANDS[@]}"; do
    for C in "${ACCS[@]}"; do
      err=$(try "$b" "$A" "$A" "$C")
      if [ -z "$err" ]; then
        found="(bool, $A, bool, $A, $C, bool)"
        break 2
      fi
      # Capture what clang demanded for the accumulator, to short-circuit.
      want=$(printf '%s' "$err" | grep -oP "parameter of type '\K[^']+" | head -1)
      if [ -n "$want" ]; then
        printf '  %s with A=%-5s C=%-5s -> wants %s\n' "$b" "$A" "$C" "$want" >&2
      fi
    done
  done
  if [ -n "$found" ]; then
    printf '%-48s WORKS %s\n' "$b" "$found"
  else
    printf '%-48s no width found\n' "$b"
  fi
done
