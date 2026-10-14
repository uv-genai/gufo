# IU4 (int4 WMMA) GEMM path for Qwen3.8-Flash-Next

Status: draft, `iu4` branch. Target: AMD Strix Halo `gfx1151` only.

## Goal

Add an int4-WMMA GEMM path for decode, and measure whether it beats the
current path on speed without an unacceptable quality loss. Nothing here
becomes default until both are measured.

## What is already established

These were measured on this toolchain, not assumed:

- `wmma_i32_16x16x16_iu4_w32(bool, v2, bool, v2, v8, bool)` compiles and runs on
  gfx1151. Throughput 566,599 GMACs/s vs 351,997 for the iu8 equivalent, a
  **1.61x** gain in a dependent-chain microbenchmark
  (`tools/bench/gfx1151_iu4_bench.hip`). Both do 4096 MACs/instruction; iu4
  carries K=16 in half the operand bytes.
- Operand packing: nibble `i` is logical `k=i`, **identical for A and B**
  (`tools/bench/gfx1151_iu4_layout.hip`, exact identity diagonal, 16/256).
- Accumulator layout (`tools/bench/gfx1151_iu4_c_layout.hip`):
  `element = row / 2`, `lane = (row % 2) * 16 + col`.
- No fp8/fp4/fp6 matrix instruction exists on gfx1151, so ROCmFP/ROCmFPX
  cannot be used.
- `wmma_i32_16x16x32_iu4_w32_gfx12` (K=32, the 2x-MAC form) is unreachable:
  it needs `wmma-128b-insts`, and enabling that feature segfaults this
  ROCm 7.2.3 clang during codegen. K=16 is the ceiling for now.

## Design decisions

**Weights: reuse Q4_K as-is. No new format, no re-quantization.**
`block_q4_K` (`src/core/quant/ggml_dequant.hpp:22`) stores 256 int4 in `qs[128]`,
which is the layout the WMMA consumes - but **not** verbatim. Gufo's reference
`Q4Value` (`src/core/quant/ggml_dequant.cpp:71`) shows the packing:

```
g = index / 32            // 8 groups of 32
qs[(g/2)*32 + lane]       // adjacent groups share a byte
nibble = (g & 1) ? high : low
```

So a K-slice inside one group reads 16 consecutive bytes and takes the same
nibble position from each - a 2:1 expand, not an arbitrary swizzle. Group `g`
is a constant `(d_g, dmin_g)`, so one group is two K=16 WMMA calls into the
same `C`, then one affine correction:

```
sum(a*w) = d*d_g*sum(a*q) - dmin*dmin_g*sum(a)
```

which needs `sum(a*q)` from the WMMA plus a separate `sum(a)` reduction. Scale
packing is in `GetQ4ScaleMin` (ggml_dequant.cpp:45): groups 0-3 take the low
6 bits of `scales[g]`/`scales[g+4]`; groups 4-7 are interleaved across the
upper bits. No new format, no re-quantization.

**Activations: int4 with a per-row fp16 scale. This is the quality risk.**
`wmma_i32_16x16x16_iu4_w32` takes iu4 x iu4; both operands are int4. There is no
mixed iu8 x iu4 instruction on this hardware, so int4 weights force int4
activations on this path. This is independent of the weight format and is the
main thing that can fail the quality gate.

**Scope: decode only.** Prefill stays on the existing path. Prefill is
FLOP-bound and int4 does not help it; CIRU makes the same split, reconstructing
BF16 tiles for long prompts.

**Rollout: behind a flag, default off**, until quality and speed are both
measured. Production paths must not regress; successful optimisations become
the default, not an extra switch.

## Phases

Each phase has one artifact and one pass/fail check. Do not start the next
phase if the check fails.

### Phase 0 - Standalone kernel correctness
Write `tools/bench/gfx1151_iu4_q4k_gemm.hip`: a decode-shaped GEMM reading real
`block_q4_K` data, applying sub-block affine corrections, with an int4
activation quantizer.
Check: output matches a CPU reference that dequantizes Q4_K to f32 and dot
products in f32, to a stated tolerance. No model code touched.

### Phase 1 - Activation quantisation gate
Measure int4 activation error against the current Q8_1 / f16 activations on
real activation tensors from a live run, not synthetic data.
Check: report int4 vs int8 activation KL/top-1 delta. **This is the decision
point.** If int4 activations cost too much, stop - the kernel is not worth
building and the plan changes to finding an iu8-compatible route.

### Phase 2 - Model integration
Wire the kernel into the qwen38_flash_next GEMM dispatch for decode only,
behind `--iu4-decode` (or the project's canonical long option), default off.
Check: existing suites still pass with the flag off; `gpu-test` covers the new
kernel.

### Phase 3 - Quality
Run the existing KL/top-1 panel against the current baseline.
Check: report the delta. If it regresses materially, do not publish as
default.

### Phase 4 - Speed
Matched decode measurements, timings mandatory per request/phase, never
averaged. Compare against the current path on the same build/settings.
Check: a real gain at the batch sizes actually served.

## Open risks

- Int4 activations may be too lossy (Phase 1 exists to find out early).
- The 1.61x is a synthetic chain; real decode gain depends on keeping matrix
  units fed at the tile sizes actually used. It may be much less, or nothing.
- The K=32 form would roughly double MACs and is blocked purely by a clang
  codegen bug, not by hardware.
