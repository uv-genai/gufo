# Decode optimisation: findings and handoff

Branch `uv-rdma`, baseline tag `research/decode-perf-baseline-2026-10-14`.
Target: AMD Strix Halo `gfx1151` (gfx11 ISA, RDNA 3.5), 20 CUs, wave32.
Model: Qwen3.8-Flash-Next `UD-Q4_K_XL`.

This records everything measured so far so work can continue without
re-deriving it. Every number here came from a real run on this host; none is
extrapolated.

## Order of work

Agreed as **3, 1, 2** - per-step cost, then concurrency, then GEMV bytes - and
then reordered by data: **the GEMV path is the real target**. See
"Dispatch map" below. Concurrency and per-step overhead are secondary.

Related: the IU4/quantisation study is a separate, closed body of work on
branch `iu4`, tagged `research/iu4-instruction-study-2026-10-14`. It is a
negative result - gfx1151 has no fp8/fp4/fp6 matrix instruction, IU4 K=16 is
the permanent ceiling, and int4 activations cost 11.4x int8. Do not reopen
quantisation format work on this GPU without new hardware.

## How to reproduce the profile

`tools/prof/prof.py` cannot be used on this host: its `show` step needs the
rocpd database, and rocprofv3's generator rejects the installed schema
(`ROCPD_STATUS_ERROR_SQL_SCHEMA_INVALID_VERSION`). Collect the raw kernel
trace instead and analyse it with `tools/prof/decode_trace.py`.

```sh
unset ROCM_PATH
export PATH=/opt/rocm/bin:/opt/rocm/core/lib/llvm/bin:$PATH

M=/home/ugo/.cache/models/Qwen3.8-Flash-Next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
rocprofv3 --kernel-trace -f csv -d /tmp/ktrace -- \
  ./build/release/gufo prompt -m "$M" -p "<prompt>" -n 64 -t 0 --no-display-prompt

python3 tools/prof/decode_trace.py /tmp/ktrace/<host>/*kernel_trace.csv --top 12
python3 tools/prof/decode_trace.py ... --shapes "mul_mat_vec_q8<1, false" --tokens 64
python3 tools/prof/decode_trace.py ... --order  "mul_mat_vec_q8<1, false"
```

`rocprofv3` writes into a hostname subdirectory. Use `-t 0` so sampling is
greedy and the token stream is reproducible.

## Model structure

From `src/models/qwen38_flash_next/config.cpp:249-258`:

```
num_layers        48
hidden_size       2560
num_experts       512
num_experts_used  10        (top-10 routing)
expert_ff         640
```

## Headline: decode is GPU-bound, and one kernel dominates

64-token single-stream decode:

| metric | value |
| --- | --- |
| decode pass span | 2463.66 ms |
| GPU busy | 2161.58 ms |
| **occupancy** | **87.74%** |
| idle between dispatches | 302.09 ms (12.26%) |

This killed the earlier hypothesis that decode was launch-bound. Only about an
eighth of a step is host or launch overhead, so launch-count reduction alone
caps at roughly 12%.

Top kernels in decode:

| kernel | total ms | calls | avg us | share |
| --- | ---: | ---: | ---: | ---: |
| `mul_mat_vec_q8<1, false, 1, false, false>` | 1198.63 | 15618 | 76.75 | 55.5% |
| `mul_mat_vec_q_moe<ggml_type 12>` (Q4_K experts) | 293.36 | 3008 | 97.53 | 13.6% |
| `mul_mat_vec_q_moe<ggml_type 7>` (Q5_1) | 181.46 | 2752 | 65.94 | 8.4% |
| `qfn_q8_hc_down_kernel<1, false>` | 108.96 | 6209 | 17.55 | 5.0% |
| `mul_mat_vec_q8<1, true, 1, false, false>` | 55.42 | 3072 | 18.04 | 2.6% |

`ggml_type 12` is Q4_K, `type 7` is Q5_1.

The largest idle gaps all precede `quantize_q8_1` or
`quantize_q8_1_by_expert`, up to 6.96 ms. That is where the remaining host-side
work sits, and it is the 12%.

## Dispatch map: the dense GEMV is five projections

`mul_mat_vec_q8<1, false, ...>` is one template instantiated for several
projections. Grouped by grid X (`--shapes`):

| gridX | per token | avg us | total ms |
| ---: | ---: | ---: | ---: |
| 327680 | 98.02 | 18.1 | 113.7 |
| 81920 | 97.00 | 44.1 | 273.8 |
| 524288 | 36.00 | 207.2 | 477.3 |
| 425984 | 12.00 | 167.8 | 128.9 |
| 7946240 | 1.02 | 3151.7 | 204.9 |  (lm_head, once per token)

All wave32 (`Workgroup_Size_X` 32).

Transitions (`--order`) resolve the chain:

```
  81920  ->  327680  6208/6208  100%   <- always
  524288 ->  81920   2304/2304  100%   <- always
  425984 ->  81920    768/768   100%   <- always
  327680 ->  81920   50.0%
  327680 ->  524288  36.7%
  327680 ->  425984  12.2%
```

So `327680` and `81920` are a fixed back-to-back pair forming the backbone of a
token, and `524288` / `425984` insert before `81920` on some iterations.

The other dense instantiation, `mul_mat_vec_q8<1, true, ...>`, has exactly one
shape (gridX 20480, 48 per token, 18.0 us). One projection, no fusion needed.
Useful as a control when measuring.

## Source confirmation: the projections share one input

`src/models/qwen38_flash_next/kernels/rocm/executor.cpp:4693-4699`:

```cpp
if (!l.attn_qkv.empty()) {
  if (!DenseBatch(l.attn_qkv, s_.mixed, s_.qg, n, error_msg))
    return false;
} else if (!DenseBatch(l.attn_q, s_.mixed, s_.qg, n, error_msg) ||
           !DenseBatch(l.attn_k, s_.mixed, s_.k,  n, error_msg) ||
           !DenseBatch(l.attn_v, s_.mixed, s_.v,  n, error_msg)) {
  return false;
}
```

Three points follow, and they are the crux of the opportunity:

1. **`attn_q`, `attn_k`, `attn_v` all read `s_.mixed`** - the same activation
   tensor. Three separate M=1 dispatches, three reads of the same 2560-element
   row.
2. **The stacked path is not taken.** `l.attn_qkv` is only populated when
   `stackable()` succeeds, and `stackable()`
   (`device_model.cpp:639-648`) requires every part to be `kQ8_0` **with equal
   `cols`**. Under GQA the Q, K and V dimensions differ, so it declines and
   they stay separate. The three distinct gridX values are that difference
   made visible.
3. **`indexer_k` also reads `s_.mixed`** (`executor.cpp:4680`), a fourth
   consumer.

The linear/GDN path has the same shape of opportunity: `ssm_in` and
`ssm_alpha_beta` both consume `x`
(`executor.cpp:2204` and `:2219`).

## What a fusion would save

The `81920`/`327680` pair is 97 pairs per token at 44.1 + 18.1 = 62.2 us, about
6.0 ms per token, roughly **32% of decode GPU time**.

`524288` is the single largest dense GEMV cost at 477 ms (22%) and always
precedes `81920`, so if Q, K, V and indexer_k come off one shared input, a
kernel handling **unequal output widths** would cover more than half of dense
GEMV time.

The gain is **not** dispatch count - occupancy is already 87.7%. It is:

- **activation memory traffic**: reading `s_.mixed` once instead of three or
  four times;
- **wave utilisation**: larger kernels over more output columns occupy the 20
  CUs better than three narrow ones.

## Open questions before writing a kernel

- Confirm the gridX shapes map to `attn_q` / `attn_k` / `attn_v` / `indexer_k`
  and not to something else. Deriving N from gridX needs the kernel's launch
  geometry in `kernels.hip.cpp`, which has not been read yet. The cadence
  (about two per layer across 48 layers) is consistent but not proof.
- Establish the arithmetic-intensity argument quantitatively. Occupancy at
  87.7% means the saving must come from bytes and utilisation; size that
  before committing, rather than assuming.
- Decide whether to extend `stackable()` to unequal `cols` (a padded or
  banded stack) or write a dedicated multi-width GEMV. The former reuses the
  existing Q8_0 path; the latter is more control.

## Files

- `tools/prof/decode_trace.py` - pass splitting, per-kernel rollup, `--shapes`
  and `--order`. Reads the raw rocprofv3 CSV; does not need rocpd.
- `tools/prof/isa_mix.py` - instruction mix, not yet used in this work.
- `tools/bench/build.sh` - standalone HIP experiment builds.
- `docs/models/qwen3.8-flash-next/IU4-PLAN.md` - the closed IU4 study.
