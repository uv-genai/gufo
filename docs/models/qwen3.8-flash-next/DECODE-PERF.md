# Decode optimisation: findings and handoff

Branch `uv-rdma`, baseline tag `research/decode-perf-baseline-2026-10-14`.
Target: AMD Strix Halo `gfx1151` (gfx11 ISA, RDNA 3.5), 20 CUs, wave32.
Model: Qwen3.8-Flash-Next `UD-Q4_K_XL`.

This records everything measured so far so work can continue without
re-deriving it. Every number here came from a real run on this host; none is
extrapolated.

## Order of work

Agreed as **3, 1, 2** - per-step cost, then concurrency, then GEMV bytes. The
profile then re-pointed at the GEMV path, and arithmetic has since narrowed it
again. The current state, in one line each:

- **Per-step / launch overhead**: measured at 12.26% of the decode pass. Real,
  small, not the main lever.
- **Concurrency**: not re-examined since. Expect it to be bounded by the same
  GPU-bound decode, so revisit only after the dense-GEMV question closes.
- **GEMV fusion**: **rejected.** Arithmetic below shows the shared activation
  is 0.019% of the traffic. Do not build it.
- **Dense GEMV, the open question**: 55.5% of decode time, and whether it is at
  the memory wall is undetermined. Resolving that is the next task.

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

**The shared input is real but does not by itself yield an optimisation** - see
the arithmetic below. Do not treat it as a work item on its own.

## The fusion idea does not survive arithmetic - do not build it

The dispatch map and the shared-input finding make a fusion look attractive,
but the memory-traffic argument for it is wrong, and it is worth being explicit
about why so nobody re-derives it.

For an M=1 GEMV the weights dominate traffic. With K=2560 and Q8_0 at 34 bytes
per 32 weights, one activation row is **2.81 KiB** against **30.64 MB** of
weights for a representative Q/K/V trio:

| N_out | weights | activation share |
| ---: | ---: | ---: |
| 2560 | 6.96 MB | 0.0413% |
| 5120 | 13.93 MB | 0.0207% |
| 10240 | 27.85 MB | 0.0103% |

Fusing three such projections to read `s_.mixed` once saves **5.62 KiB out of
30.64 MB, or 0.019%**. The activation is free either way; the weight-to-
activation ratio is about 2700:1. The "read the activation once instead of
three times" argument that the shared input suggested does not apply at M=1,
where the activation is negligible next to the weights it selects.

That leaves only two possible wins, both small:

- **launch overhead**, capped by the measured 12.26% idle, of which this pair
  is only part;
- **wave utilisation**, unmeasured.

Neither justifies a kernel against 55.5% of decode time. The dispatch map is
recorded as understanding, not as a work item.

## What the numbers actually point at

If the dense GEMV is bandwidth-bound, the only lever on 55.5% of decode time is
**fewer weight bytes** - quantising those projections, which costs quality - or
reading fewer of them. Kernel work cannot help a kernel already at the memory
wall.

Whether they are at the wall was unknown when this section was written, and
that made it the decisive question. Strix Halo LPDDR5x peaks near **256 GB/s**.
Taking that peak, the measured runtimes imply:

| shape | us | bytes at peak | implied N | effective BW |
| ---: | ---: | ---: | ---: | ---: |
| 327680 | 18.1 | 4.63 MB | 1704 | 256 GB/s |
| 81920 | 44.1 | 11.29 MB | 4151 | 256 GB/s |
| 524288 | 207.2 | 53.04 MB | 19501 | 256 GB/s |

For comparison a single N=2560 projection is 6.96 MB, or **27.2 us at peak**;
the measured 18.1 and 44.1 us bracket that. So the shapes are consistent with
running at or near peak bandwidth - which would mean there is nothing to gain -
but the reasoning is circular: N was assumed to derive the bandwidth.

## What the kernel source establishes, and what it does not

`src/models/qwen38_flash_next/kernels/rocm/mmq/mmvq.hip.cpp` holds the dense
kernel. For `tokens == 1` the launch is:

```cpp
template <int tokens>
static void launch_q8(..., int k, int rows, ...) {
    ...
    mul_mat_vec_q8<tokens, false><<<rows, 32, 0, stream>>>(...)
```

So for that path **gridX is `rows`, one wave of 32 threads per output row**.
That is a solid, reusable fact.

It does **not** reconcile with the measured gridX, and the discrepancy is
unresolved. If `rows` were the projection's output dimension and K were 2560,
shape 327680 would move

```
327680 rows x (2560/32) blocks x 34 bytes = 891 MB in 18.1 us = 49 TB/s
```

which is physically impossible on LPDDR5x. Working backwards, 18.1 us at
256 GB/s is 4.63 MB, or about 14 bytes per row - not a legal Q8_0 row width.

Two readings are possible and neither has been checked:

- the large-grid dispatches come from a **different launch**, for instance
  `mul_mat_vec_moe_grouped` (`mmvq.hip.cpp:617`, `<<<grid, 64>>>`) or the
  batched variants `launch_q8_batch` / `launch_q8_matrix`, rather than
  `launch_q8<1>`;
- or `rows` is expert-stacked rather than a single projection's width.

Note all five gridX are exact multiples of 512 (`num_experts`):

```
 327680 = 512 x 640      81920 = 512 x 160
 524288 = 512 x 1024    425984 = 512 x 832
7946240 = 512 x 15520
```

and 640 is `expert_ff`. That is suggestive of expert-stacked weights, but
synchronising every shape with a single factor is not proof and the bandwidth
arithmetic contradicts the naive reading. **Do not size any optimisation on
these gridX until the launch site for these dispatches is identified.**

The immediate next task is therefore narrower than it was: find which launch
in `mmvq.hip.cpp` (or elsewhere) produces a gridX of this magnitude, by
matching the template instantiation in the trace name
`mul_mat_vec_q8<1, false, 1, false, false>` against the call sites at
`mmvq.hip.cpp:648-651` and the batched ones at `:661-664`.

## Host overhead: 13.2% of decode, mostly diffuse dispatch latency

Decomposing the 330 ms of idle in the decode tail (82,347 gaps):

| band | gaps | total ms | share of idle |
| --- | ---: | ---: | ---: |
| > 5 ms | 4 | 45.90 | 13.9% |
| 2-5 ms | 12 | 32.40 | 9.8% |
| 1-2 ms | 6 | 8.86 | 2.7% |
| 0.5-1 ms | 15 | 13.02 | 3.9% |
| **< 0.5 ms** | **82,310** | **230.21** | **69.7%** |

Two thirds of the idle is the **diffuse** cost of issuing 82k kernels:
230 ms over 82,310 gaps, about **2.8 us per gap**. That is ordinary dispatch
latency between consecutive kernels and only falls with fewer kernels, not
with better ones. It should not be chased as a bug.

The large stalls are rare: **22 gaps above 1 ms totalling 87 ms**, which is
**3.5% of decode**. The top 10 account for 65.6 ms.

Nineteen of those 22 are preceded by `HcCombineKernel<float>` and followed by
`quantize_q8_1`. That is the **unfused** path: at
`executor.cpp:1891-1899` the combine quantizes its own output via
`HcCombineQ8_1` when the guard
`!MatrixRows(n_tokens) && UnpaddedQ8(c.HcDim())` holds, and otherwise falls
back to `HcCombine` (f32 out) plus a separate `quantize_q8_1`. The trace shows
the fallback.

But note the rate: `HcCombineKernel` fires **6,147 times** (96 per token) and
only 19 are followed by a stall above 1 ms - **0.3%**. So this is not a
systematic rejection of the fused path; it is 22 rare stalls, about 0.3 per
token. Whether they are the unfused fallback firing at a particular shape, or
something unrelated on the host that merely happens to land there, has not
been established.

**Sizing:** worth at most 3.5% of decode, and only if every one of those 22
stalls is attributable and removable. Far smaller than the dense GEMV question
and far smaller than the fusion idea looked before arithmetic killed it. Do not
start here.

- **First: which launch produces gridX in the hundreds of thousands.** Until
  that is known, N, the bandwidth estimate and therefore whether dense GEMV is
  optimizable at all are all undetermined.
- After that, whether `mul_mat_vec_q` coalesces well at M=1.

## Open questions, in priority order

1. **Which launch produces gridX in the hundreds of thousands.** Match the
   trace's `mul_mat_vec_q8<1, false, 1, false, false>` against the call sites
   at `mmvq.hip.cpp:648-651` and the batched paths at `:661-664`. Until this
   is answered, N, the bandwidth estimate and whether 55.5% of decode is
   optimizable at all are undetermined. **This gates everything.**
2. **The concurrency / occupancy contradiction.** One stream measures 87.7%
   GPU-busy, which caps aggregate concurrency gain at about 1.14x, yet 4
   streams were previously measured at 2.2x against an ideal 4x. Either the
   earlier run was in a different regime (short prompts, cold cache, more
   prefill) or one figure is being read wrong. Resolve by tracing 4 streams at
   steady-state decode with a warm cache.
3. **The 22 rare stalls after `HcCombineKernel`.** At most 3.5% of decode, and
   only if each is attributable. Lowest priority.

## Files

- `tools/prof/decode_trace.py` - pass splitting, per-kernel rollup, `--shapes`
  and `--order`. Reads the raw rocprofv3 CSV; does not need rocpd.
- `tools/prof/isa_mix.py` - instruction mix, not yet used in this work.
- `tools/bench/build.sh` - standalone HIP experiment builds.
- `docs/models/qwen3.8-flash-next/IU4-PLAN.md` - the closed IU4 study.
