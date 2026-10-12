# TP2 Q4 vs Q8 — spot check

> **Unqualified measurement.** This answers a practical question ("which
> quantisation should I use on the TP2 pair?"). It is **not** a qualified
> benchmark and is deliberately **not** part of [BENCHMARKS.md](BENCHMARKS.md)
> or the published one-host tables: one session per variant, no repetition
> control across sessions, no matched single-host baseline, no reference engine.
> Treat the decode column as ±10% noise — note the spread printed per cell.
> Promote these numbers only after a real qualification run (see
> [QUALITY.md](QUALITY.md)).

Measured on the two-host Strix Halo pair on 2026-10-12, on the **upstream
0.13.0** base (merge `f3feffbd`, tag
`research/tp2-rdma-0.13.0-2026-10-12`).

## Environment

| | |
|---|---|
| Hosts | `strix-halo-1` (rank 0) + `strix-halo-2` (rank 1) |
| CPU / GPU | AMD Ryzen AI Max+ 395, Radeon 8060S (`gfx1151`) |
| Memory | 128 GB LPDDR5X unified each |
| OS / kernel | CachyOS, `7.2.8-2-cachyos` |
| ROCm / rdma-core | 7.2.3 / 65 |
| Build | `release` + `-DGUFO_ENABLE_TP2_RDMA=ON`, identical binaries both hosts (`md5 2471d4f5…`) |
| Transport | `usb4_rdma0`, RoCE v2, `gid_index=1`, MTU 4096 |
| Link | 2 × USB4 cables, `native_write_striping=1`, rails on `path=1` + `path=257`, 10 Gb/s × 2 lanes each |
| Serving | `--sessions 4`, `--speculative mtp`, context 262144, greedy (`temperature 0`) |

Both quantisations share the build; only the model differs: `Q8_0` (6 shards)
and `UD-Q4_K_XL` (4 shards), each with the shared-Q8_0 MTP sidecar and
`mmproj-BF16.gguf`.

## Method

- **5 recorded requests per cell** after **1 warm-up** request at the same depth.
- Prompt content is **derived per request from a unique salt**, so no request can
  hit a previous one's prompt cache. Every cell reports `cached=0`; the harness
  prints a loud warning if that is ever non-zero.
- Rates and TTFT come from the server's own per-request metrics
  (`prompt_tokens_per_second`, `completion_tokens_per_second`,
  `gufo.ttft_ms`), not client wall-clock. The server's reported
  `prompt_tokens` is authoritative.
- Cells are medians of 5, with min–max decode spread shown.
- The concurrency row runs 4 simultaneous requests at the medium depth and
  reports aggregate decode tokens over wall-clock.

## Results — upstream 0.13.0 base

| Depth | Metric | **Q8_0** | **Q4_K_XL** | Q4/Q8 |
|---|---|---|---|---|
| **short** (260 tok) | prefill | 498.5 tok/s | 627.6 tok/s | **+1.26x** |
| | decode | 47.3 [44.0–49.8] | 50.5 [49.3–54.7] | +1.07x |
| | TTFT | 524.1 ms | 419.5 ms | 0.80x |
| **medium** (2,133 tok) | prefill | 1,412.5 tok/s | 1,649.5 tok/s | **+1.17x** |
| | decode | 44.9 [40.9–47.5] | 49.3 [40.7–52.2] | +1.10x |
| | TTFT | 1,517.3 ms | 1,291.2 ms | 0.85x |
| **long** (8,345 tok) | prefill | 1,367.5 tok/s | 1,645.8 tok/s | **+1.20x** |
| | decode | 48.0 [31.7–52.4] | 56.4 [44.7–58.5] | +1.18x |
| | TTFT | 6,117.2 ms | 5,101.7 ms | 0.83x |
| **4-way concurrent** | aggregate decode | 50.3 tok/s | 58.3 tok/s | **+1.16x** |

### Memory (from `event=load_completed`)

| | `gpu_device_used_mib` | fits one host? |
|---|---|---|
| Q8_0 | 85,629 | no — needs both hosts |
| Q4_K_XL | 61,222 | yes |

Load times: 23.3 s (Q8) and 17.9 s (Q4).

## Observations

- **Q4 prefill is 17–26% faster at every depth.** Prefill is weight-bandwidth
  bound, so halving the weight bytes helps directly.
- **Q4 decode is consistently ahead** (+7% to +18%), though the long-depth
  spread for Q8 (31.7–52.4) is wide — that cell is genuinely noisy and the
  +1.18x there is weaker evidence than the short/medium rows.
- **TTFT 15–20% lower for Q4** — same prefill-bandwidth story.
- **Q4 scales better under concurrency** (+16% aggregate at 4 streams).
- **Prefill rate rises with depth** (498 → 1,413 → 1,368 tok/s): fixed per-chunk
  launch overhead amortises on larger prompts. Compare depths, not absolutes.
- **The decision is really memory, not the ~20%**: Q8 needs both hosts and so
  buys the full-precision model at 262k context where a single host cannot hold
  it. Q4 would fit one host, so TP2 there is a speed choice.

## Methodology correction (why earlier numbers here were wrong)

This document previously carried a table measured with **2 repetitions and no
warm-up, on identical prompt text**. Those numbers were **contaminated and must
not be used**:

- With identical prompt text, repetition 2 hit gufo's prompt cache and reported
  `prefill = 0`. A median of `[real, 0]` halves the true rate.
- A warm-up request made it worse: all subsequent identical requests reported
  `prefill = 0.0` and `ttft ≈ 4 ms`.
- The old prompt filler was repetitive, so even a salt that only differed at the
  start still let long prompts match a shared substring.

The corrected harness therefore (a) warms up with a prompt that is never timed,
(b) gives **every** request prompt text derived from a unique salt so no two
share a prefix or long substring, and (c) reports `cached_tokens` per cell and
fails loudly if it is ever non-zero.

**Consequence:** the old table's depths were also different (578 / 5,408 /
21,578 tok) from these (260 / 2,133 / 8,345 tok), so it is not comparable
row-for-row either. The old tag
`research/tp2-rdma-merge-2026-10-12` marks the 0.12.0 base but **carries no
trustworthy numbers**; re-measure it if a 0.12.0 comparison is ever needed.

This is worth keeping: a prompt-cache hit is silent, and it looks exactly like a
2x prefill regression when it is really a measurement artifact.

## Findings

This benchmarking effort found a real defect in the TP2 forward-port that no
unit test caught.

The first two-host run returned **HTTP 500 `Qwen3.8-Flash-Next prefill failed:
SSM output F16 GEMM failed`** for every prompt ≥1024 tokens, on **both**
quantisations.

**Cause.** While resolving the `DenseF16GEMMKernel` parameter collision in the
forward-port, main's `half_stride` strided path was kept unchanged. Its guard is
hardcoded to `k == 6144`, the single-host full width. Under TP2 a rank's
`l.ssm_out.cols` is `3072`, so at `n_tokens >= 1024` the stride was enabled,
the guard rejected the shape, and prefill failed. Prompts below 1024 tokens
never take that path, which is why the defect looked absent.

**Fix.** `69ef6ae2` — scope the strided path to the width it was tuned for
(`cols == 6144`); a TP rank's half keeps the plain path, which is what the TP2
split was qualified with. Single-host behaviour is unchanged.

Worth recording because it is a pure merge artifact: it compiled, all GPU ops
tests passed (they exercise single-host geometry), and it only appeared under a
real two-host run at ≥1024 tokens.

## Raw log (0.13.0, corrected method)

```
=== variant q8 (gufo-qwen3.8-flash-next-tp2-012) ===
  pair ready
    short   prompt=   260 tok  prefill=  498.5 tok/s  decode= 47.3 [44.0-49.8]  ttft=  524.1 ms  cached=0  n=5
    medium  prompt=  2133 tok  prefill= 1412.5 tok/s  decode= 44.9 [40.9-47.5]  ttft= 1517.3 ms  cached=0  n=5
    long    prompt=  8345 tok  prefill= 1367.5 tok/s  decode= 48.0 [31.7-52.4]  ttft= 6117.2 ms  cached=0  n=5
    concurrency=4  4 ok  aggregate decode= 50.3 tok/s  wall=20.3s

=== variant q4 (gufo-qwen3.8-flash-next-tp2-012-q4) ===
  pair ready
    short   prompt=   260 tok  prefill=  627.6 tok/s  decode= 50.5 [49.3-54.7]  ttft=  419.5 ms  cached=0  n=5
    medium  prompt=  2133 tok  prefill= 1649.5 tok/s  decode= 49.3 [40.7-52.2]  ttft= 1291.2 ms  cached=0  n=5
    long    prompt=  8345 tok  prefill= 1645.8 tok/s  decode= 56.4 [44.7-58.5]  ttft= 5101.7 ms  cached=0  n=5
    concurrency=4  4 ok  aggregate decode= 58.3 tok/s  wall=17.6s
```

## Reproduce

[`tools/bench/tp2_q4_q8_bench.py`](../../../tools/bench/tp2_q4_q8_bench.py).
Rank 0 must be the local host and rank 1 reachable over ssh. Never binds 8080.

```sh
python3 tools/bench/tp2_q4_q8_bench.py
```

Tune `REPS`, `WARMUP` and `DEPTHS` at the top of the file. For a *qualified*
comparison add a matched single-host run, take medians across sessions, and
report per-request timings at 5% and 3 ms rather than medians — see
[QUALITY.md](QUALITY.md).

## Confirming RDMA is actually carrying traffic

gufo's TP2 has no TCP data path — the bootstrap socket is kept only for
peer-loss detection — so a running pair is using RDMA by construction. To see it
directly:

```sh
cat /sys/class/infiniband/usb4_rdma0/ports/1/hw_counters/{tx_bytes,rx_bytes}
# ...send a request...
cat /sys/class/infiniband/usb4_rdma0/ports/1/hw_counters/{tx_bytes,rx_bytes}

sudo cat /sys/kernel/debug/thunderbolt_ibverbs/summary | grep -E 'striping|error|lost'
```

One 220-token request moved ~279 MB in each direction; a mixed smoke session
moved 3,187 MiB. `native_write_striping: 1` with two rails on `path=1` and
`path=257` confirms both USB4 cables are striped.
