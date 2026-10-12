# TP2 Q4 vs Q8 — spot check

> **Unqualified measurement.** This is an ad-hoc comparison run to answer a
> practical question ("which quantisation should I use on the TP2 pair?").
> It is **not** a qualified benchmark and is deliberately **not** part of
> [BENCHMARKS.md](BENCHMARKS.md) or the published one-host tables. Two
> repetitions per cell, one session, no warm-up control, no repetition
> control, no matched single-host baseline, no reference engine. Treat the
> decode column as ±5% noise: the 31.6 vs 42.4 tok/s spread across depths is
> measurement variance, not a depth trend. Promote these numbers only after a
> real qualification run (see [QUALITY.md](QUALITY.md)).

Run 2026-10-12 on the two-host Strix Halo pair, on the merge commit
`58272777` plus `69ef6ae2` (see [Findings](#findings)).

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

Both quantisations are served by the same build; only the model differs:
`Q8_0` (6 shards) and `UD-Q4_K_XL` (4 shards), both with the shared-Q8_0 MTP
sidecar and `mmproj-BF16.gguf`.

## Method

Prompt depths are approximations; the server's reported `prompt_tokens` is the
authoritative value used for every rate. Rates and TTFT are taken from the
server's own per-request metrics (`prompt_tokens_per_second`,
`completion_tokens_per_second`, `gufo.ttft_ms`), not client wall-clock.
Cells are medians of 2 repetitions.

The concurrency row runs 4 simultaneous requests at the medium depth and
reports aggregate decode tokens over wall-clock.

## Results

| Depth | Metric | **Q8_0** | **Q4_K_XL** | Q4/Q8 |
|---|---|---|---|---|
| **short** (578 tok) | prefill | 278.9 tok/s | 343.9 tok/s | +1.23x |
| | decode | 34.9 tok/s | 31.6 tok/s | 0.91x |
| | TTFT | 523.9 ms | 425.2 ms | 0.81x |
| **medium** (5,408 tok) | prefill | 609.8 tok/s | 739.6 tok/s | +1.21x |
| | decode | 30.0 tok/s | 34.8 tok/s | +1.16x |
| | TTFT | 2,228.8 ms | 1,838.7 ms | 0.82x |
| **long** (21,578 tok) | prefill | 747.5 tok/s | 885.1 tok/s | +1.18x |
| | decode | 37.6 tok/s | 42.4 tok/s | +1.13x |
| | TTFT | 5,864.1 ms | 4,956.2 ms | 0.85x |
| **4-way concurrent** | aggregate decode | 80.8 tok/s | 94.5 tok/s | +1.17x |

### Memory (from `event=load_completed`)

| | `gpu_device_used_mib` | fits one host? |
|---|---|---|
| Q8_0 | 85,629 | no — needs both hosts |
| Q4_K_XL | 61,222 | yes |

Load times were 30.5 s (Q8) and 23.0 s (Q4).

## Observations

- **Q4 prefill is 18–23% faster at every depth.** Prefill is weight-bandwidth
  bound, so halving the weight bytes helps directly.
- **Decode is parity-to-slightly-better for Q4** and is mostly link-*latency*
  bound, so the weight saving matters less there.
- **TTFT 15–19% lower for Q4** — same prefill-bandwidth story.
- **Q4 scales better under concurrency** (+17% aggregate at 4 streams).
- **Prefill rate rises with depth** (279 → 610 → 748 tok/s): fixed per-chunk
  launch overhead amortises on larger prompts. Compare depths, not absolutes.
- **The decision is really memory, not the ~20%**: Q8 needs both hosts and so
  buys the full-precision model at 262k context where a single host cannot hold
  it. Q4 would fit one host, so TP2 there is a speed choice.

## Findings

This run found a real defect in the TP2 forward-port that no unit test caught.

The first pass returned **HTTP 500 `Qwen3.8-Flash-Next prefill failed: SSM
output F16 GEMM failed`** for every prompt ≥1024 tokens, on **both**
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

This is worth recording because it is a pure merge artifact: it compiled, all
GPU ops tests passed (they exercise single-host geometry), and it only appeared
under a real two-host run at ≥1024 tokens.

## Raw log

```
=== variant q8 (gufo-qwen3.8-flash-next-tp2-012) ===
  starting q8 pair on :8318 ...
  pair ready
    short   prompt=   578 tok  prefill=  278.9 tok/s  decode= 34.9 tok/s  ttft=  523.9 ms
    medium  prompt=  5408 tok  prefill=  609.8 tok/s  decode= 30.0 tok/s  ttft= 2228.8 ms
    long    prompt= 21578 tok  prefill=  747.5 tok/s  decode= 37.6 tok/s  ttft= 5864.1 ms
    concurrency=4  4 ok  aggregate decode= 80.8 tok/s  wall=12.7s

=== variant q4 (gufo-qwen3.8-flash-next-tp2-012-q4) ===
  starting q4 pair on :8318 ...
  pair ready
    short   prompt=   578 tok  prefill=  343.9 tok/s  decode= 31.6 tok/s  ttft=  425.2 ms
    medium  prompt=  5408 tok  prefill=  739.6 tok/s  decode= 34.8 tok/s  ttft= 1838.7 ms
    long    prompt= 21578 tok  prefill=  885.1 tok/s  decode= 42.4 tok/s  ttft= 4956.2 ms
    concurrency=4  4 ok  aggregate decode= 94.5 tok/s  wall=10.8s
```

## Reproduce

The harness is [`tools/bench/tp2_q4_q8_bench.py`](../../../tools/bench/tp2_q4_q8_bench.py).
It drives `~/opt/llama.cpp-cli/tp2-merged-swap.sh`, so rank 0 must be the local
host and rank 1 reachable over ssh. It never binds port 8080.

```sh
python3 tools/bench/tp2_q4_q8_bench.py
```

For a *qualified* comparison, raise `REPS` to at least 5, add a warm-up request
per depth, take medians across sessions, and report per-request timings at 5%
and 3 ms rather than medians — see [QUALITY.md](QUALITY.md).

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

One 220-token request moved ~279 MB in each direction across the link.
`native_write_striping: 1` with two rails on `path=1` and `path=257` confirms
both USB4 cables are striped.
