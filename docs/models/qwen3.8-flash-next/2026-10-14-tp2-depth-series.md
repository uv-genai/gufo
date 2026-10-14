# TP2 Q4 depth series — 2026-10-14

Post-merge (`uv-rdma` + upstream v0.15.0 / PR #571) measurement. Two Strix
Halos over USB4 Thunderbolt, compared with the same binary on one host.

- **Binary** `build/release/gufo` sha256 `09c2f34b552d307c21adda6067516331f00b18307ec178903788783167ad102a`
- **Model** `UD-Q4_K_XL` + shared-Q8_0 MTP, greedy, thinking off, MTP on
- **Method** HTTP `pp2048 / tg128`; depth is the cold prompt length minus the
  measured prefill window (prompts are `depth+2048` mixed or repetitive text)
- **Link** 2× USB4 (10 Gb/s × 2 lanes each), `native_write_striping=Y`,
  `ib_write_bw` 1 MiB ≈ **37 Gb/s** on both HCAs
- **TP2** `tools/tp2/rank0-q4.sh` + `rank1-q4.sh`, `--sessions 4`
- **One host** `tools/tp2/serve-q4.sh` with `CONTEXT=262144`

Raw logs: `artifacts/2026-10-14-tp2-depth-series/depth-tp2.txt`,
`depth-onehost.txt`, `bench-q4-rdma.txt`.

## Depth series (Q4 MTP)

| depth | kind | 1-host pp | TP2 pp | Δ pp | 1-host tg | TP2 tg | Δ tg |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | mixed | 1470.8 | 1449.5 | −1% | 38.2 | 49.3 | +29% |
| 0 | repetitive | 1666.0 | 1569.0 | −6% | 57.5 | 71.4 | +24% |
| 4,096 | mixed | 1546.6 | 1691.7 | +9% | 34.0 | 50.5 | +50% |
| 4,096 | repetitive | 1459.8 | 1560.5 | +7% | 38.6 | 59.8 | +55% |
| 8,192 | mixed | 1525.8 | 1765.9 | +16% | 36.4 | 56.5 | +58% |
| 8,192 | repetitive | 1331.8 | 1498.8 | +12% | 55.1 | 62.1 | +13% |
| 16,384 | mixed | 1544.6 | 1840.6 | +19% | 36.2 | 37.6 | +5% |
| 16,384 | repetitive | 1468.3 | 1734.1 | +18% | 36.3 | 61.1 | +69% |
| 32,768 | mixed | 1512.9 | 1813.7 | +20% | 38.9 | 48.0 | +23% |
| 32,768 | repetitive | 1469.3 | 1811.7 | +23% | 54.8 | 53.0 | −4% |
| 65,536 | mixed | 1494.6 | 1859.8 | +24% | 41.0 | 49.2 | +20% |
| 65,536 | repetitive | 1451.9 | 1811.5 | +25% | 41.5 | 78.8 | +88% |
| 131,072 | mixed | 1448.4 | 1782.2 | +23% | 33.1 | 46.3 | +39% |
| 131,072 | repetitive | 1370.7 | 1654.9 | +21% | 57.5 | 60.2 | +5% |

From depth 4K, TP2 prefill is faster than one host (~+7–25%). MTP decode is
usually faster on TP2 (mixed +5–50%, repetitive +5–88% except one −4% row).

## Earlier TP2 block (same day, pre leak-fix binary)

Cold mixed/repetitive prompts, MTP, pairing off (`PairLead=0`), pair restarted
per block. Superseded by the series above but kept for the link numbers.

| workload | pp tok/s | tg tok/s | draft acc |
| --- | ---: | ---: | ---: |
| mixed ~2K | 1103 | 79 | 93% |
| mixed ~6K | 1284 | 62 | 75% |
| mixed ~10K | 1328 | 62 | 75% |
| rep ~8K | 1342 | 57 | 70% |
| deep 10K | 1355 | 64 | 75% |
| multi-user 1/2/4 (sum tg) | — | 74 / 81 / 126 | — |
| short-prompt TTFT | — | 164–166 ms | — |

## Notes

- Not a `BENCHMARKS.md` publish pin: single rep per point, cold long prompts
  rather than warm cached-prefix depth, MTP on for both sides.
- TP2 stability fix on this binary: mirrored snapshots (no unmirrored
  component-cache restore), `PrefillThrough` == `Prefill` on rank 0,
  `in_pass_checkpoint` off when distributed. 12 unique ~1830-token prefills
  including session reuse stayed healthy.
- `tools/tp2/` holds the launchers used for both sides.

---

# Q8 TP2 depth series — 2026-10-14 (same binary)

Full `Q8_0` target (six shards) does not fit one host; only TP2 was measured.
Same method (`pp2048 / tg128`, greedy, MTP, thinking off), same link
(2× USB4, write striping, ~37 Gb/s). Launchers: `tools/tp2/rank0-q8.sh` +
`rank1-q8.sh`, `--sessions 4` (~86 GB GPU per host at 262K context).

Raw log: `artifacts/2026-10-14-tp2-depth-series/depth-tp2-q8.txt`.

| depth | kind | pp tok/s | tg tok/s | draft acc | ttft ms |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | mixed | 1058 | 44 | 69% | 1587 |
| 0 | repetitive | 1316 | 43 | 68% | 1396 |
| 4,096 | mixed | 1375 | 38 | 55% | 3628 |
| 4,096 | repetitive | 1338 | 52 | 70% | 4114 |
| 8,192 | mixed | 1606 | 38 | 69% | 5181 |
| 8,192 | repetitive | 1249 | 58 | 70% | 4043 |
| 16,384 | mixed | 1674 | 38 | 67% | 9245 |
| 16,384 | repetitive | 1568 | 58 | 70% | 5255 |
| 32,768 | mixed | 1700 | 47 | 85% | 17163 |
| 32,768 | repetitive | 1676 | 51 | 54% | 11167 |
| 65,536 | mixed | 1670 | 38 | 66% | 33897 |
| 65,536 | repetitive | 1606 | 74 | 89% | 18312 |
| 131,072 | mixed | 1633 | 38 | 68% | 70548 |
| 131,072 | repetitive | 1520 | 48 | 89% | 38816 |

### Q8 vs Q4 on TP2 (same method)

| depth | kind | Q4 pp | Q8 pp | Q4 tg | Q8 tg |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | mixed | 1450 | 1059 | 49 | 44 |
| 4,096 | mixed | 1692 | 1375 | 51 | 38 |
| 8,192 | mixed | 1766 | 1606 | 57 | 38 |
| 16,384 | mixed | 1841 | 1674 | 38 | 38 |
| 32,768 | mixed | 1814 | 1700 | 48 | 47 |
| 65,536 | mixed | 1860 | 1670 | 49 | 38 |
| 131,072 | mixed | 1782 | 1633 | 46 | 38 |

Q8 prefill is ~8–27% slower than Q4 at the same depth; decode is comparable
on repetitive text (Q8 up to 74 tok/s at 65K) and lower on mixed. No one-host
Q8 baseline: the full model does not fit a single 128 GB Strix Halo.

---

# Concurrent decode — 2026-10-14 (same binary)

Method: N simultaneous HTTP chat completions, each ~2K-token prompt and
tg128, greedy MTP, thinking off. `sum_tg` is the sum of per-user
`completion_tokens_per_second`. Runner:
`artifacts/2026-10-14-tp2-depth-series/conc-bench.py`.

| config | users | sum tok/s | wall s | per-user tok/s |
| --- | ---: | ---: | ---: | --- |
| **one-host Q4** (8K ctx, sessions=8) | 1 | 36.5 | 3.6 | 36.5 |
| | 2 | 58.1 | 4.5 | 28.2, 29.8 |
| | 4 | 86.0 | 7.4 | 20.3–22.8 |
| | 6 | 108.0 | 8.8 | 17.1–19.3 |
| | 8 | 129.6 | 10.8 | 15.0–18.5 |
| **TP2 Q4** (262K ctx, sessions=8) | 1 | 40.1 | 3.6 | 40.1 |
| | 2 | 81.0 | 7.1 | 32.1, 48.9 |
| | 4 | 112.7 | 7.4 | 27.1–29.9 |
| | 6 | **180.7** | 10.1 | 26.8–34.6 |
| | 8 | 149.6 | 9.0 | 17.8–20.6 |
| **TP2 Q8** (262K ctx, sessions=4) | 1 | 40.6 | 3.7 | 40.6 |
| | 2 | 89.2 | 6.1 | 42.5, 46.6 |
| | 4 | 93.2 | 8.8 | 21.8–26.2 |

Raw logs: `conc-onehost-q4.txt`, `conc-tp2-q4.txt`, `conc-tp2-q8.txt`.

**Notes**
- TP2 Q4 peaks at **6 users (181 tok/s sum)**; 8 users drops to 150 as
  per-user rate falls (~18 tok/s each). One-host scales more smoothly to
  **130 tok/s at 8 users**. TP2 is ahead at every width tested.
- Q8 concurrency only 1–4 (`sessions=4` at 262K; 8 sessions needs ~103 GB
  per host). Q8 at 2 users is the best sum_tg of the day (89 tok/s).
- One-host multi-user uses 8K context (8×262K does not fit). TP2 rows are
  262K-capable sessions; prompts were the same ~2K tokens.
