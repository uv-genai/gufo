# Qwen3.8 Flash-Next benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Unsloth `UD-Q4_K_XL` target and
shared-Q8_0 MTP sidecar; Gufo uses adaptive MTP. HTTP, greedy, thinking off.
Positive gain favors Gufo.
[Quality and measurement details](QUALITY.md#benchmark-method) · [Model identities](artifacts/model-identities.json)

- [Single Strix Halo](#single-strix-halo): Gufo against llama.cpp.
- [Two Strix Halos over RDMA](#two-strix-halos-over-rdma): Gufo RDMA against
  Gufo on one host, deeper contexts, and the full Q8 model.

## Single Strix Halo

Gufo measured October 16, 2026, at `fdc21849`, after merged #603 and the cache
follow-up. All six Gufo sections use the refreshed HTTP captures.
Reference results remain unchanged from September 22–23.
llama.cpp uses `b11069` for AR and `6fcaa16f` for MTP.

Gufo numerical regression checks pass; individual functional timing limits
remain documented in [QUALITY.md](QUALITY.md). Historical llama.cpp MTP values are retained as requested:
a separate check of that pin did not match its own AR output, so this review
does not requalify those comparisons. [Review evidence](artifacts/decode-kernels-review.json).

At 128K, AR pp improves 7.5% over post-#603 main, remaining 2.2% below the
older 1529.63 tok/s result. MTP pp is within 0.9% of the historical 64K/128K
results; 128K mixed/repetitive generation retains #603's gains.

### Single user, autoregressive

Approximately pp2048 / tg128; depth is the cached prefix in tokens.
Context capacities differ between engines; see the measurement details.

<!-- bench:single-ar -->
| Flash-Next Q4 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1737.24 | 489.59 | +254.8% | 28.28 | 22.20 | +27.4% |
| 4,096 | 1673.56 | 455.24 | +267.6% | 28.28 | 21.21 | +33.3% |
| 8,192 | 1667.16 | 428.85 | +288.8% | 28.26 | 20.40 | +38.5% |
| 12,288 | 1648.13 | 400.21 | +311.8% | 28.24 | 19.64 | +43.8% |
| 16,384 | 1649.96 | 375.89 | +338.9% | 28.20 | 18.95 | +48.8% |
| 32,768 | 1624.38 | 301.31 | +439.1% | 28.05 | 16.54 | +69.6% |
| 65,536 | 1554.57 | 221.94 | +600.4% | 27.84 | 11.62 | +139.6% |
| 131,072 | 1496.49 | 144.78 | +933.6% | 27.30 | 7.98 | +242.1% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar.svg)

### Single user, MTP

pp is the highest measured rate per engine and depth across mixed/repetitive
text, including Gufo predictor catch-up.

<!-- bench:single-mtp -->
| Flash-Next Q4 MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1722.87 | 468.76 | +267.5% | 35.73 | 31.73 | +12.6% | 73.45 | 48.12 | +52.6% |
| 4,096 | 1655.82 | 423.79 | +290.7% | 37.61 | 34.48 | +9.1% | 70.39 | 45.71 | +54.0% |
| 8,192 | 1652.67 | 394.98 | +318.4% | 36.82 | 32.20 | +14.3% | 58.02 | 44.31 | +30.9% |
| 12,288 | 1634.94 | 365.91 | +346.8% | 39.14 | 32.32 | +21.1% | 48.47 | 43.53 | +11.3% |
| 16,384 | 1624.24 | 341.98 | +375.0% | 40.45 | 32.07 | +26.1% | 59.88 | 43.66 | +37.2% |
| 32,768 | 1603.57 | 277.61 | +477.6% | 39.65 | 25.41 | +56.0% | 52.36 | 39.28 | +33.3% |
| 65,536 | 1547.21 | 208.04 | +643.7% | 38.68 | 19.65 | +96.8% | 52.26 | 27.75 | +88.3% |
| 131,072 | 1484.85 | 135.42 | +996.5% | 38.64 | 14.63 | +164.1% | 52.68 | 20.96 | +151.3% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp.svg)

### Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
All sessions prefilled before timed decoding; throughput sums individual rates.
llama.cpp re-evaluates its four-token checkpoint tail.

<!-- bench:multi-ar -->
| Flash-Next Q4 AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 28.22 | 22.34 | +26.3% |
| 2 | 50.67 | 37.08 | +36.7% |
| 4 | 85.78 | 54.82 | +56.5% |
| 6 | 107.98 | 65.34 | +65.3% |
| 8 | 125.14 | 68.79 | +81.9% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar.svg)

### Multiple users, MTP

Same pp2048 mixed/repetitive prompts as single-user d0, tg128, context 4096
per user. All sessions prefilled before timed decoding; rates sum individual
request decode rates. C1 cross-checks the single-user table.

<!-- bench:multi-mtp -->
| Flash-Next Q4 MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 35.62 | 31.77 | +12.1% | 73.19 | 46.93 | +56.0% |
| 2 | 59.65 | 42.79 | +39.4% | 97.43 | 52.57 | +85.3% |
| 4 | 86.53 | 48.78 | +77.4% | 157.58 | 48.23 | +226.7% |
| 6 | 108.66 | 52.59 | +106.6% | 176.47 | 48.51 | +263.8% |
| 8 | 125.42 | 61.92 | +102.6% | 194.37 | 58.13 | +234.4% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp.svg)

### Loading time

C1, context capacity 262144, MTP. Cold target/sidecar files to HTTP readiness.

<!-- bench:loading -->
| Flash-Next Q4<br>Target | Gufo ready (s) | llama.cpp ready (s) | Gain |
| --- | ---: | ---: | ---: |
| Q4 | 15.27 | 117.83 | +671.6% |
<!-- /bench -->

![Loading time](artifacts/charts/loading.svg)

### Memory occupation

C1, context capacity 133121, AR. Peak memory reported by HIP.

<!-- bench:memory -->
| Flash-Next Q4 AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 88.75 | 84.84 | -4.4% |
| 16K prefix, pp4096 + tg128 | 90.73 | 85.31 | -6.0% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory.svg)

## Two Strix Halos over RDMA

Gufo RDMA runs one model on two such hosts: every layer is split between them,
and they exchange partial sums over RDMA ([how it works](TP2.md)).
The workloads are those of the one-host tables; "Gufo" is the same build on
one host, and gain is Gufo RDMA over it. With twice the memory, single users
also reach depths one host cannot (context capacity 262144). Every RDMA table
was measured on October 9, 2026 (`c85349b5`) over USB4 RoCE v2, with the
same-build one-host runs beside them, in the balanced power mode (85 W
sustained), so prefill is lower than in the one-host tables above; decode is
not power-bound. Over USB4 a fresh 2K prompt prefills more slowly than on one
host; with a cached prefix, prefill gains 3–15%.

### Single user, autoregressive

<!-- bench:single-ar-tp2 -->
| Flash-Next Q4 AR<br>Depth (tokens) | Gufo pp (tok/s) | Gufo RDMA pp (tok/s) | Gain | Gufo tg (tok/s) | Gufo RDMA tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1492.22 | 1299.73 | -12.9% | 26.71 | 32.91 | +23.2% |
| 4,096 | 1342.09 | 1412.63 | +5.3% | 26.69 | 32.81 | +22.9% |
| 32,768 | 1270.29 | 1456.89 | +14.7% | 26.08 | 32.53 | +24.7% |
| 65,536 | 1255.97 | 1431.71 | +14.0% | 26.23 | 32.23 | +22.9% |
| 131,072 | 1216.31 | 1317.83 | +8.3% | 25.71 | 31.31 | +21.8% |
| 258,048 | — | 1221.33 | — | — | 30.10 | — |
<!-- /bench -->

![Single user, autoregressive, one host and RDMA](artifacts/charts/single-ar-tp2.svg)

### Single user, MTP

<!-- bench:single-mtp-tp2 -->
| Flash-Next Q4 MTP<br>Depth (tokens) | Gufo pp (tok/s) | Gufo RDMA pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | Gufo RDMA tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | Gufo RDMA tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1420.72 | 1346.32 | -5.2% | 32.03 | 37.88 | +18.3% | 60.11 | 79.33 | +32.0% |
| 4,096 | 1301.51 | 1333.77 | +2.5% | 33.25 | 47.46 | +42.7% | 56.03 | 60.82 | +8.5% |
| 32,768 | 1231.99 | 1389.59 | +12.8% | 33.11 | 38.62 | +16.6% | 40.61 | 65.63 | +61.6% |
| 65,536 | 1216.20 | 1314.17 | +8.1% | 33.78 | 47.55 | +40.8% | 43.33 | 57.69 | +33.1% |
| 131,072 | 1208.61 | 1275.75 | +5.6% | 34.21 | 41.46 | +21.2% | 39.90 | 55.62 | +39.4% |
| 258,048 | — | 1169.31 | — | — | 38.02 | — | — | 57.38 | — |
<!-- /bench -->

![Single user, MTP, one host and RDMA](artifacts/charts/single-mtp-tp2.svg)

### Multiple users, autoregressive

Same pp2048 prose prompt, tg128, context 4096 per user; every session
prefills before timed decoding. Rates sum individual decode rates.

<!-- bench:multi-ar-tp2 -->
| Flash-Next Q4 AR<br>Users | Gufo AR (tok/s) | Gufo RDMA AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 25.09 | 32.89 | +31.1% |
| 2 | 46.95 | 56.77 | +20.9% |
| 4 | 77.27 | 92.79 | +20.1% |
| 6 | 96.79 | 115.69 | +19.5% |
| 8 | 109.03 | 137.86 | +26.4% |
<!-- /bench -->

![Multiple users, autoregressive, one host and RDMA](artifacts/charts/multi-ar-tp2.svg)

### Multiple users, MTP

Same configuration and preparation as the AR table, with mixed/repetitive
prompts. Every prepared completion, on both topologies and in all modes, matches
its fresh one-user AR control.

<!-- bench:multi-mtp-tp2 -->
| Flash-Next Q4 MTP<br>Users | Gufo mixed (tok/s) | Gufo RDMA mixed (tok/s) | Gain | Gufo repetitive (tok/s) | Gufo RDMA repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32.04 | 37.75 | +17.8% | 60.16 | 74.16 | +23.3% |
| 2 | 53.57 | 66.98 | +25.0% | 84.28 | 113.63 | +34.8% |
| 4 | 77.13 | 92.34 | +19.7% | 124.27 | 160.23 | +28.9% |
| 6 | 96.19 | 108.78 | +13.1% | 134.85 | 176.43 | +30.8% |
| 8 | 106.50 | 127.06 | +19.3% | 149.87 | 191.36 | +27.7% |
<!-- /bench -->

![Multiple users, MTP, one host and RDMA](artifacts/charts/multi-mtp-tp2.svg)

### Full Q8 model

The full Q8_0 target does not fit one host. Single user, pp2048/tg128 by depth
as above; prefill on the left axis, generation on the right.

<!-- bench:tp2-q8 -->
| Flash-Next Q8 RDMA<br>Depth (tokens) | pp (tok/s) | tg AR (tok/s) | tg MTP mixed (tok/s) | tg MTP repetitive (tok/s) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1193.06 | 29.80 | 41.15 | 65.99 |
| 4,096 | 1169.94 | 29.52 | 37.83 | 55.31 |
| 32,768 | 1176.27 | 29.64 | 42.44 | 60.15 |
| 65,536 | 1140.63 | 28.91 | 44.19 | 52.40 |
| 131,072 | 1120.59 | 28.50 | 36.88 | 56.22 |
| 258,048 | 1041.67 | 27.60 | 39.34 | 50.64 |
<!-- /bench -->

![Flash-Next Q8 RDMA](artifacts/charts/tp2-q8.svg)

Multiple users with the configuration and preparation of the Q4 tables:

<!-- bench:tp2-q8-multi -->
| Flash-Next Q8 RDMA<br>Users | AR (tok/s) | MTP mixed (tok/s) | MTP repetitive (tok/s) |
| ---: | ---: | ---: | ---: |
| 1 | 29.97 | 41.33 | 65.99 |
| 2 | 52.82 | 66.48 | 80.25 |
| 4 | 87.52 | 105.45 | 115.17 |
| 6 | 110.17 | 92.40 | 124.27 |
| 8 | 131.79 | 117.69 | 146.72 |
<!-- /bench -->

![Flash-Next Q8 RDMA, multiple users](artifacts/charts/tp2-q8-multi.svg)

Q8 MTP rates vary more between server starts than between cohorts of one
start: a second start with three cohorts per point gave mixed 90.06 (C4) and
105.40 (C6), repetitive 115.74 (C4) and 118.65 (C6) tok/s, each cohort within
3% of its point's rate.
