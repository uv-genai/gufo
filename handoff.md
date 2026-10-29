# LIVE HANDOFF — gufo TP2 fork

**Canonical path:** `handoff.md` in **this repository** (`/home/ugo/projects/gufo/handoff.md`).
**New sessions read this first** (also see `~/gufo-merge-handoff/handoff.md` only as a
redirect until that copy is removed).
**When state changes** (merge, tag, push, script/config change, incident): **replace
or edit this file in full** — do not leave stale tips, shas, or “untested” claims.

Last updated: **2026-10-28** (session: GLM-5.3-Flash **GPU inference in progress**; quantized GEMM working, output head fix needed).

---

## 0. Read next (in order)

1. **This file** (`handoff.md` at repo root)
2. Repo `AGENTS.md` (build, remotes, TP2 hazards, **nix count must be 0**)
3. `docs/MERGE-CHECKLIST.md` (merge + TP2 runtime process)
4. `docs/models/qwen3.8-flash-next/TP2.md` (collective rules)
5. `docs/MERGE-UPSTREAM-2026-10-24.md` (v0.16→v0.23 series write-up)

Historical dated handoffs in `~/gufo-merge-handoff/HANDOFF-*.md` are **archive only**.

---

## 1. Repo & remotes

| | |
| --- | --- |
| Tree | `/home/ugo/projects/gufo` |
| Branch | **`main`** |
| Push | **`uv-genai` only** → https://github.com/uv-genai/gufo.git |
| Upstream | `origin` → gufo-org/gufo; local track **`gufo-org`** — **never push** |
| Tip | **`3ec5fe1f`** (`main` == `uv-genai/main` after last push) |
| Upstream merged through | **v0.25.0** (`16e8ed47` release — latest at last fetch) |

**Non-negotiables:** `grep -ci nix AGENTS.md` → **0**; never run that package
manager; Conventional Commits; ask before any GPU launch.

---

## 2. Binary & tested configuration

| | |
| --- | --- |
| Path | `build/release/gufo` on **both** hosts |
| sha256 | **`3d6457e93d56f41a6dd0cd2d27eddf7018e31d46a7c9c2368c99875486eac46f`** |
| Built from | `46eeaf28` era (v0.25.0; later commits are docs-only) |
| Configure | three host flags in `AGENTS.md` (HIP compiler + `-include format`) |

**TP2 serve defaults** (all four `tools/tp2/rank{0,1}-{q4,q8}.sh`):

```text
--sessions 2
--context 524288
--enable-yarn
```

- YaRN factor = **524288 / 262144 = 2.0** (`event=yarn factor=2.000000` at load).
- **Both ranks must pass identical `--context` / `--enable-yarn` / `--sessions`** or
  handshake fails (`context X here, Y on the peer`).
- **Scripts live in a separate checkout on each host.** After editing
  `tools/tp2/*.sh` on this machine, **scp or git-pull them onto `strix-halo-1`**
  (rank0 runs from strix-halo-1’s tree). Miss this → config mismatch.

---

## 3. Hosts & launch

| Role | Host | Notes |
| --- | --- | --- |
| rank0 (HTTP) | **strix-halo-1** `192.168.1.22` | llama-swap + `tools/llama-swap/tp2.sh` |
| rank1 (worker) | **strix-halo-2** (this box) | started by ssh from rank0 |

- Bootstrap `18515` / control `18516` / token `92ffa404804d336980d95f224d6bd678`
- Q8 llama-swap model id: **`gufo-qwen3.8-flash-next-tp2-q8`** → port **5809** (not 8080)
- Start: `tools/llama-swap/tp2.sh run PORT q4|q8` on strix-halo-1 (or swap `run`)
- Stop: `tools/llama-swap/tp2.sh stop`
- Logs: `~/gufo-tp2-swap.log` (rank0), `~/tp2-rank1-swap.log` (rank1), `~/tp2-swap.log`
- **One `gufo` per GPU host.** Same sha both ranks before interpreting any hang.

---

## 4. TP2 gates (must stay true)

| Gate | Where |
| --- | --- |
| `lookup_ok = TpWorldSize()==1` | `engine.cpp` (`lookup_cap`) |
| `subset_ok = allow_subset && TpWorldSize()==1` | `engine.cpp` (`draft_subset`) |
| Forced draft width from rank 0 | `request.drafts` / `batch_drafts` |
| `defer_residual_` only if `tp_world_size()==1` | `executor.cpp` |
| `UsesComponentCache()` **false** | `tp_executor.cpp` |
| Wide-prefill fused paths need **`n ≥ 1024`** under TP2 half-width | `executor.cpp` / `kernels.hip.cpp` |
| YaRN fused attention: **group-4 AND group-2** (26 tiles on TP2) | `kernels.hip.cpp` `AttentionF16Gemm` |

Prompt lookup type: **`speculative::PromptLookup<>`** (moved in v0.22), payload v18.

---

## 5. Known failures (do not regress)

| Symptom | Cause | Fix / policy |
| --- | --- | --- |
| `identity or size mismatch` | Rank skipped split-argmax or lookup cycle | Draft-subset + lookup single-host only |
| Session reuse hang | Component-cache restore rank-0 only | `UsesComponentCache()==false` |
| Short NL “scrambled text” / empty content | Wide-prefill fused `n>8` but TP2 needs `n≥1024` | `bb383a36` thresholds |
| `fused attention projection failed` under YaRN | YaRN launch required `tiles%4==0`; TP2 is 26 tiles | `80e3a7c4` group-2 yarn launch |
| Handshake `context/sessions mismatch` | Rank scripts stale on one host | Sync **all four** rank scripts on **both** hosts |
| RDMA status 5 first prefill | Often link/Thunderbolt after desync | Fix desync first; don’t loop restarts |

---

## 6. Process (each new upstream release)

1. `git fetch origin`; merge **one tag** into fork `main` per `docs/MERGE-CHECKLIST.md`.
2. Resolve conflicts **keeping TP2 gates**; README/AGENTS: manual review, no blind
   merge; AGENTS nix count **0**.
3. Format (`tools/ci/check-format.py`) + release build (host flags).
4. Ship **same sha** to strix-halo-1; no leftover `gufo`.
5. Smoke: PONG, >1024-token prompt, short NL (eBPF-style content assert),
   ≥5 @ sessions, agent-style, logs free of mismatch/status5/`tp_pair_lost`.
6. **User llama-swap** validation (Q8-tp2).
7. Tag `tp2-rdma-merge-vX.Y.Z-<YYYY-MM-DD>`; push **uv-genai only**.

Dated tags: v0.16.0-2026-10-17 … **v0.25.0-2026-10-27**. Fork is current with upstream as of last fetch.

---

## 7. Docs owned in this repo (keep current)

| Path | Purpose |
| --- | --- |
| `README.md` | fork overview; tracks upstream **through v0.25.0**; TP2 table 2026-10-14; **DeepSWE 1.1 = 90/113 (79.6%) > official leaderboard** |
| `AGENTS.md` | build, remotes, **TP2 hazard table** |
| `docs/MERGE-CHECKLIST.md` | merge + runtime checklist |
| `docs/MERGE-UPSTREAM-2026-10-24.md` | series write-up |
| `docs/models/qwen3.8-flash-next/TP2.md` | collective rules |
| `docs/SERVER.md` / `docs/CLI.md` | `--enable-yarn`, context > 262144 |

External: **uv-genai/gufo issue #1** — comment on YaRN implementation
(v0.21.0+); still open (NTK / `--rope-scale` API not implemented).

**DeepSWE 1.1 (2026-10-26):** `gufo-qwen3.8-flash-next-tp2-q8` scored
**90/113 (79.6%)** — above the official
[leaderboard](https://deepswe.datacurve.ai/) (site max **74%**).
Report: [uv-genai/deepswe-run](https://github.com/uv-genai/deepswe-run/blob/main/reports/final-report-2026-10-26.md)
(linked from README). Standalone single-host Q4 smoke on this binary also
passed (2026-10-27); GPU hosts should be **clear of `gufo`** between runs.

---

## 8. Active workstream: GLM-5.3-Flash (GPU inference in progress)

**Goal:** GLM-5.3-Flash (`glm5-next`) on **one Strix Halo** and **TP2 two-host**, inspired by
[mw00/project-maya](https://github.com/mw00/project-maya) (already installed on **strix-halo-1.local**
at `/home/ugo/software/project-maya`, models in `/home/ugo/software/Maya-data`,
user-measured **~23 tok/s gen** on Maya-S IQ2_XXS).

| Artifact | Location |
| --- | --- |
| Design study | `docs/models/glm5-flash/PLAN.md` |
| Detailed task plan | `docs/models/glm5-flash/TASKS.md` (issue map + **ADRs filled**) |
| **Tensor map** | `docs/models/glm5-flash/WEIGHTS.md` (full inventory + split-alignment) |
| Tracking issue | **[uv-genai/gufo#2](https://github.com/uv-genai/gufo/issues/2)** |
| GPU engine | `src/models/glm5_next/gpu_engine.{hpp,cpp}` |
| Quantized GEMM | `src/models/glm5_next/quant_gemv.{hpp,hip.cpp}` |
| GPU kernels | `src/models/glm5_next/kernels/rocm/` (kda, mhc, moe, mla, mtp) |

**Phase 0 COMPLETE (2026-10-27):**
- **#3** ✅ GGUF tensor map → `WEIGHTS.md`. 46 blocks (not 45!), 12 DSA layers, all quants **GREEN** for TP2 column-split.
- **#4** ✅ ADRs in `TASKS.md`. KDA full replica; expanded-KV MLA first; router piggyback; fixed MTP width; mHC Sinkhorn per-forward.
- **#5** ✅ `glm5_next` skeleton. Config/weights/engine + serve dispatch.

**Phase 1 — ALL CPU ORACLES DONE (14 test suites, 72 tests, all green):**
- **#6** ✅ Tokenizer + chat template (9 tests)
- **#7** ✅ KDA oracle (7 tests) + GPU kernel (parity 5.96e-08)
- **#8** ✅ mHC Sinkhorn oracle (7 tests) + GPU kernel (parity 2.4e-07)
- **#9** ✅ MoE sigmoid top-8 oracle (8 tests) + GPU kernel (parity 2.1e-09)
- **#10** ✅ MLA/DSA oracle (8 tests) + GPU kernel (simplified)
- **#11** ✅ NextN MTP oracle (6 tests) + GPU kernel (parity 3.9e-07)
- **#12** ✅ HTTP serve smoke PASS on gfx1151 (PONG, long prompt, short NL, sessions)

### GPU inference architecture (matching Maya)

Maya's approach (23 tok/s on same hardware):
1. **Weights stay quantized on GPU** — no dequantization for large tensors
2. **`gm_gemv` dispatches by type**: F32 → standard GEMV; quantized → `native_mmvq`
3. **`native_mmvq`**: quantize input to Q8_1, dot product against quantized weights
4. **Small weights** (norms, biases) dequantized to F32
5. **`token_embd`** stays on host, dequantized per-row as needed

Our approach:
1. **Weight arena** (~87 GB): quantized weights uploaded as-is (raw bytes),
   uploads staged through a pinned buffer (direct pageable H2D from the GGUF
   mapping stalls for tens of seconds per tensor).
2. **`QuantGemm`** (`quant_gemv.hip.cpp`): fused on-the-fly dequant for
   **F32/F16/Q8_0/Q2_K/Q3_K/Q6_K/IQ2_XXS/IQ3_XXS/IQ2_S** — exact llama.cpp
   `dequantize_row_*` semantics (parity-tested vs host `quant::Dot`).
3. **F32 only for small weights**: norms, biases, mHC, router.
4. **Output head + embedding rows** run through `QuantGemm` /
   `QuantEmbedLookup` on the quantized weights (no F32 head copy).
5. Quant types come from the loaded tensors — never hardcoded.

**STATUS 2026-10-28 late (issue #19): inputs now match Maya exactly; output
is still wrong; ROOT CAUSE LOCALIZED to the KDA (and DSA) mixer math.**
Do not trust "tests pass" — the gate is the real model answer. Reference:
Maya greedy on the golden prompt says "Simple factual question, no need for
extensive thinking. The answer is Paris." gufo says garbage.

### THE tool: seam diff vs Maya (use this first)

Maya's engine binary supports `GLM_CB_DIR` seam dumps ("so the python diff can
bisect the first divergent component"). gufo's GpuEngine dumps the same seams
per position (`<name>-<L>-<pos>.f32`). Workflow (verified this session):

```sh
# Maya reference (dumps = LAST prompt token's forward — verified: 1-vs-2-token
# --spec-follow continuations give identical dumps):
cd /home/ugo/software/project-maya && PYTHONPATH=tools .venv/bin/python -c "..."  # token ids
GLM_CB_DIR=/tmp/maya-dump ./build-hip/maya --glm-pack .../pack \
  --tokens "154822,154824,154826,25062,287,29905,371,25,7487,154827,785,6722,315,9621,374,154828,13699,15724,29" \
  --max-context 512 --greedy --spec-follow <file-with-1-or-2-token-ids>

# gufo (compare against pos=18 = last prompt token):
GLM_CB_DIR=/tmp/gufo-dump /tmp/gufo-glm5-smoke prompt --model .../...00001-of-00003.gguf \
  --prompt "The capital of France is" --max-tokens 1
# diff: load both .f32, max |a-b| / rms(a) per seam per layer.
```

**IMPORTANT: the Maya BINARY dumps a different seam set than the source tree on
disk** (binary is newer). Actual files: `mixer-L`, `attn_norm-L`, `ffn_norm-L`,
`ffn_out-L`, `hc_attn_post-L`, `l_out-L`, `dsa_q/dsa_qr/dsa_iq-L`,
`pf_attn/pf_ctx-L` (11 DSA layers), `pf_shexp-L` (42 MoE layers). The
`hc_pre/hc_comb/...` names in glm_model.cu are NOT what the binary writes.

**Result so far (relative to rms), 2026-10-28 late:**
- L0 `attn_norm` MATCHES (1e-4 T=1 / 7e-4 T=19) — inputs/framing/mHC OK.
- L0 `mixer` (KDA): **was rel 3.7 → now 1.0e-2 at T=1, 1.5e-2 at last prompt
  token** (cos 0.999997). Formula confirmed vs Maya `ref/glm.py::kda_layer`.
  Acceptance target ~1e-3 not yet met — remaining gap is GEMM/fast-math
  arithmetic (exact Python f64 ref is *farther* (5%) than gufo's f32/Q8_1).
- L3 `mixer` (DSA) T=1 rel 3e-2 cos 0.99997 — formula close; L7 mixer
  collapses (input already dirty). `dsa_q/qr/iq` dumps exist in Maya T=1
  (`/tmp/maya-t1`) but gufo does not dump them yet.
- L0 `ffn_out` T=1 rel 3.5e-2 (dense FFN OK). L3 `ffn_out` cos 0.94 — MoE
  top-k/weights still off (or flips under 3% input error).
- Acceptance prompt still garbage ("( 1. 1 point 2. What is…"), not "Paris".

**KDA bugs FIXED this session (keep these):**
1. **Conv weight layout + tap order.** GGUF `ssm_conv1d_*` is
   `(d_conv, 1, inner)` with tap fastest: `conv[tap + K*c]`, tap 0 = **oldest**
   (Maya `glm_kda_conv` / `ref/glm.py::conv1d_causal`). Old code used
   `conv[k*inner + c]` with k=0 = newest — both wrong.
2. **Conv history was never wired.** `kda_conv_off_` existed in the state
   arena but was not passed to `KdaPrefillGpu`. GpuEngine is T=1-per-token
   (`StepLayers`), so without `conv_state` every token's conv saw only itself.
   Now `KdaPrefillGpu(..., state, conv_state, ...)` slides history (Maya
   layout: 3 streams × (d_conv-1, inner), oldest first).
3. **`kernels/rocm/kda.cpp` is now a literal port** of Maya
   `glm_kda.cu` + `l2_heads_kernel`/`sigmoid_kernel`/`axpy_kernel` +
   step_layers KDA order. Recurrence is one-block-per-head with tokens
   inside, same phases, `__expf`/`dsigmoid`/`rsqrtf`. Do not re-split those
   kernels. Small GEMMs use QuantGemm Q8_1 (Maya `native_mmvq`).
   GFX11 DPP `warp_sum_lanes` ported in `quant_gemv.hip.cpp` (Maya
   `glm_fast.cu`).

**KDA intermediate localization (T=1, call 0 dumps in GLM_CB_DIR as
`kda_*-<call>.f32`; compare vs `ref/glm.py` on `attn_norm-0-0`):**
| tensor | rel vs exact | note |
| --- | --- | --- |
| proj_q (Q6_K) | 2.1e-2 | first gap — Q8_1 GEMM vs exact dequant |
| conv_q | 1.1e-1 max | cos 0.999997 (outliers near 0) |
| q/k after L2 | 6.7e-2 / 3.2e-2 | |
| fa / gpre | 8.5e-3 / 3.7e-3 | Q8_0 GEMM much tighter |
| g1 / beta | 1.4e-2 / 1.4e-3 | gate formula correct |
| scan / gated / out | 3.9e-2 / 6.2e-2 / 5.5e-2 | |
vs Maya T=1 `mixer-0`: **rel 9.9e-3**, cos 0.999997 (gate 1e-3 not met).
vs Maya T=19 last token: rel 1.7e-2.

**Compare-script pitfall:** numpy `reshape(-1)` of `(dk, nh, T)` is
`e*nh+h`, NOT ggml `e+dk*h`. Use `ref.glm.join_heads` before flattening
or g1/scan/gated look like cos 0.5 “bugs”.

**KDA math contract (verified — do not re-derive):**
```
g  = f_b @ (f_a @ x) + dt_bias          # low-rank, rank = head_dim
g1 = lb * sigmoid(-g * A)               # A = ssm_a = -exp(A_log) < 0
beta = sigmoid(ssm_beta @ x)
S = diag(exp(g1)) @ S;  delta = beta * (v - S^T k);  S += k delta^T
o = (1/sqrt(dk)) * S^T q
out = W_o @ (rms_norm_per_head(o, ssm_norm) * sigmoid(g_b @ (g_a @ x)))
q,k = l2_norm(silu(conv(attn_{q,k} @ x)))   # v NOT l2-normed
```
Do NOT use the softplus(qwen4exp) gate form — variants tested, rel 1.2.

**Maya ref oracle:** `project-maya/ref/glm.py` `kda_layer` / `conv1d_causal`
(+ `ref/load.py` GGUF dequant). Run on strix-halo-1 with
`PYTHONPATH=.../project-maya:.../tools:.../third_party/llama.cpp/gguf-py`.
At T=1 the exact ref is 5% from Maya; gufo is 1%. The gap to close is
arithmetic matching Maya's kernels, not the formula.

### Next steps (in order)

1. **Close T=1 `mixer-0` 1e-2 → 1e-3 vs `/tmp/maya-t1/mixer-0.f32`.**
   KDA elementwise kernels are already a carbon copy. The residual is
   **QuantGemm Q6_K/Q8_0 vs Maya `native_mmvq`** (proj_q is 2% from exact;
   Q8_0 fa is 0.8%). Compare `LaunchMmvq` / `RowDotQ8` / `DotQ6_K` to
   `native_mmvq.cu` `q6_q8_dot` + `native_q6_k_mmvq_kernel` (thread map,
   block-per-iter, accumulation). DPP `warp_sum_lanes` already ported.
   Keep `kda_*-<call>.f32` dumps; re-run `/tmp/cmp_kda.py` after each GEMM
   change (use `join_heads`, not `reshape(-1)`).
2. **DSA/MLA**: add gufo dumps for `dsa_q/dsa_qr/dsa_iq`; compare vs
   `/tmp/maya-t1/`. First DSA layer (3) is 3% at T=1. Maya:
   `src/kernels/cuda/glm_dsa.cu` + step_layers ~2279-2355. Layers
   3,7,…,43. **`idx_kpool_select_tail` MUST stay true** (Maya
   `idx_select_tail=1`, refuses false).
3. **MoE**: L3 `ffn_out` cos 0.94 — re-measure after mixers are tight.
4. Acceptance prompt → expect "Paris". Only then speed (117 → 43 ms).

### What is verified correct (do not re-fix)

- **Prompt framing + tokenization** — the critical fix of this session: the
  CLI fed raw BPE (out of distribution; every answer wrong). Now
  `[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>q<|assistant|><thinking>`
  per the GGUF `tokenizer.chat_template` (jinja `{%-` strips inter-turn
  newlines; `<thinking>` is BPE text, NOT a special token). gufo's token IDs
  now match Maya's pack tokenizer EXACTLY
  ([154822, 154824, 154826, 25062, 287, 29905, 371, 25, 7487, 154827, 785,
  6722, 315, 9621, 374, 154828, 13699, 15724, 29] for "The capital of France
  is"). Specials registered: [gMASK]=154822, <sop>=154824, <eop>, <|...|>
  pattern. Golden tests in `chat_template_test.cpp`.
- **Stream ordering** — every engine/kernel `hipMemcpy` raced the non-blocking
  engine stream (the earlier garbage/NaN/wild-address class). All copies now
  `hipMemcpyAsync` on `stream` with syncs at host boundaries.
- **mHC residual chains** between halves (Maya ping-pongs R); input rms-norms
  (`attn_norm`/`ffn_norm`) present — confirmed by the `attn_norm` seam match.
- **CPU engine wiring** (KDA from L.kda.*, conv1d_*, indexer tensors, mhc_ffn
  weights, dense-FFN branch).
- GPU e2e (`gpu_engine_e2e_test`) ≡ CPU engine 1e-9 multi-token — but both
  share the WRONG KDA oracle; it is a regression gate, not a correctness gate.
- Quant GEMM parity (9 types) vs host dequant — kernel arithmetic is fine;
  Q8_1/dp4a tier in place (Q6_K 8192x4096 = 131 us, 210 GB/s).

### Measured

- Real model: exit 0, 117 ms/token (8.5 tok/s), load 65-81 s. Maya: 43 ms
  (23 tok/s), load 38 s.
- Maya greedy reference output for the golden prompt:
  "Simple factual question, no need for extensive thinking. The answer is
  Paris.</thinking>" (tokens 16360, 59485, 3405, 11, 902, 1184, ...).

### Memory budget (Maya-S v2 on 124 GB GPU)

| Item | Size | Format |
| --- | --- | --- |
| Expert weights (43 MoE layers) | ~87 GB arena | IQ2_XXS/IQ2_S/IQ3_XXS/Q2_K/Q3_K |
| Embeddings + output head | (in arena) | Q6_K |
| KDA attention (34 layers) | (in arena) | Q6_K |
| Small weights (norms, biases, mHC) | ~2 GB | F32 (separate allocs) |
| State + scratch arenas | <1 GB | F32 |
| **Total** | **~90 GB** | fits 124 GB |

### IMMEDIATE NEXT STEPS

1. **Close the KDA 1% arithmetic gap** (see "Next steps" above) until
   `mixer-0` T=1 rel < 1e-3 vs `/tmp/maya-t1/mixer-0.f32`.
2. **DSA/MLA** the same way — add `dsa_q/dsa_qr/dsa_iq` dumps, compare vs
   Maya T=1. Keep `idx_kpool_select_tail=true`.
3. **MoE** once mixers are tight (top-k flips under dirty input).
4. Acceptance prompt → "Paris". Only then speed / load time / MTP.

### Key files

| File | Purpose |
| --- | --- |
| `src/models/glm5_next/gpu_engine.{hpp,cpp}` | GPU forward pass (weight arena, StepLayers) |
| `src/models/glm5_next/quant_gemv.{hpp,hip.cpp}` | Quantized GEMM (all pack types) |
| `src/models/glm5_next/gpu_kernels.{hpp,hip.cpp}` | RmsNorm, MatVec, EmbedLookup, Argmax |
| `src/core/quant/{ggml_dequant,ggml_gemm,iq_grids}.*` | Host dequant/dot + codebooks (parity reference) |
| `src/models/glm5_next/kernels/rocm/kda.cpp` | KDA kernel (uses QuantGemm for attention) |
| `src/models/glm5_next/kernels/rocm/moe.cpp` | MoE + dense FFN (uses QuantGemm) |
| `src/models/glm5_next/kernels/rocm/mhc.cpp` | mHC Sinkhorn kernel |
| `src/models/glm5_next/kernels/rocm/mla.cpp` | MLA/DSA kernels (not yet wired into StepLayers) |
| `tests/models/glm5_next/quant_gemv_gpu_test.cpp` | GPU/host quant parity suite |
| `src/models/glm5_next/weights.{hpp,cpp}` | GGUF tensor binding (block sizes!) |
| `src/cli/prompt/prompt.cpp` | `gufo prompt` CLI GLM5 path |

### Maya reference code (study these)

| File | What to study |
| --- | --- |
| `src/core/glm_model.cu:2211-2350` | `step_layers` — the forward pass |
| `src/core/glm_model.cu:469-480` | `gm_gemv` — quantized GEMM dispatch |
| `src/kernels/cuda/glm_kda.cu` | KDA kernels (conv, gate, recurrence, out_gate) |
| `src/kernels/cuda/glm_hc.cu` | mHC kernels (inv_rms, pre, post) |
| `src/kernels/cuda/glm_dsa.cu` | DSA indexer (pool, score, select, mla_attention) |
| `src/kernels/cuda/native_mmvq.cu` | Quantized matvec (IQ2_XXS/IQ3_XXS/Q6_K) |
| `src/kernels/cuda/glm_fast.cu:502-540` | `dot_iq2_xxs` / `dot_iq3_xxs` — IQ dequant |
| third_party/llama.cpp `ggml-quants.c` | dequantize_row_* (format ground truth) |

### Known issues

1. **Quality below Maya** — text is language-like but the completion is
   off (see next steps 2).
2. **Speed 117 ms/token vs 43 ms** — profile table above; GEMMs are no
   longer the bottleneck.
3. **DSA caches capped at 32K positions** (`kMaxDsaPos`).
4. Expert top-k and DSA pool selection do small D2H syncs per layer.
5. mHC Sinkhorn kernel is single-threaded per token (0.12 ms/call).

**Key geometry:** 46 blocks, hidden 4096, 34 KDA + 12 DSA (layers 3,7,…,43,45).
Block 45 = MLA+DSA + MoE + NextN MTP, **no mHC**. 288+1 experts top-8 inter 2048
(sigmoid, noaux_tc, ×2.5), mHC 4-stream Sinkhorn-20.

**Local GGUFs:** `/home/ugo/software/Maya-data/models/Maya-S-v2-IQ2_XXS/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-0000{1,2,3}-of-00003.gguf` on strix-halo-1.

## 9. How this session should use the file

1. **Start of session:** read this file; do not rely on old `SESSION-PROMPT-*` or
   `HANDOFF-2026-*.md` under `~/gufo-merge-handoff/` unless explicitly asked for
   history.
2. **End of session / after any of the above changes:** rewrite or edit
   **this** file so the next agent can cold-start without guessing.
3. Prefer one canonical path: **`handoff.md` at the repo root** (this file).
