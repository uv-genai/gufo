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

**STATUS 2026-10-28 (issue #19): GPU inference runs end-to-end with real
MLA/DSA attention; 285 ms/token (3.5 tok/s) vs Maya 43 ms (23 tok/s).**
`gufo prompt --prompt "The capital of France is" --max-tokens 16 --verbose`
exits 0 with language-like continuations and prints per-token ms.

This session's correctness work (all GPU kernels use the loaded model's
quantization):
- **DSA/MLA real attention** — `MlaStepGpu` with incremental latent/ik/ig
  caches (pool scoring, top-k selection, selected-cell softmax, v_b/wo).
  GPU↔oracle parity 2e-7 (`mla_gpu_test`). The oracle now includes the
  indexer `k_norm` on ik (it was skipped).
- **mHC FFN mixer uses hc_ffn_\*** (was reusing attn mixer weights).
- All quant types exact (Q2_K/Q3_K/Q6_K/IQ2_XXS/IQ3_XXS/IQ2_S/Q8_0/F32/F16),
  per-tensor types plumbed from the pack; output head + embed lookup
  quantized; expert byte-strides; blk.45 drafter-only.

This session's speed work (520 → 285 ms/token):
- Persistent kernel workspaces (was ~1600 hipMalloc/free per token).
- Block-per-row matvecs (thread-per-row serial chains were latency-bound;
  mHC fn matvec alone was 0.78 ms/call → 0.016 ms).
- Fused mHC Sinkhorn (39 launches → 1).
- In-order reductions (tree reduction flipped MoE top-k via rounding).

Measured profile per token (bench: `qbench2` harness):
| component | time | x count | total |
| --- | --- | --- | --- |
| KDA layer (4x Q6_K GEMM 8192x4096 @ 920 us) | 4.4 ms | 34 | 150 ms |
| MoE layer | 1.44 ms | 45 | 65 ms |
| DSA step @ 32K ctx | 5.4 ms | 12 | 65 ms |
| mHC pre/post | 0.26/0.003 ms | 90 | 24 ms |

### IMMEDIATE NEXT STEPS

1. **QuantGemm Q8_1/dp4a tier** — the fused dequant-dot achieves ~30 GB/s vs
   the ~150 GB/s effective Maya's `native_mmvq` hits. Port
   `vec_dot_{q6_K,iq2_xxs,iq3_xxs}_q8_1` (vendored in
   `qwen38_flash_next/kernels/rocm/mmq/vecdotq.hpp`) + a real
   `QuantizeQ81`. Expected 285 → ~60-80 ms/token. Tried: templated kernels
   (no change), word-load Q6_K (no change), dual-stream accumulators
   (worse, reverted) — the win is in integer dp4a dots, not layout tweaks.
2. **Quality vs Maya** — output is language-like but below Maya: check KDA
   conv/state details, swiglu clamp (`swiglu_clamp_exp` config), and
   compare logits against Maya (`maya --check-logits`).
3. **Load time 69-81 s vs Maya 38 s** — per-tensor timings print to stderr;
   likely host-side dequant + memcpy order.
4. mHC Sinkhorn is single-threaded (0.12 ms/call); parallelize per column.
5. MTP draft wiring (blk.45); expert top-k on GPU (still host D2H per layer).

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

1. **Wire real DSA/MLA attention** in `StepLayers` (layers 3,7,…,43 currently
   copy `mixed → attn_out`; kernels exist in `kernels/rocm/mla.cpp`). This is
   why output text is degenerate. Parity oracles exist (`mla_oracle_test`).
2. **Load time 81 s vs Maya 38 s** — profile the load path (`[glm5_gpu]`
   per-tensor timings are printed); likely host-side dequant and memcpy order.
3. **Performance**: target >23 tok/s (Maya baseline; measured 17.5 tok/s with
   MTP spec on a short prompt). Consider the `qfn_mmq` Q8_1/dp4a tier or
   Maya's `native_mmvq` port for the GEMM hot path.
4. **MTP draft wiring** for blk.45 (single-host only; fixed width).

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

1. **Quality below Maya** — text is language-like but garbled; remaining gaps
   in KDA conv/state details and swiglu clamp params (see next steps).
2. **QuantGemm tier is dequant-dot (~30 GB/s)** — the Q8_1/dp4a port is the
   identified path to Maya-level tok/s.
3. **Load 69-81 s vs Maya 38 s**; per-tensor load timings print to stderr.
4. **DSA caches capped at 32K positions** (`kMaxDsaPos`) — 1M-context needs
   KV streaming (issue #18).
5. Expert routing top-k runs on host (2 small D2H syncs per MoE layer);
   DSA pool selection does one D2H sync per full-attention layer.
6. mHC Sinkhorn kernel is single-threaded per token.

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
