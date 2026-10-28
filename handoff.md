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
1. **Weight arena** (~58 GB): quantized weights uploaded as-is (raw bytes)
2. **`QuantGemm`** (`quant_gemv.hip.cpp`): on-the-fly dequant for IQ2_XXS/IQ3_XXS/Q6_K
3. **F32 only for small weights**: norms, biases, mHC, router
4. **`DequantToHost`** for output head (currently slow — **needs fix**)

### Memory budget (Maya-S v2 on 124 GB GPU)

| Item | Size | Format |
| --- | --- | --- |
| Expert weights (43 MoE layers) | 94 GB | IQ2_XXS/IQ3_XXS quantized |
| Embeddings + output head | 1 GB | Q6_K quantized |
| KDA attention (34 layers) | 3.7 GB | Q6_K quantized |
| Small weights (norms, biases) | ~2 GB | F32 |
| State arena (residual + KDA state) | 0.2 GB | F32 |
| **Total** | **~100 GB** | fits 124 GB |

### IMMEDIATE NEXT STEPS (the current blocker)

**1. Fix output head GEMM** — `GpuEngine::Decode` / `Prefill` currently dequantizes
the output head `[4096, 154880]` Q6_K to F32 (634M floats = 2.5 GB, ~100 seconds).
Replace with `QuantGemm` like we did for KDA attention and MoE experts.

**2. Test forward pass** — `gufo prompt --model <gguf> --prompt "Hello" --max-tokens 4`.
Expect: weights load in ~60s, then forward pass runs on GPU.

**3. Verify quantized GEMM correctness** — The IQ2_XXS dequant in `quant_gemv.hip.cpp`
is approximate (simplified codebook). Need to port Maya's `dot_iq2_xxs` from
`glm_fast.cu:502`. Compare against `DequantToHost` + F32 GEMM on a small test case.

**4. Performance** — Target >23 tok/s. May need `qfn_mmq` kernels from
`qwen38_flash_next/kernels/rocm/mmq/` for production-quality quantized GEMM.

### Key files

| File | Purpose |
| --- | --- |
| `src/models/glm5_next/gpu_engine.{hpp,cpp}` | GPU forward pass (weight arena, StepLayers) |
| `src/models/glm5_next/quant_gemv.{hpp,hip.cpp}` | Quantized GEMM (IQ2_XXS/IQ3_XXS/Q6_K) |
| `src/models/glm5_next/gpu_kernels.{hpp,hip.cpp}` | RmsNorm, MatVec, EmbedLookup, Argmax |
| `src/models/glm5_next/kernels/rocm/kda.cpp` | KDA kernel (uses QuantGemm for attention) |
| `src/models/glm5_next/kernels/rocm/moe.cpp` | MoE kernel (uses QuantGemm for experts) |
| `src/models/glm5_next/kernels/rocm/mhc.cpp` | mHC Sinkhorn kernel |
| `src/models/glm5_next/weights.{hpp,cpp}` | GGUF tensor binding |
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

### Known issues

1. **IQ2_XXS dequant is approximate** — `quant_gemv.hip.cpp` uses simplified
   `(q-1.5)*d*sign` instead of full IQ2_XXS codebook. May produce wrong values.
2. **Output head dequant slow** — 634M floats on host. Fix with `QuantGemm`.
3. **DSA indexer not wired** — MLA/DSA layers use simplified full attention.
4. **`qfn_mmq` only supports Q4_K/Q5_K/Q5_1/Q8_0** — not IQ2_XXS. Need to extend
   or use `native_mmvq` port.

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
