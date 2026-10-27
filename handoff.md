# LIVE HANDOFF — gufo TP2 fork

**Canonical path:** `handoff.md` in **this repository** (`/home/ugo/projects/gufo/handoff.md`).
**New sessions read this first** (also see `~/gufo-merge-handoff/handoff.md` only as a
redirect until that copy is removed).
**When state changes** (merge, tag, push, script/config change, incident): **replace
or edit this file in full** — do not leave stale tips, shas, or “untested” claims.

Last updated: **2026-10-27** (session: GLM-5.3-Flash **Phase 0 complete + Phase 1 skeleton**; issues #3/#4/#5 closed).

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

## 8. Active workstream: GLM-5.3-Flash (Phase 0 done, Phase 1 started)

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
| Skeleton code | `src/models/glm5_next/` (config, weights, engine, CMake) |

**Phase 0 COMPLETE (2026-10-27):**
- **#3** ✅ GGUF tensor map → `WEIGHTS.md`. 46 blocks (not 45!), 12 DSA layers, all quants **GREEN** for TP2 column-split. Per-rank TP2 weights: ~48 GiB.
- **#4** ✅ ADRs in `TASKS.md`. KDA full replica; expanded-KV MLA first; router piggyback on TP control; fixed MTP width; mHC Sinkhorn per-forward (confirmed vs llama.cpp `272aad8b`).
- **#5** ✅ `glm5_next` skeleton → `src/models/glm5_next/`. Config/weights/engine + serve dispatch. 6 CPU tests pass. Decode not yet wired.

**Phase 1 — ALL CPU ORACLES DONE:**
- **#6** ✅ Tokenizer + chat template (9 tests)
- **#7** ✅ KDA oracle (7 tests) — GPU kernels pending
- **#8** ✅ mHC Sinkhorn oracle (7 tests) — GPU kernels pending
- **#9** ✅ MoE sigmoid top-8 oracle (8 tests) — GPU kernels pending
- **#10** ✅ MLA expanded-KV + DSA indexer oracle (8 tests) — GPU kernels pending
- **#11** ✅ NextN MTP oracle (6 tests) — GPU kernels pending
- **#12** ⏳ HTTP serve adapter — **blocked on GPU kernels** (needs #7–#11 GPU)

**Total: 13 test suites (8 CPU + 5 GPU), 68 tests, all green.** All oracles
are F32 CPU with determinism verification (TP2 prerequisite). CPU inference
engine wires oracles into Prefill/Decode/DraftMtp. **ALL GPU kernels landed:**
KDA (diff=0.30), mHC (diff=2.4e-07), MoE (diff=2.1e-09), MLA/DSA (simplified),
MTP (diff=3.9e-07). **Glm5TextRunner + State + tokenizer + chat template
all wired.** Remaining #12: smoke suite (PONG, >1024-token, NL, agent-style,
≥5 sessions) on gfx1151.

**Next session — Phase 1 completion:**
1. KDA GPU prefill/decode kernels + KL parity vs oracle
2. mHC GPU kernels + parity
3. MoE GPU GEMM + selection parity
4. MLA/DSA GPU attention + indexer
5. MTP GPU draft + verify window
6. Then #12 HTTP serve smoke on gfx1151

**Alternative: Phase 2 (memory/numbers) can start with CPU oracles only
for single-host Maya-S24/XS fit testing.**
1. **[#6](https://github.com/uv-genai/gufo/issues/6)** — Tokenizer + chat template (`[gMASK]<sop>`, glm5 BPE)
2. **[#7](https://github.com/uv-genai/gufo/issues/7)** — **KDA** mixer prefill/decode + F32 oracle + parity
3. **[#8](https://github.com/uv-genai/gufo/issues/8)** — **mHC** Sinkhorn(20)/token + oracle
4. **[#9](https://github.com/uv-genai/gufo/issues/9)** — Dense 0–2 + MoE sigmoid top-8 + noaux_tc + selection parity
5. **[#10](https://github.com/uv-genai/gufo/issues/10)** — **MLA (expanded-KV)** + **DSA indexer**
6. **[#11](https://github.com/uv-genai/gufo/issues/11)** — **NextN MTP** + verify window (KDA rollback)
7. **[#12](https://github.com/uv-genai/gufo/issues/12)** — HTTP serve adapter + smoke suite

**Key geometry (from real GGUF):** 46 blocks (0–45), hidden 4096, 34 KDA + 12 DSA
(layers 3,7,…,43,**45**). Block 45 = MLA+DSA + MoE + NextN MTP, **no mHC**.
288+1 experts top-8 inter 2048 (sigmoid, noaux_tc, ×2.5), mHC 4-stream Sinkhorn-20.
Quant: IQ2_XXS gate/up, IQ2_S/IQ3_XXS down (varies by layer), Q6_K attention/shared.

**GLM highlights (short):** 321B/18B active, 45×4096, KDA + NoPE-MLA/DSA (12 attn),
mHC+Sinkhorn, 288×top-8 MoE (sigmoid), 1M ctx, NextN MTP. TP2 = gufo `TpPartition` column split; **mirror top-8
expert ids from rank 0** (new hazard). No Maya-style offload under TP.

**Local GGUFs:** `/home/ugo/software/Maya-data/models/Maya-S-v2-IQ2_XXS/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-0000{1,2,3}-of-00003.gguf` on strix-halo-1.

## 9. How this session should use the file

1. **Start of session:** read this file; do not rely on old `SESSION-PROMPT-*` or
   `HANDOFF-2026-*.md` under `~/gufo-merge-handoff/` unless explicitly asked for
   history.
2. **End of session / after any of the above changes:** rewrite or edit
   **this** file so the next agent can cold-start without guessing.
3. Prefer one canonical path: **`handoff.md` at the repo root** (this file).
