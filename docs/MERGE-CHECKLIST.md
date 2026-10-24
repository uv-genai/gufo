# Upstream merge checklist (fork `main`)

Use this on **every** merge of `gufo-org` / `origin/main` into fork `main`.
TP2 is a **collective sequence lock**: both ranks must issue the same verbs
all-reduces (order, count, byte size). Upstream one-host optimizations often
change decode control flow and break that lock unless gated.

See also: [TP2.md](models/qwen3.8-flash-next/TP2.md) · [AGENTS.md](../AGENTS.md)

---

## 1. Before merge

- [ ] `git fetch origin && git switch gufo-org && git merge --ff-only origin/main`
- [ ] `git log --oneline main..gufo-org` — note commits that touch:
  - `src/models/qwen38_flash_next/engine.*`
  - `.../kernels/rocm/executor.*`, `kernels.hip.cpp`, `verbs.cpp`
  - `src/cli/serve/{inference_backend,component_text_cache,text_model_runner,text_generation_scheduler}.*`
  - MTP / draft / cache / checkpoint / prompt-lookup headers
    (`src/core/speculative/prompt_lookup.hpp`, `draft_vocab.hpp`,
    `mtp_policy.hpp`)
- [ ] For each such commit, ask: **does it change how many GPU collectives a
  decode/prefill cycle runs?** If yes → TP2 hazard (table below).
- [ ] Prefer **one upstream release per session**: merge → format/build →
  TP2 smoke → user llama-swap validation → tag/push, then the next tag.

## 2. TP2 hazard policy (non-negotiable)

| Feature class | Required under `TpWorldSize()>1` |
| --- | --- |
| Draft subset / small head | **Off** (`subset_ok` / `draft_subset` false) |
| Prompt lookup copies | **Off** (`lookup_cap==0`) |
| Component cache restore | **Off** (`UsesComponentCache` false) |
| `defer_residual_` | **Off** |
| Forced draft width | Rank 0 plan only (`request.drafts` / `batch_drafts`) |
| Attention/MoE “skip AR” shortcuts | Forbidden unless identical on both ranks |

Prefer **`TpWorldSize()==1` gates** in code over relying on ranks seeing the
same tokens. When a feature must work under TP, **mirror the full decision**
(or force both ranks to the neural/full path).

### Past failures (do not regress)

| Symptom | Cause | Fix |
| --- | --- | --- |
| Hang after `--sessions` N requests | Component-cache restore rank-0 only | `UsesComponentCache()==false` |
| `n-gram table read failed` | Graph pre-wait + `MixerPart` double WaitPle | `TrunkBatch.embeddings_ready` |
| `identity or size mismatch` 10240 vs N×hidden | Rank skipped split-argmax or lookup cycle | Draft-subset and lookup single-host only |
| Rapid restart loop, RDMA status 5 | Often after peer desync / bad QP | Fix desync first; then check link |
| Short NL prompts claim “scrambled text” / empty content | Wide-prefill fused paths (SSM/attention/routed tiles/`last_only`) ran for `n>8` but TP2 half-width only qualifies `n≥1024` | Keep TP2 fused thresholds at `n≥1024` (see `bb383a36`) |

## 3. After merge (code)

- [ ] `grep -ci nix AGENTS.md` → must be **0** (and never run that package manager)
- [ ] `python3 tools/ci/check-format.py`
- [ ] Release configure/build with the three host flags in AGENTS.md
- [ ] `./build/release/gufo diagnose` when GPU is free
- [ ] Confirm gates still present:
  - `UsesComponentCache` false on `TpMirroredRunner`
  - `subset_ok = … && TpWorldSize()==1`
  - `lookup_ok = … && TpWorldSize()==1`
  - `defer_residual_` gated on single-host
  - `OwnF16` includes `m==48` if TP2 half-width gates still exist

## 4. After merge (TP2 runtime — before push / before dropping `-untested`)

- [ ] **No `gufo`** on either host (`pgrep -ax gufo`)
- [ ] **Ship the same binary** to `strix-halo-1.local`; **sha256 must match**
- [ ] Start pair (rank0 then rank1 ≤30s, or `tools/tp2/swap.sh` via llama-swap;
      **not** port 8080 for raw gufo metrics)
- [ ] `/ready` OK
- [ ] **Generation text**, prompt **>1024 tokens**
- [ ] One **short natural-language** prompt (e.g. sample eBPF code) with
      content assertions — empty body / “scrambled text” is a prefill bug
- [ ] **≥5 requests** at `--sessions 4` (session reuse)
- [ ] One **repeated-prefix / agent-style** prompt (lookup / copy paths)
- [ ] `GET /metrics` on gufo HTTP: `requests_processing` / `deferred` move;
      no `tp_pair_lost` / size mismatch / `status 5` in the server log
- [ ] User **llama-swap** run (their Q8-tp2 path) before tagging
- [ ] Only then: push `uv-genai`, replace any `-untested` tag

## 5. What “green” is not

- Compile-only
- HTTP 200 with empty body
- Single short chat on a fresh pair
- One host binary newer than the other

---

Write-up template for a completed merge:
`docs/MERGE-UPSTREAM-YYYY-MM-DD.md` (see `MERGE-UPSTREAM-2026-10-17.md`).
