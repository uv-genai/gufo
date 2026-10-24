# Upstream requalify series — 2026-10-24: v0.16.0 through v0.23.0

Status: **complete.** Fork `main` == `uv-genai/main` at `632abb97`.
Latest tested tag: **`tp2-rdma-merge-v0.23.0-2026-10-24`**.

## Process

One upstream release per stage: merge into fork `main` → resolve conflicts
(TP2 gates first) → format + release build → ship identical sha to both hosts →
in-house TP2 smoke → user llama-swap validation → dated tag → push **`uv-genai`
only**. `AGENTS.md` nix count stayed **0** throughout.

## Tags

| Tag | Tip | Notes |
| --- | --- | --- |
| `tp2-rdma-merge-v0.17.1-2026-10-24` | `ae4629f0` | small cache/test delta |
| `tp2-rdma-merge-v0.18.0-2026-10-24` | `e5c45cf0` | TP2 image `shared_ptr` encode fix |
| `tp2-rdma-merge-v0.19.0-2026-10-24` | `fb03fbf9` | occupancy pricing + forced drafts |
| `tp2-rdma-merge-v0.20.0-2026-10-24` | `bb383a36` | **wide-prefill short-chat TP2 fix** |
| `tp2-rdma-merge-v0.21.0-2026-10-24` | `2578db6e` | YaRN opt-in; sampled lookup TP-gated |
| `tp2-rdma-merge-v0.22.0-2026-10-24` | `1ecdce27` | `speculative::PromptLookup<>` move |
| `tp2-rdma-merge-v0.23.0-2026-10-24` | `632abb97` | DecodeStepPlanned `budget`+`plan`; latest upstream |

Earlier: `tp2-rdma-merge-v0.16.0-2026-10-17` (pre-series baseline).

## TP2 gates kept (still true on v0.23.0)

- `lookup_ok = TpWorldSize()==1` on `lookup_cap` (path:
  `speculative::PromptLookup<>` after v0.22)
- `subset_ok = allow_subset && TpWorldSize()==1`
- Forced draft width: rank 0 `request.drafts` / `batch_drafts`
- `defer_residual_` single-host only
- `TpMirroredRunner::UsesComponentCache()==false`
- Wide-prefill fused paths (SSM / attention / routed tiles / `last_only`)
  require **`n ≥ 1024`** under TP2 half-width geometry

## Defects found during the series

1. **v0.17 lookup desync** (pre-series): prompt-lookup copies skipped neural
   cycles on one rank → `e67dd64c` single-host gate.
2. **v0.20 short-chat corruption** (`bb383a36`): upstream `#690` ran fused
   prefill routes for every `n > 8`; TP2 half-width only qualified `n ≥ 1024`.
   Symptom: “scrambled text”, empty content, unrelated replies.

## Still true

- Published TP2 depth/concurrency table in README remains **2026-10-14**
  measurements; runtime requalified through v0.23.0, numbers not re-benchmarked.
- Process: [MERGE-CHECKLIST.md](MERGE-CHECKLIST.md)
