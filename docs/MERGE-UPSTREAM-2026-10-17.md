# Upstream merges: v0.16.0 and v0.17.0 into fork `main`

Status: **merged and built; TP2 runtime requalification still pending.**
Written 2026-10-17 so the next session knows what was kept and what to test.

## Branch state

| ref | tip |
| --- | --- |
| fork `main` / `uv-genai/main` | `ed40fb3e` `chore: merge upstream v0.17.0 into main` |
| local `gufo-org` | `1f970923` (upstream `origin/main`, tag v0.17.0+) |
| previous published tip | `24536b3e` / tag `tp2-rdma-merge-v0.16.0-2026-10-17` |

Merge bases: v0.16.0 from `22e36769`; v0.17.0 from the same base lineage
after `1e7509eb` (v0.16 merge) plus TP2 fixes `2341ff55` / `24536b3e`.

## What came in

**v0.16.0** (`1e7509eb`): cache checkpoint/retention rework, draft-vocabulary
scoring (#603), tool-call dialect move, `/health` during load, long-context
checkpoint overhead (#615).

**v0.17.0** (`ed40fb3e`): prompt lookup into MTP drafts (#613), compact sparse
attention (#622), attention cost at depth (#626), publication-wait bounds
(#621), vision sidecar prepare (#616), cache stalls (#633), plus n-gram
gather scratch.

## TP2 / RDMA resolutions (keep these)

| issue | resolution |
| --- | --- |
| Draft-subset (#603) can take different `MtpHead` paths | Gate `draft_subset` on `TpWorldSize()==1` (`2341ff55`) |
| Graph pre-wait + `MixerPart` always `embeddings_ready=false` → `n-gram table read failed` | Thread `embeddings_ready` through `TrunkBatch` (`24536b3e`) |
| Upstream `OwnF16` dropped `m==48` | Restore TP2 half-width gates (`blaslt.cpp`) |
| Attention lists (#622) vs per-rank heads | Union: `heads.attn/attn_kv/attn_dim` + `attn_lists`/`attn_counts` |
| Prompt lookup (#613) vs collectives | Snapshot v18 carries lookup state; requalify session reuse |
| `UsesComponentCache` | Still false on `TpMirroredRunner` |

## Doc updates in this session

- `AGENTS.md`: remotes/merges, nix-free AGENTS rule, TP2 hard gates, metrics
  ports (gufo HTTP, not llama-swap 8080).
- `README.md`: fork tracks through v0.17.0; one-host numbers from upstream;
  TP2 depth series still 2026-10-14 (not requalified).
- `docs/models/qwen3.8-flash-next/TP2.md`: draft-subset and prompt-lookup
  collective rules.

## Still required before push

1. Wait for any running GPU benchmark to finish.
2. Ship `build/release/gufo` to both hosts (sha256 must match).
3. TP2 pair: generation text, prompt **>1024 tokens**, **≥5 requests** at
   `--sessions 4`, plus a repeated-prefix/agent prompt (lookup).
4. Optional: single-host `diagnose` + a short functional suite if decode
   paths are questioned.
5. Only then push `main` to `uv-genai` and tag (e.g.
   `tp2-rdma-merge-v0.17.0-<date>`).

## Related

- Prior merge: `/home/ugo/gufo-merge-handoff/MERGE-UPSTREAM-2026-10-14.md`
- TP2 protocol: `docs/models/qwen3.8-flash-next/TP2.md`
- Depth series (old binary): `docs/models/qwen3.8-flash-next/2026-10-14-tp2-depth-series.md`
