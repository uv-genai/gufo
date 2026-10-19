# Upstream merge — 2026-10-19: prompt-lookup TP2 gate

Status: **fix committed locally** (`e67dd64c`). Binary **`29b6fd9a…`**
shipped to both hosts. Runtime requalification still required before push.

## What broke after v0.17.0

On **matched** binaries (`99e3b9cb`), llama-swap/TP2 showed:

1. **`verbs collective identity or size mismatch`** — same scope/ordinal,
   different byte counts (e.g. 10240 vs 122880 = 1 vs 12 rows × 2560 F32).
2. After crash-restart loops: **`verbs all-reduce completion failed with
   status 5`** (`IBV_WC_MW_BIND_ERR`) on the first prefill — usually after
   the peer/QP was already poisoned.

v0.16-era failures (mismatched binaries, draft-subset, n-gram double-wait)
were already fixed. This was a **new** v0.17 path.

## Root cause

**Prompt lookup (#613)** can append copy tokens and **skip** neural
`MtpForward` cycles. Rank 0 mirrors draft **counts** only. If one rank copies
and the other does not, the pair issues **different numbers of all-reduces**
→ size mismatch → poison.

## Fix

`engine.cpp` `PrepareDecode`:

```text
lookup_ok = model_->TpWorldSize() == 1
lookup_cap = lookup_ok && lookup_.Step(base) ? cap : 0
```

Same pattern as draft-subset. Single-host keeps lookup; TP2 always takes the
neural draft path with the mirrored count.

Docs: `AGENTS.md` (hazard table + merge checklist pointer),
`docs/MERGE-CHECKLIST.md`, `TP2.md`.

## Still required

1. TP2 smoke: generation text, prompt >1024 tokens, ≥5 requests at
   `--sessions 4`, agent/repeated-prefix prompt.
2. No `tp_pair_lost` / size mismatch / status 5 in logs.
3. Push `uv-genai` and replace `-untested` tag only after that pass.

## Related

- `docs/MERGE-CHECKLIST.md` — permanent merge process
- `docs/MERGE-UPSTREAM-2026-10-17.md` — v0.16/v0.17 merge
- `docs/models/qwen3.8-flash-next/TP2.md` — collective rules
