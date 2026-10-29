# LIVE HANDOFF — gufo TP2 fork

**Canonical path:** `handoff.md` in **this repository** (`/home/ugo/projects/gufo/handoff.md`).
**New sessions read this first** (also see `~/gufo-merge-handoff/handoff.md` only as a
redirect until that copy is removed).
**When state changes** (merge, tag, push, script/config change, incident): **replace
or edit this file in full** — do not leave stale tips, shas, or “untested” claims.

Last updated: **2026-10-29** (session: **v0.26.0 merged, smoke + user llama-swap PASS; tagged and released**; user decode report **~56 vs ~45 tok/s** — see §1).

---

## 0. Read next (in order)

1. **This file** (`handoff.md` at the repo root)
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
| Tip | **`8b229542`** = tag **`tp2-rdma-merge-v0.26.0-2026-10-29`** (`main` == `uv-genai/main`) |
| Upstream merged through | **v0.26.0** (`78d74185`); README tracks through v0.26.0 |
| Release notes | **https://github.com/uv-genai/gufo/releases/tag/tp2-rdma-merge-v0.26.0-2026-10-29** |

**Non-negotiables:** `grep -ci nix AGENTS.md` → **0**; never run that package
manager; Conventional Commits; ask before any GPU launch.
**Every future tag must also get a GitHub Release with written notes** (user
directive 2026-10-29) so the tag row shows the notes link.

---

## 2. Binary & tested configuration

| | |
| --- | --- |
| Path | `build/release/gufo` on **both** hosts |
| sha256 | **`7b9c4e482ceb22dcd2e639f7a336f347d3985935dd46c2d2793025199d1e7ecf`** |
| Built from | tip **`9b0aabc2`** (v0.26.0 merge; docs-only commits after); format clean |
| TP2 smoke | **2026-10-29 PASS** on both hosts, sha `7b9c4e48…` matched; `/ready` 200; 10/10 requests (PONG, 3481-tok long, short eBPF NL w/ content asserts, 5× session, 2× agent prefix); logs free of `tp_pair_lost`/mismatch/status5; thinking off via `chat_template_kwargs` for asserts |
| User llama-swap (Q8-tp2) | **2026-10-29 PASS** — user: prefill “way faster”; token generation **~56 tok/s vs ~45** before |
| Released | **`tp2-rdma-merge-v0.26.0-2026-10-29`** on uv-genai (+ GitHub Release notes URL above); **not** pushed to origin |
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
| Hang after v0.26 side-stream MoE/DeltaNet | Side-stream work must `JoinSide` before AR/epilogue; TP2 `reduce`/`AllReduce` path kept | Merged `Moe` keeps `!reduce \|\| AllReduce`; `UsesComponentCache` still false |

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

Dated tags: v0.16.0-2026-10-17 … **v0.26.0-2026-10-29** (Release notes linked). Next: next upstream. **Each new tag: push uv-genai + create GitHub Release with notes.**

---

## 7. Docs owned in this repo (keep current)

| Path | Purpose |
| --- | --- |
| `README.md` | fork overview; tracks upstream **through v0.26.0** (no nix section; Windows build note kept; TP2 table still 2026-10-14 numbers) |
| `AGENTS.md` | build, remotes, **TP2 hazard table** |
| `docs/MERGE-CHECKLIST.md` | merge + runtime checklist |
| `docs/MERGE-UPSTREAM-2026-10-24.md` | series write-up |
| `docs/models/qwen3.8-flash-next/TP2.md` | collective rules |
| `docs/SERVER.md` / `docs/CLI.md` | `--enable-yarn`, context > 262144 |

External: **uv-genai/gufo issue #1** — comment on YaRN implementation
(v0.21.0+); still open (NTK / `--rope-scale` API not implemented).

---

## 8. How this session should use the file

1. **Start of session:** read this file; do not rely on old `SESSION-PROMPT-*` or
   `HANDOFF-2026-*.md` under `~/gufo-merge-handoff/` unless explicitly asked for
   history.
2. **End of session / after any of the above changes:** rewrite or edit
   **this** file so the next agent can cold-start without guessing.
3. Prefer one canonical path: **`handoff.md` at the repo root** (this file).
