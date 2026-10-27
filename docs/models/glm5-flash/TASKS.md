# GLM-5.3-Flash — detailed task plan (issue-tracked)

**Tracking issue:** [uv-genai/gufo#2](https://github.com/uv-genai/gufo/issues/2)
**Design:** [PLAN.md](PLAN.md) · **Status:** planning only (this file is the implementation roadmap)

Every task below maps to one GitHub issue on **uv-genai/gufo**. Work one issue at a time; merge to `main` with the usual checks (format, build, **`grep -ci nix AGENTS.md` = 0**, push **uv-genai** only).

---

## Review notes (design v2)

After re-reading PLAN.md against Maya’s GLM5-FLASH study and gufo TP2:

1. **Treat glm5_next as a new first-class model** (like `deepseek_v4_flash`), not a Flash-Next variant. Flash-Next geometry (2560/GDN/QSA) does not generalize; do not over-parameterize `qwen38_flash_next`.
2. **TP2 is the product differentiator** vs Maya (layer-split only). Phase 3 is not optional for the fork’s goal.
3. **Determinism > speed in Phase 1:** fixed MTP width, no rank-local expert shortcuts, expanded MLA before absorbed, full-replica KDA unless #4 chooses otherwise.
4. **Single-host 128 GB** cannot hold Maya-S + large KV + workspace comfortably — Phase 2 targets **Maya-S24/XS** or short context; TP2 holds Maya-S with room.
5. **Maya is a math + measurement reference** (parity oracles, 23 tok/s baseline), not a dependency to link.

---

## Phase 0 — contracts and decisions

| ID | Issue | Task | Done when |
| --- | --- | --- | --- |
| T0.1 | [#3](https://github.com/uv-genai/gufo/issues/3) | GGUF tensor inventory + TP2 column-split alignment | Tensor map doc + split-alignment matrix per quant |
| T0.2 | [#4](https://github.com/uv-genai/gufo/issues/4) | ADR: KDA split vs replicate; MLA expanded vs absorbed; router mirror; MTP policy | ADR section in this file with chosen options |

**Gate to Phase 1:** T0.1 + T0.2 complete.

---

## Phase 1 — single-host correctness

| ID | Issue | Task | Depends | Done when |
| --- | --- | --- | --- | --- |
| T1.1 | [#5](https://github.com/uv-genai/gufo/issues/5) | `src/models/glm5_next/` skeleton: config, GGUF load, CMake, backend stub | T0.1 | Load reaches “model created”; geometry unit tests |
| T1.2 | [#6](https://github.com/uv-genai/gufo/issues/6) | Tokenizer + chat template (`[gMASK]<sop>`, glm5 BPE) | T1.1 | Token-id + prompt golden tests |
| T1.3 | [#7](https://github.com/uv-genai/gufo/issues/7) | **KDA** mixer prefill/decode + F32 oracle + parity | T1.1 | KL p99 gate; state determinism |
| T1.4 | [#8](https://github.com/uv-genai/gufo/issues/8) | **mHC** Sinkhorn(20)/token + oracle | T1.1 | Parity + deterministic reruns |
| T1.5 | [#9](https://github.com/uv-genai/gufo/issues/9) | Dense 0–2 + MoE sigmoid top-8 + noaux_tc + **selection parity** | T1.1 | Selection goldens; GEMM parity |
| T1.6 | [#10](https://github.com/uv-genai/gufo/issues/10) | **MLA (expanded-KV)** + **DSA indexer** | T1.1 | Attention + indexer parity |
| T1.7 | [#11](https://github.com/uv-genai/gufo/issues/11) | **NextN MTP** + verify window (KDA rollback), fixed width | T1.3–T1.6 | Spec decode works; rollback tests |
| T1.8 | [#12](https://github.com/uv-genai/gufo/issues/12) | HTTP serve adapter + smoke suite | T1.2–T1.7 | Smoke green on one gfx1151 |

**Gate to Phase 2/3:** T1.8 green.

---

## Phase 2 — memory and one-APU numbers

| ID | Issue | Task | Depends | Done when |
| --- | --- | --- | --- | --- |
| T2.1 | [#13](https://github.com/uv-genai/gufo/issues/13) | Fit Maya-S24/XS on 128 GB; **pp/tg vs Maya ~23 tok/s**; document limits | T1.8 | `docs/models/glm5-flash/BENCHMARKS.md` |

---

## Phase 3 — TP2 (two Strix Halos)

| ID | Issue | Task | Depends | Done when |
| --- | --- | --- | --- | --- |
| T3.1 | [#14](https://github.com/uv-genai/gufo/issues/14) | `TpPartition` + split-reduce + **router id mirror** + forced MTP | T0.2, T1.8 | Identical collective digests mixed prompts |
| T3.2 | [#15](https://github.com/uv-genai/gufo/issues/15) | Two-host smoke + 256K+ qual + user llama-swap | T3.1 | User-approved; no mismatch/status 5 |
| T3.3 | [#17](https://github.com/uv-genai/gufo/issues/17) | Hazard-table docs (expert ids discrete) | any | AGENTS + MERGE-CHECKLIST updated |

---

## Phase 4 — productize

| ID | Issue | Task | Depends | Done when |
| --- | --- | --- | --- | --- |
| T4.1 | [#16](https://github.com/uv-genai/gufo/issues/16) | Launchers, llama-swap, README, handoff.md | T3.2 | One-command start; docs current |

---

## Later (not scheduled)

| ID | Issue | Item |
| --- | --- | --- |
| T5.x | [#18](https://github.com/uv-genai/gufo/issues/18) | Absorbed MLA, KV streaming 1M, vision, DFlash2, optional tiering |

---

## Implementation order (dependency graph)

```text
T0.1 ─┬─► T1.1 ─┬─► T1.3 ─┬─► T1.7 ──► T1.8 ──► T2.1
      │        ├─► T1.4 ─┤              │
      │        ├─► T1.5 ─┤              ├─► T3.1 ──► T3.2 ──► T4.1
      │        ├─► T1.6 ─┤              │         T3.3 (docs anytime)
T0.2 ─┴────────┴─────────┴──────────────┘
```

---

## Global constraints (every issue)

- Build: three host flags in `AGENTS.md`; `python3 tools/ci/check-format.py`
- **TP2 rules:** identical collectives; **mirror top-k expert ids from rank 0**; same sha both ranks; no `UsesComponentCache` under TP; no dynamic offload under TP
- `grep -ci nix AGENTS.md` = **0**; never run nix
- Push **uv-genai** only; Conventional Commits
- Quality gates: F32/KL parity where noted (not per-logit on quantized paths)

## ADR stubs (fill in T0.2)

| Decision | Options | Choice (T0.2) |
| --- | --- | --- |
| KDA under TP | full replica \| head-split + AR | _TBD_ |
| MLA first | expanded-KV \| absorbed | _TBD_ (plan recommends expanded) |
| Router mirror | TP control piggyback \| all-reduce ids | _TBD_ (plan recommends piggyback) |
| MTP under TP | fixed rank-0 width \| adaptive | _TBD_ (plan: fixed) |
