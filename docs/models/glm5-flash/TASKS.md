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

| ID | Issue | Task | Done when | Status |
| --- | --- | --- | --- | --- |
| T0.1 | [#3](https://github.com/uv-genai/gufo/issues/3) | GGUF tensor inventory + TP2 column-split alignment | Tensor map doc + split-alignment matrix per quant | **Done** → [WEIGHTS.md](WEIGHTS.md) |
| T0.2 | [#4](https://github.com/uv-genai/gufo/issues/4) | ADR: KDA split vs replicate; MLA expanded vs absorbed; router mirror; MTP policy | ADR section in this file with chosen options | **Done** → [ADRs](#adrs-t02--decided-2026-10-27) |

**Gate to Phase 1:** T0.1 + T0.2 complete.

**Key T0.1 findings** (see [WEIGHTS.md](WEIGHTS.md) for full map):
- **46 blocks** (0–45), not 45 — block 45 hosts NextN MTP, has **no mHC**.
- **12 DSA layers** (3,7,…,43,**45**), **34 KDA** layers.
- All quant types in Maya-S v2 pack are **GREEN** for TP2 column-split
  (256-element blocks; inter=2048 → 1024/rank, dense 12288 → 6144/rank).
- Per-rank TP2 weights: **~48 GiB** (41 experts split + 6.4 replicated).
- Architecture metadata key: `glm5-next.*`; chat framing `[gMASK]<sop>`.

---

## Phase 1 — single-host correctness

| ID | Issue | Task | Depends | Done when |
| --- | --- | --- | --- | --- |
| T1.1 | [#5](https://github.com/uv-genai/gufo/issues/5) | `src/models/glm5_next/` skeleton: config, GGUF load, CMake, backend stub | T0.1 | Load reaches “model created”; geometry unit tests | **Done** |
| T1.2 | [#6](https://github.com/uv-genai/gufo/issues/6) | Tokenizer + chat template (`[gMASK]<sop>`, glm5 BPE) | T1.1 | Token-id + prompt golden tests | **Done** |
| T1.3 | [#7](https://github.com/uv-genai/gufo/issues/7) | **KDA** mixer prefill/decode + F32 oracle + parity | T1.1 | KL p99 gate; state determinism | **Oracle done** (GPU pending) |
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

## ADRs (T0.2 — decided 2026-10-27)

### ADR-1: KDA under TP2 — **full replica**

**Choice:** replicate all KDA weights and state on both ranks. No head-split, no extra all-reduce.

**Rationale:**
- KDA state is a per-token recurrence over 64 heads × 128 dims. Any split would
  require all-reducing the recurrent state at every token — high frequency, small
  messages, and a new desync surface.
- Full replica is ~0.13 GiB per layer × 34 layers ≈ 4.4 GiB — negligible against
  41 GiB of experts per rank.
- Deterministic replay is the Phase 1/3 priority. Bit-identical math on both ranks
  is trivially satisfied with replication; head-split introduces rounding-order risk.
- Matches gufo Flash-Next practice: non-FFN weights are replicated; only expert
  FFN columns are split.

**Trade-off:** 4.4 GiB extra VRAM per rank vs zero extra collectives. Acceptable.

### ADR-2: MLA first — **expanded-KV** (absorbed later)

**Choice:** implement MLA with expanded K/V (materialise the latent `c_kv` into
per-head K/V projections at load/decode time) before considering the absorbed form.

**Rationale:**
- Expanded-KV reuses attention-shaped code paths (GEMM + softmax attention) with
  different projections — the same pattern as Flash-Next QSA. The absorbed form
  (fused into the attention kernel) is a large custom kernel effort.
- Per-token cost of expansion: 64 heads × 256 dims × 2 (K and V) from a 512-dim
  latent — one small GEMM, already in the critical path of MLA.
- Maya’s study recommends expanded first; llama.cpp also materialises expanded KV.
- Phase 1 exit is **correctness** (KL/parity), not peak tokens/s. Absorbed MLA is
  a Phase 2+ optimisation behind the same numerical contract.

**Trade-off:** slightly more VRAM for materialised K/V during decode (64×256×2
× context = same as latent storage, just reformatted) and a small extra GEMM
per token per DSA layer. Acceptable for determinism.

### ADR-3: Router mirroring — **piggyback on TP control** (rank-0 mirror)

**Choice:** compute MoE top-8 expert IDs on rank 0 only; ship IDs (and scores)
to rank 1 via the existing TP control channel — the same mechanism that carries
`request.drafts` / `batch_drafts` for MTP width.

**Rationale:**
- Expert selection is **discrete**: score gaps ~1e-5 can flip the 8th expert.
  Ranks that select independently will diverge → different shapes → collective
  size mismatch → QP poison.
- The TP control channel already handles rank-0→rank-1 mirroring for draft counts.
  Extending it to expert IDs is one more field in the same message — zero new
  collectives, zero new code paths for the communicator.
- An all-reduce of IDs would work but adds a collective per MoE layer per token
  (43 MoE layers × decode tokens) — high frequency, and the IDs are already
  deterministic once rank 0 computes them.

**Trade-off:** rank 1 cannot start MoE until it receives the ID list. This is
already true for batch planning / draft width; the latency is sub-microsecond
on the same RDMA link used for activations. Acceptable.

### ADR-4: MTP under TP — **fixed width from rank 0** (no adaptivity)

**Choice:** MTP draft count is forced by rank 0 (same as `request.drafts` in
Flash-Next TP2). No adaptive drafting under TP.

**Rationale:**
- Adaptive MTP means each rank independently decides how many draft tokens to
  produce — a classic desync (one rank issues more MTP collectives than the other).
- Flash-Next already solved this: forced draft width from rank 0, mirrored in the
  batch plan. GLM NextN MTP (1 layer) is even simpler than Flash-Next’s multi-draft.
- Under TP, determinism > speed. Adaptive drafting is a single-host optimisation.

**Trade-off:** slightly suboptimal draft acceptance rate vs single-host adaptive.
Acceptable — the cost of a desync (full QP teardown) dwarfs the gain.

### ADR-5: mHC — **Sinkhorn is per-forward, identical both ranks** (confirmed)

**Choice:** run the 20-iteration Sinkhorn on the 4×4 mixing matrix **per layer
per token**, inside the forward graph. Both ranks must execute identical Sinkhorn
math on identical inputs (weights are replicated).

**Evidence:** Maya’s GLM5-FLASH study (pinned against vendored llama.cpp
`glm5-next.cpp` @ `272aad8b`) confirms: *“mHC Sinkhorn normalization runs per
forward, inside the graph”* — data-dependent (softmax over dst, +eps, then
alternating src/dst normalisations on a 4×4 matrix). Must be computed per token,
not at pack time. The `hyper_connection.sinkhorn_iterations = 20` metadata key
exists because inference needs it.

**Rationale:**
- The Sinkhorn is 16 numbers × 20 iterations — trivially cheap but **data-dependent**.
  Pack-time normalisation would lose the manifold constraint.
- Both ranks share identical weights (replicated) and identical inputs (post
  collective-reduced hidden state) → Sinkhorn output is bit-identical. No extra
  collective needed.
- Must be inside the captured graph (for snapshots/replays) and in the F32 oracle
  (`ref/glm.py` parity harness).
