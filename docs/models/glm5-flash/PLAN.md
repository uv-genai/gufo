# GLM-5.3-Flash on gufo: single-host Strix Halo and TP2 — design study

**Status:** analysis and tentative plan only. **No code.**
**Sources:** [mw00/project-maya](https://github.com/mw00/project-maya) (local clone `/tmp/project-maya`, 2026-10-27), especially `docs/GLM5-FLASH.md`, `docs/AMD_*.md`, `docs/QUANT-PLAN.md`, `src/core/glm_*`; gufo tree (`src/models/qwen38_flash_next/`, `docs/MERGE-CHECKLIST.md`, `docs/models/qwen3.8-flash-next/TP2.md`).
**Author intent:** run **GLM-5.3-Flash** (the model Maya calls GLM-5.3, `glm5-next`) on **one Strix Halo** (as Maya already does) and on **two Strix Halos with gufo TP2**.

---

## 1. Executive summary

| Question | Answer |
| --- | --- |
| What is GLM-5.3-Flash? | 321 B MoE / **18 B active**, 45 layers, hidden 4096; **KDA** linear-attention (34) + **NoPE-MLA + DSA** full attention (11); **mHC** residual streams; 288×top-8 experts; **1 M** native context; one NextN **MTP** layer. |
| Does Maya already run it on Strix Halo? | **Yes** — HIP (`gfx1151`), single host, expert tiers (VRAM / RAM / SSD), ~**17.8 tok/s decode**, ~**210 tok/s prefill** (Maya measurements). |
| Can gufo do it single-host? | **Yes, in principle.** The architecture is the same *family* as Flash-Next (3:1 hybrid, HC, MoE, MTP). Work is a **new model runtime** (`src/models/glm5_next/`), not a small patch — largest pieces: KDA mixer, MLA+DSA attention, sigmoid router, dense-lead layers. |
| Can gufo do **TP2**? | **Yes, and gufo’s TP2 is a better fit than Maya’s multi-GPU.** Maya only does **layer split** (host residual hop, per-GPU expert caches). gufo TP2 already has **expert-FF column split + mirrored collectives + rank-0 draft/plan mirroring**. Two 128 GB APUs can hold Maya-S-sized experts **fully resident** with no offload — the one mode that is deterministic enough for identical-collective TP2. |
| Biggest gap vs Flash-Next | Mixers (GDN→**KDA**), attention (QSA→**MLA/DSA**), router (softmax→**sigmoid top-8**), hidden width (2560→**4096**), dense layers 0–2, **no PLE** table. |
| Biggest TP2 hazard | MoE **expert selection** is a discrete choice (not rounding). Ranks that pick different experts issue different shapes. Must **mirror selection from rank 0** (same policy as draft counts). |

**Rough verdict:** Treat this as a **new first-class model** on gufo (like `deepseek_v4_flash`), reusing gufo’s TP2 stack (`TpPartition`, `Communicator`, mirrored runner) and borrowing **math contracts and parity tests** from Maya’s GLM kernels/docs — not a port of Maya’s tiered engine.

---

## 2. What project-maya is (relevant facts)

Maya is a **single-architecture engine** for large MoE “Flash” hybrids (originally Qwen3.8-Flash-Next / `qwen4exp` lineage), extended with a **GLM-5.3-Flash** engine (`glm_*` in `src/core`, `src/kernels`). User-facing stack: GGUF quants, OpenAI/Anthropic APIs, up to 8 sequences, MTP drafting, expert offload.

### 2.1 GLM-5.3-Flash geometry (from Maya’s study + config)

| Piece | Value |
| --- | --- |
| Params | 321.3 B total / **18 B active** |
| Layers | 45, hidden **4096**; pattern **3 linear : 1 full attention** (full-attn layers 3,7,…,43 — **11** DSA) |
| Linear attn | **KDA** (Kimi Delta Attention): 64 heads × 128, conv kernel 4, **per-channel** decay gate `lb·sigmoid(−g·A)`, `gate_lower_bound = −5.0`, low-rank `f_a/f_b`, `g_a/g_b` |
| Full attn | **MLA NoPE** (q_lora 1536, kv_lora **512**, qk_nope 256, v 256, **no RoPE**) + **DSA** indexer (32×128, top-k **2048**, `index_kpool` 4) |
| KV | Latent `c_kv` 512 floats/token × 11 layers ≈ **11.3 KB/token** (half of Flash-Next-class KV) |
| Residual | **mHC**: `hc_mult=4`, **Sinkhorn-normalized mixing, 20 iters, per forward per token** |
| MoE | **288 routed + 1 shared**, top-**8**, inter **2048**, **sigmoid** scoring + **noaux_tc** bias on selection + renorm + **×2.5** scaling; router F32; layers **0–2 dense** (inter 12288), 3–44 sparse |
| MTP | 1 NextN layer (not DSpark) |
| Vision | 24-block ViT, 448 px, out 4096 |
| Vocab / context | 154,880 / **1,048,576** |
| Native weights | FP8 E4M3 (high-precision norms/router/embeds) |

**Routed experts are ~95% of parameters** (~304 B).

### 2.2 Maya runtime (single host)

- **Weights:** GGUF; Maya quants on HF (`peasantsmith/GLM-5.3-Flash-Maya-GGUF`): **Maya-S 96.5 GB** (IQ2 class experts), S24 94.7 GB, M 116 GB, L 156 GB. Standard **ggml IQ/K types** — not a private bit format.
- **Expert residency tiers:** VRAM cache → pinned RAM → **mmap SSD**, with **prefetch of routed blobs** at layer start and **no random-access mmap hints** (measured disaster: 1.93 GB/s).
- **Decode:** fused one-token HIP fast path (`glm_fast_*`); prefill chunked MMQ/hipBLASLt.
- **MTP:** NextN drafts + verify window with **KDA state rollback** on reject.
- **AMD:** `MAYA_ENABLE_HIP`, targets include **gfx1151**; custom HIP expert decode kernels; **complete-pool** mode when all experts fit (Skips disk/PCIe — deterministic).
- **Measured Strix Halo (128 GB):** ~210 tok/s prefill (3–4K prompts), **~17.8 tok/s decode** with experts in GPU pool. Gorgon Halo 192 GB + MTP: ~28–30 tok/s answers.
- **Multi-GPU:** **layer split only** (64 KB host hop per boundary). **No tensor parallel.**

### 2.3 What Maya is *not* (for our purposes)

- No mirrored collective sequence, no rank-0 plan mirroring, no TP2.
- Expert tiering is **dynamic per rank** (cache hit/miss, CPU lanes) — incompatible with gufo’s “same verbs on both ranks” rule unless replaced.
- Serve stack / quant recipes are Maya-specific; **GGUF tensors + kernels math** are what we want.

---

## 3. GLM vs gufo Flash-Next (what must be new)

Flash-Next in gufo: hidden **2560**, 48 layers, GDN linear + QSA (GQA+RoPE+indexer), HC 4 streams + 320 bottleneck, PLE/n-gram, MoE 512×top-10×640, MTP, TP2 expert-FF split.

| Area | Flash-Next (gufo today) | GLM-5.3-Flash | Effort |
| --- | --- | --- | --- |
| Layer cadence | 3:1 hybrid | same 3:1 | small (reuse pattern) |
| Linear mixer | **GDN** (per-head scalar gates) | **KDA** per-channel decay + lower bound | **medium** (kernels + oracle) |
| Full attention | QSA GQA + **RoPE** + block indexer | **MLA NoPE** + DSA lightning indexer (topk 2048) | **large** |
| KV cache | per-head K/V pools | **512-d latent** (cheaper long context) | medium (new layout; win at 256K–1M) |
| Residual | HC 4 streams | **mHC + Sinkhorn 20 iters per token** | small–medium (parity!) |
| MoE | 512 experts, softmax top-10 | **288 experts, sigmoid top-8**, scale 2.5, noaux_tc | medium (selection semantics) |
| Dense layers | all MoE | **layers 0–2 dense** | small |
| PLE / n-gram | yes (28.8 GB table) | **none** | drop path |
| MTP | shared Q8_0 sidecar | model’s own NextN | medium (geometry/names) |
| Hidden / FF | 2560 / 640 | **4096 / 2048** (dense 12288) | plumbing |
| Context | 262K native (+ YaRN 512K) | **1M native** | memory/KV streaming |
| Vision | Qwen mmproj | GLM ViT mmproj | separate |

Maya’s own study’s conclusion (GLM5-FLASH.md): *“about two thirds of the engine’s machinery applies unchanged; the remaining third is the mixer (KDA vs GDN), the full-attention mixer (MLA vs plain QKV), the router scoring, and the packaging.”*

---

## 4. Tentative design on gufo

### 4.1 Product shape

1. **`gufo serve llm` GLM-5.3-Flash, single host** (Strix Halo `gfx1151`, Linux).
2. **TP2 = two Strix Halos**, same launcher/llama-swap style as Flash-Next (`tools/tp2/`, `tools/llama-swap/tp2.sh`).
3. **Weights:** load **GGUF** (Maya-S / unsloth / future gufo-named quants). Do **not** invent a new pack unless column-split alignment demands it.

### 4.2 Runtime layout (new tree)

```text
src/models/glm5_next/           # parallel to qwen38_flash_next
  engine.{hpp,cpp}              # Model/Session, decode/prefill policy
  config.hpp                    # Glm5Geometry (45, 4096, 34/11, KDA/DSA/mHC/MoE)
  weights.{hpp,cpp}             # GGUF tensor map, high-precision lists
  kernels/rocm/                 # HIP: kda.*, mla_dsa.*, mhc.*, ffn_moe.*, mtp.*
  distributed/tp_partition.*    # extend/reuse expert-FF column split
  chat_template.*, tokenizer wiring (glm5 BPE / gguf tokenizer)
src/cli/serve/...               # runner adapter like DeepSeekTextRunner / QwenFlashNext
```

**Reuse from Flash-Next TP2 almost verbatim:**

- `rocm::Communicator` / `TwoRankAllReduce` / split-reduce (prefill pair overlap)
- `TpMirroredRunner` (digest, `kEnd` agreement, greedy argmax check)
- `TpPartition`: **column-split every expert’s intermediate dim** (`expert_ff % world == 0`); each rank computes all experts on its share; **MoE exchange sums shares** (same launch set both ranks)
- Rank-0 **forced draft width** / batch plan
- `UsesComponentCache() == false` under TP
- Disk/snapshot continuation (Flash-Next style), not Maya tiers

**Do not reuse Maya’s** doorbell + CPU expert pool + mmap tiers for TP2. For **single-host optional** later, a “complete-pool”-only path is enough for v1 (everything resident in UMA).

### 4.3 Memory budget (the good news)

| Item | Approx (Maya-S class, Q2/IQ2 experts) |
| --- | --- |
| All experts (288 × 42 × 3 × 4096 × 2048 @ ~2 bpw) | **~86–96 GB** full |
| gufo TP2 **column split** | **~43–48 GB per rank** |
| Dense lead + attention + embeds + MTP | ~4–10 GB per rank |
| KV: 11 × 512 × 2 B × context | **11.3 KB/token** → 256K ≈ **2.9 GB**; 1M ≈ **11.3 GB** |
| Workspace / prefill scratch | ~2–6 GB |

**Per 128 GB Strix Halo rank:** ~45 GB experts + ~5 GB dense + KV + scratch → **fits at 256K–512K context with room**; **1M** is tight but plausible with compact KV and `sessions=1–2`.

**Single host:** 96 GB experts + dense + KV does **not** fit all experts + large KV on 128 GB at Maya-S; options: smaller quant (Maya-S24 / XS), short context, or **Maya-style tiering** (out of scope for v1 TP2). Prefer **TP2 pair for long context / full quality**; single-host = short context or XS-class weights.

This is the structural argument for TP2 here: **two APUs turn “offload” into “replication with column split”**, which gufo’s collective lock already assumes.

### 4.4 TP2 collective design (tentative)

Invariant (from `TP2.md`): **same verbs all-reduces: order, count, bytes** on both ranks.

| Op | GLM piece | Split / mirror |
| --- | --- | --- |
| MoE FFN | gate/up cols, down rows | `TpPartition` on inter 2048 → 1024/rank; **all-reduce hidden** after each MoE (same as Flash-Next) |
| Shared expert / dense 0–2 | FF inter | same column split (dense 12288 / world) |
| KDA | heads 64 × 128 | **full replica** or split heads (see risks); state must match per rank |
| MLA + indexer | 64 heads | head-split or replica + AR on output; latent KV **replicated per rank** (same tokens) |
| mHC | 4 streams | deterministic Sinkhorn on both ranks (identical math); no extra collectives if weights replicated |
| Router | top-8 ids | **Compute on rank 0 only; mirror ids + scores** (like `request.drafts`) |
| MTP drafts | NextN | rank-0 plan / forced width (existing rule) |

**New hard rule to add to the hazard table:**  
*“MoE expert selection (top-k ids) is discrete — mirror from rank 0; never let each rank’s router re-select independently.”*  
(Measured: score gaps ~1e-5 flip the 8th expert — from Maya notes.)

**KDA state:** per-token recurrence must be **bit-identical** across ranks (or split by heads with identical math). Disable any “skip empty experts / grouped expert shortcuts” under TP.

### 4.5 Single-host Strix Halo (as Maya does)

- v1: **complete-pool** (all experts in UMA) + MTP — deterministic, simplest.
- v2 (optional): Maya-like RAM/SSD tiering behind a **single-host-only** gate (`TpWorldSize()==1`), never under TP2.
- Quant choice: **Maya-S24 / XS** for 128 GB with long context; **Maya-S/M** for quality at ≤64–128K.

---

## 5. Phased plan (no dates; order by dependency)

### Phase 0 — Paper + contracts (this doc)
- [x] Architecture extract from Maya
- [ ] Convert/inspect a real Maya-S GGUF tensor list (names, types, block alignment for column split)
- [ ] Decide: absorb/expand MLA first (Maya study recommends **expanded KV first**, absorbed later)
- [ ] Confirm mHC Sinkhorn is per-forward (Maya: yes) vs pack-time

### Phase 1 — Single-host correctness (`glm5_next` skeleton)
1. GGUF load + geometry + tokenizer/chat template (`glm5-next` metadata keys).
2. **KDA** prefill/decode + F32 oracle + parity tests (Maya: `glm_*_parity`, `ref/glm.py`).
3. **mHC** (Sinkhorn 20) + oracle.
4. Dense layers 0–2 + MoE **sigmoid top-8** (selection parity test, not just GEMM).
5. **MLA expanded-KV** path first; then absorbed MLA.
6. **DSA indexer** (topk 2048, k-pool 4).
7. NextN **MTP** (fixed draft count policy — no adaptive yet).
8. Serve via existing HTTP; smoke suite per `MERGE-CHECKLIST` style.

**Exit:** greedy decode matches a reference (Maya slow path / llama.cpp glm5) on short prompts; KL/top-1 gates on longer ones.

### Phase 2 — Memory + scale on one APU
- Fit Maya-S24/XS; measure pp/tg on gfx1151; KV streaming for >64K if needed.
- Optional expert tiering (single-host only).

### Phase 3 — TP2
1. Generalize `TpPartition` / `PlanRoutedBytes` to hidden 4096 / inter 2048 / dense 12288 (quant-block boundaries).
2. Wire `rocm::Communicator` split-reduce for MoE (and any head-split AR).
3. **Mirror router ids + MTP width from rank 0**; digest checks (`kEnd`).
4. Same-binary same-sha rule; no `UsesComponentCache` under TP.
5. Two-host smoke: PONG, >1024 tokens, agent prompt, ≥N session reuse, logs free of `tp_pair_lost` / size mismatch.
6. Long-context (256K+) quality pass; only then performance.

### Phase 4 — Productization
- Launchers `rank0-glm-q4.sh` style or one variant; llama-swap ids; README/BENCHMARKS; user llama-swap validation (process we already run).
- Optional: DFlash2 drafter, vision mmproj, 1M context qualification.

---

## 6. Risks and mitigations

| Risk | Why | Mitigation |
| --- | --- | --- |
| **Expert selection divergence** | Top-k ids are discrete | Rank-0 mirror; assert equal ids in digest |
| **KDA / mHC state drift** | Recurrence + Sinkhorn per token | Single math path both ranks; oracle tests; no rank-local skips |
| **Attention implementation size** | MLA+DSA vs QSA+RoPE | Phase 1 expanded-KV then absorbed; reuse Maya parity harnesses |
| **Quant block vs column split** | TP2 split without re-encode | Extend `PlanRoutedBytes` checks; choose quants with even inter/block (Maya-S experts at 2048 inter are designed for this) |
| **1M context memory** | Even 11 KB/token grows | `sessions` 1–2; KV streaming; YaRN not needed (native 1M) |
| **Single-host 128 GB too small for Maya-S + KV** | 96 GB experts + KV | Ship TP2 as primary; single-host XS/S24 |
| **Maya offload not ported** | Breaks collective lock | Explicit non-goal for TP2; optional later single-host-only |
| **License** | GLM MIT; MiniMax is not (if someone later wants M3.1) | GLM fine; document MiniMax license separately if ever needed |
| **Upstream/gufo churn** | Large new model vs hot Flash-Next tree | Isolate `src/models/glm5_next/`; thin serve adapter |

---

## 7. Comparison: “just use Maya” vs “native gufo”

| | Maya single host | gufo `glm5_next` 1-host | gufo TP2 |
| --- | --- | --- | --- |
| Exists today | **yes** | no | no |
| Strix Halo numbers | ~18 tg / ~210 pp (Maya-S) | unknown | unknown |
| 2×128 GB | layer split only | n/a | **column-split TP2** |
| Long context | 1M + streaming | plan 1M | **best memory story** |
| Deterministic collectives | no | n/a | **yes (goal)** |
| Agent/llama-swap process | Maya’s stack | gufo serve | **our existing TP2 process** |

**Recommendation:**  
- If the goal were only “GLM on one APU today,” **use Maya as-is**.  
- For **gufo + two-host TP2** (agents, DeepSWE-style load, 256K+ context, mirrored correctness), implement **`glm5_next` on gufo** as in Phases 1–3, **copying math contracts from Maya**, not its expert-tier engine.

---

## 8. Open questions (decide before Phase 1 code)

1. **Weight source of truth:** Maya-S GGUF vs unsloth dynamic quants vs new gufo-calibrated quants (Maya recipe is in-repo `tools/maya_quant/`).
2. **KDA/TP2 split unit:** full replica (simpler, more VRAM) vs head-split (need AR — more collectives).
3. **MLA:** expanded-KV first (compat with existing attention-shaped code) vs absorbed from day one.
4. **Router mirroring protocol:** piggyback on TP control like `request.drafts` (preferred) vs extra all-reduce of ids.
5. **MTP:** disable adaptivity under TP (hard `drafts` from rank 0) — confirm with Flash-Next practice.
6. **Whether MiniMax-M3.1 is in scope later** (separate engine in Maya; different license).

---

## 9. References

- Project Maya: https://github.com/mw00/project-maya · GLM quants: HF `peasantsmith/GLM-5.3-Flash-Maya-GGUF`
- Maya docs: `docs/GLM5-FLASH.md` (gap study), `docs/AMD_MAYA.md`, `docs/AMD_EXPERT_KERNELS.md`, `docs/QUANT-PLAN.md`, `docs/GLM-KDA-GRAPHS.md`
- Upstream model card: https://huggingface.co/zai-org/GLM-5.3-Flash
- gufo: `docs/models/qwen3.8-flash-next/TP2.md`, `docs/MERGE-CHECKLIST.md`, `src/models/qwen38_flash_next/distributed/tp_partition.hpp`
