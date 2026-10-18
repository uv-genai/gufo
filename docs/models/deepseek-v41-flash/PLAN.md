# DeepSeek V4.1 Flash — support plan

Status: **planned, not implemented.** Gufo today runs **DeepSeek V4 Flash**
(single host, resident, no TP). This plan covers **DeepSeek V4.1 Flash**
(**1017** checkpoints included): capacity on one and two Strix Halo
(`gfx1151`) hosts, **NVMe weight/Engram streaming**, and **TP2/RDMA**.

**Goals (always):**

1. **Single-host** V4.1 serving first.  
2. **TP2** (two-host RDMA) as a first-class path, not an afterthought.  
3. Start from **streaming** (Engram and/or experts on disk) because that is
   what makes the model fit.

Related: [V4 Flash](../deepseek-v4-flash/README.md) ·
[Flash-Next TP2](../qwen3.8-flash-next/TP2.md) ·
[Models index](../README.md) ·
**Reference engine: [ds4-dfm-rs](https://github.com/Baekpica/ds4-dfm-rs)**

---

## 1. Model facts (official)

| | |
| --- | --- |
| Hub | [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) · [1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) |
| Scale | **763B** total · **552B** backbone · **~196B** Engram · vision encoder |
| Active per token | **~8B** prefill · **~16B** decode |
| Architecture | Causal Encoder-Decoder (20+20), CSA2 sparse attention, Engram |
| MoE | **384** routed experts, **6** active + **1** shared |
| Context | up to **1M** tokens; FP4 KV ~**890 B/token** (~0.89 GB full 1M) |
| Official weights | **FP8** (hundreds of GB) |

V4.1 is **not** a drop-in for `src/models/deepseek_v4_flash/`. It needs a
**new model package** and encoding path.

---

## 2. Why FP8 / plain GGUF do not fit two hosts

| Representation | Approx. size | 2×128 GiB resident (~240 GiB usable) |
| --- | ---: | --- |
| Official FP8 | ~763 GB | No |
| Q4-class GGUF | ~380 GB | No |
| Q2 / IQ2-style (with Engram in-file) | ~150–360 GB | No or only with streaming |

**Resident-only TP2 is not enough.** The plan always pairs:

- **streaming** (bulk off GPU: Engram ± experts), and  
- **single-host** and **TP2** runtime modes.

---

## 3. Artifact survey (1017 and close)

Survey date: **2026-10-18**. Sizes from HF cards/trees; **measure** before
committing to a download.

| Artifact | Base / quant | Engram | Fits 1×128? | Fits 2×128? | Runtime notes |
| --- | --- | --- | --- | --- | --- |
| **[Baekpica 1017 VQ-GGUF](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-VQ-GGUF)** | **~113.6 GB** base (YoungAi `vq8sh14-q4k-mtpnative`) + ~1 GB vision | **~203 GB** (2×~101.5 GB), **read from SSD** (`--engram-dir`) | **Yes for weights** (~106 GiB loaded); Engram on disk | Yes (same or TP + half experts later) | **Primary near-term candidate.** Custom `vqblob` GGUF; **ds4-dfm-rs / YoungAi**, not stock llama.cpp |
| **[YoungAi V4.1 Flash](https://huggingface.co/wenzhouwu/YoungAi-DeepSeek-V4.1-Flash)** (and **-1017**) | Same recipe, **113.6 GB** base | Same on-disk tables | Same as Baekpica | Same | Reference recipe + `ds4-server`; ~317 GB download |
| **[mrfakename 1017 GGUF](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-GGUF)** | llama.cpp-style **Q2_K…Q8_0** (repo ~1.56 TB total) | Likely in-file / large shards | Depends on quant (Q2 may still be >100 GB + Engram) | Possible only if chosen quant+Engram ≤ budget **or** streamed | Stock **llama.cpp** path — good long-term for a gufo port shaped like other GGUFs |
| **[acrogenesis 1017 IQ2_XXS](https://huggingface.co/acrogenesis/DeepSeek-V4.1-Flash-1017-IQ2_XXS-GGUF)** | IQ2 experts + Q8 dense; tree **~359 GB** (shards incl. large Engram pieces) | In GGUF set | No if all resident | Only with **streaming** | llama.cpp-tagged; verify Engram vs base split before use |
| **[antirez V4.1 GGUF](https://huggingface.co/antirez/deepseek-v4.1-flash-gguf)** | Q2 **~366 GB**, Q4 **~519 GB** | In file | No | No without heavy streaming | ds4/Metal-oriented; not a two-host resident fit |
| Official FP8 / 4-bit community | ~380–763 GB | n/a | No | No | Out of scope for resident |

**Takeaway:** the **YoungAi VQ / Baekpica 1017** line is the only surveyed
path where **base weights** fit **one** 128 GiB host **and** the oversized
**Engram** is already designed to **live on NVMe**. That is the pattern to
productize first.

### 3.1 What the Baekpica/YoungAi card means (short)

- **“510 GB → ~114 GB base”** — aggressive **VQ** (~1.6 bit/weight on MoE)
  + q4_K on dense; quality is **lossy** (~69–79% same top-1 vs FP8).  
- **Engram `2 × ~101 GB` read in rows from local SSD** — streaming already.  
- **Resident ~106 GiB** — matches one Strix Halo with headroom if KV stays
  small (full 1M KV ~0.89 GB).  
- **Must use 1017 Engram tables with 1017 weights** — mismatch loads
  silently.  
- **Not vanilla llama.cpp** — needs YoungAi tools or `ds4-dfm-rs`.

### 3.2 Primary reference implementation: ds4-dfm-rs

The **working** engine for the VQ / 1017 artifact is **not** stock llama.cpp:

| | |
| --- | --- |
| **Repository** | **[github.com/Baekpica/ds4-dfm-rs](https://github.com/Baekpica/ds4-dfm-rs)** |
| **Model guide** | [`docs/deepseek41-flash.md`](https://github.com/Baekpica/ds4-dfm-rs/blob/main/docs/deepseek41-flash.md) |
| **Port plan / inventory** | `docs/deepseek41-port-plan.md` · `docs/deepseek41-port-inventory.md` |
| **Lineage** | Rust port of YoungAi/ds4 (JLFN PR #80; vision PR #83) |
| **Validated host** | NVIDIA **DGX Spark** (CUDA / GB10) — **not** ROCm/gfx1151 yet |

**Already implemented there (study; do not re-derive):**

| Piece | How (from their guide) |
| --- | --- |
| VQ expert decode | `ffn_exps_vq.blob` DQVL v3; 12/13-bit indices; `cuda/ds41_vq_*.cuh` |
| **Engram streaming** | Shards via `--engram-dir`; **`io_uring`** reads (`crates/ds4-core/src/uring.rs`) + cross-block prefetch |
| Resident base | ~113.5 GB GGUF; `--mem-budget-mb 110000` on 128 GB |
| MTP / DSpark | Embedded three-tower drafter; `--mtp-mode off\|auto\|on` |
| Vision | Optional `--vision-gguf` |
| APIs | OpenAI + Anthropic + Responses |
| Warm path | Sidecar + Engram held open (warm TTFT ~15 ms vs ~190 ms cold prelude) |

**Gufo strategy — reference only, never a dependency:**

> **`ds4-dfm-rs` will never be a gufo runtime dependency.**

It is a **study/oracle** target only. Gufo does not ship, link, subprocess, or
require it to build or serve.

| | |
| --- | --- |
| **Use ds4-dfm-rs for** | Behavioral oracle · Engram I/O contract (`--engram-dir`, `io_uring` prefetch) · VQ tensor/layout · serving shape · gates and recorded numbers |
| **Do not use it for** | Running on Strix Halo · production serving · ROCm/gfx1151 execution · a build or package dependency |
| **What gufo does instead** | Re-implement VQ decode + Engram store in **HIP/gfx1151** under `src/models/deepseek_v41_flash/`, plus **our TP2/RDMA** |

V4.1 there is **serial CUDA** (DGX Spark reference; other NVIDIA GPUs need
their own qualification). The tree also has inherited Metal/CPU paths; none
is ROCm. **TP2 / RDMA remains gufo-only** (single-GPU CUDA there).

Port **semantics** into gufo (HIP); do **not** depend on the Rust binary.

Secondary engines: [YoungAi](https://github.com/fodelf/YoungAi) (VQ recipe +
`ds4-server`); [antirez/ds4](https://github.com/antirez/ds4) (V4 ancestor,
not V4.1-complete).

---

## 4. Streaming design (start here)

### 4.1 Placement tiers

| Tier | Contents |
| --- | --- |
| **T0 Device** | Dense/attention/embeddings, hot expert cache, active KV/CSA2 |
| **T1 Host pin** | IO staging |
| **T2 Page cache** | Hot Engram/expert pages (OS) |
| **T3 NVMe** | Full GGUF / Engram shards (local SSD per host) |

**T0+T1 budget:** ~**40–70 GiB** for streaming mode; for the VQ artifact
much of T0 is the **~106 GiB base**, so **Engram must stay in T3** (as
designed on DGX Spark).

### 4.2 Two streaming flavors (both in plan)

| Flavor | What streams | When |
| --- | --- | --- |
| **A. Engram-only (v1)** | Engram n-gram tables from disk; base resident | **Start here** — matches Baekpica/YoungAi; simplest |
| **B. Expert streaming (v2)** | Routed experts ± Engram; dense resident | When base quant is still too big (plain Q2/IQ2) or TP needs more headroom |

### 4.3 Engram streaming (v1 — proven on ds4-dfm-rs)

- Tables on **local NVMe** (`--engram-dir`); match **ds4-dfm-rs**
  `io_uring` round reads + cross-block prefetch (`uring.rs`).  
- Double-buffer; prefetch next lookup while compute runs.  
- Wrong Engram revision vs weights = silent quality bug — **checksum** both
  (their guide records SHA-256 for 1017 shards).  
- Snapshot/session state does **not** include the whole table.

### 4.4 Expert cache (v2)

- LRU slots for active experts (6×40 layers); router top-k prefetch.  
- Side H2D stream; `WaitReady` **watchdog** (never hang TP on disk).  
- Optional **storage TP:** each rank holds/streams only its FF slice of each
  expert (quant-block aligned), same all-reduce as Flash-Next.

---

## 5. Single-host and TP2 (both required)

### 5.1 Single-host (first delivery)

| Mode | Behavior |
| --- | --- |
| **VQ / Baekpica-like** | Base resident; **Engram on NVMe**; one `gufo` process |
| **Large plain GGUF** | Resident dense + **expert stream** if needed |

No verbs, no mirror — same quality gates as other single-host models.

### 5.2 TP2 (always in scope)

Reuse [Flash-Next TP2](../qwen3.8-flash-next/TP2.md):

| Piece | Rule |
| --- | --- |
| World size | **1 or 2** |
| Compute TP | Every rank, **same** experts, **FF intermediate shard** + **all-reduce** |
| Storage | Prefer **stream local FF slice** (½ NVMe, ½ hot cache) |
| Not in v1 | Expert-**ID** partition (needs all-to-all) |
| Transport | Verbs **3×48 MiB** windows; re-derive max prefill for V4.1 hidden size |
| Mirror | Rank 1 = `TpExecutor`; **same binary/sha** both hosts |
| Disk wait | Never block a collective forever (timeout → request error) |

**Schedule:** dense/attention ∥ prefetch experts (or Engram) → MoE/GEMM →
all-reduce.

### 5.3 Failure modes

| Symptom | Likely cause |
| --- | --- |
| `verbs all-reduce timed out` | Peer stuck in disk `WaitReady`, or desync |
| Empty completion | Mirror/rank0-only bug (historical TP2 class) |
| Slow cold start | No Engram/expert prefetch |
| Silent bad text | Engram vs weight **revision mismatch** |

---

## 6. Phases

| Phase | Deliverable | GPU |
| --- | --- | --- |
| **0** | Read **[ds4-dfm-rs V4.1 guide](https://github.com/Baekpica/ds4-dfm-rs/blob/main/docs/deepseek41-flash.md)** + port plan; pin **Baekpica/YoungAi 1017 VQ**; NVMe + Engram layout/SHA audit | disk only |
| **1** | Engram store modeled on ds4-dfm-rs **`uring.rs` / `--engram-dir`** (I/O contract → HIP/ROCm) | idle machine |
| **2** | Single-host V4.1 text path: **resident base + Engram stream**, VQ decode; follow their tensor/serving contract | yes |
| **3** | Qualify vs ds4-dfm-rs Spark behavior / gates (not bit-exact across vendors) | yes |
| **4** | **TP2**: FF shard + mirrored serve; Engram on **both** NVMe; verbs all-reduce | two hosts |
| **5** | Expert streaming (v2) if a larger GGUF is needed; optional vision/DSpark | yes |
| **6** | Docs, `tools/tp2/` launchers, tags | — |

**Always ship:** single-host path **and** TP2 path before calling V4.1
“supported” in the index.

**Non-goals early:** >2 ranks, expert-ID all-to-all, training, V4 Pro.

---

## 7. Verification

1. Engram fetch correctness; **SHA** match to 1017 shards.  
2. Cold NVMe start still completes.  
3. Warm tok/s and hit rate.  
4. Greedy sample vs reference (same quant).  
5. TP2: matching shas; generation text; prompt > 1024 tokens; ≥5 requests
   at `--sessions 4`; no verbs timeout.  
6. Wrong Engram dir → **detect and refuse** (not silent load).

---

## 8. Implementation map (future)

| Area | Location |
| --- | --- |
| New model package | `src/models/deepseek_v41_flash/` |
| Engram / expert store | `stream/` — I/O contract from ds4-dfm-rs `crates/ds4-core/src/uring.rs` |
| VQ decode | Port semantics from ds4-dfm-rs `cuda/ds41_vq_*.cuh` → HIP (no Rust link) |
| TP partition pattern | `src/models/qwen38_flash_next/distributed/tp_partition.*` |
| Verbs / mirror | `verbs.cpp`, `src/cli/serve/tp_{executor,control}.*` |
| Unchanged | `src/models/deepseek_v4_flash/` (V4 Flash) |
| **Reference engine** | **https://github.com/Baekpica/ds4-dfm-rs** |
| Recipe / weights | YoungAi `vq8sh14-q4k-mtpnative` · Baekpica 1017 VQ-GGUF |

ds4-dfm-rs is a **required Phase 0 read**, not an optional side note: it is
the only full working V4.1 Flash stack published for the VQ artifact.

---

## 9. Summary

1. Official FP8 and large plain GGUFs **do not fit** two Strix Halos
   resident; **streaming is mandatory** for those.  
2. **YoungAi/Baekpica 1017 VQ (~114 GB base + Engram on SSD)** fits **one**
   host for weights and already streams Engram — **primary v1 target**.  
3. **Working reference implementation:**
   **[Baekpica/ds4-dfm-rs](https://github.com/Baekpica/ds4-dfm-rs)** —
   full VQ + Engram `io_uring` + MTP + vision on DGX Spark. Gufo studies this
   and re-implements in HIP; **TP2 remains gufo-only**.  
4. Deliver **single-host** first, then **TP2** (FF-shard all-reduce +
   optional expert storage TP); Engram stream (v1), expert stream (v2).  
5. Start at **Phase 0** (read ds4-dfm-rs, pin artifact, measure NVMe) when
   the machine is free; no GPU work until then.
