# DeepSeek V4.1 Flash — support plan

Status: **planned, not implemented.** Gufo today runs **DeepSeek V4 Flash**
(single host, no tensor parallel, no weight streaming). This document records
the capacity analysis and the intended design for **V4.1 Flash** on one or two
Strix Halo (`gfx1151`) hosts, including **NVMe weight streaming** and how it
would meet the existing **TP2 / RDMA** stack.

Related: [V4 Flash](../deepseek-v4-flash/README.md) ·
[Flash-Next TP2](../qwen3.8-flash-next/TP2.md) ·
[Models index](../README.md)

---

## 1. Model facts (official)

| | |
| --- | --- |
| Hub | [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) |
| Scale | **763B** total · **552B** backbone · **~196B** Engram · vision encoder |
| Active per token | **~8B** prefill · **~16B** decode |
| Architecture | Causal Encoder-Decoder (20 enc + 20 dec), CSA2 sparse attention, Engram memory |
| MoE | **384** routed experts, **6** active + **1** shared |
| Context | up to **1M** tokens; FP4 KV ~**890 B/token** |
| Official weights | **FP8** (safetensors); community GGUFs exist |

V4.1 is **not** a drop-in for `src/models/deepseek_v4_flash/` (V4 Flash
0731). It needs a **new model package** and encoding path.

---

## 2. Capacity on two Strix Halo hosts

| Representation | Approx. weight size | Fits 2×128 GiB resident? |
| --- | ---: | --- |
| FP8 (official) | ~763 GB | No |
| 4-bit | ~380 GB | No |
| 2-bit | ~190 GiB | At/over budget after KV, vision, RDMA, OS |

Two hosts give roughly **~240 GiB** usable unified memory together. **Resident
TP2 cannot hold V4.1.** The viable two-host strategy without more machines is
**streaming**: keep a hot working set in RAM/GPU and the rest on NVMe.

Low activated parameters reduce **compute**, not **weight residency**.

---

## 3. Streaming design

### 3.1 Placement tiers

| Tier | Contents |
| --- | --- |
| **T0 Device** | Dense/attention/embeddings (or TP shards), hot expert cache, active KV/CSA2 |
| **T1 Host pin** | IO staging, completion rings |
| **T2 Page cache** | Recently read expert/Engram blocks (OS) |
| **T3 NVMe** | Full quantized GGUF / expert-major shards (per host, local disk) |

T0+T1 budget target: about **40–70 GiB** per host, leaving room for KV and the OS.

### 3.2 Resident vs streamed

| Component | Policy |
| --- | --- |
| Embeddings, attention/CSA2, norms, routers | Resident |
| Shared expert | Resident (or pinned) |
| Routed experts (384) | **Stream** (LRU device cache) |
| Engram (~196B) | **Stream by page** (sparse) |
| Vision encoder | Optional; omit for text-only first cut |
| KV / CSA2 index | Resident budget; host managed memory if tight (not per-token NVMe) |

### 3.3 On-disk layout

Prefer **expert-major** blobs so a miss is one contiguous read:

```
model.gguf
  dense/
  experts/e{layer}.{expert}.bin   # or packed groups
  engram/page_{n}.bin
  index.json                      # offsets, quant block size, checksums
```

Align blobs to quant blocks and NVMe-friendly sizes (4 KiB pages; 64–256 KiB
optional read size). For TP, package each expert as the rank’s
`[ff_begin:ff_count]` slice so each host stores/loads **half** the expert bytes.

### 3.4 Device expert cache

- Slots: `{layer, expert_id, LRU tick, device buffer, Free|Loading|Ready}`.
- Capacity: configurable, e.g. **8–32 GiB**.
- API: `Prefetch`, `WaitReady`, `Touch`, eviction on miss.
- Inference weights are read-only (no writeback).

### 3.5 I/O pipeline

1. Router selects top‑k experts.  
2. Async prefetch: NVMe / page cache → pinned staging → H2D on a **side stream**.  
3. Overlap: prefetch layer *L+1* while layer *L* runs attention / all-reduce.  
4. `WaitReady` then MoE GEMM.  
5. Watchdog on `WaitReady`: fail the request with a clear error rather than
   leaving a TP peer blocked in a collective.

Double-buffer staging; do not serialize H2D on the same stream as compute
without events.

### 3.6 Cache policy

| Policy | Choice |
| --- | --- |
| Replacement | LRU (or decaying LFU) |
| Pin set | Shared expert + current batch’s 6 experts |
| Prefetch | Router top-k before GEMM; optional next-layer lookahead after measurement |
| Failure | Missing/corrupt blob → request error |

---

## 4. TP2 / RDMA integration

Reuse Flash-Next TP2: world size **1 or 2**, mirrored rank 1, verbs all-reduce
with **3×48 MiB** windows, same binary/sha on both hosts, component cache off
under TP.

### 4.1 Compute TP (keep Qwen’s rule)

Every rank runs the **same** selected experts on a **partition of the FF
intermediate**; MoE **all-reduce** sums partials. Ranks must not choose
different expert sets (desyncs collectives and streaming).

### 4.2 Storage TP (recommended)

| Mode | Each rank streams | Notes |
| --- | --- | --- |
| Mirror | Full expert | Wastes NVMe and hot RAM |
| **FF shard (recommended)** | **Local `[ff_begin:ff_count]` only** | Same all-reduce shape as Flash-Next |
| Expert-ID shard | Disjoint experts | Requires **all-to-all** of activations — **out of scope** (not in verbs stack) |

Do **not** shard expert IDs across ranks until an all-to-all transport exists.

### 4.3 Steady-state schedule

```
dense/attention (resident)  ∥  prefetch experts L+1
MoE GEMM on ready slots (L)
verbs all-reduce partial
residual combine
```

Expert loads are **not** all-reduce payloads; only FF partials use the RDMA
windows. Prefill batch size still must fit `MaxPartialBytes` (re-derive for
V4.1 hidden size).

### 4.4 Control plane

- Extend mirrored instructions as needed; expert IDs can be derived from
  identical routers on both ranks or shipped in the same control message.
- Device expert cache is **ephemeral**: snapshots do **not** include slots;
  restore starts cold.
- Disk **KV** continuation cache remains separate from NVMe **weight** streaming.

### 4.5 Failure modes

| Symptom | Likely cause |
| --- | --- |
| `verbs all-reduce timed out` | Peer stuck in `WaitReady` (disk) or rank desync |
| Empty completion | Rank0-only path / mirror bug (same class as historical TP2 bugs) |
| tok/s collapse when cold | Cache too small or no prefetch |
| OOM | T0 + KV over budget |

---

## 5. Phases

| Phase | Deliverable |
| --- | --- |
| **0** | Measure NVMe on both hosts; pin V4.1 GGUF revision, quant, tensor map |
| **1** | Expert/page store + device cache + standalone bench |
| **2** | Single-host V4.1 text path: resident dense + streamed experts |
| **3** | TP FF shard + stream local slices; verbs + mirrored serve |
| **4** | Engram pages; optional vision; optional DSpark |
| **5** | Qualification: cold/warm, sessions ≥ 5, generation text, tags |

**Non-goals (early phases):** more than two ranks, expert-ID all-to-all,
training/fine-tuning, V4 Pro–class models.

---

## 6. Verification (when implementing)

1. Expert fetch correctness vs file checksums.  
2. Cold start (empty page cache) still completes.  
3. Warm steady-state hit rate and tok/s.  
4. Greedy text vs offline reference on the same quant.  
5. TP2: matching binary shas; generation text; prompt > 1024 tokens; ≥ 5
   requests at `--sessions 4`; no verbs timeout.  
6. Simulated NVMe failure → clean error, peer not hung.

---

## 7. Implementation map (future)

| Area | Location |
| --- | --- |
| New model package | `src/models/deepseek_v41_flash/` |
| Expert/page store | model-local `stream/` or shared `src/core/` |
| TP partition pattern | `src/models/qwen38_flash_next/distributed/tp_partition.*` |
| Verbs / windows | `.../kernels/rocm/verbs.cpp` |
| Mirrored serve | `src/cli/serve/tp_{executor,control}.*` |
| V4 Flash (unchanged) | `src/models/deepseek_v4_flash/` |

This folder (`docs/models/deepseek-v41-flash/`) holds the plan until a
runtime and `README.md` model card exist.

---

## 8. Summary

1. V4.1 Flash does **not** fit two 128 GiB hosts as a **resident** model.  
2. **NVMe streaming** of routed experts (and Engram), with dense resident, is
   the capacity plan.  
3. **TP2:** keep compute TP (same experts, FF shard + all-reduce); add storage
   TP (stream only the local FF slice). Avoid expert-ID partitioning.  
4. Reuse verbs all-reduce and mirrored serve; add an expert store, LRU cache,
   and a WaitReady watchdog.  
5. Start with NVMe measurements and a single-host streaming path before any
   two-host work.
