# RDMA (TP=2) forward-port record

Merging `uv-genai/gufo` TP=2 RDMA support (based on upstream 0.9.1) into
`gufo-org/gufo` main (0.12.0). Executed on branch `uv-rdma`.

Source: `uv-genai/fix/tp2-mirror-decode-draft-counts` (`85646548`) — the fork's
`rdma` branch plus the TP2 decode draft-count mirroring fix and local build docs.
Common ancestor: `b2438a4a`. Main was +53 commits ahead; the fork +122.

Result: **129 files, +46,459 / -587**. Six files conflicted; all resolved by hand.

## What RDMA support is

Two-rank (TP=2) tensor-parallel serving over native InfiniBand or RoCE v2.
Rank 0 owns HTTP and the scheduler; rank 1 is a worker with no public port. Both
ranks run the same model calls in the same order; mixer and MoE output
projections produce per-rank partials that an RDMA all-reduce sums, so both
ranks hold identical results.

Gated behind the CMake option **`GUFO_ENABLE_TP2_RDMA`** (off by default), which
adds the compile definition of the same name and links `rdma-core`/`libibverbs`.
Run with `--tp-*` options; see `docs/models/qwen3.8-flash-next/TP2.md`.

### Layer stack

**Transport (model layer)**

| File | Role |
|---|---|
| `kernels/rocm/communicator.hpp` | Abstract `Communicator`: `ExchangePartial`, `StartPartial`/`FinishPartial`, `QueuePartial`/`CheckQueued`, `BeginOperation`/`EndOperation` collective scope |
| `kernels/rocm/verbs.{hpp,cpp}` | RC QP over InfiniBand/RoCE v2. RDMA write + immediate data, event-driven completion channel, TCP bootstrap retained for peer-loss detection. 3 x 48 MiB windows at fixed virtual addresses; wire magic `GUFT` v7, collective magic `GUCH` v5 |
| `distributed/rdma_gid.{hpp,cpp}` | RoCE v2 GID selection when `--tp-gid-index` is unset |
| `distributed/tp_partition.{hpp,cpp}` | Splits the routed-expert intermediate dimension evenly (rows of gate/up, columns of down); rejects splits that break quantization-block boundaries |

**Head geometry.** `MixerHeads` (`kernels/rocm/device_model.hpp`) carries
*per-rank* head counts (`ssm_k`, `ssm_v`, `ssm_dim`, `attn`, `attn_kv`,
`attn_dim`) with `All(Config)`, `SsmConvChannels()`, `AttentionQDim()`,
`AttentionKvDim()`. `DeviceLayer` gains `heads` and `mixer_split`. Compute paths
read per-rank counts; **scratch allocation deliberately keeps global sizing**, so
buffers are oversized under TP2 but never too small and keep stable addresses.

**Control plane (server layer)**

| File | Role |
|---|---|
| `src/cli/serve/tp_control.{hpp,cpp}` | Versioned framed TCP control channel. Handshake validates both ranks agree on model/MTP/context/sessions/mmproj/RAM-cache. `TpResponseBroker` routes responses and instruction acks |
| `src/cli/serve/tp_executor.{hpp,cpp}` | `TpMirroredRunner` (rank 0 decorator over `TextModelRunner`) + `TpExecutor` (rank 1). FNV-1a execution digest per request; rank 1 returns a verdict |
| `src/cli/serve/tp_disk_store.{hpp,cpp}` | Disk cache split across both hosts' local disks; rank 0's entry carries a `file_key` naming rank 1's half |

Correctness properties the design enforces:

- Every collective carries and validates its operation ordinal, so an exchange
  from one call can never pair with another call's.
- Cancellation acts only **between** model calls; a session aborting mid-layer
  would strand the peer inside an exchange.
- Rank 0 sends each instruction just before running it locally, so both ranks
  make identical calls in identical order.
- A request reusing another's live state before rank 1 judged the producer fails
  if the producer is rejected, and a rejected request's leftovers are never
  reused, so disagreement cannot spread through the cache.
- `kDecode` carries rank 0's sampler RNG state, pending draw and chosen draft
  count, so rank 1's cycles match width and decisions.

**Ops tooling:** `tools/systemd-user-service/` (Containerfile, `gufo.service`,
`gufo@.service`, `gufo-rank1@.service`, `head.env.example`).
**Bench tooling:** `tools/gufo/model_bench/tp2.py` and friends.

## Conflict resolutions

`git merge-tree` predicted 6 conflicted files / 16 hunks. All resolved.

| File | Resolution |
|---|---|
| `cmake/Checks.cmake` | Union of both test lists (`continuation_layout` + `tp_partition`/`rdma_gid`) |
| `docs/.../BENCHMARKS.md` | Fork's two-section structure (needed for the RDMA tables) with main's newer Oct-10 review evidence and dates |
| `tests/models/qwen38_flash_next/CMakeLists.txt` | Added `gufo_tp_control` to `session_test` links; kept main's `device_headroom_fault.cpp` in `snapshot_test` |
| `engine.cpp` | Union of includes; `DecodeStep` keeps the fork's `forced_drafts` parameter **and** main's `continuation_hooks_->BeforeExecution()` |
| `kernels/rocm/executor.cpp` | 7 hunks — see below |
| `kernels/rocm/kernels.hip.cpp` | 3 hunks — see below |

### executor.cpp

1. **`Reset()` zero-fill** — kept main's `zero` lambda (throws when
   `continuation_hooks_` is set) with the fork's per-rank sizing
   (`heads.SsmConvChannels()`, `heads.ssm_v * heads.ssm_dim`).
2. **Pinned n-gram buffer** — kept main's `hipHostMallocMapped |
   hipHostMallocCoherent` + `hipHostGetDevicePointer` zero-copy alias, but sized
   by `chunk` (`PrefillChunk()`) instead of `T` (`max_batch`). Required:
   `PrefillChunk() = split_reduce ? 2 * max_batch : max_batch`, so a paired
   prefill holds both lanes' rows; without TP2 it equals `T` and nothing changes.
3. **`Executor::Ple()`** — dropped the fork's explicit `hipMemcpyAsync` upload.
   With the coherent mapping, `ple_emb` is the device alias of `host_emb_`, so
   `emb_row` becomes a **device-side offset** (`s_.ple_emb + emb_row *
   PleEmbeddingDim()`). Only the lead lane calls `WaitPle` — one fetch covers both
   batches. `PairScratch` retains `ple_emb`, and both lanes only read, so there
   is no clobbering.
4. **`GatedDeltaNet` call** — per-rank `heads.ssm_k/ssm_v/ssm_dim` with main's
   new trailing args (`half_stride`, `out_q8_1 ? s_.gdn_q8_1 : nullptr`).
   Also made `value_dim` per-rank (`heads.SsmValueDim()`), matching `z_stride`.
5. **WMMA causal attention** — per-rank `heads.attn/attn_kv/attn_dim`. The
   fork's hardcoded `last=false` and main's `last` are **provably equivalent**
   here: short-circuit evaluation means `last` is false whenever that term runs.
   Kept `last`.
6. **Attention gate Q8_1 path** — main's full Q8_1 logic retained, with
   `q_dim = heads.AttentionQDim()` so the gate is sized per rank.
7. **Layer loop** — took the fork's slim loop body. Main's inline PLE + `half_only`
   + `HcMix` now lives inside the fork's `MixerPart`; re-applied the `half_only`
   computation there (`prefill_phase` is a thread_local, so it is in scope).

### kernels.hip.cpp

- Kept main's `MtpHiddenVec4Kernel`; adopted the fork's
  `AddRowsBroadcast`/`AddRowsBroadcastKernel` name, since the auto-merge had
  already updated **all** call sites (`batch.cpp`, `executor.cpp` x3,
  `kernels.hpp`) and no `MtpAddEmbedding` references remained.
- Preserved main's vectorised fast path by renaming
  `MtpAddEmbeddingVec4Kernel` → `AddRowsBroadcastVec4Kernel` and wiring it into
  `AddRowsBroadcast` (16-byte alignment + `hidden % 4` guard).
- Added the RDMA kernels `StagePartial` / `WaitValue` and their host wrappers.
- **`DenseF16GEMMKernel` parameter merge** — both sides added trailing params.
  Final order: `conv_channels`, `checkpoint_tokens`, `input_stride`. The strided
  `DenseF16Gemm` launch passed `attention, 0, input_stride` positionally and
  **had to gain an extra `0`**, otherwise `input_stride` would have landed in
  `checkpoint_tokens` — this compiles either way and silently corrupts results.

## Semantic gaps fixed (auto-merge would not catch these)

1. **`CountPromptTokens`** — new in main with a default implementation that calls
   `TextModelRunner::PreparePrompt` on *itself*. `TpMirroredRunner` did not
   override it, so under TP2 an image request would be sized by the text-only
   path, losing visual-token expansion (used at `inference_backend.cpp:3948`).
   **Fixed:** added a forwarding override to `inner_`.
2. **`continuation_hooks_`** — kept through the fork's rewritten `Reset()` and
   execution paths.
3. **TP-aware `AttentionQDim()` / `SsmValueDim()`** — audited; the remaining
   global-count uses are scratch sizing only, which is intentionally global.
4. **`DenseF16GEMMKernel` argument order** — see above.

## Verification on this host

Host: AMD Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`), 128 GB unified,
CachyOS kernel 7.2.8-2-cachyos, ROCm 7.2.3, rdma-core 65.

- `gufo diagnose` — all PASS.
- **TP2 RDMA build** (`gpu-tp2` preset): `gufo` links; 0 errors.
- **Default production build** (`release`, RDMA OFF): 0 errors — main's path
  unchanged.
- Tests passing: `tp_control_test`, `tp_executor_test`,
  `qwen38_flash_next.tp_partition`, `qwen38_flash_next.rdma_gid`,
  `qwen38_flash_next.ngram`, `qwen38_flash_next.mtp_ops`,
  `qwen38_flash_next.attention_ops`, `qwen38_flash_next.projection_ops`,
  `qwen38_flash_next.routed_wmma_ops`.
- Formatting: `tools/ci/check-format.py` introduces **no new** violations
  relative to main (the 4 files that still differ are pre-existing drift between
  local clang-format 23.1.1 and CI's pinned clang-tools).

Not yet run here: two-host TP2 end-to-end (needs both Strix Halos),
`tests/functional/tp2_constraints.py`, and matched single-host vs TP2 output
identity. A missing-model skip is not a quality pass — run these on the pair.

## Build note for CachyOS (GCC 16)

This host hits the fork's documented toolchain incompatibility: ROCm's
`hip/hip_runtime.h` defines `__noinline__` as a macro, but GCC 16's libstdc++
`<format>` uses it as an attribute name, so any TU including a HIP header before
`<chrono>`/`<format>` fails with *"expected an identifier for the attribute
name"*. This is **pre-existing on main**, not caused by this merge.

Workaround (see `BUILD-LOCAL.md`), applied to both CXX and HIP flags:

```sh
cmake --preset gpu-tp2 -DCMAKE_CXX_FLAGS="-include format" -DCMAKE_HIP_FLAGS="-include format"
```

These flags are deliberately **not** baked into CMake defaults: upstream targets a
pinned Nix/clang toolchain where they are unnecessary.
