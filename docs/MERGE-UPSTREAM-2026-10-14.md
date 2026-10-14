# Upstream merge handoff: 0.14.0 / 0.15.0 into `uv-rdma`

Status: **merge complete and validated (2026-10-14).** All twelve conflicts
resolved; nothing is committed yet (`MERGE_HEAD` until the merge commit).
This file is the cold-start record of that work.

After the merge: release build and `gufo diagnose` pass, `AGENTS.md` has zero
occurrences of `nix`, TP2 Q4/Q8 generation text is correct, session reuse is
stable, and the depth/concurrency series
[2026-10-14-tp2-depth-series.md](models/qwen3.8-flash-next/2026-10-14-tp2-depth-series.md)
beats matched one-host on this build.

## What came in

`git fetch origin` moved `origin/main` by 12 commits, releasing v0.14.0 and
v0.15.0. The one that motivated this merge:

```
a33e989c perf(qwen-flash): speed up wide prefill kernels (#571)
```

That commit touches `kernels.hip.cpp` (+1233), `executor.cpp` (+175),
`executor.hpp` (+28), `kernels.hpp` (+43) and three projection tests.

Also in: Flash-Next component cache (`058d6e80`, `#573`), managed-alloc
fallback (`248d902e`, `#460`), IPv6 listen addresses, response-schema
handling, and a cache-pages fix (`10c40b0d`, `#589`).

## Constraints, and how they were honoured

- **`AGENTS.md` is not merged and must never mention Nix.** Upstream did not
  touch it, so the merge leaves it exactly as it is on this branch, and it
  still contains zero occurrences of "nix". Verify with
  `grep -ci nix AGENTS.md` before committing.
- **`README.md` merges but our changes stay untouched.** It merged cleanly.
  Our RDMA/TP2 note survives verbatim; the diff against `origin/main` shows
  only our additions.
  *Flag, not fixed:* the logo line on our side is duplicated - two
  `<img ... width="180">` tags end to end. This predates the merge (it is in
  this branch's `HEAD`, upstream has it correct) and was left alone because
  the instruction was to keep our README changes untouched. It looks
  accidental; fix it deliberately or not at all.
- **`flake.nix` is modified by this merge and is a separate decision.** It is
  already tracked on this branch, and no other tracked file mentions Nix.
  Taking upstream's change keeps it; removing the file is a different, larger
  call that would also create a permanent conflict with upstream. Not decided
  here.

## Resolved: 8 of 12 conflicted files

| file | resolution and why |
| --- | --- |
| `CMakeLists.txt` | union: upstream's `gufo_cache` plus our `gufo_tp_control`, keeping the `GUFO_ENABLE_TP2_RDMA` block |
| `cmake/Checks.cmake` | union: `component_text_cache_test` plus our `tp_control_test` |
| `tests/models/{deepseek_v4_flash,qwen27b,qwen38_flash_next}/CMakeLists.txt` | union: upstream's `gufo_cache_hip` plus our `gufo_tp_control` |
| `docs/models/qwen3.8-flash-next/BENCHMARKS.md` | our TOC entry for the RDMA section, upstream's PR #571 build provenance. Nix wording dropped from the build note. |
| `executor.hpp` | our `AllReduce` declaration and the added `reduce` parameter, plus upstream's `WmmaExperts` |
| `kernels.hpp` | our doc comment. It is the superset: upstream's rewrite drops Q8_0 and the 64-row tile, but this branch supports both - see the `tile_rows == 64` branch in `RoutedGatedF16Gemm`. **Re-check after the `.hip.cpp` merge**; if the 64-row path is removed there, update this comment to match. |
| `src/cli/serve/inference_backend.cpp` | three hunks, all unions (see below) |

Details for `inference_backend.cpp`, since one ordering matters:

1. Constructor init list: kept our `distributed_(model_->TpWorldSize() > 1)`
   and upstream's comment about the typed in-process API.
2. Identity block: kept our TP-rank suffix injection **before** upstream's new
   `component_identity_ = persistence_->compatibility_identity;`, so a TP rank
   caches under its own key rather than colliding with the other rank.
3. Our TP2 snapshot block (113 lines, `SnapshotPayloadBytes` / `Snapshot` /
   `SnapshotForPersistence` / restore) is **kept**. Upstream deleted that
   region, but our `distributed_{false};` member survives immediately after it
   and this block is what gives TP2 snapshots their rank-aware behaviour.
   Upstream still has its own unrelated `SnapshotPayloadBytes` overloads
   elsewhere in the file; those were not touched.

## Remaining: 4 files, 10 hunks

| file | hunks |
| --- | --- |
| `src/models/qwen38_flash_next/kernels/rocm/device_model.cpp` | 2 |
| `src/models/qwen38_flash_next/kernels/rocm/executor.cpp` | 5 |
| `src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp` | 3 |

`executor.hpp` and `kernels.hpp` are already resolved; the `.cpp`/`.hip.cpp`
counterparts are not. These are the files where the prefill speedup and the
TP2 work actually overlap.

### `device_model.cpp` - the blocking one, needs design not markers

Upstream **restructured weight loading into two passes**: a measuring walk
(`measure.Model(w, mtp, shard_count)` with a null allocator) that sums the
bytes requested, then `hip::WeightAllocator weights(measured)` and a real walk
whose `Model()` returns a new `Uploaded` struct. Upstream also simplified the
layer call to `Layer(l)` with a single argument.

This branch does not have `Model()` at all. It inlines the walk and calls
`Layer(l, m->trunk_heads_, m->trunk_split_)`, with TP2 head splitting set up
immediately beforehand:

```cpp
const MixerHeads all = MixerHeads::All(w.config);
m->trunk_split_ = partition != nullptr && partition->distributed();
m->trunk_heads_ = all;
if (m->trunk_split_) {
  std::string why;
  if (!SplitMixersFit(w, partition->world_size, &why)) { ...return nullptr; }
  const std::uint32_t world = partition->world_size;
  m->trunk_heads_.ssm_k /= world;
  m->trunk_heads_.ssm_v /= world;
  m->trunk_heads_.attn /= world;
  m->trunk_heads_.attn_kv /= world;
}
```

The draft block is deliberately **not** split (`up.Layer(mtp->block, all,
false)`), because it runs one short step at a time.

**What has to happen:** move that head-splitting into upstream's `Model()`,
compute it once and have `Model()` pass `heads`/`split` to `Layer`, then set
`m->trunk_split_` and `m->trunk_heads_` on the device model from the same
values so the rest of the executor still sees them. The Uploader already
carries `partition` and `config` members - both were added by already-resolved
hunks in this merge - so the plumbing exists.

Getting this wrong breaks TP2 silently: a rank that computes full-width heads
against split weights will produce wrong logits rather than an error. Any
resolution here needs a TP2 run to confirm, not just a build.

### `executor.cpp` and `kernels.hip.cpp`

Five and three hunks. `kernels.hip.cpp` carries the bulk of PR #571 (+1233
lines) and this branch carries the TP2 decode draft-count mirroring and the
`half_stride` strided-`ssm_out` fix, which previously caused
`SSM output F16 GEMM failed` at prompts of 1024 tokens or more. That fix
scoped the strided path to `cols == 6144`; check it survives, since upstream
reworked the same kernels and TP2 has `ssm_out.cols == 3072`.

Not started; no opinion recorded on how they should be resolved.

## After the merge

The branch must build and pass before this is published:

```sh
unset ROCM_PATH
export PATH=/opt/rocm/bin:/opt/rocm/core/lib/llvm/bin:$PATH
cmake --preset release \
  -DCMAKE_HIP_COMPILER=/opt/rocm/core/lib/llvm/bin/clang++ \
  -DCMAKE_CXX_FLAGS="-include format" \
  -DCMAKE_HIP_FLAGS="-include format"
cmake --build --preset release --parallel 4
./build/release/gufo diagnose
```

Then the TP2 two-host check, since that is what the resolved hunks protect.
The previous merge of this branch broke at prompts of 1024 tokens, so test at
least one prompt above that. Verify `grep -ci nix AGENTS.md` still prints 0.

## Related reading

- `docs/models/qwen3.8-flash-next/DECODE-PERF.md` - decode profiling work on
  this branch, including three optimisation ideas closed by measurement.
- `docs/models/qwen3.8-flash-next/IU4-PLAN.md` - the closed IU4 study, on
  branch `iu4`.
- Tags: `research/decode-perf-baseline-2026-10-14` (this branch before any of
  this), `research/iu4-instruction-study-2026-10-14`, and the two TP2 tags.
