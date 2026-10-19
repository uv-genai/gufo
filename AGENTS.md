# AGENTS.md

## Platform and build

Production targets Linux x86-64 AMD Strix Halo (`gfx1151`) only. CMake owns
compiler flags, dependencies and installation; the system ROCm toolchain at
`/opt/rocm` supplies the compiler. Linux source-build prerequisites are in
[README.md](README.md#build-from-source).

Every configure on this host needs the same environment and the same three
flags. Set the environment once per shell:

```sh
unset ROCM_PATH
export PATH=/opt/rocm/bin:/opt/rocm/core/lib/llvm/bin:$PATH
```

Pass these three flags at every `cmake --preset` invocation:

| flag | why |
| --- | --- |
| `-DCMAKE_HIP_COMPILER=/opt/rocm/core/lib/llvm/bin/clang++` | CMake's default HIP compiler autodetection does not resolve on this host |
| `-DCMAKE_CXX_FLAGS="-include format"` | works around HIP's `__noinline__` macro colliding with GCC 16's `<format>` |
| `-DCMAKE_HIP_FLAGS="-include format"` | same collision on the HIP side |

```sh
cmake --preset release \
  -DCMAKE_HIP_COMPILER=/opt/rocm/core/lib/llvm/bin/clang++ \
  -DCMAKE_CXX_FLAGS="-include format" \
  -DCMAKE_HIP_FLAGS="-include format"
cmake --build --preset release --parallel 4
./build/release/gufo diagnose
```

The last two flags are host workarounds, not properties of the project, so
they are passed at configure time and are deliberately absent from the
presets. Do not bake them in.

Measure performance with `build/release/gufo`. Preserve compiler and
dependency versions when comparing results.

## Focused tests

Formatting and Python repository checks run on the editing host; they do not
need the GPU. Before committing C++ changes, run the shared CI check:

```sh
python3 tools/ci/check-format.py
# Add --fix to apply formatting, then rerun the check.
```

Run the smallest check covering the change. `gpu-test` is RelWithDebInfo with
assertions enabled. Build only the affected target during iteration, passing
the same three flags as the release configure above:

```sh
cmake --preset gpu-test \
  -DCMAKE_HIP_COMPILER=/opt/rocm/core/lib/llvm/bin/clang++ \
  -DCMAKE_CXX_FLAGS="-include format" \
  -DCMAKE_HIP_FLAGS="-include format"
cmake --build --preset gpu-test --target <test-target>
ctest --preset gpu-full -R '^<test-name>$' --output-on-failure
```

`cmake --build --preset pr`
runs the hosted contract suite after configuring `cpu-test`. Full CPU checks,
sanitisers, GPU/operator/model checks and H3 quality tools remain available
locally; see [docs/TESTING.md](docs/TESTING.md). Do not run full model sweeps,
video generation or duplicate suites for routine edits. A missing-model skip
is not a quality pass. Broaden checks when shared behavior changes or failures
expose risk.

Text API functional/regression tests live in [tests/functional/](tests/functional/README.md).
Select suites explicitly. Model-specific changes need the affected model/modes;
use all four profiles only for shared behavior changes. Run correctness on the
candidate; use main for matched affected timing controls and suspected regressions,
not a duplicate full correctness run. Reuse baseline evidence only when revision,
toolchain, harness, model, settings and cache history match. Include `long-context`
and `cache` for continuation changes; verify loaded mode and actual drafts.
Correctness and expected prefill/cache work are strict; timings are
mandatory per request/phase at 5% and 3 ms, never averaged across requests.
Run once, investigate flags, then alternate main/PR only for affected histories;
use unchanged-main controls when needed and retain every result. Fix confirmed
regressions before publishing. Noisy evidence stays visibly inconclusive and
unqualified; do not widen margins or stop at reporting failures. Use the code
diff to select affected metrics; unrelated timing variance does not justify
another full matrix.
For reported coding-agent loops, also replay the affected workflow with
`tests/functional/pi_agent.py`; retain its sessions and inspect actual tool results.
Retain numerical quality tests and the standard speed benchmark. Keep these
tests in `tests/functional/`, outside hosted model CI; avoid full sweeps.

## Profiling and kernels

Apply [.agents/skills/optimize-kernel/SKILL.md](.agents/skills/optimize-kernel/SKILL.md).
Use `tools/bench/build.sh` for standalone HIP experiments,
`tools/bench/gfx1151_peak.hip` for measured hardware ceilings,
`tools/prof/prof.py` for pipeline/wall-time profiles, and
`tools/prof/isa_mix.py` for instruction analysis. Production paths must retain
quality; successful optimizations become the default, without extra switches.

## Development

- Keep model code, tests, tools and numerical contracts with their model.
- Use the shared Qwen control-token constants for framing; see
  [src/models/qwen/AGENTS.md](src/models/qwen/AGENTS.md). Inspect their definitions
  when reviewing or changing behavior, and keep reference fixtures independent.
- Use one canonical long option and backend name per behavior; avoid aliases.
- Use `gh` for GitHub operations after checking `gh auth status`.
- Follow Conventional Commits with a single-line message. Title pull requests
  according to [the release policy](docs/RELEASING.md); squash merging retains
  that title on `main`. Mark breaking changes with `!` in the title, and leave
  version and changelog updates to the release workflow.
- Prefer `jj` when available (`jj version`); otherwise use Git.
- Follow the user's remote workflow and preserve unrelated work.

## Remotes and merges

- Upstream is **`origin`** → `gufo-org/gufo`. Track it on local branch
  **`gufo-org`** (`git fetch origin && git switch gufo-org && git merge
  --ff-only origin/main`). Never push to `origin` / `gufo-org`.
- Work and publishes go on fork **`main`**, remote **`uv-genai`** only.
- **`AGENTS.md` is fork-owned.** Do not merge upstream `AGENTS.md` changes
  that reintroduce the packaged environment. Before any commit, count case-
  insensitive matches of that environment's name in this file; the count must
  stay **0**. Never run that package manager on this host.
- After a merge: format check, release build, then TP2 validation below
  before any push. Compile-only is not a TP2 pass.
- Before merging upstream, read [docs/MERGE-CHECKLIST.md](docs/MERGE-CHECKLIST.md)
  and scan the diff for TP2 hazards (below). Prefer a hard
  `TpWorldSize()==1` gate over hoping ranks agree.

## TP2 (two-host Flash-Next)

Launchers live in [tools/tp2/](tools/tp2/); ship the same `build/release/gufo`
to `strix-halo-1.local` before starting a pair. Start rank 0, then rank 1
within 30 s. Verify TP2 with **generation text**, not HTTP 200: a rank-0-only
restore or a residual that never settles returns an empty completion with no
error. Session reuse is the usual desync point; see
[TP2.md](docs/models/qwen3.8-flash-next/TP2.md).

### Collective invariant

Both ranks must issue **the same** verbs collectives: **order, count, and
byte size**. Rank 0 mirrors calls and draft **counts**, not every decode
micro-decision. Any path that skips or adds an all-reduce on one rank poisons
the QP (`verbs all-reduce timed out`, `identity or size mismatch`, or RDMA
completion status errors). Size mismatches often show as row counts of
hidden (e.g. 10240 vs 122880 = 1 vs 12 rows × 2560 F32).

### Hard rules

- **One `gufo` per GPU host.** Never start a second model load while a TP2
  rank is up. Ask before any GPU launch.
- **Both ranks must run the same binary** (same git revision and sha256).
  A mismatched pair desyncs collectives and hangs in `verbs all-reduce`.
- **`TpMirroredRunner::UsesComponentCache()` stays false.** Component-cache
  restore is rank-0 only; TP2 uses mirrored `CreateState` / `Snapshot` /
  `RestoreOrFork`.
- **MTP draft-subset is single-host only** (`TpWorldSize()==1`). The small-head
  path skips the split-argmax all-reduce; ranks that disagree hang.
- **Prompt lookup copies are single-host only** (`TpWorldSize()==1`). Lookup
  can skip neural `MtpForward` cycles; under TP both ranks always run the
  neural draft path with the mirrored count. Snapshots still carry lookup
  state (payload v18).
- `defer_residual_` only when `tp_world_size()==1`. Forced draft counts come
  from rank 0 (`request.drafts` / `batch_drafts`).
- Watch live concurrency on rank 0: `GET /metrics`
  (`llamacpp:requests_processing`, `requests_deferred`) and `GET /slots`.
  Port is the HTTP port in `tools/tp2/rank0-*.sh`, not llama-swap's 8080.

### TP2 hazards in upstream merges

If a merge or feature changes any of the following, treat it as a **TP2
hazard** until gated or mirrored:

| Hazard | Why it breaks verbs | Policy |
| --- | --- | --- |
| Draft subset / small head | Skips split-argmax all-reduce | `TpWorldSize()==1` only |
| Prompt lookup / n-gram copy | Skips neural `MtpForward` | `TpWorldSize()==1` only |
| New speculative/MTP modes | Different cycle width per rank | Mirror count from rank 0 **and** path shape |
| Component-cache / slot restore | Rank-0-only restore | Keep `UsesComponentCache` false |
| Prefill checkpoint / in-pass | Extra collectives on one rank | `in_pass_checkpoint` / `PrefillThrough` rules in TP2.md |
| Attention/MoE “fast paths” | Different row counts or skipped AR | Same geometry both ranks; no host-only skip |
| Serve batching / plan | Different `batch_drafts` or member widths | Ship plan from rank 0 |
| Snapshot payload version | Rank state drift after reuse | Bump version; both ranks same binary |

**After any merge that touches** `engine.cpp`, `executor.*`,
`inference_backend.cpp`, `component_text_cache.*`, or MTP/cache: run the
TP2 smoke in [MERGE-CHECKLIST.md](docs/MERGE-CHECKLIST.md) before push.
