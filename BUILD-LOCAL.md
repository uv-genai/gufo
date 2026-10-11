# Local TP2 build on the twin Strix Halo hosts (CachyOS)

How the gufo executables on `strix-halo-1` and `strix-halo-2` were built, and
why the naive build fails on these machines.

The README is written for two toolchains: Nix (not installed here) or
"With Nix-less" with the *qualified* toolchain **GCC 15.3 + ROCm 7.2.3**.
These CachyOS boxes have **GCC 16.2.1** and the `rocm-gfx1151-bin` package
layout and no Nix, so three extra problems had to be worked around. Those are
almost certainly why a naive build attempt fails.

## What both builds have in common

Both use the `release` preset plus `-DGUFO_ENABLE_TP2_RDMA=ON` (that's the
whole point: the RDMA TP2 code is off by default).

```sh
cmake --preset release -DGUFO_ENABLE_TP2_RDMA=ON \
  -DCMAKE_CXX_FLAGS="-include format" \
  -DCMAKE_HIP_FLAGS="-include format"
cmake --build --preset release --parallel 16
```

**The `-include format` workaround (fails on both hosts without it):** ROCm's
`hip/hip_runtime.h` defines `__noinline__` as a *macro*
(`#define __noinline__ __attribute__((noinline))`), but GCC 16's libstdc++
`<format>` uses it as an attribute name (`[[__gnu__::__noinline__]]`). Any
translation unit that includes a HIP header before `<chrono>/<format>` dies
with `error: expected an identifier for the attribute name`. GCC 15 (the
qualified compiler) didn't have that `<format>` code, so upstream never hit
it. `-include format` force-includes the header *before* HIP can define the
macro, and its include guard prevents the later broken parse. It is applied
to both CXX and HIP compile flags.

## strix-halo-2 (the easy one)

Everything was already installed (gcc, cmake, ninja, `rocm-gfx1151-bin`,
`rdma-core`), repo at `~/software/gufo` on branch `rdma`. Just the two
commands above. Done.

## strix-halo-1 (needed three fixes)

1. **Missing `ninja`** — `sudo pacman -S ninja`
2. **Cloned the repo to `~/software/gufo-rdma`** — the existing
   `~/software/gufo` there held `config.yaml`, launch scripts and models, so
   it wasn't touched at the time. *(Since aligned: those files now live in
   `~/software/gufo-podman/` and the clone is at `~/software/gufo`, like on
   strix-halo-2.)*
3. **ROCm toolchain discovery — the tricky one.** Plain `cmake --preset
   release` failed with *"Failed to find HIP root directory"*, because the
   non-interactive PATH has no ROCm. But simply setting
   `export ROCM_PATH=/opt/rocm` made it worse: clang then failed with
   *"cannot find ROCm device library"*. The reason: this package layout is
   `/opt/rocm/{bin,lib,include} → /opt/rocm/core/...` (symlinks), with device
   bitcode in `/opt/rocm/core/lib/llvm/amdgcn/bitcode`. Clang's `--rocm-path`
   resolution walks the symlinked path and can't find the bitcode there
   (verified: `ROCM_PATH=/opt/rocm clang++ --offload-arch=gfx1151 ...` fails,
   same command without it succeeds). The fix was to **not set `ROCM_PATH`**,
   put the real clang on the PATH, and pin it explicitly:

```sh
cd ~/software/gufo-rdma
export PATH=/opt/rocm/bin:/opt/rocm/core/lib/llvm/bin:$PATH   # note: NO ROCM_PATH
cmake --preset release -DGUFO_ENABLE_TP2_RDMA=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/core/lib/llvm/bin/clang++ \
  -DCMAKE_CXX_FLAGS="-include format" \
  -DCMAKE_HIP_FLAGS="-include format"
cmake --build --preset release --parallel 16
```

(`-DCMAKE_HIP_COMPILER=...core/lib/llvm/bin/clang++` uses the real path;
`/opt/rocm/lib/llvm/bin/clang++` is the same binary via symlink and hits the
same device-lib problem.)

## Summary of the three gotchas

| Symptom | Cause | Fix |
|---|---|---|
| `expected an identifier for the attribute name` in `<format>` | HIP's `__noinline__` macro vs GCC 16 libstdc++ | `-DCMAKE_CXX_FLAGS="-include format" -DCMAKE_HIP_FLAGS="-include format"` |
| `Failed to find HIP root directory` | ROCm not on non-interactive PATH | `export PATH=/opt/rocm/bin:/opt/rocm/core/lib/llvm/bin:$PATH` |
| `cannot find ROCm device library` | `ROCM_PATH`/symlinked `/opt/rocm/lib` breaks clang's bitcode lookup | Don't set `ROCM_PATH`; pin `-DCMAKE_HIP_COMPILER=/opt/rocm/core/lib/llvm/bin/clang++` |

If you ever rebuild from scratch, wipe the stale cache first
(`rm -rf build/release`) — a configure cache that already recorded a broken
HIP compiler or `ROCM_PATH` will keep failing even after you fix the
environment.

## Unified workflow (current state)

The two hosts now have the same environment: identical packages and kernel
(toolchain is the same: gcc 16.2.1, cmake 4.4.4, ninja, `rocm-gfx1151-bin`
10.0.0-2, `rdma-core` 65) and the same repo path `~/software/gufo` on branch
`rdma`. All workarounds above are encapsulated in one script that works
unchanged on both hosts:

```sh
~/software/gufo/build-gufo.sh          # configure + build + gufo diagnose
~/software/gufo/build-gufo.sh --clean  # wipe build/release first
```
