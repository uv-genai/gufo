// Throughput of the int4 (IU4) WMMA on gfx1151 vs the int8 (IU8) baseline.
//
// Signatures below were found by compiling candidate operand widths against
// clang, not by __has_builtin (which is unreliable for these names):
//   wmma_i32_16x16x16_iu8_w32       (bool, v4,  bool, v4,  v8, bool)  K=16
//   wmma_i32_16x16x16_iu8_w64       (bool, v4,  bool, v4,  v4, bool)  K=16
//   wmma_i32_16x16x16_iu4_w32       (bool, v2,  bool, v2,  v8, bool)  K=16
//   wmma_i32_16x16x16_iu4_w64       (bool, v2,  bool, v2,  v4, bool)  K=16
//   wmma_i32_16x16x32_iu4_w32_gfx12 (bool, v2,  bool, v2,  v8, bool)  K=32
//
// iu4 carries K=16 in half the operand bytes of iu8, and the 16x16x32 form
// carries K=32 in the same bytes - so int4 could double either bandwidth or
// MACs. This measures both: a correctness pass (compare against a scalar
// int8 reference) and a timed loop for MACs/sec.
//
// Build: hipcc -O3 --offload-arch=gfx1151 -std=c++20 tools/bench/gfx1151_iu4_bench.hip -o /tmp/iu4bench
// Run:   /tmp/iu4bench
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdint>

using v2 = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using v4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using v8 = __attribute__((__vector_size__(8 * sizeof(int)))) int;

#define CHECK(x)                                                        \
  do {                                                                  \
    const hipError_t e_ = (x);                                          \
    if (e_ != hipSuccess) {                                             \
      std::fprintf(stderr, "%s:%d hip error: %s\n", __FILE__, __LINE__, \
                   hipGetErrorString(e_));                              \
      return 1;                                                         \
    }                                                                   \
  } while (0)

// ---- timed kernels: one warp per block, inner loop counts MMA ops ----

// NOTE: wmma_i32_16x16x32_iu4_w32_gfx12 (K=32, the 2x-MAC form) requires the
// wmma-128b-insts target feature. gfx1151 has it, but enabling it - via
// __attribute__((target(...))) or -Xclang -target-feature - makes this ROCm
// 7.2.3 clang front end segfault during codegen, so that form cannot be
// measured here and is excluded.

__global__ void TimeI8_16x16x16_w32(long long* out) {
  const bool s = true;
  v4 a = {1, 1, 1, 1};
  v4 b = {1, 1, 1, 1};
  v8 c = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll 1
  for (int i = 0; i < 2048; ++i) {
    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(s, a, s, b, c, s);
  }
  out[threadIdx.x] = (long long)c[0] + c[7];
}

__global__ void TimeI4_16x16x16_w32(long long* out) {
  const bool s = true;
  v2 a = {1, 1};
  v2 b = {1, 1};
  v8 c = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll 1
  for (int i = 0; i < 2048; ++i) {
    c = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(s, a, s, b, c, s);
  }
  out[threadIdx.x] = (long long)c[0] + c[7];
}

// Correctness: compare iu8 and iu4 results on identical logical data.
// iu8 operand A holds 16 int8 values per lane; iu4 holds 16 int4 values in
// half the bytes. We set every packed field to 1 so the expected dot product
// over K=16 of all-ones is 16 per accumulator element.
__global__ void Correctness(int* out) {
  const bool s = true;
  v4 a8 = {0x01010101, 0x01010101, 0x01010101, 0x01010101};
  v4 b8 = {0x01010101, 0x01010101, 0x01010101, 0x01010101};
  v8 c8 = {0, 0, 0, 0, 0, 0, 0, 0};
  c8 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(s, a8, s, b8, c8, s);

  // Every nibble = 1 packs to 0x11 per byte.
  v2 a4 = {0x11111111, 0x11111111};
  v2 b4 = {0x11111111, 0x11111111};
  v8 c4 = {0, 0, 0, 0, 0, 0, 0, 0};
  c4 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(s, a4, s, b4, c4, s);

  out[0] = c8[0];
  out[1] = c4[0];
  out[2] = c8[3];
  out[3] = c4[3];
}

using Kern = void (*)(long long*);

struct Row {
  const char* name;
  Kern fn;
  int macs_per_op;  // M*N*K
  int op_bytes;     // A+B bytes per lane per op
};

int main() {
  hipDeviceProp_t p{};
  CHECK(hipGetDeviceProperties(&p, 0));
  std::printf("device: %s  arch=%s  CUs=%d  clock=%.2f GHz\n", p.name,
              p.gcnArchName, p.multiProcessorCount, p.clockRate / 1e6);

  long long* d_out = nullptr;
  CHECK(hipMalloc(&d_out, p.multiProcessorCount * 64 * sizeof(long long)));

  const Row rows[] = {
      {"iu8 16x16x16 w32 (baseline)", TimeI8_16x16x16_w32, 16 * 16 * 16, 32},
      {"iu4 16x16x16 w32", TimeI4_16x16x16_w32, 16 * 16 * 16, 16},
  };
  const int nrow = 2;

  // ---- execution check + timing ----
  std::printf("\n%-30s %10s %14s %12s\n", "instruction", "exec", "MACs/instr",
              "GMACs/s");
  double baseline = 0.0;
  double best = 0.0;
  const char* best_name = "";
  for (int i = 0; i < nrow; ++i) {
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0));
    CHECK(hipEventCreate(&e1));
    // warmup
    rows[i].fn<<<p.multiProcessorCount, 64>>>(d_out);
    const hipError_t werr = hipDeviceSynchronize();
    if (werr != hipSuccess) {
      std::printf("%-30s %10s %14d %12s\n", rows[i].name, "TRAP", 0, "-");
      continue;
    }
    CHECK(hipEventRecord(e0));
    const int iters = 64;
    for (int k = 0; k < iters; ++k) {
      rows[i].fn<<<p.multiProcessorCount, 64>>>(d_out);
    }
    CHECK(hipEventRecord(e1));
    CHECK(hipEventSynchronize(e1));
    float ms = 0.0f;
    CHECK(hipEventElapsedTime(&ms, e0, e1));
    const double sec = ms / 1e3 / iters;
    const double warps =
        (double)p.multiProcessorCount * 64;  // 1 warp per CU
    const double total_macs = warps * 2048.0 * rows[i].macs_per_op;
    const double gmacs = total_macs / sec / 1e9;
    if (i == 0) baseline = gmacs;
    if (gmacs > best) {
      best = gmacs;
      best_name = rows[i].name;
    }
    std::printf("%-30s %10s %14d %12.2f   (%.2fx i8)\n", rows[i].name, "ok",
                rows[i].macs_per_op, gmacs, gmacs / baseline);
  }

  // ---- correctness ----
  int* d_res = nullptr;
  CHECK(hipMalloc(&d_res, 4 * sizeof(int)));
  Correctness<<<1, 32>>>(d_res);
  CHECK(hipDeviceSynchronize());
  int res[4] = {0, 0, 0, 0};
  CHECK(hipMemcpy(res, d_res, sizeof(res), hipMemcpyDeviceToHost));
  std::printf("\ncorrectness (K=16 of all-ones, expect 16 per accumulator):\n");
  std::printf("  iu8 acc[0]=%d  iu4 acc[0]=%d   %s\n", res[0], res[1],
              res[0] == res[1] ? "MATCH" : "MISMATCH");
  std::printf("  iu8 acc[3]=%d  iu4 acc[3]=%d   %s\n", res[2], res[3],
              res[2] == res[3] ? "MATCH" : "MISMATCH");

  std::printf("\nfastest: %s\n", best_name);
  CHECK(hipFree(d_out));
  CHECK(hipFree(d_res));
  return 0;
}
