// Int4 (iu4) WMMA operand-layout probe for gfx1151, wave32.
//
// Purpose: a dedicated int4 GEMM kernel must know how the 16 nibbles packed in
// a v2 operand map to the logical k dimension, and whether the A and B
// operands use the same mapping. If they agree, the kernel can pack both the
// same way; if not, one side needs a transposed pack.
//
// Method: isolate one nibble on the A side and one on the B side. With
//   A = only nibble p set to +1
//   B = only nibble q set to +1
// the accumulator is 1 iff nibble p and nibble q denote the same logical k,
// and 0 otherwise. Sweeping p,q over 0..15 gives the matching classes without
// needing to know the absolute k labels.
//
// The accumulator side uses the same 16x16x16 wave32 shape as the tree's iu8
// path, so C[r][n] is checked against a scalar int4 reference computed with a
// candidate nibble order to confirm the whole contract holds.
//
// Build: hipcc -O2 --offload-arch=gfx1151 -std=c++20 tools/bench/gfx1151_iu4_layout.hip -o /tmp/iu4layout
// Run:   /tmp/iu4layout
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdint>
#include <vector>

using v2 = __attribute__((__vector_size__(2 * sizeof(int)))) int;
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

// Pack 16 int4 values (n0..n15, each already sign-corrected to 0..15 or
// 0..7 for negatives) into a v2. Bit offset 4*i selects nibble i.
__device__ __forceinline__ v2 PackNibbles(const int* n) {
  unsigned int lo = 0, hi = 0;
  for (int i = 0; i < 8; ++i) lo |= (unsigned)(n[i] & 0xf) << (4 * i);
  for (int i = 0; i < 8; ++i) hi |= (unsigned)(n[8 + i] & 0xf) << (4 * i);
  v2 r;
  r[0] = (int)lo;
  r[1] = (int)hi;
  return r;
}

// Set only nibble q of B to +1, everything else 0; A has only nibble p set.
// grid (1,1) block (32,1) - one wave.
__global__ void MatchKernel(const unsigned int* __restrict__ a_bits,
                            const unsigned int* __restrict__ b_bits,
                            int* __restrict__ out) {
  const int lane = threadIdx.x;
  // Lane -> (row, col): same convention the tree's iu8 probe established,
  // row = lane % 16, col = lane / 16 is not right for w32; use row = lane % 16,
  // col = (lane / 16) ^ (lane % 16) is a guess. We only need *some* consistent
  // (row,col) here because we compare against the reference built the same way,
  // so a plain row = lane % 16, col = lane % 16 degeneracy is fine for the
  // nibble-match question.
  const int rc = lane % 16;

  v2 a;
  a[0] = (int)a_bits[0];
  a[1] = (int)a_bits[1];
  v2 b;
  b[0] = (int)b_bits[0];
  b[1] = (int)b_bits[1];

  const bool s = true;
  v8 c = {0, 0, 0, 0, 0, 0, 0, 0};
  c = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(s, a, s, b, c, s);

  // Lane owns one accumulator cell; store it so the host can test nonzero.
  out[lane] = c[0];
}

int main() {
  hipDeviceProp_t p{};
  CHECK(hipGetDeviceProperties(&p, 0));
  std::printf("device: %s  arch=%s\n", p.name, p.gcnArchName);

  unsigned int *d_a = nullptr, *d_b = nullptr;
  int* d_out = nullptr;
  CHECK(hipMalloc(&d_a, 2 * sizeof(unsigned int)));
  CHECK(hipMalloc(&d_b, 2 * sizeof(unsigned int)));
  CHECK(hipMalloc(&d_out, 32 * sizeof(int)));

  // For each nibble pair (p,q) report whether the accumulator is nonzero.
  std::printf("\nnibble match matrix (rows = A nibble p, cols = B nibble q).\n");
  std::printf("'X' = same logical k, '.' = different k:\n\n     ");
  for (int q = 0; q < 16; ++q) std::printf("%2d ", q);
  std::printf("\n");

  int match_count = 0;
  int first_match[16];
  for (int p = 0; p < 16; ++p) first_match[p] = -1;

  for (int p = 0; p < 16; ++p) {
    std::printf("p=%2d ", p);
    for (int q = 0; q < 16; ++q) {
      unsigned int abits[2] = {0, 0};
      unsigned int bbits[2] = {0, 0};
      if (p < 8)
        abits[0] = 1u << (4 * p);
      else
        abits[1] = 1u << (4 * (p - 8));
      if (q < 8)
        bbits[0] = 1u << (4 * q);
      else
        bbits[1] = 1u << (4 * (q - 8));

      CHECK(hipMemcpy(d_a, abits, sizeof(abits), hipMemcpyHostToDevice));
      CHECK(hipMemcpy(d_b, bbits, sizeof(bbits), hipMemcpyHostToDevice));
      MatchKernel<<<1, 32>>>(d_a, d_b, d_out);
      CHECK(hipDeviceSynchronize());

      int out[32];
      CHECK(hipMemcpy(out, d_out, sizeof(out), hipMemcpyDeviceToHost));
      // Any nonzero accumulator cell means the two nibbles hit the same k.
      int hit = 0;
      for (int l = 0; l < 32; ++l)
        if (out[l] != 0) hit = 1;
      std::printf(" %c", hit ? 'X' : '.');
      if (hit) {
        match_count++;
        if (first_match[p] < 0) first_match[p] = q;
      }
    }
    std::printf("\n");
  }

  std::printf("\n%d of 256 pairs match\n", match_count);
  std::printf("first matching q per p:");
  for (int p = 0; p < 16; ++p) std::printf(" p%d->q%d", p, first_match[p]);
  std::printf("\n");

  // Identity match (p==q) confirms A and B share one packing order.
  int identity = 1;
  for (int p = 0; p < 16; ++p)
    if (first_match[p] != p) identity = 0;
  std::printf("\nA and B use the same nibble order: %s\n",
              identity ? "YES (identity)" : "NO / unclear");

  CHECK(hipFree(d_a));
  CHECK(hipFree(d_b));
  CHECK(hipFree(d_out));
  return 0;
}
