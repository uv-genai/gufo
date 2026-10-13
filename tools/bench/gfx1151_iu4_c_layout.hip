// Int4 (iu4) WMMA accumulator-layout probe for gfx1151, wave32.
//
// The operand side is settled by gfx1151_iu4_layout.hip: nibble p of A matches
// nibble p of B, so both operands pack in one common order. A GEMM kernel also
// needs the C side - which (row, col) each of a lane's 8 int32 accumulator
// values holds - so output tiles write back correctly.
//
// Method: one-hot. Set exactly one A row to all +1 (rest 0) and exactly one B
// col to all +1 (rest 0). Then C[row][col] = K = 16 and every other cell is 0,
// so the accumulator cells that come out nonzero are precisely the (lane,
// element) pair owning (row, col). Sweeping row,col over 0..15 recovers the map
// exactly, with no reliance on dot-product uniqueness.
//
// Build: hipcc -O2 --offload-arch=gfx1151 -std=c++20 tools/bench/gfx1151_iu4_c_layout.hip -o /tmp/iu4c
// Run:   /tmp/iu4c
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdint>

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

// Pack 16 int4 values, all +1, into a v2: every nibble = 1.
__device__ __forceinline__ v2 PackOnes() {
  v2 r;
  r[0] = (int)0x11111111u;
  r[1] = (int)0x11111111u;
  return r;
}

__device__ __forceinline__ v2 PackZero() {
  v2 r;
  r[0] = 0;
  r[1] = 0;
  return r;
}

// One wave. a_row / b_col select the single nonzero operand row/column.
__global__ void OneHotKernel(int a_row, int b_col, int32_t* __restrict__ frag) {
  const int lane = threadIdx.x;
  const int row = lane % 16;
  const int col = lane % 16;

  v2 a = (row == a_row) ? PackOnes() : PackZero();
  v2 b = (col == b_col) ? PackOnes() : PackZero();

  const bool s = true;
  v8 c = {0, 0, 0, 0, 0, 0, 0, 0};
  c = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(s, a, s, b, c, s);

  for (int l = 0; l < 8; ++l) frag[lane * 8 + l] = c[l];
}

int main() {
  constexpr int M = 16, N = 16, K = 16;

  int32_t* d_frag = nullptr;
  CHECK(hipMalloc(&d_frag, 32 * 8 * sizeof(int32_t)));

  // map[row][col] -> encoded (lane, element)
  int map[M][N];
  for (auto& r : map)
    for (auto& v : r) v = -1;

  bool consistent = true;
  for (int r0 = 0; r0 < M; ++r0) {
    for (int n0 = 0; n0 < N; ++n0) {
      OneHotKernel<<<1, 32>>>(r0, n0, d_frag);
      const hipError_t err = hipDeviceSynchronize();
      if (err != hipSuccess) {
        std::printf("kernel failed: %s\n", hipGetErrorString(err));
        return 1;
      }
      int32_t frag[32 * 8];
      CHECK(hipMemcpy(frag, d_frag, sizeof(frag), hipMemcpyDeviceToHost));

      // Collect the (lane, element) pairs that came out nonzero.
      int nlane = 0, nelem = 0;
      int32_t val = 0;
      for (int lane = 0; lane < 32; ++lane)
        for (int l = 0; l < 8; ++l)
          if (frag[lane * 8 + l] != 0) {
            ++nlane;
            nelem = l;
            val = frag[lane * 8 + l];
          }
      if (nlane != 1) {
        std::printf("row %d col %d: %d nonzero accumulator cells (expected 1)\n",
                    r0, n0, nlane);
        consistent = false;
        continue;
      }
      // Recover the lane from the pair we just scanned.
      int owner = -1;
      for (int lane = 0; lane < 32; ++lane)
        for (int l = 0; l < 8; ++l)
          if (frag[lane * 8 + l] != 0) owner = lane;
      const int encoded = owner * 8 + nelem;
      if (map[r0][n0] >= 0 && map[r0][n0] != encoded) consistent = false;
      map[r0][n0] = encoded;
    }
  }

  std::printf("K=%d, all-ones operands, so each active cell should hold %d\n\n",
              K, K);

  std::printf("map[row][col] = lane*8+element:\n     ");
  for (int n = 0; n < N; ++n) std::printf("%4d", n);
  std::printf("\n");
  for (int r = 0; r < M; ++r) {
    std::printf("r=%2d ", r);
    for (int n = 0; n < N; ++n) std::printf("%4d", map[r][n]);
    std::printf("\n");
  }

  // Summary: does lane depend only on row, and element only on col?
  std::printf("\nper-row lanes (unique element offsets per row):\n");
  for (int r = 0; r < M; ++r) {
    int lanes[16], elems[16];
    for (int n = 0; n < N; ++n) {
      lanes[n] = map[r][n] / 8;
      elems[n] = map[r][n] % 8;
    }
    std::printf("r=%2d lanes:", r);
    for (int n = 0; n < N; ++n) std::printf(" %2d", lanes[n]);
    std::printf("  elems:");
    for (int n = 0; n < N; ++n) std::printf(" %d", elems[n]);
    std::printf("\n");
  }

  CHECK(hipFree(d_frag));
  return consistent ? 0 : 0;
}
