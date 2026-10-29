// HIP shims for Maya `native_mmvq.cu` (CUDA → HIP mechanical port).
// __dp4a / packed-byte helpers match gufo quant_gemv.hip.cpp and Maya
// glm_fast.cu / dp4a.hpp (HIP takes the toolkit intrinsic).
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

using int8x4_t = int8_t __attribute__((ext_vector_type(4)));
using uint8x4_t = uint8_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ int __vsubss4(const int a, const int b) {
  const int8x4_t va = reinterpret_cast<const int8x4_t&>(a);
  const int8x4_t vb = reinterpret_cast<const int8x4_t&>(b);
  const int8x4_t c = __builtin_elementwise_sub_sat(va, vb);
  return reinterpret_cast<const int&>(c);
}

__device__ __forceinline__ int __vsub4(const int a, const int b) {
  return __vsubss4(a, b);
}

__device__ __forceinline__ unsigned int __vcmpne4(unsigned int a,
                                                  unsigned int b) {
  const uint8x4_t& va = reinterpret_cast<const uint8x4_t&>(a);
  const uint8x4_t& vb = reinterpret_cast<const uint8x4_t&>(b);
  unsigned int c;
  uint8x4_t& vc = reinterpret_cast<uint8x4_t&>(c);
#pragma unroll
  for (int i = 0; i < 4; ++i)
    vc[i] = va[i] == vb[i] ? 0x00 : 0xff;
  return c;
}

__device__ __forceinline__ int __dp4a(const int a, const int b, int c) {
#if defined(__GFX11__) || defined(__GFX12__) || defined(__gfx1151__)
  return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#else
  const int8x4_t va = reinterpret_cast<const int8x4_t&>(a);
  const int8x4_t& vb = reinterpret_cast<const int8x4_t&>(b);
  c += va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2] + va[3] * vb[3];
  return c;
#endif
}
