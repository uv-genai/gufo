#include "src/models/glm5_next/quant_gemv.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/quant/iq_grids.hpp"

// Quantized GEMM for GLM5 weights. Dequantization is fused with the dot
// product (no per-thread dequant buffers) and matches the host reference in
// src/core/quant/ggml_dequant.cpp — itself a port of llama.cpp
// `dequantize_row_{q8_0,q6_K,iq2_xxs,iq3_xxs}`.
//
// Layout: one wavefront-sized workgroup (64 threads) computes 64/threads_per_
// row output rows. The threads of a row stride over that row's quant blocks
// (contiguous bytes -> coalesced reads) and reduce through LDS.

namespace gufo::models::glm5_next::gpu {
namespace {

using quant::block_iq2_s;
using quant::block_iq2_xxs;
using quant::block_iq3_xxs;
using quant::block_q2_K;
using quant::block_q3_K;
using quant::block_q6_K;
using quant::block_q8_0;
using quant::IqSigns;
using quant::kIq2sGrid;
using quant::kIq2xxsGrid;
using quant::kIq3xxsGrid;

constexpr int kTypeF32 = 0;
constexpr int kTypeF16 = 1;
constexpr int kTypeQ8_0 = 8;
constexpr int kTypeQ6_K = 14;
constexpr int kTypeIQ2_XXS = 16;
constexpr int kTypeIQ3_XXS = 18;
constexpr int kTypeQ2_K = 10;
constexpr int kTypeQ3_K = 11;
constexpr int kTypeIQ2_S = 22;

__device__ __forceinline__ float H2F(std::uint16_t bits) {
  return __half2float(*reinterpret_cast<const __half*>(&bits));
}

struct TypeTraits {
  int qk;           ///< elements per quant block
  int block_bytes;  ///< storage bytes per quant block
};

constexpr TypeTraits Traits(int type) {
  switch (type) {
    case kTypeF32:
      return {1, 4};
    case kTypeF16:
      return {1, 2};
    case kTypeQ8_0:
      return {32, 34};
    case kTypeQ6_K:
      return {256, 210};
    case kTypeIQ2_XXS:
      return {256, 66};
    case kTypeIQ3_XXS:
      return {256, 98};
    case kTypeIQ2_S:
      return {256, 82};
    case kTypeQ2_K:
      return {256, 84};
    case kTypeQ3_K:
      return {256, 110};
    default:
      return {0, 0};
  }
}

// ---- fused block dot products -----------------------------------------------

__device__ float BlockDotF32(const void* blk, const float* x) {
  return *static_cast<const float*>(blk) * x[0];
}

__device__ float BlockDotF16(const void* blk, const float* x) {
  return H2F(*static_cast<const std::uint16_t*>(blk)) * x[0];
}

__device__ float BlockDotQ8_0(const void* blk, const float* x) {
  const auto* b = static_cast<const block_q8_0*>(blk);
  const float d = H2F(b->d);
  float acc = 0.0F;
  for (int i = 0; i < 32; ++i)
    acc += static_cast<float>(b->qs[i]) * x[i];
  return d * acc;
}

// llama.cpp dequantize_row_q6_K fused with the activation dot product.
// Weights are loaded as 32-bit words (byte loads serialize the VALU); values
// are packed int8 so the per-16-element scale pairs accumulate in groups.
__device__ float BlockDotQ6_K(const void* blk, const float* x) {
  const auto* b = static_cast<const block_q6_K*>(blk);
  const float d = H2F(b->d);
  const auto* ql = reinterpret_cast<const std::uint32_t*>(b->ql);
  const auto* qh = reinterpret_cast<const std::uint32_t*>(b->qh);
  const auto* sc = reinterpret_cast<const std::int32_t*>(b->scales);
  float acc = 0.0F;
  for (int half = 0; half < 2; ++half) {
    const std::uint32_t* qlw = ql + half * 16;  // 64 bytes = 16 words
    const std::uint32_t* qhw = qh + half * 8;   // 32 bytes = 8 words
    // 4 int8 scales per int32 word.
    const std::int8_t* scr =
        reinterpret_cast<const std::int8_t*>(sc + half * 2);
    const float* xx = x + half * 128;
    for (int l = 0; l < 32; ++l) {
      const std::uint32_t qll = qlw[l >> 2];
      const std::uint32_t qhh = qhw[l >> 2];
      const int sh = (l & 3) * 8;
      const std::uint8_t b0 = (qll >> sh) & 0xFFU;
      const std::uint8_t b1 = (qlw[8 + (l >> 2)] >> sh) & 0xFFU;
      const std::uint8_t h0 = (qhh >> sh) & 0xFFU;
      const int is = l >> 4;
      const int v1 =
          (static_cast<int>(b0 & 0x0FU) | (static_cast<int>(h0 & 0x03U) << 4)) -
          32;
      const int v2 = (static_cast<int>(b1 & 0x0FU) |
                      (static_cast<int>((h0 >> 2U) & 0x03U) << 4)) -
                     32;
      const int v3 = (static_cast<int>(b0 >> 4U) |
                      (static_cast<int>((h0 >> 4U) & 0x03U) << 4)) -
                     32;
      const int v4 = (static_cast<int>(b1 >> 4U) |
                      (static_cast<int>((h0 >> 6U) & 0x03U) << 4)) -
                     32;
      const float s0 = static_cast<float>(scr[is]);
      const float s1 = static_cast<float>(scr[is + 2]);
      acc += s0 * static_cast<float>(v1) * xx[l];
      acc += s1 * static_cast<float>(v2) * xx[l + 32];
      acc +=
          static_cast<float>(scr[is + 4]) * static_cast<float>(v3) * xx[l + 64];
      acc +=
          static_cast<float>(scr[is + 6]) * static_cast<float>(v4) * xx[l + 96];
    }
  }
  return d * acc;
}

// llama.cpp dequantize_row_iq2_xxs fused with the activation dot product.
__device__ float BlockDotIQ2_XXS(const void* blk, const float* x) {
  const auto* b = static_cast<const block_iq2_xxs*>(blk);
  const float d = H2F(b->d);
  const auto* qs = reinterpret_cast<const std::uint8_t*>(b->qs);
  float acc = 0.0F;
  for (int ib32 = 0; ib32 < 8; ++ib32) {
    std::uint64_t group = 0;
    __builtin_memcpy(&group, qs + 8 * ib32, 8);
    const std::uint32_t aux32_1 = static_cast<std::uint32_t>(group >> 32);
    const float db = d * (0.5F + static_cast<float>(aux32_1 >> 28)) * 0.25F;
    for (int l = 0; l < 4; ++l) {
      const std::uint8_t code = static_cast<std::uint8_t>(group >> (8 * l));
      const std::uint64_t grid = kIq2xxsGrid[code];
      const std::uint8_t signs = IqSigns((aux32_1 >> (7 * l)) & 127U);
      const float* xx = x + ib32 * 32 + l * 8;
      for (int j = 0; j < 8; ++j) {
        const float mag =
            static_cast<float>(static_cast<std::uint8_t>(grid >> (8 * j)));
        acc += db * mag * ((signs & (1U << j)) ? -xx[j] : xx[j]);
      }
    }
  }
  return acc;
}

// llama.cpp dequantize_row_iq3_xxs fused with the activation dot product.
__device__ float BlockDotIQ3_XXS(const void* blk, const float* x) {
  const auto* b = static_cast<const block_iq3_xxs*>(blk);
  const float d = H2F(b->d);
  const std::uint8_t* qs = b->qs;
  float acc = 0.0F;
  for (int ib32 = 0; ib32 < 8; ++ib32) {
    std::uint32_t aux32 = 0;
    __builtin_memcpy(&aux32, qs + 64 + 4 * ib32, 4);
    const float db = d * (0.5F + static_cast<float>(aux32 >> 28)) * 0.5F;
    const std::uint8_t* codes = qs + 8 * ib32;
    const float* xx = x + ib32 * 32;
    for (int l = 0; l < 4; ++l) {
      const std::uint8_t signs = IqSigns((aux32 >> (7 * l)) & 127U);
      const std::uint32_t g1 = kIq3xxsGrid[codes[2 * l + 0]];
      const std::uint32_t g2 = kIq3xxsGrid[codes[2 * l + 1]];
      for (int j = 0; j < 4; ++j) {
        const float m1 =
            static_cast<float>(static_cast<std::uint8_t>(g1 >> (8 * j)));
        const float m2 =
            static_cast<float>(static_cast<std::uint8_t>(g2 >> (8 * j)));
        acc += db * m1 * ((signs & (1U << j)) ? -xx[l * 8 + j] : xx[l * 8 + j]);
        acc += db * m2 *
               ((signs & (1U << (j + 4))) ? -xx[l * 8 + j + 4]
                                          : xx[l * 8 + j + 4]);
      }
    }
  }
  return acc;
}

// llama.cpp dequantize_row_q2_K fused with the activation dot product.
__device__ float BlockDotQ2_K(const void* blk, const float* x) {
  const auto* b = static_cast<const block_q2_K*>(blk);
  const float d = H2F(b->d);
  const float mn = H2F(b->dmin);
  float acc = 0.0F;
  for (int half = 0; half < 2; ++half) {
    const std::uint8_t* q = b->qs + half * 32;
    const float* xx = x + half * 128;
    int is = half * 8;
    for (int j = 0; j < 4; ++j) {
      for (int grp = 0; grp < 2; ++grp) {
        const std::uint8_t sc = b->scales[is++];
        const float dl = d * static_cast<float>(sc & 0x0FU);
        const float ml = mn * static_cast<float>(sc >> 4U);
        const std::uint8_t* qq = q + grp * 16;
        const float* xt = xx + j * 32 + grp * 16;
        for (int l = 0; l < 16; ++l) {
          const int qv = (qq[l] >> (2 * j)) & 0x03U;
          acc += dl * static_cast<float>(qv) * xt[l] - ml * xt[l];
        }
      }
    }
  }
  return acc;
}

// llama.cpp dequantize_row_q3_K fused with the activation dot product.
__device__ float BlockDotQ3_K(const void* blk, const float* x) {
  const auto* b = static_cast<const block_q3_K*>(blk);
  const float d_all = H2F(b->d);
  std::uint32_t aux[4];
  __builtin_memcpy(aux, b->scales, 12);
  const std::uint32_t kmask1 = 0x03030303U;
  const std::uint32_t kmask2 = 0x0f0f0f0fU;
  const std::uint32_t tmp = aux[2];
  aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
  aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
  aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
  aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
  const auto* scales = reinterpret_cast<const std::int8_t*>(aux);
  const std::uint8_t* q = b->qs;
  const std::uint8_t* hm = b->hmask;
  float acc = 0.0F;
  std::uint8_t m = 1;
  int is = 0;
  for (int half = 0; half < 2; ++half) {
    for (int j = 0; j < 4; ++j) {
      for (int grp = 0; grp < 2; ++grp) {
        const float dl = d_all * static_cast<float>(scales[is++] - 32);
        const std::uint8_t* qq = q + grp * 16;
        const std::uint8_t* hh = hm + grp * 16;
        const float* xt = x + half * 128 + j * 32 + grp * 16;
        for (int l = 0; l < 16; ++l) {
          const int qv = (qq[l] >> (2 * j)) & 0x03U;
          const int hi = (hh[l] & m) ? 0 : 4;
          acc += dl * static_cast<float>(qv - hi) * xt[l];
        }
      }
      m = static_cast<std::uint8_t>(m << 1);
    }
    q += 32;
  }
  return acc;
}

// llama.cpp dequantize_row_iq2_s fused with the activation dot product.
__device__ float BlockDotIQ2_S(const void* blk, const float* x) {
  const auto* b = static_cast<const block_iq2_s*>(blk);
  const float d = H2F(b->d);
  float acc = 0.0F;
  for (int ib32 = 0; ib32 < 8; ++ib32) {
    const std::uint8_t sc = b->scales[ib32];
    const float db0 = d * (0.5F + static_cast<float>(sc & 0x0FU)) * 0.25F;
    const float db1 = d * (0.5F + static_cast<float>(sc >> 4U)) * 0.25F;
    for (int l = 0; l < 4; ++l) {
      const float dl = (l < 2) ? db0 : db1;
      const std::uint16_t code = static_cast<std::uint16_t>(
          b->qs[(4 * ib32) + l] |
          ((static_cast<std::uint16_t>(b->qh[ib32]) << (8 - 2 * l)) & 0x300U));
      const std::uint64_t grid = kIq2sGrid[code];
      const std::uint8_t signs = b->qs[32 + (4 * ib32) + l];
      const float* xx = x + ib32 * 32 + l * 8;
      for (int j = 0; j < 8; ++j) {
        const float mag =
            static_cast<float>(static_cast<std::uint8_t>(grid >> (8 * j)));
        acc += dl * mag * ((signs & (1U << j)) ? -xx[j] : xx[j]);
      }
    }
  }
  return acc;
}

/// Compile-time dispatched block dot (a runtime switch here would keep every
/// type's registers live and spill to scratch — a measured 10x slowdown).
template<int TYPE>
__device__ __forceinline__ float BlockDotT(const void* blk, const float* x) {
  if constexpr (TYPE == kTypeF32)
    return BlockDotF32(blk, x);
  else if constexpr (TYPE == kTypeF16)
    return BlockDotF16(blk, x);
  else if constexpr (TYPE == kTypeQ8_0)
    return BlockDotQ8_0(blk, x);
  else if constexpr (TYPE == kTypeQ6_K)
    return BlockDotQ6_K(blk, x);
  else if constexpr (TYPE == kTypeIQ2_XXS)
    return BlockDotIQ2_XXS(blk, x);
  else if constexpr (TYPE == kTypeIQ3_XXS)
    return BlockDotIQ3_XXS(blk, x);
  else if constexpr (TYPE == kTypeIQ2_S)
    return BlockDotIQ2_S(blk, x);
  else if constexpr (TYPE == kTypeQ2_K)
    return BlockDotQ2_K(blk, x);
  else
    return BlockDotQ3_K(blk, x);
}

// ---- full-block dequant (embedding lookup) ---------------------------------

__device__ void DequantBlockQ6_K(const void* blk, float* out) {
  const auto* b = static_cast<const block_q6_K*>(blk);
  const float d = H2F(b->d);
  for (int half = 0; half < 2; ++half) {
    const std::uint8_t* ql = b->ql + half * 64;
    const std::uint8_t* qh = b->qh + half * 32;
    const std::int8_t* sc = b->scales + half * 8;
    float* yy = out + half * 128;
    for (int l = 0; l < 32; ++l) {
      const int is = l / 16;
      const int v1 = (static_cast<int>(ql[l] & 0x0FU) |
                      (static_cast<int>(qh[l] & 0x03U) << 4)) -
                     32;
      const int v2 = (static_cast<int>(ql[l + 32] & 0x0FU) |
                      (static_cast<int>((qh[l] >> 2U) & 0x03U) << 4)) -
                     32;
      const int v3 = (static_cast<int>(ql[l] >> 4U) |
                      (static_cast<int>((qh[l] >> 4U) & 0x03U) << 4)) -
                     32;
      const int v4 = (static_cast<int>(ql[l + 32] >> 4U) |
                      (static_cast<int>((qh[l] >> 6U) & 0x03U) << 4)) -
                     32;
      yy[l] = d * static_cast<float>(sc[is + 0]) * static_cast<float>(v1);
      yy[l + 32] = d * static_cast<float>(sc[is + 2]) * static_cast<float>(v2);
      yy[l + 64] = d * static_cast<float>(sc[is + 4]) * static_cast<float>(v3);
      yy[l + 96] = d * static_cast<float>(sc[is + 6]) * static_cast<float>(v4);
    }
  }
}

__device__ void DequantBlockIQ2_XXS(const void* blk, float* out) {
  const auto* b = static_cast<const block_iq2_xxs*>(blk);
  const float d = H2F(b->d);
  const auto* qs = reinterpret_cast<const std::uint8_t*>(b->qs);
  for (int ib32 = 0; ib32 < 8; ++ib32) {
    std::uint64_t group = 0;
    __builtin_memcpy(&group, qs + 8 * ib32, 8);
    const std::uint32_t aux32_1 = static_cast<std::uint32_t>(group >> 32);
    const float db = d * (0.5F + static_cast<float>(aux32_1 >> 28)) * 0.25F;
    for (int l = 0; l < 4; ++l) {
      const std::uint8_t code = static_cast<std::uint8_t>(group >> (8 * l));
      const std::uint64_t grid = kIq2xxsGrid[code];
      const std::uint8_t signs = IqSigns((aux32_1 >> (7 * l)) & 127U);
      float* yy = out + ib32 * 32 + l * 8;
      for (int j = 0; j < 8; ++j) {
        const float mag =
            static_cast<float>(static_cast<std::uint8_t>(grid >> (8 * j)));
        yy[j] = db * mag * ((signs & (1U << j)) ? -1.0F : 1.0F);
      }
    }
  }
}

__device__ void DequantBlockIQ3_XXS(const void* blk, float* out) {
  const auto* b = static_cast<const block_iq3_xxs*>(blk);
  const float d = H2F(b->d);
  const std::uint8_t* qs = b->qs;
  for (int ib32 = 0; ib32 < 8; ++ib32) {
    std::uint32_t aux32 = 0;
    __builtin_memcpy(&aux32, qs + 64 + 4 * ib32, 4);
    const float db = d * (0.5F + static_cast<float>(aux32 >> 28)) * 0.5F;
    const std::uint8_t* codes = qs + 8 * ib32;
    float* yy = out + ib32 * 32;
    for (int l = 0; l < 4; ++l) {
      const std::uint8_t signs = IqSigns((aux32 >> (7 * l)) & 127U);
      const std::uint32_t g1 = kIq3xxsGrid[codes[2 * l + 0]];
      const std::uint32_t g2 = kIq3xxsGrid[codes[2 * l + 1]];
      for (int j = 0; j < 4; ++j) {
        const float m1 =
            static_cast<float>(static_cast<std::uint8_t>(g1 >> (8 * j)));
        const float m2 =
            static_cast<float>(static_cast<std::uint8_t>(g2 >> (8 * j)));
        yy[l * 8 + j] = db * m1 * ((signs & (1U << j)) ? -1.0F : 1.0F);
        yy[l * 8 + j + 4] =
            db * m2 * ((signs & (1U << (j + 4))) ? -1.0F : 1.0F);
      }
    }
  }
}

__device__ void DequantBlockQ2_K(const void* blk, float* out) {
  const auto* b = static_cast<const block_q2_K*>(blk);
  const float d = H2F(b->d);
  const float mn = H2F(b->dmin);
  for (int half = 0; half < 2; ++half) {
    const std::uint8_t* q = b->qs + half * 32;
    float* yy = out + half * 128;
    int is = half * 8;
    for (int j = 0; j < 4; ++j) {
      for (int grp = 0; grp < 2; ++grp) {
        const std::uint8_t sc = b->scales[is++];
        const float dl = d * static_cast<float>(sc & 0x0FU);
        const float ml = mn * static_cast<float>(sc >> 4U);
        const std::uint8_t* qq = q + grp * 16;
        float* yt = yy + j * 32 + grp * 16;
        for (int l = 0; l < 16; ++l) {
          const int qv = (qq[l] >> (2 * j)) & 0x03U;
          yt[l] = dl * static_cast<float>(qv) - ml;
        }
      }
    }
  }
}

__device__ void DequantBlockQ3_K(const void* blk, float* out) {
  const auto* b = static_cast<const block_q3_K*>(blk);
  const float d_all = H2F(b->d);
  std::uint32_t aux[4];
  __builtin_memcpy(aux, b->scales, 12);
  const std::uint32_t kmask1 = 0x03030303U;
  const std::uint32_t kmask2 = 0x0f0f0f0fU;
  const std::uint32_t tmp = aux[2];
  aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
  aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
  aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
  aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
  const auto* scales = reinterpret_cast<const std::int8_t*>(aux);
  const std::uint8_t* q = b->qs;
  const std::uint8_t* hm = b->hmask;
  std::uint8_t m = 1;
  int is = 0;
  for (int half = 0; half < 2; ++half) {
    for (int j = 0; j < 4; ++j) {
      for (int grp = 0; grp < 2; ++grp) {
        const float dl = d_all * static_cast<float>(scales[is++] - 32);
        const std::uint8_t* qq = q + grp * 16;
        const std::uint8_t* hh = hm + grp * 16;
        float* yt = out + half * 128 + j * 32 + grp * 16;
        for (int l = 0; l < 16; ++l) {
          const int qv = (qq[l] >> (2 * j)) & 0x03U;
          const int hi = (hh[l] & m) ? 0 : 4;
          yt[l] = dl * static_cast<float>(qv - hi);
        }
      }
      m = static_cast<std::uint8_t>(m << 1);
    }
    q += 32;
  }
}

__device__ void DequantBlockIQ2_S(const void* blk, float* out) {
  const auto* b = static_cast<const block_iq2_s*>(blk);
  const float d = H2F(b->d);
  for (int ib32 = 0; ib32 < 8; ++ib32) {
    const std::uint8_t sc = b->scales[ib32];
    const float db0 = d * (0.5F + static_cast<float>(sc & 0x0FU)) * 0.25F;
    const float db1 = d * (0.5F + static_cast<float>(sc >> 4U)) * 0.25F;
    for (int l = 0; l < 4; ++l) {
      const float dl = (l < 2) ? db0 : db1;
      const std::uint16_t code = static_cast<std::uint16_t>(
          b->qs[(4 * ib32) + l] |
          ((static_cast<std::uint16_t>(b->qh[ib32]) << (8 - 2 * l)) & 0x300U));
      const std::uint64_t grid = kIq2sGrid[code];
      const std::uint8_t signs = b->qs[32 + (4 * ib32) + l];
      float* yy = out + ib32 * 32 + l * 8;
      for (int j = 0; j < 8; ++j) {
        const float mag =
            static_cast<float>(static_cast<std::uint8_t>(grid >> (8 * j)));
        yy[j] = dl * mag * ((signs & (1U << j)) ? -1.0F : 1.0F);
      }
    }
  }
}

__device__ void DequantBlockQ8_0(const void* blk, float* out) {
  const auto* b = static_cast<const block_q8_0*>(blk);
  const float d = H2F(b->d);
  for (int i = 0; i < 32; ++i)
    out[i] = d * static_cast<float>(b->qs[i]);
}

__device__ void DequantBlock(int type, const void* blk, float* out) {
  switch (type) {
    case kTypeF32:
      *out = *static_cast<const float*>(blk);
      return;
    case kTypeF16:
      *out = H2F(*static_cast<const std::uint16_t*>(blk));
      return;
    case kTypeQ8_0:
      DequantBlockQ8_0(blk, out);
      return;
    case kTypeQ6_K:
      DequantBlockQ6_K(blk, out);
      return;
    case kTypeIQ2_XXS:
      DequantBlockIQ2_XXS(blk, out);
      return;
    case kTypeIQ3_XXS:
      DequantBlockIQ3_XXS(blk, out);
      return;
    case kTypeIQ2_S:
      DequantBlockIQ2_S(blk, out);
      return;
    case kTypeQ2_K:
      DequantBlockQ2_K(blk, out);
      return;
    case kTypeQ3_K:
      DequantBlockQ3_K(blk, out);
      return;
    default:
      return;
  }
}

// ---- kernels ----------------------------------------------------------------

/// y[row] = W[row] @ x over `total_rows` output rows.
/// With `ids != nullptr` output row r maps to expert ids[r / m], row r % m of
/// a [n_experts, m, n] weight tensor (MoE, shared x).
/// With `x_group > 0`, output rows of group g read x at x + g*n (per-group
/// activations, e.g. per-head k_b/v_b projections).
template<int TYPE>
__global__ void RowDotKernel(const void* W, const float* x,
                             const int32_t* __restrict__ ids, float* y,
                             int total_rows, int m, int n_blocks, int qk,
                             int block_bytes, int tpr, int x_group) {
  __shared__ float red[64];
  const int r = static_cast<int>(threadIdx.x) / tpr;
  const int t = static_cast<int>(threadIdx.x) % tpr;
  const int out_row = static_cast<int>(blockIdx.x) * (64 / tpr) + r;
  const int row_bytes = n_blocks * block_bytes;

  float acc = 0.0F;
  bool valid = out_row < total_rows;
  int row_in_matrix = out_row;
  const auto* base = static_cast<const std::uint8_t*>(W);
  const float* x_base = x;
  if (valid && ids != nullptr) {
    const int expert = ids[out_row / m];
    row_in_matrix = out_row % m;
    valid = expert >= 0;
    if (valid)
      base += static_cast<std::size_t>(expert) * m * row_bytes;
    if (valid && x_group > 0)
      x_base = x + static_cast<std::size_t>(out_row / x_group) * n_blocks * qk;
  } else if (valid && x_group > 0) {
    x_base = x + static_cast<std::size_t>(out_row / x_group) * n_blocks * qk;
  }
  if (valid) {
    const std::uint8_t* w_row =
        base + static_cast<std::size_t>(row_in_matrix) * row_bytes;
    for (int b = t; b < n_blocks; b += tpr) {
      acc += BlockDotT<TYPE>(w_row + static_cast<std::size_t>(b) * block_bytes,
                             x_base + static_cast<std::size_t>(b) * qk);
    }
  }
  red[threadIdx.x] = acc;
  __syncthreads();
  // In-order reduction per row: matches a serial dot when a thread holds at
  // most one block (a tree reduction can flip MoE top-k selection through
  // rounding).
  if (t == 0) {
    float s = 0.0F;
    for (int i = 0; i < tpr; ++i)
      s += red[r * tpr + i];
    if (out_row < total_rows)
      y[out_row] = s;
  }
}

/// out[hidden] = dequantized row `row` of a [vocab, hidden] weight matrix.
__global__ void EmbedRowKernel(const void* W, float* out, int row, int n_blocks,
                               int qk, int block_bytes, int type) {
  const int b = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                static_cast<int>(threadIdx.x);
  if (b >= n_blocks)
    return;
  const auto* base = static_cast<const std::uint8_t*>(W);
  const std::uint8_t* w_row =
      base + static_cast<std::size_t>(row) * n_blocks * block_bytes;
  DequantBlock(type, w_row + static_cast<std::size_t>(b) * block_bytes,
               out + static_cast<std::size_t>(b) * qk);
}

constexpr int kThreads = 64;

bool LaunchMmvq(const void* W, const float* x, const int32_t* ids, float* y,
                int total_rows, int m, int n, int ggml_type, int x_group,
                hipStream_t stream);

bool LaunchRowDot(const void* W, const float* x, const int32_t* ids, float* y,
                  int total_rows, int m, int n, int ggml_type, int x_group,
                  hipStream_t stream) {
  // Quantized weights: Maya's Q8_1 / dp4a tier. Dense F32/F16 stay on the
  // direct F32-activation kernel below.
  switch (ggml_type) {
    case kTypeQ8_0:
    case kTypeQ2_K:
    case kTypeQ3_K:
    case kTypeQ6_K:
    case kTypeIQ2_XXS:
    case kTypeIQ3_XXS:
    case kTypeIQ2_S:
      return LaunchMmvq(W, x, ids, y, total_rows, m, n, ggml_type, x_group,
                        stream);
    default:
      break;
  }
  const TypeTraits tr = Traits(ggml_type);
  if (W == nullptr || x == nullptr || y == nullptr || tr.qk == 0 ||
      total_rows <= 0 || n <= 0 || (n % tr.qk) != 0 ||
      (ids != nullptr && m <= 0))
    return false;
  const int n_blocks = n / tr.qk;
  int tpr = 1;
  while (tpr < n_blocks && tpr < kThreads)
    tpr *= 2;
  const int rpw = kThreads / tpr;
  const int grid = (total_rows + rpw - 1) / rpw;
  const auto launch = [&]<int TYPE>() {
    hipLaunchKernelGGL(RowDotKernel<TYPE>, dim3(grid), dim3(kThreads), 0,
                       stream, W, x, ids, y, total_rows, m, n_blocks, tr.qk,
                       tr.block_bytes, tpr, x_group);
  };
  switch (ggml_type) {
    case kTypeF32:
      launch.template operator()<kTypeF32>();
      break;
    case kTypeF16:
      launch.template operator()<kTypeF16>();
      break;
    case kTypeQ8_0:
      launch.template operator()<kTypeQ8_0>();
      break;
    case kTypeQ2_K:
      launch.template operator()<kTypeQ2_K>();
      break;
    case kTypeQ3_K:
      launch.template operator()<kTypeQ3_K>();
      break;
    case kTypeQ6_K:
      launch.template operator()<kTypeQ6_K>();
      break;
    case kTypeIQ2_XXS:
      launch.template operator()<kTypeIQ2_XXS>();
      break;
    case kTypeIQ3_XXS:
      launch.template operator()<kTypeIQ3_XXS>();
      break;
    case kTypeIQ2_S:
      launch.template operator()<kTypeIQ2_S>();
      break;
    default:
      return false;
  }
  return true;
}

// ---- Q8_1 activation tier -------------------------------------------------
// CUDA SIMD-in-word helpers, ported from the vendored mmq vendors/hip.h.
typedef std::int8_t int8x4_t __attribute__((ext_vector_type(4)));
typedef std::uint8_t uint8x4_t __attribute__((ext_vector_type(4)));

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
// Ports of Maya glm_fast.cu (itself a transcription of llama.cpp vecdotq):
// activations quantize to Q8_1 once per GEMV, then each row's dots run in
// packed int8 (__dp4a) with the dequant scales applied per 8/32-element call.
// This is the arithmetic Maya's native_mmvq runs; the F32-activation block
// dots above are only the F32/F16-weight fallback and the embed lookup.

struct block_q8_1 {
  __half2 ds;  // d = scale (low), s = sum(x) (high)
  std::int8_t qs[32];
};

// Maya glm_fast.cu warp_sum / warp_max — RDNA3 DPP lane path on GFX11
// (bit-identical to __shfl_xor_sync per Maya's comments).
template<int Ctrl>
__device__ __forceinline__ float LaneDpp(float v) {
  return __builtin_bit_cast(
      float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), Ctrl,
                                         0xf, 0xf, false));
}

__device__ __forceinline__ float LaneXor16(float v) {
  const int i = __builtin_bit_cast(int, v);
  return __builtin_bit_cast(
      float,
      __builtin_amdgcn_permlanex16(i, i, 0x76543210, 0xfedcba98, false, false));
}

__device__ __forceinline__ float WarpSum32(float v) {
  // Maya glm_fast.cu warp_sum_lanes (GFX11 / Strix Halo).
  v += LaneXor16(v);
  v += LaneDpp<0x168>(v);
  v += LaneDpp<0x164>(v);
  v += LaneDpp<0x162>(v);
  v += LaneDpp<0x161>(v);
  return v;
}

__device__ __forceinline__ float WarpMax32(float v) {
  v = fmaxf(v, LaneXor16(v));
  v = fmaxf(v, LaneDpp<0x168>(v));
  v = fmaxf(v, LaneDpp<0x164>(v));
  v = fmaxf(v, LaneDpp<0x162>(v));
  v = fmaxf(v, LaneDpp<0x161>(v));
  return v;
}

// q8_1 quantization of one 32-element activation group (q8_1_store_warp).
__device__ __forceinline__ void Q81StoreWarp(float v, block_q8_1* dst,
                                             int lane) {
  const float amax = WarpMax32(fabsf(v));
  const float sum = WarpSum32(v);
  const float d = fminf(amax / 127.0f, 65504.0f);
  const auto q = static_cast<std::int8_t>(
      amax == 0.0f ? 0 : fmaxf(-127.0f, fminf(127.0f, roundf(v / d))));
  dst->qs[lane] = q;
  if (lane == 0) {
    __half2 ds;
    ds.x = __float2half(d);
    ds.y = __float2half(fmaxf(-65504.0f, fminf(65504.0f, sum)));
    dst->ds = ds;
  }
}

__device__ __forceinline__ int GetIntB2(const void* x, int i32) {
  const auto* x16 = static_cast<const std::uint16_t*>(x);
  return static_cast<int>(x16[2 * i32 + 0]) |
         (static_cast<int>(x16[2 * i32 + 1]) << 16);
}

__device__ __forceinline__ int GetIntB4(const void* x, int i32) {
  return static_cast<const int*>(x)[i32];
}

// 7-bit sign pattern -> per-byte sign masks (llama unpack_ksigns).
__device__ __forceinline__ std::uint32_t UnpackKsigns(std::uint8_t v) {
  const std::uint32_t p = __popc(v) & 1;
  const std::uint32_t s = v ^ (p << 7);
  return s * 0x01010101U;
}

__device__ __forceinline__ float H2Lo(__half2 h) {
  return __low2float(h);
}

// ---- dots: one call handles 8 or 32 weights of one block ------------------

__device__ __forceinline__ float DotQ8_0(const void* vbq,
                                         const block_q8_1* bq8_1, int kbx,
                                         int iqs) {
  const auto* w = static_cast<const block_q8_0*>(vbq) + kbx;
  int sumi = 0;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int v = GetIntB2(w->qs, iqs + i);
    const int u = GetIntB4(bq8_1->qs, iqs + i);
    sumi = __dp4a(v, u, sumi);
  }
  return H2F(w->d) * H2Lo(bq8_1->ds) * static_cast<float>(sumi);
}

__device__ __forceinline__ float DotQ2_K(const void* vbq,
                                         const block_q8_1* bq8_1, int kbx,
                                         int iqs) {
  const auto* b = static_cast<const block_q2_K*>(vbq) + kbx;
  const int bq8_offset = 4 * (iqs / 8);
  const std::uint8_t* scales =
      reinterpret_cast<const std::uint8_t*>(b->scales) +
      (iqs - iqs % 8 + (iqs % 8) / 4);
  const int v = GetIntB4(b->qs, iqs);
  float sumf_d = 0.0f, sumf_m = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const block_q8_1* q8 = bq8_1 + bq8_offset + i;
    const int u = GetIntB4(q8->qs, iqs % 8);
    const float d8 = H2Lo(q8->ds);
    const int sc = scales[2 * i];
    const int vi = (v >> (2 * i)) & 0x03030303;
    sumf_d += d8 * (__dp4a(vi, u, 0) * (sc & 0xF));
    int m = sc >> 4;
    m |= m << 8;
    m |= m << 16;
    sumf_m += d8 * __dp4a(m, u, 0);
  }
  return H2F(b->d) * sumf_d - H2F(b->dmin) * sumf_m;
}

__device__ __forceinline__ float DotQ3_K(const void* vbq,
                                         const block_q8_1* bq8_1, int kbx,
                                         int iqs) {
  const auto* b = static_cast<const block_q3_K*>(vbq) + kbx;
  const int bq8_offset = 4 * (iqs / 8);
  const int scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
  const float d = H2F(b->d);
  const int vl = GetIntB2(b->qs, iqs);
  const int vh = ~GetIntB2(b->hmask, iqs % 8) >> bq8_offset;
  float sumf = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const block_q8_1* q8 = bq8_1 + bq8_offset + i;
    const int u = GetIntB4(q8->qs, iqs % 8);
    const float d8 = H2Lo(q8->ds);
    const int isc = scale_offset + 2 * i;
    const int sc_low = (b->scales[isc % 8] >> (4 * (isc / 8))) & 0xF;
    const int sc_high = ((b->scales[8 + isc % 4] >> (2 * (isc / 4))) & 3) << 4;
    const int sc = (sc_low | sc_high) - 32;
    const int vil = (vl >> (2 * i)) & 0x03030303;
    const int vih = ((vh >> i) << 2) & 0x04040404;
    sumf += d8 * (__dp4a(static_cast<int>(__vsubss4(vil, vih)), u, 0) * sc);
  }
  return d * sumf;
}

__device__ __forceinline__ float DotQ6_K(const void* vbq,
                                         const block_q8_1* bq8_1, int kbx,
                                         int iqs) {
  const auto* w = static_cast<const block_q6_K*>(vbq) + kbx;
  const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
  const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
  const int vh_shift = 2 * ((iqs % 16) / 8);
  const int vl = GetIntB2(w->ql, iqs);
  const int vh = GetIntB2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
  const std::int8_t* scales = w->scales + scale_offset;
  float sumf = 0.0f;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int u = GetIntB4(bq8_1[bq8_offset + 2 * i].qs, iqs % 8);
    const float d8 = H2Lo(bq8_1[bq8_offset + 2 * i].ds);
    const int sc = scales[4 * i];
    const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
    const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
    const int vi = __vsubss4(vil | vih, 0x20202020);
    sumf += d8 * (__dp4a(vi, u, 0) * sc);
  }
  return H2F(w->d) * sumf;
}

__device__ __forceinline__ float DotIQ2_XXS(const void* vbq,
                                            const block_q8_1* bq8_1, int kbx,
                                            int iqs) {
  const auto* bq2 = static_cast<const block_iq2_xxs*>(vbq) + kbx;
  const int q2 = GetIntB2(bq2->qs, iqs);
  const auto* aux8 = reinterpret_cast<const std::uint8_t*>(&q2);
  const std::uint32_t aux32 =
      static_cast<std::uint32_t>(GetIntB2(bq2->qs, iqs + 1));
  int sumi = 0;
#pragma unroll
  for (int k0 = 0; k0 < 8; k0 += 2) {
    const uint2 grid_pos =
        reinterpret_cast<const uint2*>(kIq2xxsGrid)[aux8[k0 / 2]];
    const std::uint32_t signs = UnpackKsigns((aux32 >> (7 * k0 / 2)) & 0xFF);
    const int signs0 = __vcmpne4(signs & 0x08040201, 0);
    const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);
    const int u0 = GetIntB4(bq8_1[iqs / 2].qs, k0 + 0);
    sumi = __dp4a(grid0, u0, sumi);
    const int signs1 = __vcmpne4(signs & 0x80402010, 0);
    const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);
    const int u1 = GetIntB4(bq8_1[iqs / 2].qs, k0 + 1);
    sumi = __dp4a(grid1, u1, sumi);
  }
  const int ls = (aux32 >> 27) | 1;
  sumi = sumi * ls / 8;
  return H2F(bq2->d) * H2Lo(bq8_1[iqs / 2].ds) * static_cast<float>(sumi);
}

__device__ __forceinline__ float DotIQ3_XXS(const void* vbq,
                                            const block_q8_1* bq8_1, int kbx,
                                            int iqs) {
  const auto* bq3 = static_cast<const block_iq3_xxs*>(vbq) + kbx;
  const int2 q3_packed =
      make_int2(GetIntB2(bq3->qs, iqs), GetIntB2(bq3->qs, iqs + 1));
  const auto* q3 = reinterpret_cast<const std::uint8_t*>(&q3_packed);
  const std::uint32_t aux32 =
      static_cast<std::uint32_t>(GetIntB2(bq3->qs, 16 + iqs / 2));
  int sumi = 0;
#pragma unroll
  for (int l0 = 0; l0 < 8; l0 += 2) {
    const int2 grid_pos = make_int2(static_cast<int>(kIq3xxsGrid[q3[l0 + 0]]),
                                    static_cast<int>(kIq3xxsGrid[q3[l0 + 1]]));
    const std::uint32_t signs = UnpackKsigns((aux32 >> (7 * l0 / 2)) & 0xFF);
    const int signs0 = __vcmpne4(signs & 0x08040201, 0);
    const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
    const int u0 = GetIntB4(bq8_1[iqs / 2].qs, l0 + 0);
    const int signs1 = __vcmpne4(signs & 0x80402010, 0);
    const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
    const int u1 = GetIntB4(bq8_1[iqs / 2].qs, l0 + 1);
    sumi = __dp4a(grid_l, u0, sumi);
    sumi = __dp4a(grid_h, u1, sumi);
  }
  const int ls = aux32 >> 28;
  sumi = (ls * sumi + sumi / 2) / 2;
  return H2F(bq3->d) * H2Lo(bq8_1[iqs / 2].ds) * static_cast<float>(sumi);
}

__device__ __forceinline__ float DotIQ2_S(const void* vbq,
                                          const block_q8_1* bq8_1, int kbx,
                                          int iqs) {
  const auto* b = static_cast<const block_iq2_s*>(vbq) + kbx;
  const int ib = iqs / 2;  // 32-element group
  const block_q8_1* q8 = bq8_1 + ib;
  const int qs_packed = GetIntB2(b->qs, ib);
  const auto* qs = reinterpret_cast<const std::uint8_t*>(&qs_packed);
  const int qh = b->qh[ib];
  const int signs_packed = GetIntB2(b->qs, 8 + ib);
  const auto* signs8 = reinterpret_cast<const std::uint8_t*>(&signs_packed);
  const int ls0 = b->scales[ib] & 0x0F;
  const int ls1 = b->scales[ib] >> 4;
  int sumi0 = 0, sumi1 = 0;
#pragma unroll
  for (int l0 = 0; l0 < 8; l0 += 2) {
    const uint2 grid_pos = reinterpret_cast<const uint2*>(
        kIq2sGrid)[qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)];
    const int signs0 = __vcmpne4(
        ((signs8[l0 / 2] & 0x03) << 7) | ((signs8[l0 / 2] & 0x0C) << 21),
        0x00000000);
    const int signs1 = __vcmpne4(
        ((signs8[l0 / 2] & 0x30) << 3) | ((signs8[l0 / 2] & 0xC0) << 17),
        0x00000000);
    const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
    const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
    if (l0 < 4) {
      sumi0 = __dp4a(grid_l, GetIntB4(q8->qs, l0 + 0), sumi0);
      sumi0 = __dp4a(grid_h, GetIntB4(q8->qs, l0 + 1), sumi0);
    } else {
      sumi1 = __dp4a(grid_l, GetIntB4(q8->qs, l0 + 0), sumi1);
      sumi1 = __dp4a(grid_h, GetIntB4(q8->qs, l0 + 1), sumi1);
    }
  }
  const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
  return H2F(b->d) * H2Lo(q8->ds) * static_cast<float>(sumi);
}

// ---- F<T>: qk / calls-per-block / iqs stride / block bytes / dot ----------
template<int T>
struct FTrait;
template<>
struct FTrait<kTypeQ8_0> {
  static constexpr int qk = 32, ipb = 4, step = 2, bytes = 34;
};
template<>
struct FTrait<kTypeQ2_K> {
  static constexpr int qk = 256, ipb = 16, step = 1, bytes = 84;
};
template<>
struct FTrait<kTypeQ3_K> {
  static constexpr int qk = 256, ipb = 16, step = 1, bytes = 110;
};
template<>
struct FTrait<kTypeQ6_K> {
  static constexpr int qk = 256, ipb = 32, step = 1, bytes = 210;
};
template<>
struct FTrait<kTypeIQ2_XXS> {
  static constexpr int qk = 256, ipb = 8, step = 2, bytes = 66;
};
template<>
struct FTrait<kTypeIQ3_XXS> {
  static constexpr int qk = 256, ipb = 8, step = 2, bytes = 98;
};
template<>
struct FTrait<kTypeIQ2_S> {
  static constexpr int qk = 256, ipb = 8, step = 2, bytes = 82;
};

template<int T>
__device__ __forceinline__ float DotCall(const void* v, const block_q8_1* y,
                                         int kbx, int iqs) {
  if constexpr (T == kTypeQ8_0)
    return DotQ8_0(v, y, kbx, iqs);
  else if constexpr (T == kTypeQ2_K)
    return DotQ2_K(v, y, kbx, iqs);
  else if constexpr (T == kTypeQ3_K)
    return DotQ3_K(v, y, kbx, iqs);
  else if constexpr (T == kTypeQ6_K)
    return DotQ6_K(v, y, kbx, iqs);
  else if constexpr (T == kTypeIQ2_XXS)
    return DotIQ2_XXS(v, y, kbx, iqs);
  else if constexpr (T == kTypeIQ3_XXS)
    return DotIQ3_XXS(v, y, kbx, iqs);
  else
    return DotIQ2_S(v, y, kbx, iqs);
}

// One row against one Q8_1 activation vector, a 32-lane group lane-strided
// over the row's calls (Maya row_dot), reduced with warp_sum.
template<int T>
__device__ __forceinline__ float RowDotQ8(const std::uint8_t* row,
                                          const block_q8_1* x, int n_in,
                                          int lane) {
  using F = FTrait<T>;
  const int nb = n_in / F::qk;
  float s = 0.0f;
#pragma unroll 1
  for (int k = lane; k < nb * F::ipb; k += 32) {
    const int kbx = k / F::ipb;
    const int iqs = F::step * (k % F::ipb);
    s += DotCall<T>(row, x + kbx * (F::qk / 32), kbx, iqs);
  }
  return WarpSum32(s);
}

// 32-lane rows: 2 rows per wave64. ids != nullptr -> MoE gather
// (y[slot] = W[ids[slot/m]][slot%m] @ x); x_group > 0 -> per-group x.
template<int T>
__global__ void MmvqKernel(const void* W, const block_q8_1* x,
                           const std::int32_t* __restrict__ ids, float* y,
                           int total_rows, int m, int n, int x_group) {
  using F = FTrait<T>;
  const std::size_t row_bytes = static_cast<std::size_t>(n / F::qk) * F::bytes;
  const int lane = static_cast<int>(threadIdx.x);
  const int slot =
      static_cast<int>(blockIdx.x) * 2 + static_cast<int>(threadIdx.y);
  if (slot >= total_rows)
    return;
  int row_in_matrix = slot;
  const auto* base = static_cast<const std::uint8_t*>(W);
  const block_q8_1* xq = x;
  if (ids != nullptr) {
    const int expert = ids[slot / m];
    row_in_matrix = slot % m;
    if (expert < 0) {
      if (lane == 0)
        y[slot] = 0.0f;
      return;
    }
    base += static_cast<std::size_t>(expert) * m * row_bytes;
    if (x_group > 0)
      xq = x + static_cast<std::size_t>(slot / x_group) * (n / 32);
  } else if (x_group > 0) {
    xq = x + static_cast<std::size_t>(slot / x_group) * (n / 32);
  }
  const std::uint8_t* w_row =
      base + static_cast<std::size_t>(row_in_matrix) * row_bytes;
  const float s = RowDotQ8<T>(w_row, xq, n, lane);
  if (lane == 0)
    y[slot] = s;
}

__global__ void QuantizeQ81Kernel(const float* __restrict__ x,
                                  block_q8_1* __restrict__ y, int n) {
  const int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                static_cast<int>(threadIdx.x);
  if (i >= n)
    return;  // caller guarantees n % 32 == 0
  Q81StoreWarp(x[i], y + i / 32, static_cast<int>(threadIdx.x) & 31);
}

// Activation Q8_1 scratch: grown on demand (single-stream use, like Maya's
// q8_scratch).
void* g_q81 = nullptr;
std::size_t g_q81_bytes = 0;

block_q8_1* Q81Scratch(std::size_t bytes) {
  if (bytes > g_q81_bytes) {
    // Grow only — never free the old buffer: earlier kernels on the stream
    // may still be reading it (a hipFree here faulted the next row's read).
    // The stale buffers are a few KB each and bounded by the growth steps.
    void* fresh = nullptr;
    if (hipMalloc(&fresh, bytes) != hipSuccess)
      return nullptr;
    g_q81 = fresh;
    g_q81_bytes = bytes;
  }
  return static_cast<block_q8_1*>(g_q81);
}

bool LaunchMmvq(const void* W, const float* x, const int32_t* ids, float* y,
                int total_rows, int m, int n, int ggml_type, int x_group,
                hipStream_t stream) {
  const TypeTraits tr = Traits(ggml_type);
  if (W == nullptr || x == nullptr || y == nullptr || tr.qk == 0 ||
      total_rows <= 0 || n <= 0 || (n % 32) != 0 || (ids != nullptr && m <= 0))
    return false;
  block_q8_1* xq =
      Q81Scratch(static_cast<std::size_t>(n / 32) * sizeof(block_q8_1));
  if (!xq)
    return false;
  // x_group > 0: each group of slots shares one x vector; quantize all of
  // them (one vector would leave the kernel reading past the scratch).
  const int x_vecs = x_group > 0 ? (total_rows + x_group - 1) / x_group : 1;
  const int x_elems = x_vecs * n;
  xq = Q81Scratch(static_cast<std::size_t>(x_elems / 32) * sizeof(block_q8_1));
  hipLaunchKernelGGL(QuantizeQ81Kernel, dim3((x_elems + 255) / 256), dim3(256),
                     0, stream, x, xq, x_elems);
  const int grid = (total_rows + 1) / 2;
  const auto launch = [&]<int TYPE>() {
    hipLaunchKernelGGL(MmvqKernel<TYPE>, dim3(grid), dim3(32, 2), 0, stream, W,
                       xq, ids, y, total_rows, m, n, x_group);
  };
  switch (ggml_type) {
    case kTypeQ8_0:
      launch.template operator()<kTypeQ8_0>();
      break;
    case kTypeQ2_K:
      launch.template operator()<kTypeQ2_K>();
      break;
    case kTypeQ3_K:
      launch.template operator()<kTypeQ3_K>();
      break;
    case kTypeQ6_K:
      launch.template operator()<kTypeQ6_K>();
      break;
    case kTypeIQ2_XXS:
      launch.template operator()<kTypeIQ2_XXS>();
      break;
    case kTypeIQ3_XXS:
      launch.template operator()<kTypeIQ3_XXS>();
      break;
    case kTypeIQ2_S:
      launch.template operator()<kTypeIQ2_S>();
      break;
    default:
      return false;
  }
  return true;
}

}  // namespace

bool QuantGemmSupported(int ggml_type) noexcept {
  return Traits(ggml_type).qk != 0;
}

std::size_t QuantMatrixBytes(int m, int n, int ggml_type) noexcept {
  const TypeTraits tr = Traits(ggml_type);
  if (tr.qk == 0 || m <= 0 || n <= 0 || (n % tr.qk) != 0)
    return 0;
  return static_cast<std::size_t>(m) * (n / tr.qk) * tr.block_bytes;
}

bool QuantGemm(const void* W, const float* x_f32, float* y, int m, int n,
               int ggml_type, hipStream_t stream) {
  return LaunchRowDot(W, x_f32, nullptr, y, m, m, n, ggml_type, 0, stream);
}

bool QuantGemmGroups(const void* W, const float* x, float* y, int groups, int m,
                     int n, int ggml_type, hipStream_t stream) {
  if (groups <= 0)
    return false;
  return LaunchRowDot(W, x, nullptr, y, groups * m, m, n, ggml_type, m, stream);
}

bool QuantGemmMoe(const void* W, const float* x_f32, const int32_t* ids,
                  float* y, int m, int n, int n_experts, int top_k,
                  int ggml_type, hipStream_t stream, int x_group) {
  if (n_experts <= 0 || top_k <= 0)
    return false;
  return LaunchRowDot(W, x_f32, ids, y, top_k * m, m, n, ggml_type, x_group,
                      stream);
}

bool QuantEmbedLookup(const void* embd, float* out, int hidden, int token,
                      int ggml_type, hipStream_t stream) {
  const TypeTraits tr = Traits(ggml_type);
  if (tr.qk == 0 || hidden <= 0 || token < 0 || (hidden % tr.qk) != 0)
    return false;
  const int n_blocks = hidden / tr.qk;
  const int grid = (n_blocks + kThreads - 1) / kThreads;
  hipLaunchKernelGGL(EmbedRowKernel, dim3(grid), dim3(kThreads), 0, stream,
                     embd, out, token, n_blocks, tr.qk, tr.block_bytes,
                     ggml_type);
  return true;
}

}  // namespace gufo::models::glm5_next::gpu
