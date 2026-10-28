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
__device__ float BlockDotQ6_K(const void* blk, const float* x) {
  const auto* b = static_cast<const block_q6_K*>(blk);
  const float d = H2F(b->d);
  float acc = 0.0F;
  for (int half = 0; half < 2; ++half) {
    const std::uint8_t* ql = b->ql + half * 64;
    const std::uint8_t* qh = b->qh + half * 32;
    const std::int8_t* sc = b->scales + half * 8;
    const float* xx = x + half * 128;
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
      acc += static_cast<float>(sc[is + 0]) * static_cast<float>(v1) * xx[l];
      acc +=
          static_cast<float>(sc[is + 2]) * static_cast<float>(v2) * xx[l + 32];
      acc +=
          static_cast<float>(sc[is + 4]) * static_cast<float>(v3) * xx[l + 64];
      acc +=
          static_cast<float>(sc[is + 6]) * static_cast<float>(v4) * xx[l + 96];
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

__device__ float BlockDot(int type, const void* blk, const float* x) {
  switch (type) {
    case kTypeF32:
      return BlockDotF32(blk, x);
    case kTypeF16:
      return BlockDotF16(blk, x);
    case kTypeQ8_0:
      return BlockDotQ8_0(blk, x);
    case kTypeQ6_K:
      return BlockDotQ6_K(blk, x);
    case kTypeIQ2_XXS:
      return BlockDotIQ2_XXS(blk, x);
    case kTypeIQ3_XXS:
      return BlockDotIQ3_XXS(blk, x);
    case kTypeIQ2_S:
      return BlockDotIQ2_S(blk, x);
    case kTypeQ2_K:
      return BlockDotQ2_K(blk, x);
    case kTypeQ3_K:
      return BlockDotQ3_K(blk, x);
    default:
      return 0.0F;
  }
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
/// a [n_experts, m, n] weight tensor (MoE).
__global__ void RowDotKernel(const void* W, const float* x,
                             const int32_t* __restrict__ ids, float* y,
                             int total_rows, int m, int n_blocks, int qk,
                             int block_bytes, int type, int tpr) {
  __shared__ float red[64];
  const int r = static_cast<int>(threadIdx.x) / tpr;
  const int t = static_cast<int>(threadIdx.x) % tpr;
  const int out_row = static_cast<int>(blockIdx.x) * (64 / tpr) + r;
  const int row_bytes = n_blocks * block_bytes;

  float acc = 0.0F;
  bool valid = out_row < total_rows;
  int row_in_matrix = out_row;
  const auto* base = static_cast<const std::uint8_t*>(W);
  if (valid && ids != nullptr) {
    const int expert = ids[out_row / m];
    row_in_matrix = out_row % m;
    valid = expert >= 0;
    if (valid)
      base += static_cast<std::size_t>(expert) * m * row_bytes;
  }
  if (valid) {
    const std::uint8_t* w_row =
        base + static_cast<std::size_t>(row_in_matrix) * row_bytes;
    for (int b = t; b < n_blocks; b += tpr) {
      acc += BlockDot(type, w_row + static_cast<std::size_t>(b) * block_bytes,
                      x + static_cast<std::size_t>(b) * qk);
    }
  }
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int s = tpr / 2; s > 0; s >>= 1) {
    if (t < s)
      red[r * tpr + t] += red[r * tpr + t + s];
    __syncthreads();
  }
  if (t == 0 && out_row < total_rows)
    y[out_row] = red[r * tpr];
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

bool LaunchRowDot(const void* W, const float* x, const int32_t* ids, float* y,
                  int total_rows, int m, int n, int ggml_type,
                  hipStream_t stream) {
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
  hipLaunchKernelGGL(RowDotKernel, dim3(grid), dim3(kThreads), 0, stream, W, x,
                     ids, y, total_rows, m, n_blocks, tr.qk, tr.block_bytes,
                     ggml_type, tpr);
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
  return LaunchRowDot(W, x_f32, nullptr, y, m, m, n, ggml_type, stream);
}

bool QuantGemmMoe(const void* W, const float* x_f32, const int32_t* ids,
                  float* y, int m, int n, int n_experts, int top_k,
                  int ggml_type, hipStream_t stream) {
  if (n_experts <= 0 || top_k <= 0)
    return false;
  return LaunchRowDot(W, x_f32, ids, y, top_k * m, m, n, ggml_type, stream);
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
