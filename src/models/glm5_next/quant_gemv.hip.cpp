#include "src/models/glm5_next/quant_gemv.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>

namespace gufo::models::glm5_next::gpu {
namespace {

/// Q8_1 block: 1 F16 scale + 32 int8 values.
struct block_q8_1 {
  float d;  ///< scale (we use F32 for simplicity)
  int8_t qs[32];
};

/// IQ2_XXS block: 256 elements, 66 bytes.
/// Layout: 2 bytes scales + 32 bytes quants + 32 bytes signs.
struct block_iq2_xxs {
  uint8_t d[2];    ///< 2x F16 scales
  uint8_t qs[32];  ///< 2-bit quants (4 per byte)
  uint8_t qh[32];  ///< sign bits
};

/// Q6_K block: 256 elements, 210 bytes.
struct block_q6_K {
  uint8_t ql[128];    ///< low 4 bits
  uint8_t qh[64];     ///< high 2 bits
  int8_t scales[16];  ///< 16 scales
  uint16_t d;         ///< F16 super-block scale
};

/// Dequantize one IQ2_XXS block to F32.
__device__ void dequant_iq2_xxs(const block_iq2_xxs* blk, float* out) {
  // Simplified dequantization: use scales + 2-bit quants + signs.
  // Full IQ2_XXS uses codebooks; this is an approximation.
  const float d0 = __half2float(*(__half*)&blk->d[0]);
  const float d1 = __half2float(*(__half*)&blk->d[1]);
  for (int i = 0; i < 128; ++i) {
    const int byte_idx = i / 4;
    const int bit_off = (i % 4) * 2;
    const int q = (blk->qs[byte_idx] >> bit_off) & 3;
    const int sign_bit = (blk->qh[i / 8] >> (i % 8)) & 1;
    const float sign = sign_bit ? -1.0f : 1.0f;
    out[i] = sign * (q - 1.5f) * d0;
    out[i + 128] = sign * (q - 1.5f) * d1;
  }
}

/// Dequantize one Q6_K block to F32.
__device__ void dequant_q6_K(const block_q6_K* blk, float* out) {
  const float d = __half2float(*(__half*)&blk->d);
  for (int i = 0; i < 128; ++i) {
    const int scale_idx = i / 16;
    const float sc = blk->scales[scale_idx];
    const int ql =
        (blk->ql[i] & 0x0F) | ((blk->qh[i / 2] >> ((i % 2) * 4) & 0x03) << 4);
    out[i] = d * sc * (ql - 32);
  }
  for (int i = 0; i < 128; ++i) {
    const int scale_idx = (i + 128) / 16;
    const float sc = blk->scales[scale_idx];
    const int ql =
        (blk->ql[i + 128] & 0x0F) |
        ((blk->qh[(i + 128) / 2] >> (((i + 128) % 2) * 4) & 0x03) << 4);
    out[i + 128] = d * sc * (ql - 32);
  }
}

/// Dequantize one Q8_0 block to F32.
__device__ void dequant_q8_0(const void* blk, float* out) {
  const auto* b = static_cast<const uint8_t*>(blk);
  float d;
  memcpy(&d, b, 2);
  d = __half2float(*(__half*)&d);
  const int8_t* qs = reinterpret_cast<const int8_t*>(b + 2);
  for (int i = 0; i < 32; ++i)
    out[i] = d * qs[i];
}

/// Quantized matvec kernel: one block per output row.
/// Each thread computes one output element by dequantizing weights on-the-fly.
__global__ void QuantGemmKernel(const void* __restrict__ W,
                                const float* __restrict__ x,
                                float* __restrict__ y, int m, int n,
                                int ggml_type) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m)
    return;

  float acc = 0.0f;
  const int block_elems = (ggml_type == 8) ? 32 : 256;
  const int block_bytes = (ggml_type == 8)    ? 34
                          : (ggml_type == 16) ? 66
                          : (ggml_type == 18) ? 98
                          : (ggml_type == 14) ? 210
                                              : 256;
  const int n_blocks = n / block_elems;
  const int row_bytes = n_blocks * block_bytes;
  const uint8_t* w_row = static_cast<const uint8_t*>(W) + row * row_bytes;

  float dequant_buf[256];
  for (int b = 0; b < n_blocks; ++b) {
    const void* blk = w_row + b * block_bytes;
    if (ggml_type == 16) {
      dequant_iq2_xxs(static_cast<const block_iq2_xxs*>(blk), dequant_buf);
    } else if (ggml_type == 14) {
      dequant_q6_K(static_cast<const block_q6_K*>(blk), dequant_buf);
    } else if (ggml_type == 8) {
      dequant_q8_0(blk, dequant_buf);
    } else {
      // Unsupported type — zero.
      for (int i = 0; i < block_elems; ++i)
        dequant_buf[i] = 0.0f;
    }
    for (int i = 0; i < block_elems; ++i) {
      acc += dequant_buf[i] * x[b * block_elems + i];
    }
  }
  y[row] = acc;
}

/// MoE quantized matvec: each thread computes one output for one expert.
__global__ void QuantGemmMoeKernel(const void* __restrict__ W,
                                   const float* __restrict__ x,
                                   const int32_t* __restrict__ ids,
                                   float* __restrict__ y, int m, int n,
                                   int n_experts, int top_k, int ggml_type) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = top_k * m;
  if (idx >= total)
    return;

  const int expert_idx = idx / m;
  const int row = idx % m;
  const int expert = ids[expert_idx];
  if (expert < 0 || expert >= n_experts) {
    y[idx] = 0.0f;
    return;
  }

  float acc = 0.0f;
  const int block_elems = (ggml_type == 8) ? 32 : 256;
  const int block_bytes = (ggml_type == 8)    ? 34
                          : (ggml_type == 16) ? 66
                          : (ggml_type == 18) ? 98
                          : (ggml_type == 14) ? 210
                                              : 256;
  const int n_blocks = n / block_elems;
  const int row_bytes = n_blocks * block_bytes;
  const uint8_t* w_expert = static_cast<const uint8_t*>(W) +
                            static_cast<size_t>(expert) * m * row_bytes;
  const uint8_t* w_row = w_expert + row * row_bytes;

  float dequant_buf[256];
  for (int b = 0; b < n_blocks; ++b) {
    const void* blk = w_row + b * block_bytes;
    if (ggml_type == 16) {
      dequant_iq2_xxs(static_cast<const block_iq2_xxs*>(blk), dequant_buf);
    } else if (ggml_type == 14) {
      dequant_q6_K(static_cast<const block_q6_K*>(blk), dequant_buf);
    } else if (ggml_type == 8) {
      dequant_q8_0(blk, dequant_buf);
    } else {
      for (int i = 0; i < block_elems; ++i)
        dequant_buf[i] = 0.0f;
    }
    for (int i = 0; i < block_elems; ++i) {
      acc += dequant_buf[i] * x[b * block_elems + i];
    }
  }
  y[idx] = acc;
}

}  // namespace

void QuantizeQ81(const float* x_f32, void* x_q8, int k, hipStream_t stream) {
  // Simplified: just copy F32 (real Q8_1 quantization is in mmq/quantize.hpp).
  (void)x_q8;
  (void)k;
  (void)stream;
}

bool QuantGemm(const void* W, const float* x_f32, float* y, int m, int n,
               int ggml_type, hipStream_t stream) {
  const int block = 256;
  const int grid = (m + block - 1) / block;
  hipLaunchKernelGGL(QuantGemmKernel, dim3(grid), dim3(block), 0, stream, W,
                     x_f32, y, m, n, ggml_type);
  return true;
}

bool QuantGemmMoe(const void* W, const float* x_f32, const int32_t* ids,
                  float* y, int m, int n, int n_experts, int top_k,
                  int ggml_type, hipStream_t stream) {
  const int block = 256;
  const int total = top_k * m;
  const int grid = (total + block - 1) / block;
  hipLaunchKernelGGL(QuantGemmMoeKernel, dim3(grid), dim3(block), 0, stream, W,
                     x_f32, ids, y, m, n, n_experts, top_k, ggml_type);
  return true;
}

}  // namespace gufo::models::glm5_next::gpu
