#ifndef GUFO_MODELS_GLM5_NEXT_QUANT_GEMV_HPP_
#define GUFO_MODELS_GLM5_NEXT_QUANT_GEMV_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::glm5_next::gpu {

/// Quantized matrix-vector multiply for GLM5 weights.
/// Handles IQ2_XXS, IQ3_XXS, Q6_K, Q8_0 — the types in Maya-S packs.
/// Weights stay quantized on GPU; activations quantized to Q8_1 on-the-fly.
/// Follows Maya's gm_gemv dispatch pattern.

/// Quantize F32 activation row to Q8_1 blocks on GPU.
/// x_f32: [k] device. x_q8: [k/32] Q8_1 blocks device.
void QuantizeQ81(const float* x_f32, void* x_q8, int k, hipStream_t stream);

/// Quantized matvec: y[m] = W[m, n] @ x[n] where W is quantized.
/// W: quantized weight matrix (n = inner dim, m = output dim).
/// x_f32: [n] device (F32 activation).
/// y: [m] device (F32 output).
/// ggml_type: 16=IQ2_XXS, 18=IQ3_XXS, 14=Q6_K, 8=Q8_0.
bool QuantGemm(const void* W, const float* x_f32, float* y, int m, int n,
               int ggml_type, hipStream_t stream);

/// Quantized matvec for MoE experts (3D weight tensor).
/// W: [n_experts, m, n] quantized. ids: [top_k] expert indices.
/// x_f32: [n] device. y: [top_k * m] device.
bool QuantGemmMoe(const void* W, const float* x_f32, const int32_t* ids,
                  float* y, int m, int n, int n_experts, int top_k,
                  int ggml_type, hipStream_t stream);

}  // namespace gufo::models::glm5_next::gpu

#endif  // GUFO_MODELS_GLM5_NEXT_QUANT_GEMV_HPP_
