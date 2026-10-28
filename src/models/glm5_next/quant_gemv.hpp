#ifndef GUFO_MODELS_GLM5_NEXT_QUANT_GEMV_HPP_
#define GUFO_MODELS_GLM5_NEXT_QUANT_GEMV_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::glm5_next::gpu {

/// Quantized matrix-vector multiply for GLM5 weights.
///
/// Weights stay quantized on GPU in their GGUF storage format; activations are
/// F32 and dequantization happens on-the-fly per block. The dequant paths match
/// llama.cpp `dequantize_row_*` exactly (see the host reference in
/// src/core/quant/ggml_dequant.cpp and the parity tests).
///
/// `ggml_type` is the GGUF type id of the device-side weight bytes:
/// 0=F32, 1=F16, 8=Q8_0, 10=Q2_K, 11=Q3_K, 14=Q6_K, 16=IQ2_XXS, 18=IQ3_XXS,
/// 22=IQ2_S.

/// True when QuantGemm supports weights stored in this ggml_type.
[[nodiscard]] bool QuantGemmSupported(int ggml_type) noexcept;

/// Byte size of one [m, n] row-major quantized matrix in this ggml_type
/// (0 if unsupported or not block aligned). Use for 3D expert-tensor strides.
[[nodiscard]] std::size_t QuantMatrixBytes(int m, int n,
                                           int ggml_type) noexcept;

/// Quantized matvec: y[m] = W[m, n] @ x[n] where W is quantized row-major.
/// W: quantized weight matrix (n = input dim, m = output dim).
/// x_f32: [n] device (F32 activation).
/// y: [m] device (F32 output).
/// Returns false for unsupported ggml_type (no silent zeros).
bool QuantGemm(const void* W, const float* x_f32, float* y, int m, int n,
               int ggml_type, hipStream_t stream);

/// Quantized matvec per group: y[g*m + r] = W[g] row r @ x[g*n + :].
/// W: [groups*m, n] quantized row-major (e.g. per-head k_b/v_b projections).
/// x: [groups*n] device, y: [groups*m] device.
bool QuantGemmGroups(const void* W, const float* x, float* y, int groups, int m,
                     int n, int ggml_type, hipStream_t stream);

/// Quantized matvec for MoE experts (3D weight tensor).
/// W: [n_experts, m, n] quantized. ids: [top_k] expert indices.
/// x_f32: [n] device (x_group=0) or [top_k/x_group, n] with one x per
/// x_group slots (e.g. x_group=m for per-slot down projections).
/// y: [top_k * m] device (expert-major).
bool QuantGemmMoe(const void* W, const float* x_f32, const int32_t* ids,
                  float* y, int m, int n, int n_experts, int top_k,
                  int ggml_type, hipStream_t stream, int x_group = 0);

/// Dequantize one row of a quantized embedding matrix to F32.
/// embd: [vocab, hidden] quantized. out: [hidden] device.
bool QuantEmbedLookup(const void* embd, float* out, int hidden, int token,
                      int ggml_type, hipStream_t stream);

}  // namespace gufo::models::glm5_next::gpu

#endif  // GUFO_MODELS_GLM5_NEXT_QUANT_GEMV_HPP_
