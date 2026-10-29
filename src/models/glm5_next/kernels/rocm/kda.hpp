#ifndef GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_KDA_HPP_
#define GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_KDA_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::glm5_next::rocm {

/// KDA (Kimi Delta Attention) GPU kernel interface.
/// Implements the recurrent delta rule with per-channel decay gates.
///
/// Math (matches CPU oracle in reference.cpp):
///   1. Q/K/V projections (GEMM)
///   2. Causal conv kernel 4 + SiLU
///   3. L2 norm q, k (per head)
///   4. Decay gate: lb * sigmoid(-(f_b @ (f_a @ x) + dt_bias) * A)
///   5. Beta: sigmoid(ssm_beta @ x)
///   6. Recurrent: S = diag(exp(g1)) @ S; S += k @ (beta * (v - S^T k))^T
///   7. Output: (1/sqrt(dk)) * S^T @ q
///   8. Output gate: sigmoid(g_b @ (g_a @ x))
///   9. RMS norm + output projection

struct KdaGpuGeometry {
  std::uint32_t hidden_size;
  std::uint32_t num_heads;
  std::uint32_t head_dim;
  std::uint32_t conv_kernel;
  float gate_lower_bound;
  float norm_eps;
};

/// Device pointers for KDA weights (all on GPU).
struct KdaGpuWeights {
  const float* w_q;
  const float* w_k;
  const float* w_v;
  const float* w_out;
  const float* conv_q;
  const float* conv_k;
  const float* conv_v;
  const float* f_a;
  const float* f_b;
  const float* dt_bias;
  const float* ssm_a;
  const float* ssm_beta;
  const float* g_a;
  const float* g_b;
  const float* ssm_norm;
  // Quant types (0=F32, 8=Q8_0, 14=Q6_K, ...). Q8_0/Q6_K go through
  // QuantGemm's Q8_1 tier — the arithmetic Maya's native_mmvq runs.
  int w_q_type{0};
  int w_k_type{0};
  int w_v_type{0};
  int w_out_type{0};
  int f_a_type{0};
  int f_b_type{0};
  int beta_type{0};
  int g_a_type{0};
  int g_b_type{0};
};

/// Device scratch floats KdaPrefillGpu needs for T tokens.
[[nodiscard]] std::size_t KdaScratchFloats(const KdaGpuGeometry& geom,
                                           std::uint32_t seq_len);

/// Run KDA prefill on GPU for a chunk of tokens.
/// input: [hidden, T] on device.
/// state: [num_heads, head_dim, head_dim] on device (recurrent state).
/// conv_state: 3 × (d_conv-1, inner) history floats (oldest first), updated
/// in place — or nullptr to treat history as zero and skip the slide.
/// out: [hidden, T] on device.
/// scratch: KdaScratchFloats() floats of device memory, or nullptr to
/// allocate internally (slow — tests only).
bool KdaPrefillGpu(const KdaGpuGeometry& geom, const KdaGpuWeights& w,
                   const float* input, float* state, float* conv_state,
                   float* out, std::uint32_t seq_len, float* scratch,
                   hipStream_t stream);

/// Run KDA decode on GPU for one token.
/// token_input: [hidden] on device.
/// state: [num_heads, head_dim, head_dim] on device (updated in place).
/// conv_state: same layout as KdaPrefillGpu (updated in place).
/// out: [hidden] on device.
bool KdaDecodeGpu(const KdaGpuGeometry& geom, const KdaGpuWeights& w,
                  const float* token_input, float* state, float* conv_state,
                  float* out, float* scratch, hipStream_t stream);

}  // namespace gufo::models::glm5_next::rocm

#endif  // GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_KDA_HPP_
