#ifndef GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_MOE_HPP_
#define GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_MOE_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::glm5_next::rocm {

/// MoE (Mixture of Experts) GPU kernel interface.
///
/// Math (matches CPU oracle in moe_reference.cpp):
///   1. logits = gate_inp @ x  (router F32)
///   2. probs = sigmoid(logits)  — UNBIASED, used for weights
///   3. sel_scores = probs + bias  — bias added ONLY for selection
///   4. sel = top_k_desc(sel_scores, k)  — ties → lower index
///   5. w = probs[sel]; renorm: w /= max(sum(w), 2^-14); w *= scale
///   6. expert output: down @ swiglu_clamp(gate @ x, up @ x, limit)
///      swiglu_clamp: silu(min(gate, limit)) * clamp(up, ±limit)
///   7. shared expert: same swiglu_clamp path
///   8. output = moe + shexp

struct MoeGpuGeometry {
  std::uint32_t hidden_size;
  std::uint32_t num_experts;
  std::uint32_t top_k;
  std::uint32_t expert_ff;
  std::uint32_t shared_ff;
  float expert_weights_scale;
  bool expert_weights_norm;
  float swiglu_limit;
  float renorm_clamp;
};

/// Device pointers for MoE weights.
struct MoeGpuWeights {
  const float* gate_inp;     ///< [num_experts, hidden] F32
  const float* exp_probs_b;  ///< [num_experts] F32
  const float* gate_exps;    ///< [num_experts, hidden, expert_ff] quantized
  const float* up_exps;      ///< [num_experts, hidden, expert_ff] quantized
  const float* down_exps;    ///< [num_experts, expert_ff, hidden] quantized
  const float* gate_shexp;   ///< [hidden, shared_ff] F32
  const float* up_shexp;     ///< [hidden, shared_ff] F32
  const float* down_shexp;   ///< [shared_ff, hidden] F32
  int gate_exps_type{0};     ///< ggml_type for gate_exps (0=F32, 16=IQ2_XXS)
  int up_exps_type{0};       ///< ggml_type for up_exps
  int down_exps_type{0};     ///< ggml_type for down_exps
};

/// Run MoE forward on GPU for one token.
/// token_input: [hidden] on device.
/// out: [hidden] on device (output).
/// Returns true on success.
bool MoeForwardGpu(const MoeGpuGeometry& geom, const MoeGpuWeights& w,
                   const float* token_input, float* out, hipStream_t stream);

/// Run dense FFN forward on GPU for one token.
/// gate/up: [hidden, inter], down: [inter, hidden] on device.
/// out: [hidden] on device (output).
bool DenseFfnForwardGpu(std::uint32_t hidden, std::uint32_t inter, float limit,
                        const float* gate, const float* up, const float* down,
                        const float* token_input, float* out,
                        hipStream_t stream);

}  // namespace gufo::models::glm5_next::rocm

#endif  // GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_MOE_HPP_
