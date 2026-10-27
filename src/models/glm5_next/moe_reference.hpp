#ifndef GUFO_MODELS_GLM5_NEXT_MOE_REFERENCE_HPP_
#define GUFO_MODELS_GLM5_NEXT_MOE_REFERENCE_HPP_

#include <cstdint>
#include <span>
#include <vector>

namespace gufo::models::glm5_next {

/// MoE (Mixture of Experts) with sigmoid gating + noaux_tc bias — F32 CPU
/// oracle.
///
/// Math contract (Maya ref/glm.py moe_ffn):
///   1. logits = gate_inp @ x  (router F32)
///   2. probs = sigmoid(logits)  — UNBIASED, used for weights
///   3. sel_scores = probs + bias  — bias added ONLY for selection
///   4. sel = top_k_desc(sel_scores, k)  — ties → lower index
///   5. w = probs[sel]; renorm: w /= max(sum(w), 2^-14); w *= scale
///   6. expert output: down @ swiglu_clamp(gate @ x, up @ x, limit)
///      swiglu_clamp: silu(min(gate, limit)) * clamp(up, ±limit)
///   7. shared expert: same swiglu_clamp path
///   8. output = moe + shexp

struct MoeGeometry {
  std::uint32_t hidden_size{4096};
  std::uint32_t num_experts{288};
  std::uint32_t top_k{8};
  std::uint32_t expert_ff{2048};
  std::uint32_t shared_ff{2048};
  float expert_weights_scale{2.5F};
  bool expert_weights_norm{true};
  float swiglu_limit{10.0F};
  float renorm_clamp{6.103515625e-5F};  // 2^-14
};

/// MoE weights for one layer — F32 for the oracle.
struct MoeWeightsF32 {
  /// Router: [num_experts, hidden_size].
  std::vector<float> gate_inp;
  /// noaux_tc bias: [num_experts].
  std::vector<float> exp_probs_b;
  /// Routed experts: gate/up/down per expert.
  /// gate_exps[e]: [hidden, expert_ff], up_exps[e]: [hidden, expert_ff],
  /// down_exps[e]: [expert_ff, hidden].
  std::vector<float> gate_exps;  ///< [num_experts * hidden * expert_ff]
  std::vector<float> up_exps;    ///< [num_experts * hidden * expert_ff]
  std::vector<float> down_exps;  ///< [num_experts * expert_ff * hidden]
  /// Shared expert: gate/up/down [hidden, shared_ff] / [shared_ff, hidden].
  std::vector<float> gate_shexp;  ///< [hidden * shared_ff]
  std::vector<float> up_shexp;
  std::vector<float> down_shexp;  ///< [shared_ff * hidden]
};

/// Selection result for one token.
struct MoeSelection {
  std::vector<std::uint32_t> expert_ids;  ///< [top_k]
  std::vector<float> weights;             ///< [top_k] (renormalized + scaled)
};

/// Select top-k experts for one token (the selection parity boundary).
[[nodiscard]] MoeSelection MoESelect(const MoeGeometry& geom,
                                     const MoeWeightsF32& w,
                                     std::span<const float> token_input);

/// Run MoE FFN for one token. Returns [hidden] output.
[[nodiscard]] std::vector<float> MoEForward(const MoeGeometry& geom,
                                            const MoeWeightsF32& w,
                                            std::span<const float> token_input);

/// Dense FFN (layers 0-2): clamped SwiGLU.
/// gate/up: [hidden, inter], down: [inter, hidden].
[[nodiscard]] std::vector<float> DenseFfnForward(
    std::span<const float> token_input, std::span<const float> gate,
    std::span<const float> up, std::span<const float> down,
    std::uint32_t hidden, std::uint32_t inter, float limit);

}  // namespace gufo::models::glm5_next

#endif  // GUFO_MODELS_GLM5_NEXT_MOE_REFERENCE_HPP_
