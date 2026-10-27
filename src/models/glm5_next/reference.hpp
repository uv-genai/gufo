#ifndef GUFO_MODELS_GLM5_NEXT_REFERENCE_HPP_
#define GUFO_MODELS_GLM5_NEXT_REFERENCE_HPP_

#include <cstdint>
#include <span>
#include <vector>

namespace gufo::models::glm5_next {

/// KDA (Kimi Delta Attention) linear-attention mixer — F32 CPU oracle.
/// Implements the recurrent delta rule with per-channel decay gates,
/// low-rank f/g gates, causal conv kernel 4, and output gating.
///
/// Math contract (Maya ref/glm.py kda_layer, llama.cpp glm5-next.cpp L687-763):
///   1. q, k, v = W_{q,k,v} @ x
///   2. causal conv kernel 4 on q, k, v separately
///   3. q, k = l2_norm (v NOT normalized)
///   4. g1 = lb * sigmoid(-(f_b @ (f_a @ x) + dt_bias) * A),  A = -exp(ssm_a)
///   5. beta = sigmoid(ssm_beta @ x) per head
///   6. recurrent: S = diag(exp(g1)) @ S; S += k @ (beta * (v - S^T k))^T
///   7. o = (1/sqrt(dk)) * S^T @ q
///   8. g2 = sigmoid(g_b @ (g_a @ x)) per channel
///   9. out = W_o @ (rms_norm(o) * sigmoid(g2))

struct KdaGeometry {
  std::uint32_t hidden_size{4096};
  std::uint32_t num_heads{64};
  std::uint32_t head_dim{128};  ///< dk = dv = 128
  std::uint32_t conv_kernel{4};
  float gate_lower_bound{-5.0F};
  float norm_eps{1e-5F};
};

/// KDA weights for one layer (F32 for the oracle).
struct KdaWeightsF32 {
  // Q/K/V projections [hidden, num_heads*head_dim] (or equivalent).
  std::vector<float> w_q;  ///< [hidden, inner] where inner = num_heads*head_dim
  std::vector<float> w_k;
  std::vector<float> w_v;
  std::vector<float> w_out;  ///< [inner, hidden]

  // Causal conv [conv_kernel, inner] per stream.
  std::vector<float> conv_q;  ///< [conv_kernel * inner]
  std::vector<float> conv_k;
  std::vector<float> conv_v;

  // Decay gate: low-rank f_a [hidden, rank], f_b [rank, inner], dt_bias
  // [inner].
  std::vector<float> f_a;      ///< [hidden, head_dim] (rank = head_dim)
  std::vector<float> f_b;      ///< [head_dim, inner]
  std::vector<float> dt_bias;  ///< [inner]

  // ssm_a: [num_heads] holding -exp(A_log).
  std::vector<float> ssm_a;

  // Beta: [hidden, num_heads].
  std::vector<float> ssm_beta;

  // Output gate: g_a [hidden, head_dim], g_b [head_dim, inner].
  std::vector<float> g_a;
  std::vector<float> g_b;

  // Norm: [head_dim].
  std::vector<float> ssm_norm;
};

/// Run KDA prefill on a sequence of tokens.
/// input: [hidden, T] column-major (each column is one token).
/// Returns [hidden, T] output.
[[nodiscard]] std::vector<float> KdaPrefill(const KdaGeometry& geom,
                                            const KdaWeightsF32& w,
                                            std::span<const float> input,
                                            std::uint32_t seq_len);

/// Run KDA decode for one token given prior state.
/// state: [num_heads, head_dim, head_dim] recurrent state.
/// Returns output [hidden] and updates state in place.
[[nodiscard]] std::vector<float> KdaDecode(const KdaGeometry& geom,
                                           const KdaWeightsF32& w,
                                           std::span<const float> token_input,
                                           std::span<float> state);

/// Compute KL divergence between two probability distributions.
/// Returns KL(p || q) using log-sum-exp for stability.
[[nodiscard]] float KLDivergence(std::span<const float> p,
                                 std::span<const float> q);

/// Compute max absolute difference between two vectors.
[[nodiscard]] float MaxAbsDiff(std::span<const float> a,
                               std::span<const float> b);

}  // namespace gufo::models::glm5_next

#endif  // GUFO_MODELS_GLM5_NEXT_REFERENCE_HPP_
