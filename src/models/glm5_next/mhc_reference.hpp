#ifndef GUFO_MODELS_GLM5_NEXT_MHC_REFERENCE_HPP_
#define GUFO_MODELS_GLM5_NEXT_MHC_REFERENCE_HPP_

#include <cstdint>
#include <span>
#include <vector>

namespace gufo::models::glm5_next {

/// mHC (manifold-constrained Hyper-Connections) — F32 CPU oracle.
///
/// Math contract (Maya ref/glm.py hc_pre/hc_sinkhorn/hc_post):
///   Read half (per mixer, attn or ffn):
///     1. Joint inverse-RMS: ONE norm over all hc streams (not per-stream).
///     2. mixes = hc_fn @ flat_norm  (24-row GEMV).
///     3. pre = sigmoid(mixes[0:hc]*scale[0] + base[0:hc]) + eps
///     4. post = 2*sigmoid(mixes[hc:2hc]*scale[1] + base[hc:2hc])
///     5. comb[dst,src] = sinkhorn(mixes[2hc+dst+hc*src]*scale[2] +
///     base[2hc+...])
///        NOTE: src-major row order (dst + hc*src).
///     6. mixed = sum_s pre_s * R_s
///   Sinkhorn (per token, per layer):
///     softmax over dst, +eps, dst-norm, then (iters-1)×[src-norm, dst-norm].
///   Write half:
///     out_dst = block_out * post_dst + sum_src comb[dst,src] * residual_src

struct MhcGeometry {
  std::uint32_t hidden_size{4096};
  std::uint32_t hc_count{4};     ///< Number of parallel streams.
  std::uint32_t hc_mix_dim{24};  ///< 24 = hc_count * 6? (from GGUF fn shape).
  std::uint32_t sinkhorn_iters{20};
  float eps{1e-6F};
  float norm_eps{1e-5F};
};

/// mHC weights for one mixer (attn or ffn) — F32 for the oracle.
struct MhcWeightsF32 {
  /// fn: [hc_mix_dim, hc_count * hidden_size] — the projection.
  std::vector<float> fn;
  /// scale: [3] — per-gate scale factors.
  std::vector<float> scale;
  /// base: [2*hc_count + hc_count*hc_count] — per-gate biases.
  std::vector<float> base;
};

/// Run mHC read half: given residual streams R [hidden, hc, T],
/// returns (mixed [hidden, T], post [hc, T], comb [hc, hc, T]).
struct MhcReadOutput {
  std::vector<float> mixed;  ///< [hidden, T]
  std::vector<float> post;   ///< [hc, T]
  std::vector<float> comb;   ///< [hc * hc * T] (dst-major per token)
};

[[nodiscard]] MhcReadOutput MhcPre(const MhcGeometry& geom,
                                   const MhcWeightsF32& w,
                                   std::span<const float> residual,
                                   std::uint32_t seq_len);

/// Sinkhorn normalization on a single [hc, hc] matrix (one token).
/// Input: raw comb values. Output: normalized comb (dst-stochastic).
[[nodiscard]] std::vector<float> MhcSinkhorn(const MhcGeometry& geom,
                                             std::span<const float> raw_comb);

/// Run mHC write half: out_dst = block_out * post_dst + sum_src comb[dst,src] *
/// residual_src. block_out: [hidden, T], residual: [hidden, hc, T], post: [hc,
/// T], comb: [hc*hc*T]. Returns [hidden, hc, T].
[[nodiscard]] std::vector<float> MhcPost(const MhcGeometry& geom,
                                         std::span<const float> block_out,
                                         std::span<const float> residual,
                                         std::span<const float> post,
                                         std::span<const float> comb,
                                         std::uint32_t seq_len);

}  // namespace gufo::models::glm5_next

#endif  // GUFO_MODELS_GLM5_NEXT_MHC_REFERENCE_HPP_
