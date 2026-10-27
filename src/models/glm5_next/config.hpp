#ifndef GUFO_MODELS_GLM5_NEXT_CONFIG_HPP_
#define GUFO_MODELS_GLM5_NEXT_CONFIG_HPP_

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::glm5_next {

/// Architecture parameters of a `glm5-next` GGUF artifact (GLM-5.3-Flash).
/// Every value is read from the file; the fixed limits below only bound what
/// this runtime was written and validated for.
///
/// Canonical geometry (Maya-S v2, confirmed against real GGUF 2026-10-27):
///   46 blocks (0-45), hidden 4096, 34 KDA / 12 MLA+DSA layers,
///   288 routed + 1 shared experts, top-8, inter 2048,
///   mHC 4-stream Sinkhorn-20, NextN MTP 1 layer, NoPE, 1M context.
struct Config {
  static constexpr std::uint32_t kMaxLayers = 64;
  static constexpr std::uint32_t kMaxExperts = 512;

  std::string architecture;  ///< "glm5-next"

  // Transformer core.
  std::uint32_t num_layers{0};            ///< Total blocks (46 for Maya-S).
  std::uint32_t hidden_size{0};           ///< 4096.
  std::uint32_t vocab_size{0};            ///< 154880.
  std::uint32_t context_length{0};        ///< 1048576.
  std::uint32_t leading_dense_layers{0};  ///< 3 (blk.0-2 dense FFN).
  float layer_norm_rms_eps{1e-5F};
  float layer_norm_eps{1e-6F};

  // Hyper-connections (mHC): Sinkhorn-normalized residual mixing.
  std::uint32_t hc_count{0};           ///< 4 parallel streams.
  std::uint32_t hc_sinkhorn_iters{0};  ///< 20 Sinkhorn iterations.
  float hc_epsilon{1e-6F};

  // KDA (Kimi Delta Attention) linear mixer.
  std::uint32_t kda_head_dim{0};     ///< 128.
  std::uint32_t kda_conv_kernel{0};  ///< 4.
  float kda_gate_lower_bound{0.0F};  ///< -5.0.

  // Full attention (MLA + DSA indexer) every 4th layer starting at 3.
  std::uint32_t num_attention_heads{0};  ///< 64.
  std::uint32_t q_lora_rank{0};          ///< 1536.
  std::uint32_t kv_lora_rank{0};         ///< 512.
  std::uint32_t qk_nope_dim{0};          ///< 256 (key_length_mla).
  std::uint32_t v_head_dim{0};           ///< 256 (value_length_mla).
  std::uint32_t indexer_heads{0};        ///< 32.
  std::uint32_t indexer_head_dim{0};     ///< 128.
  std::uint32_t indexer_top_k{0};        ///< 2048.
  std::uint32_t indexer_kpool{0};        ///< 4.

  // MoE.
  std::uint32_t num_experts{0};         ///< 288 routed.
  std::uint32_t num_experts_used{0};    ///< 8 top-k.
  std::uint32_t expert_ff{0};           ///< 2048 routed inter.
  std::uint32_t num_shared_experts{0};  ///< 1.
  std::uint32_t shared_expert_ff{0};    ///< 2048 shared inter.
  std::uint32_t expert_gating_func{0};  ///< 2 = sigmoid.
  float expert_weights_scale{0.0F};     ///< 2.5.
  bool expert_weights_norm{false};      ///< true = renorm over selected.

  // Dense FFN (leading layers only).
  std::uint32_t dense_ff{0};  ///< 12288.

  // MTP.
  std::uint32_t nextn_layers{0};  ///< 1 NextN predictor.

  // SwiGLU clamping.
  std::vector<float> swiglu_clamp_exp;
  std::vector<float> swiglu_clamp_shexp;

  // Per-layer attention type from `head_count_kv` pattern (1 = MLA/DSA).
  std::vector<bool> is_full_attention;

  /// Parse a `glm5-next` GGUF artifact's metadata into a Config.
  [[nodiscard]] static std::optional<Config> FromGguf(
      const core::GgufReader& gguf, std::string* error_msg = nullptr);

  // Derived geometry.

  /// KDA heads = attention heads when head_dim divides the QKV size.
  [[nodiscard]] constexpr std::uint32_t KdaNumHeads() const noexcept {
    return kda_head_dim > 0 ? (num_attention_heads * 2) / 2 : 0;
  }
  /// MLA expanded query dim (all heads × qk_nope).
  [[nodiscard]] constexpr std::uint32_t MlaQDim() const noexcept {
    return num_attention_heads * qk_nope_dim;
  }
  /// MLA latent KV dim.
  [[nodiscard]] constexpr std::uint32_t MlaKvLatentDim() const noexcept {
    return kv_lora_rank;
  }
  /// Number of full-attention (DSA) layers.
  [[nodiscard]] std::uint32_t FullAttentionCount() const noexcept {
    std::uint32_t n = 0;
    for (bool b : is_full_attention)
      if (b)
        ++n;
    return n;
  }
  /// Number of KDA linear-attention layers.
  [[nodiscard]] std::uint32_t KdaLayerCount() const noexcept {
    return num_layers - FullAttentionCount();
  }
  /// True if the layer uses MLA+DSA (full attention).
  [[nodiscard]] bool IsFullAttention(std::uint32_t layer) const noexcept {
    return layer < is_full_attention.size() && is_full_attention[layer];
  }
  /// True if the layer has dense FFN (leading layers).
  [[nodiscard]] bool IsDenseFfn(std::uint32_t layer) const noexcept {
    return layer < leading_dense_layers;
  }
  /// True if the layer carries NextN MTP weights (last block).
  [[nodiscard]] bool IsMtpLayer(std::uint32_t layer) const noexcept {
    return nextn_layers > 0 && layer == num_layers - 1;
  }

  /// Estimate resident weight bytes from the config (before any load).
  /// Covers: embeddings, all layer weights, router, mHC, MTP head.
  [[nodiscard]] std::uint64_t EstimateResidentBytes() const noexcept;
};

}  // namespace gufo::models::glm5_next

#endif  // GUFO_MODELS_GLM5_NEXT_CONFIG_HPP_
