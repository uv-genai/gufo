#ifndef GUFO_MODELS_GLM5_NEXT_CPU_ENGINE_HPP_
#define GUFO_MODELS_GLM5_NEXT_CPU_ENGINE_HPP_

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/glm5_next/config.hpp"
#include "src/models/glm5_next/mhc_reference.hpp"
#include "src/models/glm5_next/mla_reference.hpp"
#include "src/models/glm5_next/moe_reference.hpp"
#include "src/models/glm5_next/mtp_reference.hpp"
#include "src/models/glm5_next/reference.hpp"
#include "src/models/glm5_next/weights.hpp"

namespace gufo::models::glm5_next {

/// CPU inference engine using the F32 oracles as the execution path.
/// This is the reference implementation for GPU kernel parity testing
/// and enables smoke tests before GPU kernels land.
class CpuEngine {
public:
  [[nodiscard]] static std::shared_ptr<CpuEngine> Create(
      const Config& config, const ModelWeights& weights,
      std::string* error_msg = nullptr);

  /// Prefill a prompt and return the hidden state after the last token.
  /// tokens: [seq_len] input token IDs.
  /// Returns: [hidden_size] final hidden state.
  [[nodiscard]] std::vector<float> Prefill(
      std::span<const std::uint32_t> tokens);

  /// Decode one token from the current hidden state.
  /// hidden: [hidden_size] from prefill or previous decode.
  /// Returns: token ID.
  [[nodiscard]] std::uint32_t Decode(const std::vector<float>& hidden);

  /// Run MTP draft generation.
  /// embedding: [hidden_size], hidden: [hidden_size].
  /// Returns draft token IDs (fixed width).
  [[nodiscard]] std::vector<std::uint32_t> DraftMtp(
      const std::vector<float>& embedding, const std::vector<float>& hidden);

  [[nodiscard]] const Config& GetConfig() const noexcept { return config_; }

private:
  CpuEngine() = default;

  /// Run one transformer layer (KDA or MLA+DSA + FFN).
  [[nodiscard]] std::vector<float> ForwardLayer(std::uint32_t layer,
                                                std::span<const float> input,
                                                std::uint32_t seq_len);

  Config config_;
  ModelWeights weights_;

  // Per-layer weight caches (F32 for the oracles).
  struct LayerCache {
    KdaWeightsF32 kda;
    MhcWeightsF32 mhc_attn;
    MhcWeightsF32 mhc_ffn;
    MoeWeightsF32 moe;
    MlaWeightsF32 mla;
  };
  std::vector<LayerCache> layer_cache_;

  // Embedding/output head.
  std::vector<float> token_embd_;   ///< [vocab, hidden]
  std::vector<float> output_head_;  ///< [vocab, hidden]
  std::vector<float> output_norm_;  ///< [hidden]

  // MTP weights.
  MtpWeightsF32 mtp_;
};

}  // namespace gufo::models::glm5_next

#endif  // GUFO_MODELS_GLM5_NEXT_CPU_ENGINE_HPP_
