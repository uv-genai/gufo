#ifndef GUFO_MODELS_GLM5_NEXT_MTP_REFERENCE_HPP_
#define GUFO_MODELS_GLM5_NEXT_MTP_REFERENCE_HPP_

#include <cstdint>
#include <span>
#include <vector>

namespace gufo::models::glm5_next {

/// NextN MTP (Multi-Token Prediction) — F32 CPU oracle.
///
/// Math contract (from GGUF tensor shapes + Maya study):
///   1. enorm = rms_norm(embedding)
///   2. hnorm = rms_norm(hidden_state)
///   3. eh_proj @ [enorm | hnorm]  (8192 = 2*4096, packed across rows)
///   4. shared_head_norm on the projection output
///   5. Output head (shared with target model) produces draft logits
///
/// The MTP layer predicts the next token from the current embedding and
/// hidden state. Draft width is FIXED (no adaptive — TP2 mirrors rank 0).

struct MtpGeometry {
  std::uint32_t hidden_size{4096};
  std::uint32_t vocab_size{154880};
  float norm_eps{1e-5F};
  std::uint32_t fixed_draft_width{4};  ///< Fixed number of draft tokens.
};

/// MTP weights — F32 for the oracle.
struct MtpWeightsF32 {
  std::vector<float> enorm;    ///< [hidden]
  std::vector<float> hnorm;    ///< [hidden]
  std::vector<float> eh_proj;  ///< [hidden, 2*hidden] ([embed|hidden] packed)
  std::vector<float> shared_head_norm;  ///< [hidden]
  /// Shared output head: [vocab, hidden] (same as target model's
  /// output.weight).
  std::vector<float> output_head;
};

/// MTP state for verify window (KDA rollback support).
struct MtpState {
  /// KDA states for each layer (for rollback on reject).
  std::vector<float> kda_states;
  /// Current draft width (fixed under TP).
  std::uint32_t draft_width{0};
};

/// Run MTP draft generation for one step.
/// embedding: [hidden], hidden_state: [hidden].
/// Returns draft logits [vocab].
[[nodiscard]] std::vector<float> MtpDraft(const MtpGeometry& geom,
                                          const MtpWeightsF32& w,
                                          std::span<const float> embedding,
                                          std::span<const float> hidden_state);

/// Verify a draft token against the target model's greedy output.
/// Returns true if draft matches, false if rejected (triggers rollback).
[[nodiscard]] bool MtpVerify(std::uint32_t draft_token,
                             std::uint32_t target_token);

/// Rollback KDA state to the last verified position.
/// state: [num_layers * num_heads * head_dim * head_dim]
/// layer_states: per-layer checkpoint (copied at last verify).
void MtpRollbackKdaState(std::span<float> state,
                         std::span<const float> checkpoint);

}  // namespace gufo::models::glm5_next

#endif  // GUFO_MODELS_GLM5_NEXT_MTP_REFERENCE_HPP_
