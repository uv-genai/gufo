#ifndef GUFO_MODELS_GLM5_NEXT_GPU_ENGINE_HPP_
#define GUFO_MODELS_GLM5_NEXT_GPU_ENGINE_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/glm5_next/config.hpp"
#include "src/models/glm5_next/kernels/rocm/mla.hpp"
#include "src/models/glm5_next/weights.hpp"

namespace gufo::models::glm5_next {

/// GPU inference engine matching Maya's Glm5Model architecture.
///
/// Arenas (all device memory, contiguous):
///   w_arena_   — all F32 weights, 256-byte aligned per tensor
///   state_     — residual streams R (2*hc_dim), KDA state, conv state,
///                DSA latents/indexer buffers
///   sc_        — scratch activations (one token at a time)
///
/// Forward pass (per token, matching Maya step_layers):
///   for each layer:
///     glm_hc_pre → rms_norm → KDA|DSA mixer → glm_hc_post
///     glm_hc_pre → rms_norm → MoE → glm_hc_post
///   final rms_norm + output GEMV + argmax
class GpuEngine {
public:
  ~GpuEngine();

  GpuEngine(const GpuEngine&) = delete;
  GpuEngine& operator=(const GpuEngine&) = delete;
  GpuEngine(GpuEngine&&) = delete;
  GpuEngine& operator=(GpuEngine&&) = delete;

  [[nodiscard]] static std::shared_ptr<GpuEngine> Create(
      const Config& config, const ModelWeights& weights,
      std::string* error_msg = nullptr);

  /// Prefill prompt tokens on GPU. Returns the final hidden state (pre-norm).
  [[nodiscard]] std::vector<float> Prefill(
      std::span<const std::uint32_t> tokens);

  /// Advance one token through the layers. Returns the new hidden state.
  [[nodiscard]] std::vector<float> Step(std::uint32_t token);

  /// Run the output head on a hidden state. Returns token ID.
  [[nodiscard]] std::uint32_t Decode(const std::vector<float>& hidden);

  [[nodiscard]] const Config& GetConfig() const noexcept { return config_; }
  [[nodiscard]] std::uint64_t WeightBytes() const noexcept {
    return w_arena_bytes_;
  }
  [[nodiscard]] std::uint64_t StateBytes() const noexcept {
    return state_bytes_;
  }

private:
  GpuEngine() = default;

  /// Embed one token and run it through all layers; returns the new hidden.
  [[nodiscard]] std::vector<float> RunToken(std::uint32_t token,
                                            std::uint32_t pos);

  /// Run one token through all layers (Maya step_layers).
  bool StepLayers(std::uint32_t pos, hipStream_t stream);

  Config config_;
  hipStream_t stream_{nullptr};

  // Weight arena: contiguous device buffer, 256-byte aligned per tensor.
  void* w_arena_{nullptr};
  std::uint64_t w_arena_bytes_{0};
  std::vector<float*> w_ptrs_;  ///< Per-tensor device pointers into arena.
  std::vector<int> w_types_;    ///< Device-side ggml type per slot (0=F32).

  // Per-layer DSA state (caches + scratch), one entry per layer (unused for
  // KDA layers), plus its shared geometry.
  std::vector<rocm::MlaGpuState> mla_state_;
  rocm::MlaGpuGeometry mla_geom_{};

  // State arena: residual + recurrent state + DSA buffers.
  float* state_{nullptr};
  std::uint64_t state_bytes_{0};
  std::vector<std::int64_t> kda_S_off_;     ///< KDA state offsets per layer.
  std::vector<std::int64_t> kda_conv_off_;  ///< KDA conv state offsets.
  std::vector<std::int64_t> dsa_lat_off_;   ///< DSA latent offsets.

  // Scratch arena: activations for one token.
  float* sc_{nullptr};
  std::uint64_t sc_bytes_{0};
  // Shared kernel workspace (KDA/mHC/MoE scratch) — sized once at Create.
  float* ws_{nullptr};
  std::uint64_t ws_bytes_{0};

  // Embedded token lookup.
  float* token_embd_{nullptr};
  int token_embd_type_{0};
  float* output_head_{nullptr};
  int output_head_type_{0};
  float* output_norm_{nullptr};

  // Geometry shortcuts.
  std::uint32_t H_{0};      ///< hidden_size
  std::uint32_t hc_{0};     ///< hc_count
  std::uint32_t nh_{0};     ///< num_heads
  std::uint32_t dk_{0};     ///< kda_head_dim
  std::uint32_t inner_{0};  ///< d_inner = nh * dk
  std::uint32_t pos_{0};    ///< next token position
};

}  // namespace gufo::models::glm5_next

#endif  // GUFO_MODELS_GLM5_NEXT_GPU_ENGINE_HPP_
