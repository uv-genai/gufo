#ifndef GUFO_MODELS_GLM5_NEXT_GPU_ENGINE_HPP_
#define GUFO_MODELS_GLM5_NEXT_GPU_ENGINE_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/glm5_next/config.hpp"
#include "src/models/glm5_next/weights.hpp"

namespace gufo::models::glm5_next {

/// GPU inference engine. 100% GPU execution — no CPU inference path.
/// Manages device memory and runs the full forward pass using GPU kernels.
class GpuEngine {
public:
  ~GpuEngine();

  GpuEngine(const GpuEngine&) = delete;
  GpuEngine& operator=(const GpuEngine&) = delete;
  GpuEngine(GpuEngine&&) = delete;
  GpuEngine& operator=(GpuEngine&&) = delete;

  /// Create GPU engine and upload all weights to device.
  [[nodiscard]] static std::shared_ptr<GpuEngine> Create(
      const Config& config, const ModelWeights& weights,
      std::string* error_msg = nullptr);

  /// Prefill a prompt on GPU. Returns hidden state [hidden_size] on host.
  [[nodiscard]] std::vector<float> Prefill(
      std::span<const std::uint32_t> tokens);

  /// Decode one token on GPU from hidden state. Returns token ID.
  [[nodiscard]] std::uint32_t Decode(const std::vector<float>& hidden);

  [[nodiscard]] const Config& GetConfig() const noexcept { return config_; }
  [[nodiscard]] std::uint64_t DeviceBytes() const noexcept;

private:
  GpuEngine() = default;

  Config config_;
  hipStream_t stream_{nullptr};

  // Device weight pointers (one per tensor category).
  struct DeviceLayer {
    float* w_q{nullptr};
    float* w_k{nullptr};
    float* w_v{nullptr};
    float* w_out{nullptr};
    float* conv_q{nullptr};
    float* conv_k{nullptr};
    float* conv_v{nullptr};
    float* f_a{nullptr};
    float* f_b{nullptr};
    float* dt_bias{nullptr};
    float* ssm_a{nullptr};
    float* ssm_beta{nullptr};
    float* g_a{nullptr};
    float* g_b{nullptr};
    float* ssm_norm{nullptr};
    // mHC
    float* hc_attn_fn{nullptr};
    float* hc_attn_scale{nullptr};
    float* hc_attn_base{nullptr};
    // MoE
    float* gate_inp{nullptr};
    float* exp_probs_b{nullptr};
    float* gate_exps{nullptr};
    float* up_exps{nullptr};
    float* down_exps{nullptr};
    float* gate_shexp{nullptr};
    float* up_shexp{nullptr};
    float* down_shexp{nullptr};
  };
  std::vector<DeviceLayer> layers_;

  // Embeddings / output head.
  float* token_embd_{nullptr};
  float* output_head_{nullptr};
  float* output_norm_{nullptr};

  // Device scratch buffers.
  float* d_input_{nullptr};
  float* d_output_{nullptr};
  float* d_hidden_{nullptr};
  std::size_t scratch_bytes_{0};
};

}  // namespace gufo::models::glm5_next

#endif  // GUFO_MODELS_GLM5_NEXT_GPU_ENGINE_HPP_
