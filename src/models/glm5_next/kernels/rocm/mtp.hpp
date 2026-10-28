#ifndef GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_MTP_HPP_
#define GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_MTP_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::glm5_next::rocm {

/// NextN MTP (Multi-Token Prediction) GPU kernel interface.
///
/// Math (matches CPU oracle in mtp_reference.cpp):
///   1. enorm = rms_norm(embedding)
///   2. hnorm = rms_norm(hidden_state)
///   3. concat [enorm | hnorm] → [2*H]
///   4. proj = eh_proj @ concat  [H]  (eh_proj is [H, 2H])
///   5. shared_head_norm on proj
///   6. logits = output_head @ normed  [vocab]
///   Draft width is FIXED (no adaptive — TP2 mirrors rank 0).

struct MtpGpuGeometry {
  std::uint32_t hidden_size;
  std::uint32_t vocab_size;
  float norm_eps;
  std::uint32_t fixed_draft_width;
};

/// Device pointers for MTP weights.
struct MtpGpuWeights {
  const float* enorm;
  const float* hnorm;
  const float* eh_proj;
  const float* shared_head_norm;
  const float* output_head;
  /// ggml type of the device-side output_head bytes (0=F32; the shared
  /// output head is quantized in real packs).
  int output_head_type{0};
};

/// Run MTP draft generation on GPU for one step.
/// embedding: [hidden] on device.
/// hidden_state: [hidden] on device.
/// out_logits: [vocab] on device (output).
/// Returns true on success.
bool MtpDraftGpu(const MtpGpuGeometry& geom, const MtpGpuWeights& w,
                 const float* embedding, const float* hidden_state,
                 float* out_logits, hipStream_t stream);

/// Verify a draft token against the target model's greedy output.
/// Runs on host (simple comparison).
bool MtpVerify(std::uint32_t draft_token, std::uint32_t target_token);

/// Rollback KDA state to the last verified position.
void MtpRollbackKdaState(float* state, const float* checkpoint,
                         std::size_t size);

}  // namespace gufo::models::glm5_next::rocm

#endif  // GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_MTP_HPP_
