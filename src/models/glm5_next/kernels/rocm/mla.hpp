#ifndef GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_MLA_HPP_
#define GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_MLA_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::glm5_next::rocm {

/// NoPE-MLA + DSA lightning indexer GPU kernels.
///
/// Math (matches the CPU oracle in mla_reference.cpp):
///   1. qr = rms_norm(q_a @ cur, q_a_norm)
///   2. q = split_heads(q_b @ qr, qk_nope, n_head)
///   3. latent = rms_norm(kv_a_mqa @ cur, kv_a_norm)  — cached per position
///   4. q_abs[h] = k_b[h] @ q[:, h]  — absorb Q through k_b
///   5. DSA indexer selects cells for the query (pools of idx_kpool keys,
///      tail included when idx_kpool_select_tail):
///        ik = layer_norm(idx_k @ cur, k_norm)  — cached per position
///        ig = idx_gate @ cur                   — cached per position
///        pooled[p] = softmax-weighted avg of ik over pool members
///        score[p] = sum_h relu(iq_h . pooled_p) * w_h,  w = idx_proj @ cur
///        cells = top-(idx_top_k/kpool) pools expanded to members (+ tail)
///   6. score[h, i] = q_abs[h] . latent[cell_i] * (1/sqrt(qk_nope))
///      p = softmax(score), ctx = latent @ p^T, oh[h] = v_b[h] @ ctx
///   7. out = wo @ oh

struct MlaGpuGeometry {
  std::uint32_t hidden_size;
  std::uint32_t num_heads;
  std::uint32_t qk_nope;
  std::uint32_t v_head_dim;
  std::uint32_t q_lora_rank;
  std::uint32_t kv_lora_rank;
  float norm_eps;

  // DSA indexer.
  std::uint32_t idx_n_head;
  std::uint32_t idx_key_dim;
  std::uint32_t idx_top_k;
  std::uint32_t idx_kpool;
  bool idx_kpool_select_tail;
};

/// Device pointers for MLA + DSA weights. Quantized tensors keep their GGUF
/// storage on the device and carry its ggml type (0 = F32).
struct MlaGpuWeights {
  // MLA projections.
  const void* q_a;
  int q_a_type{0};
  const float* q_a_norm;
  const void* q_b;
  int q_b_type{0};
  const void* kv_a_mqa;
  int kv_a_mqa_type{0};
  const float* kv_a_norm;
  const void* k_b;
  int k_b_type{0};
  const void* v_b;
  int v_b_type{0};
  const void* output;
  int output_type{0};

  // DSA indexer.
  const void* idx_q_b;
  int idx_q_b_type{0};
  const void* idx_k;
  int idx_k_type{0};
  const float* idx_k_norm_w;
  const float* idx_k_norm_b;
  const void* idx_proj;
  int idx_proj_type{0};
  const float* idx_ape;
  const void* idx_gate;
  int idx_gate_type{0};
};

/// Persistent per-layer decode state (device memory owned by the caller).
struct MlaGpuState {
  float* latents;  ///< [max_pos][kv_lora] cached normalized latents.
  float* idx_ik;   ///< [max_pos][idx_key_dim] cached layer-normed keys.
  float* idx_ig;   ///< [max_pos][idx_key_dim] cached gate logits.
  float* scratch;  ///< see MlaStateFloats().
  std::uint32_t max_pos;
};

/// Device bytes needed for the three caches of one layer.
[[nodiscard]] std::size_t MlaCacheBytes(std::uint32_t max_pos,
                                        const MlaGpuGeometry& geom);

/// Floats of scratch MlaStepGpu needs for one layer.
[[nodiscard]] std::size_t MlaStateFloats(std::uint32_t max_pos,
                                         const MlaGpuGeometry& geom);

/// Run one token of a DSA layer (prefill/decode step).
/// cur: [hidden] device. out: [hidden] device. pos: token position
/// (0-based, dense). Caches of positions [0, pos] are updated in state.
bool MlaStepGpu(const MlaGpuGeometry& geom, const MlaGpuWeights& w,
                MlaGpuState& state, const float* cur, float* out,
                std::uint32_t pos, hipStream_t stream);

/// Batched convenience over tokens 0..seq_len-1 (allocates its own state;
/// used by tests and full-prefill callers). cur: [seq_len, hidden] device,
/// out: [seq_len, hidden] device.
bool DsaLayerGpu(const MlaGpuGeometry& geom, const MlaGpuWeights& w,
                 const float* cur, float* out, std::uint32_t seq_len,
                 hipStream_t stream);

}  // namespace gufo::models::glm5_next::rocm

#endif  // GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_MLA_HPP_
