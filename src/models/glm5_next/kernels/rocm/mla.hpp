#ifndef GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_MLA_HPP_
#define GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_MLA_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::glm5_next::rocm {

/// NoPE-MLA + DSA lightning indexer GPU kernel interface.
///
/// Math (matches CPU oracle in mla_reference.cpp):
///   MLA (expanded-KV first):
///     1. qr = rms_norm(q_a @ cur, q_a_norm)
///     2. q = split_heads(q_b @ qr, qk_nope, n_head)
///     3. kv_cmpr = rms_norm(kv_a_mqa @ cur, kv_a_norm)  — latent [kv_lora, T]
///     4. q_abs[h] = k_b[h] @ q[:, h]  — absorb Q through k_b
///     5. For each t: select pools via DSA indexer, then:
///        score = q_abs^T @ latents * (1/sqrt(qk_nope))
///        p = softmax(score)
///        ctx = latents @ p^T
///        oh[h] = v_b[h] @ ctx[:, h]
///     6. out = wo @ oh
///
///   DSA indexer (pool selection):
///     1. iq = split_heads(indexer_q_b @ qr, idx_key_dim, idx_n_head)
///     2. ik = layer_norm(indexer_k @ cur, k_norm)
///     3. ig = indexer_compressor_gate @ cur
///     4. Pool: per key-dim softmax over (ig + ape) weighted avg of ik members
///     5. score = sum_h relu(iq_h . pooled) * weights_h
///     6. Select top_k/kpool pools by descending score

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

/// Device pointers for MLA + DSA weights.
struct MlaGpuWeights {
  // MLA projections.
  const float* q_a;
  const float* q_a_norm;
  const float* q_b;
  const float* kv_a_mqa;
  const float* kv_a_norm;
  const float* k_b;
  const float* v_b;
  const float* output;

  // DSA indexer.
  const float* idx_q_b;
  const float* idx_k;
  const float* idx_k_norm_w;
  const float* idx_k_norm_b;
  const float* idx_proj;
  const float* idx_ape;
  const float* idx_gate;
};

/// Run DSA layer forward on GPU for a sequence.
/// cur: [hidden, T] on device.
/// out: [hidden, T] on device (output).
/// Returns true on success.
bool DsaLayerGpu(const MlaGpuGeometry& geom, const MlaGpuWeights& w,
                 const float* cur, float* out, std::uint32_t seq_len,
                 hipStream_t stream);

}  // namespace gufo::models::glm5_next::rocm

#endif  // GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_MLA_HPP_
