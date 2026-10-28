#ifndef GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_DSA_HPP_
#define GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_DSA_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::glm5_next::rocm {

/// DSA lightning indexer GPU kernels.
///
/// Math (matches CPU oracle in mla_reference.cpp DsaSelectPools):
///   1. iq = split_heads(idx_q_b @ qr, idx_key_dim, idx_n_head)
///   2. ik = layer_norm(idx_k @ cur, k_norm_w, k_norm_b)
///   3. ig = idx_gate @ cur  (per key-dim element)
///   4. Pool: per key-dim softmax over (ig + ape) weighted avg of ik members
///   5. score[t, p] = sum_h relu(iq_h . pooled_p) * weights_h
///   6. weights = idx_proj @ cur / sqrt(idx_key_dim * idx_n_head)
///   7. Select top_k/kpool pools by descending score

struct DsaGpuGeometry {
  std::uint32_t hidden_size;
  std::uint32_t idx_n_head;
  std::uint32_t idx_key_dim;
  std::uint32_t idx_top_k;
  std::uint32_t idx_kpool;
  bool idx_kpool_select_tail;
  float norm_eps;
};

/// Device pointers for DSA indexer weights.
struct DsaGpuWeights {
  const float* idx_q_b;       ///< [idx_n_head * idx_key_dim, q_lora]
  const float* idx_k;         ///< [idx_key_dim, hidden]
  const float* idx_k_norm_w;  ///< [idx_key_dim]
  const float* idx_k_norm_b;  ///< [idx_key_dim]
  const float* idx_proj;      ///< [idx_n_head, hidden]
  const float* idx_ape;       ///< [idx_key_dim, idx_kpool]
  const float* idx_gate;      ///< [idx_key_dim, hidden]
};

/// Run DSA pool selection for one query token.
/// cur: [hidden, T] on device.
/// Returns: number of selected cells written to out_cells.
/// out_cells: device buffer for selected token indices.
bool DsaSelectPoolsGpu(const DsaGpuGeometry& geom, const DsaGpuWeights& w,
                       const float* cur, std::uint32_t T, std::uint32_t t,
                       std::int64_t* out_cells, std::uint32_t* out_count,
                       hipStream_t stream);

}  // namespace gufo::models::glm5_next::rocm

#endif  // GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_DSA_HPP_
