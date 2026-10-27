#ifndef GUFO_MODELS_GLM5_NEXT_MLA_REFERENCE_HPP_
#define GUFO_MODELS_GLM5_NEXT_MLA_REFERENCE_HPP_

#include <cstdint>
#include <span>
#include <vector>

namespace gufo::models::glm5_next {

/// NoPE-MLA + DSA lightning indexer — F32 CPU oracle.
///
/// Math contract (Maya ref/glm.py dsa_layer):
///   MLA (expanded-KV first):
///     1. qr = rms_norm(q_a @ cur, q_a_norm)
///     2. q = split_heads(q_b @ qr, qk_nope, n_head)
///     3. kv_cmpr = rms_norm(kv_a_mqa @ cur, kv_a_norm)  — latent [kv_lora, T]
///     4. q_abs[h] = wk_b[h] @ q[:, h]  — absorb Q through wk_b
///     5. For each t: select pools via DSA indexer, then:
///        score = q_abs^T @ latents * (1/sqrt(qk_nope))
///        p = softmax(score)
///        ctx = latents @ p^T
///        oh[h] = wv_b[h] @ ctx[:, h]
///     6. out = wo @ oh
///
///   DSA indexer (pool selection):
///     1. iq = split_heads(indexer_q_b @ qr, idx_key_dim, idx_n_head)
///     2. ik = layer_norm(indexer_k @ cur, k_norm)
///     3. ig = indexer_compressor_gate @ cur
///     4. Pool: per key-dim softmax over (ig + ape) weighted avg of ik members
///     5. score = sum_h relu(iq_h . pooled) * weights_h
///     6. Select top_k/kpool pools by descending score

struct MlaGeometry {
  std::uint32_t hidden_size{4096};
  std::uint32_t num_heads{64};
  std::uint32_t qk_nope{256};     ///< Per-head query/key dim (no RoPE).
  std::uint32_t v_head_dim{256};  ///< Per-head value dim.
  std::uint32_t q_lora_rank{1536};
  std::uint32_t kv_lora_rank{512};  ///< Latent dim.
  float norm_eps{1e-5F};

  // DSA indexer.
  std::uint32_t idx_n_head{32};
  std::uint32_t idx_key_dim{128};
  std::uint32_t idx_top_k{2048};
  std::uint32_t idx_kpool{4};
  bool idx_kpool_select_tail{true};
};

/// MLA + DSA weights for one layer — F32 for the oracle.
struct MlaWeightsF32 {
  // MLA projections.
  std::vector<float> q_a;        ///< [q_lora, hidden]
  std::vector<float> q_a_norm;   ///< [q_lora]
  std::vector<float> q_b;        ///< [n_head * qk_nope, q_lora]
  std::vector<float> kv_a_mqa;   ///< [kv_lora, hidden]
  std::vector<float> kv_a_norm;  ///< [kv_lora]
  std::vector<float> k_b;        ///< [n_head, qk_nope, kv_lora] (per-head)
  std::vector<float> v_b;        ///< [n_head, v_head_dim, kv_lora] (per-head)
  std::vector<float> output;     ///< [hidden, n_head * v_head_dim]

  // DSA indexer.
  std::vector<float> idx_q_b;       ///< [idx_n_head * idx_key_dim, q_lora]
  std::vector<float> idx_k;         ///< [idx_key_dim, hidden]
  std::vector<float> idx_k_norm_w;  ///< [idx_key_dim]
  std::vector<float> idx_k_norm_b;  ///< [idx_key_dim]
  std::vector<float> idx_proj;      ///< [idx_n_head, hidden]
  std::vector<float> idx_ape;       ///< [idx_key_dim, idx_kpool]
  std::vector<float> idx_gate;      ///< [idx_key_dim, hidden]
};

/// DSA pool selection result for one query token.
struct DsaSelection {
  std::vector<std::int64_t>
      cells;  ///< Selected token indices (pools * kpool + offsets).
};

/// Select DSA pools for one query token t.
[[nodiscard]] DsaSelection DsaSelectPools(const MlaGeometry& geom,
                                          const MlaWeightsF32& w,
                                          std::span<const float> cur,
                                          std::uint32_t T, std::uint32_t t);

/// Run DSA attention for one token (expanded-KV path).
/// cur: [hidden, T], returns [n_head * v_head_dim].
[[nodiscard]] std::vector<float> DsaAttentionToken(const MlaGeometry& geom,
                                                   const MlaWeightsF32& w,
                                                   std::span<const float> cur,
                                                   std::uint32_t T,
                                                   std::uint32_t t);

/// Run full DSA layer for a sequence. cur: [hidden, T], returns [hidden, T].
[[nodiscard]] std::vector<float> DsaLayer(const MlaGeometry& geom,
                                          const MlaWeightsF32& w,
                                          std::span<const float> cur,
                                          std::uint32_t T);

}  // namespace gufo::models::glm5_next

#endif  // GUFO_MODELS_GLM5_NEXT_MLA_REFERENCE_HPP_
