#include "src/models/glm5_next/mla_reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace gufo::models::glm5_next {
namespace {

float Sigmoid(float x) {
  if (x >= 0) {
    const float z = std::exp(-x);
    return 1.0F / (1.0F + z);
  }
  const float z = std::exp(x);
  return z / (1.0F + z);
}

void Softmax(float* x, std::size_t n) {
  float max_val = x[0];
  for (std::size_t i = 1; i < n; ++i)
    max_val = std::max(max_val, x[i]);
  float sum = 0;
  for (std::size_t i = 0; i < n; ++i) {
    x[i] = std::exp(x[i] - max_val);
    sum += x[i];
  }
  for (std::size_t i = 0; i < n; ++i)
    x[i] /= sum;
}

void RmsNorm(float* x, const float* gamma, std::size_t n, float eps) {
  float ms = 0;
  for (std::size_t i = 0; i < n; ++i)
    ms += x[i] * x[i];
  const float inv_rms = 1.0F / std::sqrt(ms / n + eps);
  for (std::size_t i = 0; i < n; ++i)
    x[i] = x[i] * inv_rms * gamma[i];
}

void LayerNorm(float* x, const float* gamma, const float* beta, std::size_t n,
               float eps) {
  float mean = 0;
  for (std::size_t i = 0; i < n; ++i)
    mean += x[i];
  mean /= n;
  float var = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const float d = x[i] - mean;
    var += d * d;
  }
  var /= n;
  const float inv_std = 1.0F / std::sqrt(var + eps);
  for (std::size_t i = 0; i < n; ++i) {
    x[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
  }
}

void MatVec(const float* W, const float* x, float* y, std::size_t m,
            std::size_t n) {
  for (std::size_t i = 0; i < m; ++i) {
    float acc = 0;
    const float* row = W + i * n;
    for (std::size_t j = 0; j < n; ++j)
      acc += row[j] * x[j];
    y[i] = acc;
  }
}

std::vector<std::uint32_t> TopKDesc(std::span<const float> scores,
                                    std::uint32_t k) {
  k = std::min(k, static_cast<std::uint32_t>(scores.size()));
  std::vector<std::pair<float, std::uint32_t>> indexed(scores.size());
  for (std::uint32_t i = 0; i < scores.size(); ++i) {
    indexed[i] = {-scores[i], i};
  }
  std::stable_sort(indexed.begin(), indexed.end());
  std::vector<std::uint32_t> result(k);
  for (std::uint32_t i = 0; i < k; ++i)
    result[i] = indexed[i].second;
  return result;
}

}  // namespace

DsaSelection DsaSelectPools(const MlaGeometry& geom, const MlaWeightsF32& w,
                            std::span<const float> cur, std::uint32_t T,
                            std::uint32_t t) {
  const std::uint32_t H = geom.hidden_size;
  const std::uint32_t qd = geom.idx_key_dim;
  const std::uint32_t qh = geom.idx_n_head;
  const std::uint32_t kpool = geom.idx_kpool;

  // Compute indexer Q, K, gate for all tokens up to t.
  // iq = split_heads(idx_q_b @ qr, qd, qh) — we need qr first.
  // For simplicity, compute per-token.
  // qr = rms_norm(q_a @ cur_t, q_a_norm).
  std::vector<float> q_a_out(geom.q_lora_rank);
  MatVec(w.q_a.data(), cur.data() + t * H, q_a_out.data(), geom.q_lora_rank, H);
  std::vector<float> qr(q_a_out);
  RmsNorm(qr.data(), w.q_a_norm.data(), geom.q_lora_rank, geom.norm_eps);

  // iq = idx_q_b @ qr  [qh * qd].
  std::vector<float> iq(qh * qd);
  MatVec(w.idx_q_b.data(), qr.data(), iq.data(), qh * qd, geom.q_lora_rank);

  // Compute pooled keys for all completed pools.
  const std::uint32_t n_pool_total = T / kpool;
  const std::uint32_t n_vis = (t + 1) / kpool;

  // Tail-only path: no complete pools visible, but kpool_select_tail
  // appends the incomplete prefix [0, t].
  if (n_vis == 0 && geom.idx_kpool_select_tail) {
    DsaSelection result;
    for (std::uint32_t tok = 0; tok <= t; ++tok) {
      result.cells.push_back(static_cast<std::int64_t>(tok));
    }
    return result;
  }
  if (n_vis == 0) {
    return DsaSelection{};
  }

  // ik[t] = layer_norm(idx_k @ cur_t, k_norm_w, k_norm_b)  [qd] and
  // ig[t] = idx_gate @ cur_t  [qd] — precomputed for all visible tokens.
  std::vector<float> ik_all(static_cast<std::size_t>(qd) * T);
  std::vector<float> ig_all(static_cast<std::size_t>(qd) * T);
  for (std::uint32_t tok = 0; tok < T; ++tok) {
    MatVec(w.idx_k.data(), cur.data() + tok * H,
           ik_all.data() + static_cast<std::size_t>(tok) * qd, qd, H);
    LayerNorm(ik_all.data() + static_cast<std::size_t>(tok) * qd,
              w.idx_k_norm_w.data(), w.idx_k_norm_b.data(), qd, geom.norm_eps);
    MatVec(w.idx_gate.data(), cur.data() + tok * H,
           ig_all.data() + static_cast<std::size_t>(tok) * qd, qd, H);
  }

  // For each pool p, compute the pooled key.
  // w = softmax(ig[:, members] + ape, axis=1) per key-dim element;
  // pooled[:, p] = (ik[:, members] * w).sum(axis=1).
  std::vector<float> pooled(qd * n_vis, 0.0F);
  for (std::uint32_t p = 0; p < n_vis; ++p) {
    for (std::uint32_t d = 0; d < qd; ++d) {
      float member_logits[kpool];
      for (std::uint32_t m = 0; m < kpool; ++m) {
        const std::uint32_t tok = p * kpool + m;
        member_logits[m] = ig_all[static_cast<std::size_t>(tok) * qd + d] +
                           w.idx_ape[static_cast<std::size_t>(d) * kpool + m];
      }
      Softmax(member_logits, kpool);
      float acc = 0;
      for (std::uint32_t m = 0; m < kpool; ++m) {
        const std::uint32_t tok = p * kpool + m;
        acc +=
            member_logits[m] * ik_all[static_cast<std::size_t>(tok) * qd + d];
      }
      pooled[p * qd + d] = acc;
    }
  }

  // Compute score[t, p] = sum_h relu(iq_h . pooled_p) * weights_h.
  // weights = idx_proj @ cur / sqrt(qd * qh).
  std::vector<float> weights(qh);
  MatVec(w.idx_proj.data(), cur.data() + t * H, weights.data(), qh, H);
  const float scale = 1.0F / std::sqrt(static_cast<float>(qd * qh));
  for (auto& v : weights)
    v *= scale;

  std::vector<float> scores(n_vis);
  for (std::uint32_t p = 0; p < n_vis; ++p) {
    float score = 0;
    for (std::uint32_t h = 0; h < qh; ++h) {
      float dot = 0;
      for (std::uint32_t d = 0; d < qd; ++d) {
        dot += iq[h * qd + d] * pooled[p * qd + d];
      }
      score += std::max(dot, 0.0F) * weights[h];
    }
    scores[p] = score;
  }

  // Select top_k/kpool pools.
  const std::uint32_t n_top_pool = std::min(n_vis, geom.idx_top_k / kpool);
  auto sel_pools = TopKDesc(scores, n_top_pool);

  // Expand to cells: sel_pools[:, None] * kpool + arange(kpool).
  DsaSelection result;
  for (std::uint32_t i = 0; i < sel_pools.size(); ++i) {
    for (std::uint32_t m = 0; m < kpool; ++m) {
      result.cells.push_back(
          static_cast<std::int64_t>(sel_pools[i] * kpool + m));
    }
  }

  // Append incomplete tail if kpool_select_tail.
  if (geom.idx_kpool_select_tail) {
    const std::uint32_t tail_start = n_vis * kpool;
    for (std::uint32_t tok = tail_start; tok <= t; ++tok) {
      result.cells.push_back(static_cast<std::int64_t>(tok));
    }
  }

  return result;
}

std::vector<float> DsaAttentionToken(const MlaGeometry& geom,
                                     const MlaWeightsF32& w,
                                     std::span<const float> cur,
                                     std::uint32_t T, std::uint32_t t) {
  const std::uint32_t H = geom.hidden_size;
  const std::uint32_t nh = geom.num_heads;
  const std::uint32_t qd = geom.qk_nope;
  const std::uint32_t vd = geom.v_head_dim;
  const std::uint32_t kv_lora = geom.kv_lora_rank;
  const float kq_scale = 1.0F / std::sqrt(static_cast<float>(qd));

  // qr = rms_norm(q_a @ cur_t, q_a_norm).
  std::vector<float> q_a_out(geom.q_lora_rank);
  MatVec(w.q_a.data(), cur.data() + t * H, q_a_out.data(), geom.q_lora_rank, H);
  std::vector<float> qr(q_a_out);
  RmsNorm(qr.data(), w.q_a_norm.data(), geom.q_lora_rank, geom.norm_eps);

  // q = split_heads(q_b @ qr, qd, nh)  [qd, nh].
  std::vector<float> q_flat(nh * qd);
  MatVec(w.q_b.data(), qr.data(), q_flat.data(), nh * qd, geom.q_lora_rank);

  // kv_cmpr = rms_norm(kv_a_mqa @ cur_t, kv_a_norm)  [kv_lora].
  std::vector<float> kv_lat(kv_lora);
  MatVec(w.kv_a_mqa.data(), cur.data() + t * H, kv_lat.data(), kv_lora, H);
  RmsNorm(kv_lat.data(), w.kv_a_norm.data(), kv_lora, geom.norm_eps);

  // q_abs[h] = k_b[h] @ q[:, h]  [kv_lora, nh].
  std::vector<float> q_abs(kv_lora * nh);
  for (std::uint32_t h = 0; h < nh; ++h) {
    const float* kb = w.k_b.data() + static_cast<std::size_t>(h) * qd * kv_lora;
    const float* qh = q_flat.data() + h * qd;
    for (std::uint32_t j = 0; j < kv_lora; ++j) {
      float acc = 0;
      for (std::uint32_t i = 0; i < qd; ++i) {
        acc += kb[j * qd + i] * qh[i];
      }
      q_abs[h * kv_lora + j] = acc;
    }
  }

  // Select pools for this query.
  auto sel = DsaSelectPools(geom, w, cur, T, t);
  if (sel.cells.empty()) {
    return std::vector<float>(nh * vd, 0.0F);
  }

  // Compute latents for selected cells.
  const std::uint32_t n_valid = static_cast<std::uint32_t>(sel.cells.size());
  std::vector<float> latents(kv_lora * n_valid);
  for (std::uint32_t i = 0; i < n_valid; ++i) {
    const std::int64_t cell = sel.cells[i];
    if (cell < 0)
      continue;
    const std::uint32_t tok = static_cast<std::uint32_t>(cell);
    // kv_latent for token tok.
    std::vector<float> lat(kv_lora);
    MatVec(w.kv_a_mqa.data(), cur.data() + tok * H, lat.data(), kv_lora, H);
    RmsNorm(lat.data(), w.kv_a_norm.data(), kv_lora, geom.norm_eps);
    for (std::uint32_t j = 0; j < kv_lora; ++j) {
      latents[i * kv_lora + j] = lat[j];
    }
  }

  // score[h, i] = q_abs[h] . latents[i] * kq_scale.
  // p[h, i] = softmax(score[h, :]).
  // ctx[:, h] = latents @ p[h, :].
  // oh[h] = v_b[h] @ ctx[:, h].
  std::vector<float> out(nh * vd);
  for (std::uint32_t h = 0; h < nh; ++h) {
    std::vector<float> scores(n_valid);
    for (std::uint32_t i = 0; i < n_valid; ++i) {
      float dot = 0;
      for (std::uint32_t j = 0; j < kv_lora; ++j) {
        dot += q_abs[h * kv_lora + j] * latents[i * kv_lora + j];
      }
      scores[i] = dot * kq_scale;
    }
    Softmax(scores.data(), n_valid);

    // ctx = latents @ p^T  [kv_lora].
    std::vector<float> ctx(kv_lora, 0.0F);
    for (std::uint32_t i = 0; i < n_valid; ++i) {
      for (std::uint32_t j = 0; j < kv_lora; ++j) {
        ctx[j] += latents[i * kv_lora + j] * scores[i];
      }
    }

    // oh[h] = v_b[h] @ ctx  [vd].
    const float* vb = w.v_b.data() + static_cast<std::size_t>(h) * vd * kv_lora;
    for (std::uint32_t j = 0; j < vd; ++j) {
      float acc = 0;
      for (std::uint32_t i = 0; i < kv_lora; ++i) {
        acc += vb[j * kv_lora + i] * ctx[i];
      }
      out[h * vd + j] = acc;
    }
  }
  return out;
}

std::vector<float> DsaLayer(const MlaGeometry& geom, const MlaWeightsF32& w,
                            std::span<const float> cur, std::uint32_t T) {
  const std::uint32_t H = geom.hidden_size;
  const std::uint32_t nh = geom.num_heads;
  const std::uint32_t vd = geom.v_head_dim;

  std::vector<float> out(H * T);
  for (std::uint32_t t = 0; t < T; ++t) {
    auto oh = DsaAttentionToken(geom, w, cur, T, t);
    // out_t = wo @ oh  [H].
    MatVec(w.output.data(), oh.data(), out.data() + t * H, H, nh * vd);
  }
  return out;
}

}  // namespace gufo::models::glm5_next
