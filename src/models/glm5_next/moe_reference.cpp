#include "src/models/glm5_next/moe_reference.hpp"

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

float Silu(float x) {
  return x * Sigmoid(x);
}

/// ggml_swiglu_clamp: silu(min(gate, limit)) * clamp(up, ±limit).
float SwigluClamp(float gate, float up, float limit) {
  const float g = std::min(gate, limit);
  const float u = std::clamp(up, -limit, limit);
  return Silu(g) * u;
}

/// top_k_desc: k indices ordered by descending score, ties → lower index.
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

}  // namespace

MoeSelection MoESelect(const MoeGeometry& geom, const MoeWeightsF32& w,
                       std::span<const float> token_input) {
  const std::uint32_t H = geom.hidden_size;
  const std::uint32_t E = geom.num_experts;
  const std::uint32_t k = geom.top_k;

  // logits = gate_inp @ x
  std::vector<float> logits(E);
  MatVec(w.gate_inp.data(), token_input.data(), logits.data(), E, H);

  // probs = sigmoid(logits) — unbiased.
  std::vector<float> probs(E);
  for (std::uint32_t i = 0; i < E; ++i)
    probs[i] = Sigmoid(logits[i]);

  // sel_scores = probs + bias (bias for selection only).
  std::vector<float> sel_scores(E);
  for (std::uint32_t i = 0; i < E; ++i) {
    sel_scores[i] = probs[i] + w.exp_probs_b[i];
  }

  // top-k descending.
  auto sel = TopKDesc(sel_scores, k);

  // Weights from unbiased probs, renorm + scale.
  MoeSelection result;
  result.expert_ids = sel;
  result.weights.resize(k);
  float w_sum = 0;
  for (std::uint32_t i = 0; i < k; ++i) {
    result.weights[i] = probs[sel[i]];
    w_sum += result.weights[i];
  }
  if (geom.expert_weights_norm) {
    const float inv = 1.0F / std::max(w_sum, geom.renorm_clamp);
    for (auto& v : result.weights)
      v *= inv;
  }
  for (auto& v : result.weights)
    v *= geom.expert_weights_scale;
  return result;
}

std::vector<float> MoEForward(const MoeGeometry& geom, const MoeWeightsF32& w,
                              std::span<const float> token_input) {
  const std::uint32_t H = geom.hidden_size;
  const std::uint32_t E = geom.num_experts;
  const std::uint32_t I = geom.expert_ff;
  const std::uint32_t SI = geom.shared_ff;
  const float limit = geom.swiglu_limit;

  auto sel = MoESelect(geom, w, token_input);

  // Routed experts.
  std::vector<float> moe(H, 0.0F);
  for (std::uint32_t idx = 0; idx < sel.expert_ids.size(); ++idx) {
    const std::uint32_t e = sel.expert_ids[idx];
    const float weight = sel.weights[idx];

    // gate = gate_exps[e] @ x  [I]
    std::vector<float> gate(I), up(I);
    const float* gW = w.gate_exps.data() + static_cast<std::size_t>(e) * H * I;
    const float* uW = w.up_exps.data() + static_cast<std::size_t>(e) * H * I;
    MatVec(gW, token_input.data(), gate.data(), I, H);
    MatVec(uW, token_input.data(), up.data(), I, H);

    // h = swiglu_clamp(gate, up)
    std::vector<float> h(I);
    for (std::uint32_t i = 0; i < I; ++i) {
      h[i] = SwigluClamp(gate[i], up[i], limit);
    }

    // down = down_exps[e] @ h  [H]
    const float* dW = w.down_exps.data() + static_cast<std::size_t>(e) * I * H;
    std::vector<float> dh(H);
    MatVec(dW, h.data(), dh.data(), H, I);

    for (std::uint32_t i = 0; i < H; ++i) {
      moe[i] += weight * dh[i];
    }
  }

  // Shared expert.
  std::vector<float> gate_s(SI), up_s(SI);
  MatVec(w.gate_shexp.data(), token_input.data(), gate_s.data(), SI, H);
  MatVec(w.up_shexp.data(), token_input.data(), up_s.data(), SI, H);
  std::vector<float> h_s(SI);
  for (std::uint32_t i = 0; i < SI; ++i) {
    h_s[i] = SwigluClamp(gate_s[i], up_s[i], limit);
  }
  std::vector<float> shexp(H);
  MatVec(w.down_shexp.data(), h_s.data(), shexp.data(), H, SI);

  // out = moe + shexp.
  std::vector<float> out(H);
  for (std::uint32_t i = 0; i < H; ++i)
    out[i] = moe[i] + shexp[i];
  return out;
}

std::vector<float> DenseFfnForward(std::span<const float> token_input,
                                   std::span<const float> gate,
                                   std::span<const float> up,
                                   std::span<const float> down,
                                   std::uint32_t hidden, std::uint32_t inter,
                                   float limit) {
  // gate = gate @ x, up = up @ x.
  std::vector<float> g(inter), u(inter);
  MatVec(gate.data(), token_input.data(), g.data(), inter, hidden);
  MatVec(up.data(), token_input.data(), u.data(), inter, hidden);

  // h = swiglu_clamp(g, u).
  std::vector<float> h(inter);
  for (std::uint32_t i = 0; i < inter; ++i) {
    h[i] = SwigluClamp(g[i], u[i], limit);
  }

  // out = down @ h.
  std::vector<float> out(hidden);
  MatVec(down.data(), h.data(), out.data(), hidden, inter);
  return out;
}

}  // namespace gufo::models::glm5_next
