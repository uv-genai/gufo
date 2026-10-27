#include "src/models/glm5_next/mtp_reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace gufo::models::glm5_next {
namespace {

void RmsNorm(float* x, const float* gamma, std::size_t n, float eps) {
  float ms = 0;
  for (std::size_t i = 0; i < n; ++i)
    ms += x[i] * x[i];
  const float inv_rms = 1.0F / std::sqrt(ms / n + eps);
  for (std::size_t i = 0; i < n; ++i)
    x[i] = x[i] * inv_rms * gamma[i];
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

std::vector<float> MtpDraft(const MtpGeometry& geom, const MtpWeightsF32& w,
                            std::span<const float> embedding,
                            std::span<const float> hidden_state) {
  const std::uint32_t H = geom.hidden_size;

  // 1. enorm = rms_norm(embedding).
  std::vector<float> en(embedding.begin(), embedding.end());
  RmsNorm(en.data(), w.enorm.data(), H, geom.norm_eps);

  // 2. hnorm = rms_norm(hidden_state).
  std::vector<float> hn(hidden_state.begin(), hidden_state.end());
  RmsNorm(hn.data(), w.hnorm.data(), H, geom.norm_eps);

  // 3. Concatenate [enorm | hnorm] → [2*H].
  std::vector<float> eh(2 * H);
  std::memcpy(eh.data(), en.data(), H * sizeof(float));
  std::memcpy(eh.data() + H, hn.data(), H * sizeof(float));

  // 4. proj = eh_proj @ eh  [H]  (eh_proj is [H, 2H]).
  std::vector<float> proj(H);
  MatVec(w.eh_proj.data(), eh.data(), proj.data(), H, 2 * H);

  // 5. shared_head_norm on proj.
  std::vector<float> normed(proj);
  RmsNorm(normed.data(), w.shared_head_norm.data(), H, geom.norm_eps);

  // 6. logits = output_head @ normed  [vocab].
  std::vector<float> logits(geom.vocab_size);
  MatVec(w.output_head.data(), normed.data(), logits.data(), geom.vocab_size,
         H);
  return logits;
}

bool MtpVerify(std::uint32_t draft_token, std::uint32_t target_token) {
  return draft_token == target_token;
}

void MtpRollbackKdaState(std::span<float> state,
                         std::span<const float> checkpoint) {
  const std::size_t n = std::min(state.size(), checkpoint.size());
  std::memcpy(state.data(), checkpoint.data(), n * sizeof(float));
}

}  // namespace gufo::models::glm5_next
