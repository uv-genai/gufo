#include "src/models/glm5_next/mhc_reference.hpp"

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

/// Softmax over one axis of a [rows] vector.
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

/// Joint inverse-RMS over the entire flattened stack (hc streams together).
/// flat is [hc*hidden, T]. Norms across ALL rows jointly, not per-stream.
void JointRmsNorm(float* flat, std::uint32_t rows, std::uint32_t T, float eps) {
  // Compute mean-square over ALL elements (joint norm).
  double ms = 0;
  const std::size_t total = static_cast<std::size_t>(rows) * T;
  for (std::size_t i = 0; i < total; ++i)
    ms += flat[i] * flat[i];
  const float inv_rms = static_cast<float>(1.0 / std::sqrt(ms / total + eps));
  for (std::size_t i = 0; i < total; ++i)
    flat[i] *= inv_rms;
}

/// Per-row inverse-RMS (for Sinkhorn normalization).
void NormDst(float* comb, std::uint32_t hc, float eps) {
  // Each dst row sums to 1.
  for (std::uint32_t d = 0; d < hc; ++d) {
    float sum = 0;
    for (std::uint32_t s = 0; s < hc; ++s)
      sum += comb[d * hc + s];
    const float inv = 1.0F / (sum + eps);
    for (std::uint32_t s = 0; s < hc; ++s)
      comb[d * hc + s] *= inv;
  }
}

/// Per-column inverse-RMS (for Sinkhorn normalization).
void NormSrc(float* comb, std::uint32_t hc, float eps) {
  // Each src column sums to 1.
  for (std::uint32_t s = 0; s < hc; ++s) {
    float sum = 0;
    for (std::uint32_t d = 0; d < hc; ++d)
      sum += comb[d * hc + s];
    const float inv = 1.0F / (sum + eps);
    for (std::uint32_t d = 0; d < hc; ++d)
      comb[d * hc + s] *= inv;
  }
}

}  // namespace

std::vector<float> MhcSinkhorn(const MhcGeometry& geom,
                               std::span<const float> raw_comb) {
  const std::uint32_t hc = geom.hc_count;
  const float eps = geom.eps;
  const std::uint32_t iters = geom.sinkhorn_iters;

  std::vector<float> comb(raw_comb.begin(), raw_comb.end());

  // Softmax over dst (axis=0, i.e. rows).
  for (std::uint32_t s = 0; s < hc; ++s) {
    // Extract column s (dst values for fixed src).
    float col[64];  // max hc
    for (std::uint32_t d = 0; d < hc; ++d)
      col[d] = comb[d * hc + s];
    Softmax(col, hc);
    for (std::uint32_t d = 0; d < hc; ++d)
      comb[d * hc + s] = col[d];
  }

  // +eps.
  for (auto& v : comb)
    v += eps;

  // dst-norm.
  NormDst(comb.data(), hc, eps);

  // (iters-1) × [src-norm, dst-norm].
  for (std::uint32_t i = 1; i < iters; ++i) {
    NormSrc(comb.data(), hc, eps);
    NormDst(comb.data(), hc, eps);
  }
  return comb;
}

MhcReadOutput MhcPre(const MhcGeometry& geom, const MhcWeightsF32& w,
                     std::span<const float> residual, std::uint32_t seq_len) {
  const std::uint32_t H = geom.hidden_size;
  const std::uint32_t hc = geom.hc_count;
  const std::uint32_t mix_dim = geom.hc_mix_dim;
  const float eps = geom.eps;

  // residual: [H, hc, T] → flatten stream-major to [H*hc, T].
  // GGUF layout: element (e, s, t) at e + H*s + H*hc*t (stream-major).
  std::vector<float> flat(H * hc * seq_len);
  for (std::uint32_t t = 0; t < seq_len; ++t) {
    for (std::uint32_t s = 0; s < hc; ++s) {
      for (std::uint32_t e = 0; e < H; ++e) {
        flat[t * H * hc + s * H + e] = residual[t * H * hc + s * H + e];
      }
    }
  }

  // Joint inverse-RMS over ALL streams.
  JointRmsNorm(flat.data(), H * hc, seq_len, geom.norm_eps);

  // mixes = fn @ flat_norm  (mix_dim × (H*hc)) @ (H*hc, T) → (mix_dim, T).
  std::vector<float> mixes(mix_dim * seq_len, 0.0F);
  for (std::uint32_t t = 0; t < seq_len; ++t) {
    const float* x = flat.data() + t * H * hc;
    for (std::uint32_t m = 0; m < mix_dim; ++m) {
      float acc = 0;
      const float* row = w.fn.data() + m * H * hc;
      for (std::uint32_t j = 0; j < H * hc; ++j)
        acc += row[j] * x[j];
      mixes[t * mix_dim + m] = acc;
    }
  }

  // pre = sigmoid(mixes[0:hc]*scale[0] + base[0:hc]) + eps
  std::vector<float> pre(hc * seq_len);
  for (std::uint32_t t = 0; t < seq_len; ++t) {
    for (std::uint32_t s = 0; s < hc; ++s) {
      const float raw = mixes[t * mix_dim + s] * w.scale[0] + w.base[s];
      pre[t * hc + s] = Sigmoid(raw) + eps;
    }
  }

  // post = 2*sigmoid(mixes[hc:2hc]*scale[1] + base[hc:2hc])
  std::vector<float> post(hc * seq_len);
  for (std::uint32_t t = 0; t < seq_len; ++t) {
    for (std::uint32_t d = 0; d < hc; ++d) {
      const float raw =
          mixes[t * mix_dim + hc + d] * w.scale[1] + w.base[hc + d];
      post[t * hc + d] = 2.0F * Sigmoid(raw);
    }
  }

  // comb[dst,src] = sinkhorn(mixes[2hc+dst+hc*src]*scale[2] + base[...])
  // src-major row order: row index = dst + hc*src.
  MhcReadOutput out;
  out.comb.resize(hc * hc * seq_len);
  for (std::uint32_t t = 0; t < seq_len; ++t) {
    float raw[64 * 64];  // max hc*hc
    for (std::uint32_t src = 0; src < hc; ++src) {
      for (std::uint32_t dst = 0; dst < hc; ++dst) {
        const std::uint32_t row = dst + hc * src;
        const float val = mixes[t * mix_dim + 2 * hc + row] * w.scale[2] +
                          w.base[2 * hc + row];
        raw[row] = val;
      }
    }
    // Sinkhorn expects [dst, src] layout (dst-major for the matrix).
    // raw is src-major (row = dst + hc*src). Transpose to dst-major.
    float raw_dst[64 * 64];
    for (std::uint32_t src = 0; src < hc; ++src) {
      for (std::uint32_t dst = 0; dst < hc; ++dst) {
        raw_dst[dst * hc + src] = raw[dst + hc * src];
      }
    }
    auto normalized =
        MhcSinkhorn(geom, std::span<const float>(raw_dst, hc * hc));
    for (std::uint32_t i = 0; i < hc * hc; ++i) {
      out.comb[t * hc * hc + i] = normalized[i];
    }
  }

  // mixed = sum_s pre_s * R_s  → [H, T].
  out.mixed.resize(H * seq_len);
  for (std::uint32_t t = 0; t < seq_len; ++t) {
    for (std::uint32_t e = 0; e < H; ++e) {
      float acc = 0;
      for (std::uint32_t s = 0; s < hc; ++s) {
        acc += pre[t * hc + s] * residual[t * H * hc + s * H + e];
      }
      out.mixed[t * H + e] = acc;
    }
  }

  out.post = std::move(post);
  return out;
}

std::vector<float> MhcPost(const MhcGeometry& geom,
                           std::span<const float> block_out,
                           std::span<const float> residual,
                           std::span<const float> post,
                           std::span<const float> comb, std::uint32_t seq_len) {
  const std::uint32_t H = geom.hidden_size;
  const std::uint32_t hc = geom.hc_count;

  // out_dst = block_out * post_dst + sum_src comb[dst,src] * residual_src
  std::vector<float> out(H * hc * seq_len);
  for (std::uint32_t t = 0; t < seq_len; ++t) {
    for (std::uint32_t d = 0; d < hc; ++d) {
      const float post_val = post[t * hc + d];
      for (std::uint32_t e = 0; e < H; ++e) {
        float acc = block_out[t * H + e] * post_val;
        for (std::uint32_t s = 0; s < hc; ++s) {
          acc +=
              comb[t * hc * hc + d * hc + s] * residual[t * H * hc + s * H + e];
        }
        out[t * H * hc + d * H + e] = acc;
      }
    }
  }
  return out;
}

}  // namespace gufo::models::glm5_next
