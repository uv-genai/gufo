#include "src/models/glm5_next/reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace gufo::models::glm5_next {
namespace {

constexpr float kSqrt2Pi = 2.5066282746310002F;

float Sigmoid(float x) {
  if (x >= 0) {
    const float z = std::exp(-x);
    return 1.0F / (1.0F + z);
  }
  const float z = std::exp(x);
  return z / (1.0F + z);
}

/// L2 normalize a vector in place.
void L2Normalize(float* v, std::size_t n) {
  float sum = 0;
  for (std::size_t i = 0; i < n; ++i)
    sum += v[i] * v[i];
  const float inv = 1.0F / std::sqrt(sum + 1e-6F);
  for (std::size_t i = 0; i < n; ++i)
    v[i] *= inv;
}

/// RMS norm: x * gamma / rms(x).
void RmsNorm(float* x, const float* gamma, std::size_t n, float eps) {
  float ms = 0;
  for (std::size_t i = 0; i < n; ++i)
    ms += x[i] * x[i];
  const float inv_rms = 1.0F / std::sqrt(ms / n + eps);
  for (std::size_t i = 0; i < n; ++i)
    x[i] = x[i] * inv_rms * gamma[i];
}

/// Matrix-vector multiply: y[m] = W[m, n] @ x[n] (row-major W).
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

/// Causal conv1d matching Maya `glm_kda_conv` (SiLU applied separately).
/// conv is GGUF (d_conv, 1, inner): element (tap, c) at tap + K*c, tap 0 =
/// oldest. out[t] = sum_tap conv[tap + K*c] * x[t - (K-1) + tap]; taps before
/// 0 are zero (fresh sequence). input/output are [inner, T].
void CausalConv1d(const float* input, const float* conv, float* output,
                  std::uint32_t inner, std::uint32_t T, std::uint32_t K) {
  const std::uint32_t hist = K - 1;
  for (std::uint32_t t = 0; t < T; ++t) {
    for (std::uint32_t c = 0; c < inner; ++c) {
      float acc = 0;
      for (std::uint32_t tap = 0; tap < K; ++tap) {
        const int src = static_cast<int>(t) - static_cast<int>(hist) +
                        static_cast<int>(tap);
        if (src >= 0) {
          acc += conv[tap + K * c] *
                 input[static_cast<std::uint32_t>(src) * inner + c];
        }
      }
      output[t * inner + c] = acc;
    }
  }
}

/// Apply SiLU (swish) elementwise.
void Silu(float* x, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    const float s = Sigmoid(x[i]);
    x[i] = x[i] * s;
  }
}

}  // namespace

std::vector<float> KdaPrefill(const KdaGeometry& geom, const KdaWeightsF32& w,
                              std::span<const float> input,
                              std::uint32_t seq_len) {
  const std::uint32_t H = geom.hidden_size;
  const std::uint32_t nh = geom.num_heads;
  const std::uint32_t dk = geom.head_dim;
  const std::uint32_t inner = nh * dk;  // d_inner
  const std::uint32_t K = geom.conv_kernel;
  const float lb = geom.gate_lower_bound;

  // 1. Q/K/V projections: [inner, T] from [H, T].
  std::vector<float> wq(inner * seq_len), wk(inner * seq_len),
      wv(inner * seq_len);
  for (std::uint32_t t = 0; t < seq_len; ++t) {
    const float* x = input.data() + t * H;
    MatVec(w.w_q.data(), x, wq.data() + t * inner, inner, H);
    MatVec(w.w_k.data(), x, wk.data() + t * inner, inner, H);
    MatVec(w.w_v.data(), x, wv.data() + t * inner, inner, H);
  }

  // 2. Causal conv + SiLU.
  std::vector<float> q_c(inner * seq_len), k_c(inner * seq_len),
      v_c(inner * seq_len);
  CausalConv1d(wq.data(), w.conv_q.data(), q_c.data(), inner, seq_len, K);
  CausalConv1d(wk.data(), w.conv_k.data(), k_c.data(), inner, seq_len, K);
  CausalConv1d(wv.data(), w.conv_v.data(), v_c.data(), inner, seq_len, K);
  Silu(q_c.data(), q_c.size());
  Silu(k_c.data(), k_c.size());
  Silu(v_c.data(), v_c.size());

  // 3. L2 normalize q, k per head (v NOT normalized).
  for (std::uint32_t t = 0; t < seq_len; ++t) {
    for (std::uint32_t h = 0; h < nh; ++h) {
      L2Normalize(q_c.data() + t * inner + h * dk, dk);
      L2Normalize(k_c.data() + t * inner + h * dk, dk);
    }
  }

  // 4. Decay gate g1: [inner, T].
  // g1 = lb * sigmoid(-(f_b @ (f_a @ x) + dt_bias) * A)
  // A = -exp(ssm_a) per head, applied as elementwise on KEY axis.
  std::vector<float> g1(inner * seq_len);
  for (std::uint32_t t = 0; t < seq_len; ++t) {
    const float* x = input.data() + t * H;
    // f_a @ x -> [dk] (rank = head_dim).
    std::vector<float> fa_out(dk);
    MatVec(w.f_a.data(), x, fa_out.data(), dk, H);
    // f_b @ fa_out -> [inner].
    std::vector<float> fb_out(inner);
    MatVec(w.f_b.data(), fa_out.data(), fb_out.data(), inner, dk);
    // Add dt_bias and compute gate.
    for (std::uint32_t i = 0; i < inner; ++i) {
      const std::uint32_t h = i / dk;
      const float a = w.ssm_a[h];  // -exp(A_log)
      const float arg = -(fb_out[i] + w.dt_bias[i]) * a;
      g1[t * inner + i] = lb * Sigmoid(arg);
    }
  }

  // 5. Beta: [nh, T].
  std::vector<float> beta(nh * seq_len);
  for (std::uint32_t t = 0; t < seq_len; ++t) {
    const float* x = input.data() + t * H;
    std::vector<float> b_out(nh);
    MatVec(w.ssm_beta.data(), x, b_out.data(), nh, H);
    for (std::uint32_t h = 0; h < nh; ++h) {
      beta[t * nh + h] = Sigmoid(b_out[h]);
    }
  }

  // 6. Recurrent delta rule: state per head [dk, dk].
  // o[t] = (1/sqrt(dk)) * S^T @ q[t]
  const float scale = 1.0F / std::sqrt(static_cast<float>(dk));
  std::vector<float> o(inner * seq_len);
  std::vector<float> state(nh * dk * dk, 0.0F);

  for (std::uint32_t h = 0; h < nh; ++h) {
    float* S = state.data() + h * dk * dk;
    for (std::uint32_t t = 0; t < seq_len; ++t) {
      const float* kt = k_c.data() + t * inner + h * dk;
      const float* vt = v_c.data() + t * inner + h * dk;
      const float* qt = q_c.data() + t * inner + h * dk;
      const float* g1t = g1.data() + t * inner + h * dk;
      const float bt = beta[t * nh + h];

      // S = diag(exp(g1)) @ S  (gate scales KEY axis = rows).
      for (std::uint32_t i = 0; i < dk; ++i) {
        const float g = std::exp(g1t[i]);
        for (std::uint32_t j = 0; j < dk; ++j) {
          S[i * dk + j] *= g;
        }
      }

      // delta = beta * (v - S^T @ k), then S += outer(k, delta).
      // S^T @ k: result[j] = sum_i S[i,j] * k[i]  (column j of S dotted with
      // k).
      std::vector<float> stk(dk, 0.0F);
      for (std::uint32_t j = 0; j < dk; ++j) {
        float acc = 0;
        for (std::uint32_t i = 0; i < dk; ++i) {
          acc += S[i * dk + j] * kt[i];
        }
        stk[j] = acc;
      }
      std::vector<float> delta(dk);
      for (std::uint32_t j = 0; j < dk; ++j) {
        delta[j] = bt * (vt[j] - stk[j]);
      }
      // S += outer(k, delta): S[i,j] += k[i] * delta[j].
      for (std::uint32_t i = 0; i < dk; ++i) {
        for (std::uint32_t j = 0; j < dk; ++j) {
          S[i * dk + j] += kt[i] * delta[j];
        }
      }

      // o = scale * S^T @ q: out[j] = sum_i S[i,j] * q[i].
      float* ot = o.data() + t * inner + h * dk;
      for (std::uint32_t j = 0; j < dk; ++j) {
        float acc = 0;
        for (std::uint32_t i = 0; i < dk; ++i) {
          acc += S[i * dk + j] * qt[i];
        }
        ot[j] = acc * scale;
      }
    }
  }

  // 7. Output gate g2: [inner, T].
  std::vector<float> g2(inner * seq_len);
  for (std::uint32_t t = 0; t < seq_len; ++t) {
    const float* x = input.data() + t * H;
    std::vector<float> ga_out(dk);
    MatVec(w.g_a.data(), x, ga_out.data(), dk, H);
    std::vector<float> gb_out(inner);
    MatVec(w.g_b.data(), ga_out.data(), gb_out.data(), inner, dk);
    for (std::uint32_t i = 0; i < inner; ++i) {
      g2[t * inner + i] = Sigmoid(gb_out[i]);
    }
  }

  // 8. RMS norm + output gate.
  std::vector<float> normed(inner * seq_len);
  for (std::uint32_t t = 0; t < seq_len; ++t) {
    std::memcpy(normed.data() + t * inner, o.data() + t * inner,
                inner * sizeof(float));
    RmsNorm(normed.data() + t * inner, w.ssm_norm.data(), dk, geom.norm_eps);
    // Apply per-head norm (ssm_norm is per head_dim channel).
    // Wait: ssm_norm is [head_dim], applied per head.
    for (std::uint32_t h = 0; h < nh; ++h) {
      float* nt = normed.data() + t * inner + h * dk;
      // Re-run norm per head with the shared gamma.
      std::memcpy(nt, o.data() + t * inner + h * dk, dk * sizeof(float));
      RmsNorm(nt, w.ssm_norm.data(), dk, geom.norm_eps);
    }
    // Multiply by sigmoid(g2).
    for (std::uint32_t i = 0; i < inner; ++i) {
      normed[t * inner + i] *= g2[t * inner + i];
    }
  }

  // 9. Output projection: [H, T].
  std::vector<float> out(H * seq_len, 0.0F);
  for (std::uint32_t t = 0; t < seq_len; ++t) {
    // out_t = W_o @ normed_t  (W_o is [H, inner]).
    MatVec(w.w_out.data(), normed.data() + t * inner, out.data() + t * H, H,
           inner);
  }
  return out;
}

std::vector<float> KdaDecode(const KdaGeometry& geom, const KdaWeightsF32& w,
                             std::span<const float> token_input,
                             std::span<float> state) {
  // Single-token decode: same math as prefill with T=1 and prior state.
  const std::uint32_t H = geom.hidden_size;
  const std::uint32_t nh = geom.num_heads;
  const std::uint32_t dk = geom.head_dim;
  const std::uint32_t inner = nh * dk;
  const std::uint32_t K = geom.conv_kernel;
  const float lb = geom.gate_lower_bound;

  // Q/K/V projections.
  std::vector<float> wq(inner), wk(inner), wv(inner);
  MatVec(w.w_q.data(), token_input.data(), wq.data(), inner, H);
  MatVec(w.w_k.data(), token_input.data(), wk.data(), inner, H);
  MatVec(w.w_v.data(), token_input.data(), wv.data(), inner, H);

  // For decode, the causal conv needs K-1 history tokens. For a fresh
  // sequence with only one token, the conv uses just the current token
  // (zero-padded left). In a real system the caller would maintain a conv
  // ring buffer; for the oracle we approximate with zero left-pad.
  std::vector<float> q_c(inner), k_c(inner), v_c(inner);
  for (std::uint32_t c = 0; c < inner; ++c) {
    float acc_q = w.conv_q[0 * inner + c] * wq[c];
    float acc_k = w.conv_k[0 * inner + c] * wk[c];
    float acc_v = w.conv_v[0 * inner + c] * wv[c];
    q_c[c] = acc_q;
    k_c[c] = acc_k;
    v_c[c] = acc_v;
  }
  Silu(q_c.data(), inner);
  Silu(k_c.data(), inner);
  Silu(v_c.data(), inner);

  // L2 norm q, k per head.
  for (std::uint32_t h = 0; h < nh; ++h) {
    L2Normalize(q_c.data() + h * dk, dk);
    L2Normalize(k_c.data() + h * dk, dk);
  }

  // Decay gate.
  std::vector<float> fa_out(dk);
  MatVec(w.f_a.data(), token_input.data(), fa_out.data(), dk, H);
  std::vector<float> fb_out(inner);
  MatVec(w.f_b.data(), fa_out.data(), fb_out.data(), inner, dk);
  std::vector<float> g1(inner);
  for (std::uint32_t i = 0; i < inner; ++i) {
    const std::uint32_t h = i / dk;
    const float a = w.ssm_a[h];
    g1[i] = lb * Sigmoid(-(fb_out[i] + w.dt_bias[i]) * a);
  }

  // Beta.
  std::vector<float> b_out(nh);
  MatVec(w.ssm_beta.data(), token_input.data(), b_out.data(), nh, H);
  std::vector<float> beta(nh);
  for (std::uint32_t h = 0; h < nh; ++h)
    beta[h] = Sigmoid(b_out[h]);

  // Recurrent update + output.
  const float scale = 1.0F / std::sqrt(static_cast<float>(dk));
  std::vector<float> o(inner);
  for (std::uint32_t h = 0; h < nh; ++h) {
    float* S = state.data() + h * dk * dk;
    const float* kt = k_c.data() + h * dk;
    const float* vt = v_c.data() + h * dk;
    const float* qt = q_c.data() + h * dk;
    const float* g1t = g1.data() + h * dk;
    const float bt = beta[h];

    // S = diag(exp(g1)) @ S.
    for (std::uint32_t i = 0; i < dk; ++i) {
      const float g = std::exp(g1t[i]);
      for (std::uint32_t j = 0; j < dk; ++j)
        S[i * dk + j] *= g;
    }

    // delta = beta * (v - S^T @ k).
    std::vector<float> stk(dk, 0.0F);
    for (std::uint32_t j = 0; j < dk; ++j) {
      float acc = 0;
      for (std::uint32_t i = 0; i < dk; ++i)
        acc += S[i * dk + j] * kt[i];
      stk[j] = acc;
    }
    std::vector<float> delta(dk);
    for (std::uint32_t j = 0; j < dk; ++j)
      delta[j] = bt * (vt[j] - stk[j]);

    // S += outer(k, delta).
    for (std::uint32_t i = 0; i < dk; ++i)
      for (std::uint32_t j = 0; j < dk; ++j)
        S[i * dk + j] += kt[i] * delta[j];

    // o = scale * S^T @ q.
    float* ot = o.data() + h * dk;
    for (std::uint32_t j = 0; j < dk; ++j) {
      float acc = 0;
      for (std::uint32_t i = 0; i < dk; ++i)
        acc += S[i * dk + j] * qt[i];
      ot[j] = acc * scale;
    }
  }

  // Output gate g2.
  std::vector<float> ga_out(dk);
  MatVec(w.g_a.data(), token_input.data(), ga_out.data(), dk, H);
  std::vector<float> gb_out(inner);
  MatVec(w.g_b.data(), ga_out.data(), gb_out.data(), inner, dk);
  std::vector<float> g2(inner);
  for (std::uint32_t i = 0; i < inner; ++i)
    g2[i] = Sigmoid(gb_out[i]);

  // RMS norm + gate.
  std::vector<float> normed(inner);
  for (std::uint32_t h = 0; h < nh; ++h) {
    std::memcpy(normed.data() + h * dk, o.data() + h * dk, dk * sizeof(float));
    RmsNorm(normed.data() + h * dk, w.ssm_norm.data(), dk, geom.norm_eps);
    for (std::uint32_t j = 0; j < dk; ++j) {
      normed[h * dk + j] *= g2[h * dk + j];
    }
  }

  // Output projection.
  std::vector<float> out(H);
  MatVec(w.w_out.data(), normed.data(), out.data(), H, inner);
  return out;
}

float KLDivergence(std::span<const float> p, std::span<const float> q) {
  // KL(p || q) = sum p_i * (log p_i - log q_i).
  // Use log-sum-exp on q for stability if q is logits; here q is probs.
  float kl = 0;
  for (std::size_t i = 0; i < p.size(); ++i) {
    if (p[i] > 1e-10F) {
      kl += p[i] * (std::log(p[i]) - std::log(q[i] + 1e-10F));
    }
  }
  return kl;
}

float MaxAbsDiff(std::span<const float> a, std::span<const float> b) {
  float max_diff = 0;
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
    max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
  }
  return max_diff;
}

}  // namespace gufo::models::glm5_next
