#include "src/models/glm5_next/kernels/rocm/mla.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

#include "src/models/glm5_next/quant_gemv.hpp"

namespace gufo::models::glm5_next::rocm {
namespace {

#define HIP_CHECK(expr)            \
  do {                             \
    const hipError_t err = (expr); \
    if (err != hipSuccess) {       \
      return false;                \
    }                              \
  } while (0)

__global__ void RmsNormKernel(float* __restrict__ x,
                              const float* __restrict__ gamma, int n,
                              float eps) {
  float ms = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    ms += x[i] * x[i];
  __shared__ float sdata[256];
  sdata[threadIdx.x] = ms;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  const float inv_rms = rsqrtf(sdata[0] / n + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    x[i] = x[i] * inv_rms * gamma[i];
}

__global__ void LayerNormKernel(float* __restrict__ x,
                                const float* __restrict__ gamma,
                                const float* __restrict__ beta, int n,
                                float eps) {
  __shared__ float mean, var;
  if (threadIdx.x == 0) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i)
      sum += x[i];
    mean = sum / n;
    var = 0.0f;
    for (int i = 0; i < n; ++i) {
      const float d = x[i] - mean;
      var += d * d;
    }
    var /= n;
  }
  __syncthreads();
  const float inv_std = rsqrtf(var + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    x[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
}

__global__ void ScaleKernel(float* __restrict__ x, int n, float s) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] *= s;
}

/// pooled[p, d] = softmax_m(ig[p*kpool+m, d] + ape[d, m]) . ik[p*kpool+m, d].
__global__ void PoolKernel(const float* __restrict__ ig,
                           const float* __restrict__ ik,
                           const float* __restrict__ ape,
                           float* __restrict__ pooled, int qd, int kpool,
                           int n_vis) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_vis * qd;
  if (idx >= total)
    return;
  const int p = idx / qd;
  const int d = idx % qd;

  float logits[8];  // kpool <= 8
  float max_val = -1e30f;
  for (int m = 0; m < kpool; ++m) {
    const int tok = p * kpool + m;
    logits[m] = ig[tok * qd + d] + ape[d * kpool + m];
    if (logits[m] > max_val)
      max_val = logits[m];
  }
  float sum = 0.0f;
  for (int m = 0; m < kpool; ++m) {
    logits[m] = expf(logits[m] - max_val);
    sum += logits[m];
  }
  float acc = 0.0f;
  for (int m = 0; m < kpool; ++m) {
    const int tok = p * kpool + m;
    acc += (logits[m] / sum) * ik[tok * qd + d];
  }
  pooled[p * qd + d] = acc;
}

/// score[p] = sum_h relu(iq_h . pooled_p) * weights_h.
__global__ void ScoreKernel(const float* __restrict__ iq,
                            const float* __restrict__ pooled,
                            const float* __restrict__ weights,
                            float* __restrict__ scores, int qh, int qd,
                            int n_vis) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n_vis)
    return;
  float score = 0.0f;
  for (int h = 0; h < qh; ++h) {
    float dot = 0.0f;
    for (int d = 0; d < qd; ++d)
      dot += iq[h * qd + d] * pooled[p * qd + d];
    score += fmaxf(dot, 0.0f) * weights[h];
  }
  scores[p] = score;
}

/// attn_scores[h, i] = (q_abs[h] . latent[cells[i]]) * scale.
__global__ void AttnScoreKernel(const float* __restrict__ q_abs,
                                const float* __restrict__ latents,
                                const std::int32_t* __restrict__ cells,
                                float* __restrict__ scores, int kv, int n_cells,
                                float scale) {
  const int h = blockIdx.x;
  const int i = threadIdx.x;
  if (i >= n_cells)
    return;
  const float* qa = q_abs + static_cast<std::size_t>(h) * kv;
  const float* lat = latents + static_cast<std::size_t>(cells[i]) * kv;
  float dot = 0.0f;
  for (int j = 0; j < kv; ++j)
    dot += qa[j] * lat[j];
  scores[static_cast<std::size_t>(h) * n_cells + i] = dot * scale;
}

/// Per-head softmax over cells, then ctx[h, j] = sum_i p[i] * latent[cells[i],
/// j]. scores is normalized in place and holds p afterwards.
__global__ void AttnCtxKernel(float* __restrict__ scores,
                              const float* __restrict__ latents,
                              const std::int32_t* __restrict__ cells,
                              float* __restrict__ ctx, int kv, int n_cells) {
  const int h = blockIdx.x;
  float* sc = scores + static_cast<std::size_t>(h) * n_cells;
  __shared__ float red[256];
  __shared__ float s_max, s_sum;

  float local = -1e30f;
  for (int i = threadIdx.x; i < n_cells; i += blockDim.x)
    local = fmaxf(local, sc[i]);
  red[threadIdx.x] = local;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    s_max = red[0];
  __syncthreads();

  float lsum = 0.0f;
  for (int i = threadIdx.x; i < n_cells; i += blockDim.x) {
    const float e = expf(sc[i] - s_max);
    sc[i] = e;
    lsum += e;
  }
  red[threadIdx.x] = lsum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    s_sum = red[0];
  __syncthreads();

  float* ctx_h = ctx + static_cast<std::size_t>(h) * kv;
  for (int j = threadIdx.x; j < kv; j += blockDim.x) {
    float acc = 0.0f;
    for (int i = 0; i < n_cells; ++i) {
      acc += sc[i] * latents[static_cast<std::size_t>(cells[i]) * kv + j];
    }
    ctx_h[j] = acc / s_sum;
  }
}

/// Top-k indices by descending score; ties keep the lower index first
/// (stable) — identical to the CPU oracle's TopKDesc.
std::vector<std::uint32_t> TopKDesc(const std::vector<float>& scores,
                                    std::uint32_t k) {
  k = std::min(k, static_cast<std::uint32_t>(scores.size()));
  std::vector<std::pair<float, std::uint32_t>> indexed(scores.size());
  for (std::uint32_t i = 0; i < scores.size(); ++i)
    indexed[i] = {-scores[i], i};
  std::stable_sort(indexed.begin(), indexed.end());
  std::vector<std::uint32_t> result(k);
  for (std::uint32_t i = 0; i < k; ++i)
    result[i] = indexed[i].second;
  return result;
}

}  // namespace

std::size_t MlaCacheBytes(std::uint32_t max_pos, const MlaGpuGeometry& geom) {
  return static_cast<std::size_t>(max_pos) * sizeof(float) *
         (geom.kv_lora_rank + 2ULL * geom.idx_key_dim);
}

std::size_t MlaStateFloats(std::uint32_t max_pos, const MlaGpuGeometry& geom) {
  const std::size_t qd = geom.idx_key_dim;
  const std::size_t qh = geom.idx_n_head;
  const std::size_t kpool = geom.idx_kpool ? geom.idx_kpool : 1;
  const std::size_t max_pools = static_cast<std::size_t>(max_pos) / kpool + 1;
  const std::size_t max_cells = geom.idx_top_k + geom.idx_kpool;
  return geom.q_lora_rank +                    // qr
         geom.num_heads * geom.qk_nope +       // q_flat
         qh * qd +                             // iq
         qh +                                  // index weights
         max_pools * qd +                      // pooled
         max_pools +                           // pool scores
         max_cells +                           // cells (int32 in bytes of 1)
         geom.num_heads * geom.kv_lora_rank +  // q_abs
         geom.num_heads * max_cells +          // attn scores
         geom.num_heads * geom.kv_lora_rank +  // ctx
         geom.num_heads * geom.v_head_dim;     // oh
}

bool MlaStepGpu(const MlaGpuGeometry& geom, const MlaGpuWeights& w,
                MlaGpuState& state, const float* cur, float* out,
                std::uint32_t pos, hipStream_t stream) {
  const std::uint32_t H = geom.hidden_size;
  const std::uint32_t nh = geom.num_heads;
  const std::uint32_t qd = geom.qk_nope;
  const std::uint32_t vd = geom.v_head_dim;
  const std::uint32_t qlora = geom.q_lora_rank;
  const std::uint32_t kv = geom.kv_lora_rank;
  const std::uint32_t qh = geom.idx_n_head;
  const std::uint32_t ikd = geom.idx_key_dim;
  const std::uint32_t kpool = geom.idx_kpool ? geom.idx_kpool : 1;
  if (pos >= state.max_pos || state.scratch == nullptr)
    return false;

  const std::size_t max_pools =
      static_cast<std::size_t>(state.max_pos) / kpool + 1;
  const std::size_t max_cells = geom.idx_top_k + geom.idx_kpool;

  // Scratch slices.
  std::size_t off = 0;
  float* qr = state.scratch + off;
  off += qlora;
  float* q_flat = state.scratch + off;
  off += static_cast<std::size_t>(nh) * qd;
  float* iq = state.scratch + off;
  off += static_cast<std::size_t>(qh) * ikd;
  float* w8 = state.scratch + off;
  off += qh;
  float* pooled = state.scratch + off;
  off += max_pools * ikd;
  float* scores_dev = state.scratch + off;
  off += max_pools;
  auto* cells_dev = reinterpret_cast<std::int32_t*>(state.scratch + off);
  off += max_cells;
  float* q_abs = state.scratch + off;
  off += static_cast<std::size_t>(nh) * kv;
  float* attn_scores = state.scratch + off;
  off += static_cast<std::size_t>(nh) * max_cells;
  float* ctx = state.scratch + off;
  off += static_cast<std::size_t>(nh) * kv;
  float* oh = state.scratch + off;

  const float kq_scale = 1.0f / sqrtf(static_cast<float>(qd));

  // 1. qr = rms_norm(q_a @ cur).
  if (!gpu::QuantGemm(w.q_a, cur, qr, qlora, H, w.q_a_type, stream))
    return false;
  hipLaunchKernelGGL(RmsNormKernel, dim3(1), dim3(256), 0, stream, qr,
                     w.q_a_norm, qlora, geom.norm_eps);

  // 2. q = q_b @ qr.
  if (!gpu::QuantGemm(w.q_b, qr, q_flat, static_cast<int>(nh) * qd, qlora,
                      w.q_b_type, stream))
    return false;

  // 3. latent = rms_norm(kv_a_mqa @ cur) — cached at pos.
  float* lat_row = state.latents + static_cast<std::size_t>(pos) * kv;
  if (!gpu::QuantGemm(w.kv_a_mqa, cur, lat_row, kv, H, w.kv_a_mqa_type, stream))
    return false;
  hipLaunchKernelGGL(RmsNormKernel, dim3(1), dim3(256), 0, stream, lat_row,
                     w.kv_a_norm, kv, geom.norm_eps);

  // 4. ik = layer_norm(idx_k @ cur), ig = idx_gate @ cur — cached at pos.
  float* ik_row = state.idx_ik + static_cast<std::size_t>(pos) * ikd;
  if (!gpu::QuantGemm(w.idx_k, cur, ik_row, ikd, H, w.idx_k_type, stream))
    return false;
  hipLaunchKernelGGL(LayerNormKernel, dim3(1), dim3(256), 0, stream, ik_row,
                     w.idx_k_norm_w, w.idx_k_norm_b, ikd, geom.norm_eps);
  float* ig_row = state.idx_ig + static_cast<std::size_t>(pos) * ikd;
  if (!gpu::QuantGemm(w.idx_gate, cur, ig_row, ikd, H, w.idx_gate_type, stream))
    return false;

  // 5. iq = idx_q_b @ qr; weights = idx_proj @ cur / sqrt(qd*qh).
  if (!gpu::QuantGemm(w.idx_q_b, qr, iq, static_cast<int>(qh) * ikd, qlora,
                      w.idx_q_b_type, stream))
    return false;
  if (!gpu::QuantGemm(w.idx_proj, cur, w8, qh, H, w.idx_proj_type, stream))
    return false;
  {
    const float s =
        1.0f / sqrtf(static_cast<float>(ikd) * static_cast<float>(qh));
    hipLaunchKernelGGL(ScaleKernel, dim3(1), dim3(qh), 0, stream, w8, qh, s);
  }

  // 6. DSA pool selection (host top-k over GPU-computed scores).
  std::vector<std::int32_t> cells_host;
  const std::uint32_t n_vis = (pos + 1) / kpool;
  if (n_vis == 0) {
    if (geom.idx_kpool_select_tail) {
      for (std::uint32_t tok = 0; tok <= pos; ++tok)
        cells_host.push_back(static_cast<std::int32_t>(tok));
    }
  } else {
    {
      const int block = 256;
      const int total = static_cast<int>(n_vis) * ikd;
      hipLaunchKernelGGL(PoolKernel, dim3((total + block - 1) / block),
                         dim3(block), 0, stream, state.idx_ig, state.idx_ik,
                         w.idx_ape, pooled, ikd, kpool, n_vis);
      hipLaunchKernelGGL(
          ScoreKernel, dim3((static_cast<int>(n_vis) + block - 1) / block),
          dim3(block), 0, stream, iq, pooled, w8, scores_dev, qh, ikd, n_vis);
    }
    std::vector<float> scores_host(n_vis);
    HIP_CHECK(hipMemcpyAsync(scores_host.data(), scores_dev,
                             n_vis * sizeof(float), hipMemcpyDeviceToHost,
                             stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    const std::uint32_t n_top_pool = std::min(n_vis, geom.idx_top_k / kpool);
    const auto sel_pools = TopKDesc(scores_host, n_top_pool);
    for (const std::uint32_t p : sel_pools) {
      for (std::uint32_t m = 0; m < kpool; ++m) {
        cells_host.push_back(static_cast<std::int32_t>(p * kpool + m));
      }
    }
    if (geom.idx_kpool_select_tail) {
      for (std::uint32_t tok = n_vis * kpool; tok <= pos; ++tok)
        cells_host.push_back(static_cast<std::int32_t>(tok));
    }
  }
  const std::uint32_t n_cells = static_cast<std::uint32_t>(cells_host.size());
  if (n_cells > max_cells)
    return false;
  if (n_cells > 0) {
    HIP_CHECK(hipMemcpyAsync(cells_dev, cells_host.data(),
                             n_cells * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream));
  }

  // 7. q_abs[h] = k_b[h] @ q[:, h].
  if (!gpu::QuantGemmGroups(w.k_b, q_flat, q_abs, nh, kv, qd, w.k_b_type,
                            stream))
    return false;

  // 8. Attention over the selected cells, then oh[h] = v_b[h] @ ctx[h].
  if (n_cells == 0) {
    HIP_CHECK(hipMemsetAsync(
        oh, 0, static_cast<std::size_t>(nh) * vd * sizeof(float), stream));
  } else {
    hipLaunchKernelGGL(AttnScoreKernel, dim3(nh), dim3(nh > 0 ? 256 : 1), 0,
                       stream, q_abs, state.latents, cells_dev, attn_scores, kv,
                       n_cells, kq_scale);
    hipLaunchKernelGGL(AttnCtxKernel, dim3(nh), dim3(256), 0, stream,
                       attn_scores, state.latents, cells_dev, ctx, kv, n_cells);
    if (!gpu::QuantGemmGroups(w.v_b, ctx, oh, nh, vd, kv, w.v_b_type, stream))
      return false;
  }

  // 9. out = wo @ oh.
  if (!gpu::QuantGemm(w.output, oh, out, H, static_cast<int>(nh) * vd,
                      w.output_type, stream))
    return false;
  return true;
}

bool DsaLayerGpu(const MlaGpuGeometry& geom, const MlaGpuWeights& w,
                 const float* cur, float* out, std::uint32_t seq_len,
                 hipStream_t stream) {
  if (seq_len == 0)
    return true;
  MlaGpuState st{};
  st.max_pos = seq_len;
  const std::size_t cache_bytes = MlaCacheBytes(seq_len, geom);
  const std::size_t scratch_bytes =
      MlaStateFloats(seq_len, geom) * sizeof(float);
  void* caches = nullptr;
  void* scratch = nullptr;
  HIP_CHECK(hipMalloc(&caches, cache_bytes));
  HIP_CHECK(hipMalloc(&scratch, scratch_bytes));
  HIP_CHECK(hipMemset(caches, 0, cache_bytes));
  const std::size_t per_pos =
      sizeof(float) * (geom.kv_lora_rank + 2ULL * geom.idx_key_dim);
  st.latents = static_cast<float*>(caches);
  st.idx_ik = reinterpret_cast<float*>(static_cast<char*>(caches) +
                                       static_cast<std::size_t>(seq_len) *
                                           sizeof(float) * geom.kv_lora_rank);
  st.idx_ig = reinterpret_cast<float*>(
      static_cast<char*>(caches) + static_cast<std::size_t>(seq_len) *
                                       sizeof(float) *
                                       (geom.kv_lora_rank + geom.idx_key_dim));
  (void)per_pos;
  st.scratch = static_cast<float*>(scratch);

  bool ok = true;
  for (std::uint32_t t = 0; t < seq_len && ok; ++t) {
    ok = MlaStepGpu(
        geom, w, st, cur + static_cast<std::size_t>(t) * geom.hidden_size,
        out + static_cast<std::size_t>(t) * geom.hidden_size, t, stream);
  }
  hipFree(caches);
  hipFree(scratch);
  HIP_CHECK(hipStreamSynchronize(stream));
  return ok;
}

}  // namespace gufo::models::glm5_next::rocm
