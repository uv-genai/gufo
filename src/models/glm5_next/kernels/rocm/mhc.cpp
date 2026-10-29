#include "src/models/glm5_next/kernels/rocm/mhc.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "strata/kernels/glm_hc.hpp"

namespace gufo::models::glm5_next::rocm {
namespace {

#define HIP_CHECK(expr)            \
  do {                             \
    const hipError_t err = (expr); \
    if (err != hipSuccess) {       \
      return false;                \
    }                              \
  } while (0)

/// Joint inverse-RMS over the entire flattened stack (hc streams together).
__global__ void JointRmsNormKernel(float* __restrict__ flat, int rows, int T,
                                   float eps) {
  // Compute mean-square over ALL elements (joint norm).
  const int total = rows * T;
  const int tid = threadIdx.x;
  const int stride = blockDim.x;

  float sum = 0.0f;
  for (int i = tid; i < total; i += stride) {
    sum += flat[i] * flat[i];
  }
  __shared__ float sdata[256];
  sdata[tid] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  const float inv_rms = rsqrtf(sdata[0] / total + eps);
  for (int i = tid; i < total; i += stride) {
    flat[i] *= inv_rms;
  }
}

/// GEMV: y[m] = W[m, n] @ x[n] (row-major W).
__global__ void MatVecKernel(const float* __restrict__ W,
                             const float* __restrict__ x, float* __restrict__ y,
                             int m, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m) {
    float acc = 0.0f;
    const float* row = W + i * n;
    for (int j = 0; j < n; ++j) {
      acc += row[j] * x[j];
    }
    y[i] = acc;
  }
}

/// Batched MatVec for [m, T] output from [m, n] @ [n, T].
__global__ void BatchedMatVecKernel(const float* __restrict__ W,
                                    const float* __restrict__ x,
                                    float* __restrict__ y, int m, int n,
                                    int T) {
  // One workgroup per output row (t*m+i); threads split the dot — a single
  // thread with a serial n-long chain is latency-bound (measured 0.78 ms for
  // n=16384 vs ~20 us here).
  __shared__ float red[256];
  const int row = blockIdx.x;
  if (row >= m * T)
    return;
  const int t = row / m;
  const int i = row % m;
  float acc = 0.0f;
  const float* wrow = W + i * n;
  const float* xcol = x + t * n;
  for (int j = threadIdx.x; j < n; j += blockDim.x)
    acc += wrow[j] * xcol[j];
  red[threadIdx.x] = acc;
  __syncthreads();
  // In-order reduction on one thread: summation order then matches the CPU
  // oracle when a thread holds at most one term (a tree reduction can flip
  // MoE top-k selection through rounding).
  if (threadIdx.x == 0) {
    float s = 0.0f;
    for (int i = 0; i < blockDim.x; ++i)
      s += red[i];
    y[row] = s;
  }
}

/// Pre gate: pre_s = sigmoid(mixes_s * scale0 + base_s) + eps.
__global__ void PreGateKernel(const float* __restrict__ mixes,
                              const float* __restrict__ scale,
                              const float* __restrict__ base,
                              float* __restrict__ pre, int hc, int T,
                              float eps) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= hc * T)
    return;
  const int s = idx % hc;
  const int t = idx / hc;
  const float raw = mixes[t * 24 + s] * scale[0] + base[s];
  pre[t * hc + s] = 1.0f / (1.0f + expf(-raw)) + eps;
}

/// Post gate: post_d = 2 * sigmoid(mixes_{hc+d} * scale1 + base_{hc+d}).
__global__ void PostGateKernel(const float* __restrict__ mixes,
                               const float* __restrict__ scale,
                               const float* __restrict__ base,
                               float* __restrict__ post, int hc, int T) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= hc * T)
    return;
  const int d = idx % hc;
  const int t = idx / hc;
  const float raw = mixes[t * 24 + hc + d] * scale[1] + base[hc + d];
  post[t * hc + d] = 2.0f / (1.0f + expf(-raw));
}

/// Comb raw values kernel (before Sinkhorn).
__global__ void CombRawKernel(const float* __restrict__ mixes,
                              const float* __restrict__ scale,
                              const float* __restrict__ base,
                              float* __restrict__ comb_raw, int hc, int T) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= hc * hc * T)
    return;
  const int row = idx % (hc * hc);
  const int t = idx / (hc * hc);
  const float raw =
      mixes[t * 24 + 2 * hc + row] * scale[2] + base[2 * hc + row];
  // Transpose src-major to dst-major: row = dst + hc*src -> dst*hc + src.
  const int src = row / hc;
  const int dst = row % hc;
  comb_raw[t * hc * hc + dst * hc + src] = raw;
}

/// Sinkhorn softmax over dst (axis=0, i.e. rows).
__global__ void SinkhornSoftmaxDstKernel(float* __restrict__ comb, int hc,
                                         int T) {
  const int t = blockIdx.x;
  if (t >= T)
    return;
  float* c = comb + t * hc * hc;

  // Each column (src) softmax over dst (rows).
  for (int s = threadIdx.x; s < hc; s += blockDim.x) {
    float max_val = c[0 * hc + s];
    for (int d = 1; d < hc; ++d) {
      max_val = fmaxf(max_val, c[d * hc + s]);
    }
    float sum = 0.0f;
    for (int d = 0; d < hc; ++d) {
      c[d * hc + s] = expf(c[d * hc + s] - max_val);
      sum += c[d * hc + s];
    }
    for (int d = 0; d < hc; ++d) {
      c[d * hc + s] /= sum;
    }
  }
}

/// Sinkhorn +eps.
__global__ void SinkhornAddEpsKernel(float* __restrict__ comb, int hc, int T,
                                     float eps) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= hc * hc * T)
    return;
  comb[idx] += eps;
}

/// Sinkhorn dst-norm: each dst row sums to 1.
__global__ void SinkhornNormDstKernel(float* __restrict__ comb, int hc, int T,
                                      float eps) {
  const int t = blockIdx.x;
  if (t >= T)
    return;
  float* c = comb + t * hc * hc;
  for (int d = threadIdx.x; d < hc; d += blockDim.x) {
    float sum = 0.0f;
    for (int s = 0; s < hc; ++s) {
      sum += c[d * hc + s];
    }
    const float inv = 1.0f / (sum + eps);
    for (int s = 0; s < hc; ++s) {
      c[d * hc + s] *= inv;
    }
  }
}

/// Sinkhorn src-norm: each src column sums to 1.
__global__ void SinkhornNormSrcKernel(float* __restrict__ comb, int hc, int T,
                                      float eps) {
  const int t = blockIdx.x;
  if (t >= T)
    return;
  float* c = comb + t * hc * hc;
  for (int s = threadIdx.x; s < hc; s += blockDim.x) {
    float sum = 0.0f;
    for (int d = 0; d < hc; ++d) {
      sum += c[d * hc + s];
    }
    const float inv = 1.0f / (sum + eps);
    for (int d = 0; d < hc; ++d) {
      c[d * hc + s] *= inv;
    }
  }
}

/// Fused Sinkhorn: softmax over dst, +eps, dst-norm, then
/// (iters-1) x [src-norm, dst-norm] — the same math as the individual
/// SoftmaxDst / AddEps / NormDst / NormSrc kernels, in one launch.
__global__ void SinkhornKernel(float* __restrict__ comb, int hc, int T,
                               int iters, float eps) {
  const int t = blockIdx.x;
  if (t >= T || threadIdx.x != 0)
    return;
  float* c = comb + static_cast<std::size_t>(t) * hc * hc;

  // Softmax per column (src) over dst rows.
  for (int s = 0; s < hc; ++s) {
    float max_val = c[0 * hc + s];
    for (int d = 1; d < hc; ++d)
      max_val = fmaxf(max_val, c[d * hc + s]);
    float sum = 0.0f;
    for (int d = 0; d < hc; ++d) {
      c[d * hc + s] = expf(c[d * hc + s] - max_val);
      sum += c[d * hc + s];
    }
    for (int d = 0; d < hc; ++d)
      c[d * hc + s] /= sum;
  }
  for (int i = 0; i < hc * hc; ++i)
    c[i] += eps;

  auto norm_dst = [&]() {
    for (int d = 0; d < hc; ++d) {
      float sum = 0.0f;
      for (int s = 0; s < hc; ++s)
        sum += c[d * hc + s];
      const float inv = 1.0f / (sum + eps);
      for (int s = 0; s < hc; ++s)
        c[d * hc + s] *= inv;
    }
  };
  auto norm_src = [&]() {
    for (int s = 0; s < hc; ++s) {
      float sum = 0.0f;
      for (int d = 0; d < hc; ++d)
        sum += c[d * hc + s];
      const float inv = 1.0f / (sum + eps);
      for (int d = 0; d < hc; ++d)
        c[d * hc + s] *= inv;
    }
  };

  norm_dst();
  for (int i = 1; i < iters; ++i) {
    norm_src();
    norm_dst();
  }
}

/// Mixed computation: mixed = sum_s pre_s * R_s.
__global__ void MixedKernel(const float* __restrict__ residual,
                            const float* __restrict__ pre,
                            float* __restrict__ mixed, int H, int hc, int T) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= H * T)
    return;
  const int e = idx % H;
  const int t = idx / H;
  float acc = 0.0f;
  for (int s = 0; s < hc; ++s) {
    acc += pre[t * hc + s] * residual[t * H * hc + s * H + e];
  }
  mixed[t * H + e] = acc;
}

/// Post computation: out_dst = block_out * post_dst + sum_src comb[dst,src] *
/// residual_src.
__global__ void PostKernel(const float* __restrict__ block_out,
                           const float* __restrict__ residual,
                           const float* __restrict__ post,
                           const float* __restrict__ comb,
                           float* __restrict__ out, int H, int hc, int T) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= H * hc * T)
    return;
  const int e = idx % H;
  const int d = (idx / H) % hc;
  const int t = idx / (H * hc);
  float acc = block_out[t * H + e] * post[t * hc + d];
  for (int s = 0; s < hc; ++s) {
    acc += comb[t * hc * hc + d * hc + s] * residual[t * H * hc + s * H + e];
  }
  out[t * H * hc + d * H + e] = acc;
}

}  // namespace

std::size_t MhcScratchFloats(const MhcGpuGeometry& geom,
                             std::uint32_t seq_len) {
  return geom.hidden_size * geom.hc_count * seq_len +
         geom.hc_mix_dim * seq_len + geom.hc_count * geom.hc_count * seq_len +
         geom.hc_count * seq_len;
}

bool MhcPreGpu(const MhcGpuGeometry& geom, const MhcGpuWeights& w,
               const float* residual, float* mixed, float* post, float* comb,
               std::uint32_t seq_len, float* scratch, hipStream_t stream) {
  // Maya glm_hc_pre — one token (T=1 engine). residual is (hc*n_embd).
  (void)scratch;
  strata::kernels::GlmHcShapes s;
  s.n_embd = geom.hidden_size;
  s.hc = geom.hc_count;
  s.sinkhorn_iters = geom.sinkhorn_iters;
  strata::kernels::GlmHcWorkspace ws;
  const std::size_t need = strata::kernels::glm_hc_workspace_bytes(s);
  float* owned = nullptr;
  if (scratch == nullptr) {
    if (hipMalloc(&owned, need) != hipSuccess)
      return false;
    scratch = owned;
  }
  strata::kernels::glm_hc_workspace_init(s, scratch, ws);
  float* pre = nullptr;
  if (hipMalloc(&pre, (std::size_t)geom.hc_count * sizeof(float)) !=
      hipSuccess) {
    if (owned)
      hipFree(owned);
    return false;
  }
  for (std::uint32_t t = 0; t < seq_len; ++t) {
    strata::kernels::glm_hc_pre(
        residual + (std::size_t)t * geom.hidden_size * geom.hc_count, w.fn,
        w.scale, w.base, geom.norm_eps, geom.eps, s, ws,
        mixed + (std::size_t)t * geom.hidden_size, pre,
        post + (std::size_t)t * geom.hc_count,
        comb + (std::size_t)t * geom.hc_count * geom.hc_count, stream);
  }
  hipFree(pre);
  if (owned)
    hipFree(owned);
  return true;
}

bool MhcPostGpu(const MhcGpuGeometry& geom, const float* block_out,
                const float* residual, const float* post, const float* comb,
                float* out, std::uint32_t seq_len, hipStream_t stream) {
  strata::kernels::GlmHcShapes s;
  s.n_embd = geom.hidden_size;
  s.hc = geom.hc_count;
  s.sinkhorn_iters = geom.sinkhorn_iters;
  for (std::uint32_t t = 0; t < seq_len; ++t) {
    const std::size_t hc_dim = (std::size_t)geom.hidden_size * geom.hc_count;
    strata::kernels::glm_hc_post(
        block_out + (std::size_t)t * geom.hidden_size, residual + t * hc_dim,
        post + (std::size_t)t * geom.hc_count,
        comb + (std::size_t)t * geom.hc_count * geom.hc_count, s,
        out + t * hc_dim, stream);
  }
  return true;
}

}  // namespace gufo::models::glm5_next::rocm
