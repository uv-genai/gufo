#include "src/models/glm5_next/kernels/rocm/mhc.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstring>
#include <vector>

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
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = m * T;
  if (idx >= total)
    return;
  const int t = idx / m;
  const int i = idx % m;
  float acc = 0.0f;
  const float* row = W + i * n;
  const float* x_col = x + t * n;
  for (int j = 0; j < n; ++j) {
    acc += row[j] * x_col[j];
  }
  y[t * m + i] = acc;
}

/// Sigmoid kernel.
__global__ void SigmoidKernel(const float* __restrict__ x,
                              float* __restrict__ y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = 1.0f / (1.0f + expf(-x[i]));
  }
}

/// Pre gate kernel: pre = sigmoid(mixes*scale + base) + eps.
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

/// Post gate kernel: post = 2*sigmoid(mixes*scale + base).
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

bool MhcPreGpu(const MhcGpuGeometry& geom, const MhcGpuWeights& w,
               const float* residual, float* mixed, float* post, float* comb,
               std::uint32_t seq_len, hipStream_t stream) {
  const std::uint32_t H = geom.hidden_size;
  const std::uint32_t hc = geom.hc_count;
  const std::uint32_t mix_dim = geom.hc_mix_dim;
  const int T = static_cast<int>(seq_len);
  const std::size_t flat_size = H * hc * T;

  // Allocate scratch.
  float *flat, *mixes, *comb_raw, *pre;
  HIP_CHECK(hipMalloc(&flat, flat_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&mixes, mix_dim * T * sizeof(float)));
  HIP_CHECK(hipMalloc(&comb_raw, hc * hc * T * sizeof(float)));
  HIP_CHECK(hipMalloc(&pre, hc * T * sizeof(float)));

  // Copy residual to flat (stream-major).
  HIP_CHECK(hipMemcpy(flat, residual, flat_size * sizeof(float),
                      hipMemcpyDeviceToDevice));

  // 1. Joint inverse-RMS.
  {
    const int block = 256;
    hipLaunchKernelGGL(JointRmsNormKernel, dim3(1), dim3(block), 0, stream,
                       flat, H * hc, T, geom.norm_eps);
  }

  // 2. mixes = fn @ flat  [mix_dim, T].
  {
    const int block = 256;
    const int grid = (mix_dim * T + block - 1) / block;
    hipLaunchKernelGGL(BatchedMatVecKernel, dim3(grid), dim3(block), 0, stream,
                       w.fn, flat, mixes, mix_dim, H * hc, T);
  }

  // 3. Pre gate.
  {
    const int block = 256;
    const int grid = (hc * T + block - 1) / block;
    hipLaunchKernelGGL(PreGateKernel, dim3(grid), dim3(block), 0, stream, mixes,
                       w.scale, w.base, pre, hc, T, geom.eps);
  }

  // 4. Post gate.
  {
    const int block = 256;
    const int grid = (hc * T + block - 1) / block;
    hipLaunchKernelGGL(PostGateKernel, dim3(grid), dim3(block), 0, stream,
                       mixes, w.scale, w.base, post, hc, T);
  }

  // 5. Comb raw + Sinkhorn.
  {
    const int block = 256;
    const int grid = (hc * hc * T + block - 1) / block;
    hipLaunchKernelGGL(CombRawKernel, dim3(grid), dim3(block), 0, stream, mixes,
                       w.scale, w.base, comb_raw, hc, T);
  }

  // Sinkhorn: softmax dst, +eps, dst-norm, (iters-1)×[src-norm, dst-norm].
  {
    const int block = 256;
    const int grid = T;
    hipLaunchKernelGGL(SinkhornSoftmaxDstKernel, dim3(grid), dim3(block), 0,
                       stream, comb_raw, hc, T);
    const int grid2 = (hc * hc * T + block - 1) / block;
    hipLaunchKernelGGL(SinkhornAddEpsKernel, dim3(grid2), dim3(block), 0,
                       stream, comb_raw, hc, T, geom.eps);
    hipLaunchKernelGGL(SinkhornNormDstKernel, dim3(grid), dim3(block), 0,
                       stream, comb_raw, hc, T, geom.eps);
    for (std::uint32_t i = 1; i < geom.sinkhorn_iters; ++i) {
      hipLaunchKernelGGL(SinkhornNormSrcKernel, dim3(grid), dim3(block), 0,
                         stream, comb_raw, hc, T, geom.eps);
      hipLaunchKernelGGL(SinkhornNormDstKernel, dim3(grid), dim3(block), 0,
                         stream, comb_raw, hc, T, geom.eps);
    }
    HIP_CHECK(hipMemcpy(comb, comb_raw, hc * hc * T * sizeof(float),
                        hipMemcpyDeviceToDevice));
  }

  // 6. Mixed = sum_s pre_s * R_s.
  {
    const int block = 256;
    const int grid = (H * T + block - 1) / block;
    hipLaunchKernelGGL(MixedKernel, dim3(grid), dim3(block), 0, stream,
                       residual, pre, mixed, H, hc, T);
  }

  // Cleanup.
  HIP_CHECK(hipFree(flat));
  HIP_CHECK(hipFree(mixes));
  HIP_CHECK(hipFree(comb_raw));
  HIP_CHECK(hipFree(pre));
  HIP_CHECK(hipStreamSynchronize(stream));
  return true;
}

bool MhcPostGpu(const MhcGpuGeometry& geom, const float* block_out,
                const float* residual, const float* post, const float* comb,
                float* out, std::uint32_t seq_len, hipStream_t stream) {
  const std::uint32_t H = geom.hidden_size;
  const std::uint32_t hc = geom.hc_count;
  const int T = static_cast<int>(seq_len);

  // out_dst = block_out * post_dst + sum_src comb[dst,src] * residual_src.
  {
    const int block = 256;
    const int grid = (H * hc * T + block - 1) / block;
    hipLaunchKernelGGL(PostKernel, dim3(grid), dim3(block), 0, stream,
                       block_out, residual, post, comb, out, H, hc, T);
  }

  HIP_CHECK(hipStreamSynchronize(stream));
  return true;
}

}  // namespace gufo::models::glm5_next::rocm
