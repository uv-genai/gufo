#include "src/models/glm5_next/kernels/rocm/mla.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
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

/// RMS norm kernel.
__global__ void RmsNormKernel(float* __restrict__ x,
                              const float* __restrict__ gamma, int n,
                              float eps) {
  float ms = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    ms += x[i] * x[i];
  }
  __shared__ float sdata[256];
  sdata[threadIdx.x] = ms;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  const float inv_rms = rsqrtf(sdata[0] / n + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    x[i] = x[i] * inv_rms * gamma[i];
  }
}

/// Layer norm kernel.
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
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    x[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
  }
}

/// MatVec: y[m] = W[m, n] @ x[n] (row-major W).
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

/// Absorb Q: q_abs[h] = k_b[h] @ q[:, h].
__global__ void AbsorbQKernel(const float* __restrict__ k_b,
                              const float* __restrict__ q,
                              float* __restrict__ q_abs, int qk_nope,
                              int kv_lora, int n_head) {
  const int h = blockIdx.x;
  if (h >= n_head)
    return;
  const float* kb = k_b + h * qk_nope * kv_lora;
  const float* qh = q + h * qk_nope;
  float* qa = q_abs + h * kv_lora;
  for (int j = threadIdx.x; j < kv_lora; j += blockDim.x) {
    float acc = 0.0f;
    for (int i = 0; i < qk_nope; ++i) {
      acc += kb[j * qk_nope + i] * qh[i];
    }
    qa[j] = acc;
  }
}

/// Attention score + softmax + context per head.
__global__ void AttentionKernel(const float* __restrict__ q_abs,
                                const float* __restrict__ latents,
                                float* __restrict__ out, int kv_lora,
                                int v_head_dim, int n_head, int n_valid,
                                float kq_scale) {
  const int h = blockIdx.x;
  if (h >= n_head)
    return;

  const float* qa = q_abs + h * kv_lora;
  float* oh = out + h * v_head_dim;

  // Compute scores.
  __shared__ float scores[256];
  for (int i = threadIdx.x; i < n_valid; i += blockDim.x) {
    float dot = 0.0f;
    for (int j = 0; j < kv_lora; ++j) {
      dot += qa[j] * latents[i * kv_lora + j];
    }
    scores[i] = dot * kq_scale;
  }
  __syncthreads();

  // Softmax.
  if (threadIdx.x == 0) {
    float max_val = scores[0];
    for (int i = 1; i < n_valid; ++i)
      max_val = fmaxf(max_val, scores[i]);
    float sum = 0.0f;
    for (int i = 0; i < n_valid; ++i) {
      scores[i] = expf(scores[i] - max_val);
      sum += scores[i];
    }
    for (int i = 0; i < n_valid; ++i)
      scores[i] /= sum;
  }
  __syncthreads();

  // Context: ctx = latents @ p^T.
  __shared__ float ctx[512];  // max kv_lora.
  for (int j = threadIdx.x; j < kv_lora; j += blockDim.x) {
    float acc = 0.0f;
    for (int i = 0; i < n_valid; ++i) {
      acc += latents[i * kv_lora + j] * scores[i];
    }
    ctx[j] = acc;
  }
  __syncthreads();

  // Output: oh = v_b[h] @ ctx.
  // v_b layout: [n_head, v_head_dim, kv_lora].
  const float* vb = nullptr;  // Would need v_b pointer.
  // Simplified: output ctx directly (placeholder).
  for (int j = threadIdx.x; j < kv_lora && j < v_head_dim; j += blockDim.x) {
    oh[j] = ctx[j];
  }
}

}  // namespace

bool DsaLayerGpu(const MlaGpuGeometry& geom, const MlaGpuWeights& w,
                 const float* cur, float* out, std::uint32_t seq_len,
                 hipStream_t stream) {
  const std::uint32_t H = geom.hidden_size;
  const std::uint32_t nh = geom.num_heads;
  const std::uint32_t qd = geom.qk_nope;
  const std::uint32_t vd = geom.v_head_dim;
  const std::uint32_t qlora = geom.q_lora_rank;
  const std::uint32_t kv = geom.kv_lora_rank;
  const int T = static_cast<int>(seq_len);
  const float kq_scale = 1.0f / sqrtf(static_cast<float>(qd));

  // Allocate scratch.
  float *qr, *q_flat, *kv_lat, *q_abs;
  HIP_CHECK(hipMalloc(&qr, qlora * T * sizeof(float)));
  HIP_CHECK(hipMalloc(&q_flat, nh * qd * T * sizeof(float)));
  HIP_CHECK(hipMalloc(&kv_lat, kv * T * sizeof(float)));
  HIP_CHECK(hipMalloc(&q_abs, nh * kv * T * sizeof(float)));

  const int block = 256;

  // 1. qr = rms_norm(q_a @ cur, q_a_norm).
  {
    const int grid = (qlora * T + block - 1) / block;
    hipLaunchKernelGGL(BatchedMatVecKernel, dim3(grid), dim3(block), 0, stream,
                       w.q_a, cur, qr, qlora, H, T);
    // Per-token RMS norm.
    for (int t = 0; t < T; ++t) {
      hipLaunchKernelGGL(RmsNormKernel, dim3(1), dim3(block), 0, stream,
                         qr + t * qlora, w.q_a_norm, qlora, geom.norm_eps);
    }
  }

  // 2. q = split_heads(q_b @ qr, qd, nh).
  {
    const int grid = (nh * qd * T + block - 1) / block;
    hipLaunchKernelGGL(BatchedMatVecKernel, dim3(grid), dim3(block), 0, stream,
                       w.q_b, qr, q_flat, nh * qd, qlora, T);
  }

  // 3. kv_cmpr = rms_norm(kv_a_mqa @ cur, kv_a_norm).
  {
    const int grid = (kv * T + block - 1) / block;
    hipLaunchKernelGGL(BatchedMatVecKernel, dim3(grid), dim3(block), 0, stream,
                       w.kv_a_mqa, cur, kv_lat, kv, H, T);
    for (int t = 0; t < T; ++t) {
      hipLaunchKernelGGL(RmsNormKernel, dim3(1), dim3(block), 0, stream,
                         kv_lat + t * kv, w.kv_a_norm, kv, geom.norm_eps);
    }
  }

  // 4. q_abs[h] = k_b[h] @ q[:, h].
  for (int t = 0; t < T; ++t) {
    hipLaunchKernelGGL(AbsorbQKernel, dim3(nh), dim3(block), 0, stream, w.k_b,
                       q_flat + t * nh * qd, q_abs + t * nh * kv, qd, kv, nh);
  }

  // 5. For each token, compute attention (simplified: use all tokens as
  // context). Full DSA indexer selection would be implemented here. For now,
  // use a simple full-attention approach.
  for (int t = 0; t < T; ++t) {
    hipLaunchKernelGGL(AttentionKernel, dim3(nh), dim3(block), 0, stream,
                       q_abs + t * nh * kv, kv_lat, out + t * nh * vd, kv, vd,
                       nh, T, kq_scale);
  }

  // 6. Output projection.
  {
    const int grid = (H * T + block - 1) / block;
    hipLaunchKernelGGL(BatchedMatVecKernel, dim3(grid), dim3(block), 0, stream,
                       w.output, out, out, H, nh * vd, T);
  }

  // Cleanup.
  HIP_CHECK(hipFree(qr));
  HIP_CHECK(hipFree(q_flat));
  HIP_CHECK(hipFree(kv_lat));
  HIP_CHECK(hipFree(q_abs));
  HIP_CHECK(hipStreamSynchronize(stream));
  return true;
}

}  // namespace gufo::models::glm5_next::rocm
