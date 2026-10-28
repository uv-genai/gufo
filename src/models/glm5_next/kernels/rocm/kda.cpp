#include "src/models/glm5_next/kernels/rocm/kda.hpp"

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cmath>
#include <cstring>
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

/// Sigmoid activation kernel.
__global__ void SigmoidKernel(const float* __restrict__ x,
                              float* __restrict__ y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = 1.0f / (1.0f + expf(-x[i]));
  }
}

/// SiLU (swish) activation kernel.
__global__ void SiluKernel(const float* __restrict__ x, float* __restrict__ y,
                           int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = x[i] / (1.0f + expf(-x[i]));
  }
}

/// L2 normalize per head: x[i] /= sqrt(sum(x[i]^2) + eps).
__global__ void L2NormKernel(float* __restrict__ x, int head_dim, int num_heads,
                             int seq_len, float eps) {
  const int h = blockIdx.x;
  const int t = blockIdx.y;
  if (h >= num_heads || t >= seq_len)
    return;

  float* head = x + (t * num_heads + h) * head_dim;
  float sum = 0.0f;
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    sum += head[i] * head[i];
  }
  // Block-level reduction.
  __shared__ float sdata[256];
  sdata[threadIdx.x] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = rsqrtf(sdata[0] + eps);
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    head[i] *= inv;
  }
}

/// Decay gate kernel: g1 = lb * sigmoid(-(fb_out + dt_bias) * A).
/// Processes all inner*T elements (batched over tokens).
__global__ void DecayGateKernel(const float* __restrict__ fb_out,
                                const float* __restrict__ dt_bias,
                                const float* __restrict__ ssm_a,
                                float* __restrict__ g1, int inner, int head_dim,
                                int num_heads, int total, float lb) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) {
    const int c = i % inner;  // channel index within token
    const int h = c / head_dim;
    const float a = ssm_a[h];
    const float arg = -(fb_out[i] + dt_bias[c]) * a;
    g1[i] = lb / (1.0f + expf(-arg));
  }
}

/// Beta kernel: beta = sigmoid(ssm_beta @ x). Batched over tokens.
__global__ void BetaKernel(const float* __restrict__ beta_out,
                           float* __restrict__ beta, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) {
    beta[i] = 1.0f / (1.0f + expf(-beta_out[i]));
  }
}

/// Recurrent state update kernel (per head).
/// S = diag(exp(g1)) @ S; S += outer(k, delta) where delta = beta * (v - S^T
/// k).
__global__ void StateUpdateKernel(float* __restrict__ state,
                                  const float* __restrict__ k,
                                  const float* __restrict__ v,
                                  const float* __restrict__ g1,
                                  const float* __restrict__ beta, int head_dim,
                                  int num_heads, int t, int inner) {
  const int h = blockIdx.x;
  if (h >= num_heads)
    return;

  float* S = state + h * head_dim * head_dim;
  const float* kt = k + t * inner + h * head_dim;
  const float* vt = v + t * inner + h * head_dim;
  const float* g1t = g1 + t * inner + h * head_dim;
  const float bt = beta[t * num_heads + h];

  // S = diag(exp(g1)) @ S  (scale rows).
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    const float g = expf(g1t[i]);
    for (int j = 0; j < head_dim; ++j) {
      S[i * head_dim + j] *= g;
    }
  }
  __syncthreads();

  // Compute S^T @ k (column j of S dotted with k).
  __shared__ float stk[256];
  for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
    float acc = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
      acc += S[i * head_dim + j] * kt[i];
    }
    stk[j] = acc;
  }
  __syncthreads();

  // S += outer(k, delta) where delta = beta * (v - stk).
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    for (int j = 0; j < head_dim; ++j) {
      const float delta = bt * (vt[j] - stk[j]);
      S[i * head_dim + j] += kt[i] * delta;
    }
  }
}

/// Output computation kernel: o = (1/sqrt(dk)) * S^T @ q.
__global__ void OutputKernel(const float* __restrict__ state,
                             const float* __restrict__ q, float* __restrict__ o,
                             int head_dim, int num_heads, int t, int inner,
                             float scale) {
  const int h = blockIdx.x;
  if (h >= num_heads)
    return;

  const float* S = state + h * head_dim * head_dim;
  const float* qt = q + t * inner + h * head_dim;
  float* ot = o + t * inner + h * head_dim;

  for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
    float acc = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
      acc += S[i * head_dim + j] * qt[i];
    }
    ot[j] = acc * scale;
  }
}

/// RMS norm kernel (per head).
__global__ void RmsNormKernel(float* __restrict__ x,
                              const float* __restrict__ gamma, int head_dim,
                              int num_heads, int seq_len, float eps) {
  const int h = blockIdx.x;
  const int t = blockIdx.y;
  if (h >= num_heads || t >= seq_len)
    return;

  float* head = x + (t * num_heads + h) * head_dim;
  float ms = 0.0f;
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    ms += head[i] * head[i];
  }
  __shared__ float sdata[256];
  sdata[threadIdx.x] = ms;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  const float inv_rms = rsqrtf(sdata[0] / head_dim + eps);
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    head[i] = head[i] * inv_rms * gamma[i];
  }
}

/// Output gate kernel: out = normed * sigmoid(g2).
__global__ void OutputGateKernel(float* __restrict__ normed,
                                 const float* __restrict__ g2_out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float g2 = 1.0f / (1.0f + expf(-g2_out[i]));
    normed[i] *= g2;
  }
}

/// Causal conv1d kernel (per channel).
__global__ void CausalConv1dKernel(const float* __restrict__ input,
                                   const float* __restrict__ conv,
                                   float* __restrict__ output, int inner, int T,
                                   int K) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= inner)
    return;

  for (int t = 0; t < T; ++t) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      if (t >= k) {
        acc += conv[k * inner + c] * input[(t - k) * inner + c];
      }
    }
    output[t * inner + c] = acc;
  }
}

/// Simple GEMM: y[m] = W[m,n] @ x[n] (row-major W).
/// Uses one thread per output element.
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

/// Batched MatVec for [inner, T] output from [inner, H] @ [H, T].
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

}  // namespace

bool KdaPrefillGpu(const KdaGpuGeometry& geom, const KdaGpuWeights& w,
                   const float* input, float* state, float* out,
                   std::uint32_t seq_len, hipStream_t stream) {
  const std::uint32_t H = geom.hidden_size;
  const std::uint32_t nh = geom.num_heads;
  const std::uint32_t dk = geom.head_dim;
  const std::uint32_t inner = nh * dk;
  const std::uint32_t K = geom.conv_kernel;
  const float lb = geom.gate_lower_bound;
  const int T = static_cast<int>(seq_len);

  // Allocate device scratch.
  float *wq, *wk, *wv, *q_c, *k_c, *v_c;
  float *g1, *beta, *o, *g2, *normed;
  float *fa_out, *fb_out, *ga_out, *gb_out;

  const std::size_t inner_T = inner * T;
  HIP_CHECK(hipMalloc(&wq, inner_T * sizeof(float)));
  HIP_CHECK(hipMalloc(&wk, inner_T * sizeof(float)));
  HIP_CHECK(hipMalloc(&wv, inner_T * sizeof(float)));
  HIP_CHECK(hipMalloc(&q_c, inner_T * sizeof(float)));
  HIP_CHECK(hipMalloc(&k_c, inner_T * sizeof(float)));
  HIP_CHECK(hipMalloc(&v_c, inner_T * sizeof(float)));
  HIP_CHECK(hipMalloc(&g1, inner_T * sizeof(float)));
  HIP_CHECK(hipMalloc(&beta, nh * T * sizeof(float)));
  HIP_CHECK(hipMalloc(&o, inner_T * sizeof(float)));
  HIP_CHECK(hipMalloc(&g2, inner_T * sizeof(float)));
  HIP_CHECK(hipMalloc(&normed, inner_T * sizeof(float)));
  HIP_CHECK(hipMalloc(&fa_out, dk * T * sizeof(float)));
  HIP_CHECK(hipMalloc(&fb_out, inner_T * sizeof(float)));
  HIP_CHECK(hipMalloc(&ga_out, dk * T * sizeof(float)));
  HIP_CHECK(hipMalloc(&gb_out, inner_T * sizeof(float)));

  const int block = 256;

  // 1. Q/K/V projections (quantized GEMM per token).
  for (std::uint32_t t = 0; t < T; ++t) {
    const float* x = input + t * H;
    if (!gpu::QuantGemm(w.w_q, x, wq + t * inner, inner, H, w.w_q_type,
                        stream) ||
        !gpu::QuantGemm(w.w_k, x, wk + t * inner, inner, H, w.w_k_type,
                        stream) ||
        !gpu::QuantGemm(w.w_v, x, wv + t * inner, inner, H, w.w_v_type, stream))
      return false;
  }

  // 2. Causal conv + SiLU.
  {
    const int grid_conv = (inner + block - 1) / block;
    hipLaunchKernelGGL(CausalConv1dKernel, dim3(grid_conv), dim3(block), 0,
                       stream, wq, w.conv_q, q_c, inner, T, K);
    hipLaunchKernelGGL(CausalConv1dKernel, dim3(grid_conv), dim3(block), 0,
                       stream, wk, w.conv_k, k_c, inner, T, K);
    hipLaunchKernelGGL(CausalConv1dKernel, dim3(grid_conv), dim3(block), 0,
                       stream, wv, w.conv_v, v_c, inner, T, K);
    // SiLU must cover all inner*T elements, not just inner.
    const int grid_act = (inner_T + block - 1) / block;
    hipLaunchKernelGGL(SiluKernel, dim3(grid_act), dim3(block), 0, stream, q_c,
                       q_c, inner_T);
    hipLaunchKernelGGL(SiluKernel, dim3(grid_act), dim3(block), 0, stream, k_c,
                       k_c, inner_T);
    hipLaunchKernelGGL(SiluKernel, dim3(grid_act), dim3(block), 0, stream, v_c,
                       v_c, inner_T);
  }

  // 3. L2 norm q, k per head.
  {
    const dim3 grid(nh, T);
    hipLaunchKernelGGL(L2NormKernel, grid, dim3(block), 0, stream, q_c, dk, nh,
                       T, 1e-6f);
    hipLaunchKernelGGL(L2NormKernel, grid, dim3(block), 0, stream, k_c, dk, nh,
                       T, 1e-6f);
  }

  // 4. Decay gate.
  {
    // fa_out = f_a @ x  [dk, T].
    const int grid_fa = (dk * T + block - 1) / block;
    hipLaunchKernelGGL(BatchedMatVecKernel, dim3(grid_fa), dim3(block), 0,
                       stream, w.f_a, input, fa_out, dk, H, T);
    // fb_out = f_b @ fa_out  [inner, T].
    const int grid_fb = (inner * T + block - 1) / block;
    hipLaunchKernelGGL(BatchedMatVecKernel, dim3(grid_fb), dim3(block), 0,
                       stream, w.f_b, fa_out, fb_out, inner, dk, T);
    // g1 = lb * sigmoid(-(fb_out + dt_bias) * A).
    const int grid_g1 = (inner * T + block - 1) / block;
    hipLaunchKernelGGL(DecayGateKernel, dim3(grid_g1), dim3(block), 0, stream,
                       fb_out, w.dt_bias, w.ssm_a, g1, inner, dk, nh, inner * T,
                       lb);
  }

  // 5. Beta.
  {
    // beta_out = ssm_beta @ x  [nh, T].
    float* beta_out;
    HIP_CHECK(hipMalloc(&beta_out, nh * T * sizeof(float)));
    const int grid = (nh * T + block - 1) / block;
    hipLaunchKernelGGL(BatchedMatVecKernel, dim3(grid), dim3(block), 0, stream,
                       w.ssm_beta, input, beta_out, nh, H, T);
    hipLaunchKernelGGL(BetaKernel, dim3(grid), dim3(block), 0, stream, beta_out,
                       beta, nh * T);
    hipFree(beta_out);
  }

  // 6. Recurrent delta rule.
  const float scale = 1.0f / sqrtf(static_cast<float>(dk));
  for (int t = 0; t < T; ++t) {
    hipLaunchKernelGGL(StateUpdateKernel, dim3(nh), dim3(block), 0, stream,
                       state, k_c, v_c, g1, beta, dk, nh, t, inner);
    hipLaunchKernelGGL(OutputKernel, dim3(nh), dim3(block), 0, stream, state,
                       q_c, o, dk, nh, t, inner, scale);
  }

  // 7. Output gate g2.
  {
    const int grid_ga = (dk * T + block - 1) / block;
    hipLaunchKernelGGL(BatchedMatVecKernel, dim3(grid_ga), dim3(block), 0,
                       stream, w.g_a, input, ga_out, dk, H, T);
    const int grid_gb = (inner * T + block - 1) / block;
    hipLaunchKernelGGL(BatchedMatVecKernel, dim3(grid_gb), dim3(block), 0,
                       stream, w.g_b, ga_out, gb_out, inner, dk, T);
  }

  // 8. RMS norm + output gate.
  {
    const dim3 grid(nh, T);
    hipLaunchKernelGGL(RmsNormKernel, grid, dim3(block), 0, stream, o,
                       w.ssm_norm, dk, nh, T, geom.norm_eps);
    hipMemcpy(normed, o, inner_T * sizeof(float), hipMemcpyDeviceToDevice);
    const int grid_gate = (inner * T + block - 1) / block;
    hipLaunchKernelGGL(OutputGateKernel, dim3(grid_gate), dim3(block), 0,
                       stream, normed, gb_out, inner_T);
  }

  // 9. Output projection (quantized GEMM per token).
  for (std::uint32_t t = 0; t < T; ++t) {
    if (!gpu::QuantGemm(w.w_out, normed + t * inner, out + t * H, H, inner,
                        w.w_out_type, stream))
      return false;
  }

  // Cleanup.
  hipFree(wq);
  hipFree(wk);
  hipFree(wv);
  hipFree(q_c);
  hipFree(k_c);
  hipFree(v_c);
  hipFree(g1);
  hipFree(beta);
  hipFree(o);
  hipFree(g2);
  hipFree(normed);
  hipFree(fa_out);
  hipFree(fb_out);
  hipFree(ga_out);
  hipFree(gb_out);

  HIP_CHECK(hipStreamSynchronize(stream));
  return true;
}

bool KdaDecodeGpu(const KdaGpuGeometry& geom, const KdaGpuWeights& w,
                  const float* token_input, float* state, float* out,
                  hipStream_t stream) {
  // Single-token decode: same as prefill with T=1.
  return KdaPrefillGpu(geom, w, token_input, state, out, 1, stream);
}

}  // namespace gufo::models::glm5_next::rocm
