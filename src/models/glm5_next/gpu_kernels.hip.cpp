#include "src/models/glm5_next/gpu_kernels.hpp"

namespace gufo::models::glm5_next::gpu {
namespace {

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
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    x[i] = x[i] * inv_rms * gamma[i];
  }
}

__global__ void MatVecKernel(const float* __restrict__ W,
                             const float* __restrict__ x, float* __restrict__ y,
                             int m, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m) {
    float acc = 0.0f;
    const float* row = W + i * n;
    for (int j = 0; j < n; ++j)
      acc += row[j] * x[j];
    y[i] = acc;
  }
}

__global__ void EmbedLookupKernel(const float* __restrict__ embd,
                                  float* __restrict__ out, int hidden,
                                  int token) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < hidden)
    out[i] = embd[token * hidden + i];
}

__global__ void ArgmaxKernel(const float* __restrict__ logits, int n,
                             int* __restrict__ result) {
  __shared__ float best_val[256];
  __shared__ int best_idx[256];
  const int tid = threadIdx.x;
  best_val[tid] = -1e30f;
  best_idx[tid] = -1;
  for (int i = tid; i < n; i += blockDim.x) {
    if (logits[i] > best_val[tid]) {
      best_val[tid] = logits[i];
      best_idx[tid] = i;
    }
  }
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s && best_val[tid + s] > best_val[tid]) {
      best_val[tid] = best_val[tid + s];
      best_idx[tid] = best_idx[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0)
    *result = best_idx[0];
}

}  // namespace

void LaunchRmsNorm(float* x, const float* gamma, int n, float eps,
                   hipStream_t stream) {
  hipLaunchKernelGGL(RmsNormKernel, dim3(1), dim3(256), 0, stream, x, gamma, n,
                     eps);
}

void LaunchMatVec(const float* W, const float* x, float* y, int m, int n,
                  hipStream_t stream) {
  const int block = 256;
  const int grid = (m + block - 1) / block;
  hipLaunchKernelGGL(MatVecKernel, dim3(grid), dim3(block), 0, stream, W, x, y,
                     m, n);
}

void LaunchEmbedLookup(const float* embd, float* out, int hidden, int token,
                       hipStream_t stream) {
  const int block = 256;
  const int grid = (hidden + block - 1) / block;
  hipLaunchKernelGGL(EmbedLookupKernel, dim3(grid), dim3(block), 0, stream,
                     embd, out, hidden, token);
}

void LaunchArgmax(const float* logits, int n, int* host_result,
                  hipStream_t stream) {
  int* d_result = nullptr;
  hipMalloc(&d_result, sizeof(int));
  hipLaunchKernelGGL(ArgmaxKernel, dim3(1), dim3(256), 0, stream, logits, n,
                     d_result);
  hipStreamSynchronize(stream);
  hipMemcpy(host_result, d_result, sizeof(int), hipMemcpyDeviceToHost);
  hipFree(d_result);
}

}  // namespace gufo::models::glm5_next::gpu
