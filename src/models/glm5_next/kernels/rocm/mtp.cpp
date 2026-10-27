#include "src/models/glm5_next/kernels/rocm/mtp.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstring>

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

/// Concatenate [enorm | hnorm] → [2*H].
__global__ void ConcatKernel(const float* __restrict__ a,
                             const float* __restrict__ b,
                             float* __restrict__ out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i];
    out[n + i] = b[i];
  }
}

}  // namespace

bool MtpDraftGpu(const MtpGpuGeometry& geom, const MtpGpuWeights& w,
                 const float* embedding, const float* hidden_state,
                 float* out_logits, hipStream_t stream) {
  const std::uint32_t H = geom.hidden_size;

  // Allocate scratch.
  float *en, *hn, *eh, *proj, *normed;
  HIP_CHECK(hipMalloc(&en, H * sizeof(float)));
  HIP_CHECK(hipMalloc(&hn, H * sizeof(float)));
  HIP_CHECK(hipMalloc(&eh, 2 * H * sizeof(float)));
  HIP_CHECK(hipMalloc(&proj, H * sizeof(float)));
  HIP_CHECK(hipMalloc(&normed, H * sizeof(float)));

  const int block = 256;

  // 1. enorm = rms_norm(embedding).
  HIP_CHECK(
      hipMemcpy(en, embedding, H * sizeof(float), hipMemcpyDeviceToDevice));
  hipLaunchKernelGGL(RmsNormKernel, dim3(1), dim3(block), 0, stream, en,
                     w.enorm, H, geom.norm_eps);

  // 2. hnorm = rms_norm(hidden_state).
  HIP_CHECK(
      hipMemcpy(hn, hidden_state, H * sizeof(float), hipMemcpyDeviceToDevice));
  hipLaunchKernelGGL(RmsNormKernel, dim3(1), dim3(block), 0, stream, hn,
                     w.hnorm, H, geom.norm_eps);

  // 3. concat [enorm | hnorm].
  {
    const int grid = (H + block - 1) / block;
    hipLaunchKernelGGL(ConcatKernel, dim3(grid), dim3(block), 0, stream, en, hn,
                       eh, H);
  }

  // 4. proj = eh_proj @ eh  [H].
  {
    const int grid = (H + block - 1) / block;
    hipLaunchKernelGGL(MatVecKernel, dim3(grid), dim3(block), 0, stream,
                       w.eh_proj, eh, proj, H, 2 * H);
  }

  // 5. shared_head_norm on proj.
  HIP_CHECK(
      hipMemcpy(normed, proj, H * sizeof(float), hipMemcpyDeviceToDevice));
  hipLaunchKernelGGL(RmsNormKernel, dim3(1), dim3(block), 0, stream, normed,
                     w.shared_head_norm, H, geom.norm_eps);

  // 6. logits = output_head @ normed  [vocab].
  {
    const int grid = (geom.vocab_size + block - 1) / block;
    hipLaunchKernelGGL(MatVecKernel, dim3(grid), dim3(block), 0, stream,
                       w.output_head, normed, out_logits, geom.vocab_size, H);
  }

  // Cleanup.
  HIP_CHECK(hipFree(en));
  HIP_CHECK(hipFree(hn));
  HIP_CHECK(hipFree(eh));
  HIP_CHECK(hipFree(proj));
  HIP_CHECK(hipFree(normed));
  HIP_CHECK(hipStreamSynchronize(stream));
  return true;
}

bool MtpVerify(std::uint32_t draft_token, std::uint32_t target_token) {
  return draft_token == target_token;
}

void MtpRollbackKdaState(float* state, const float* checkpoint,
                         std::size_t size) {
  std::memcpy(state, checkpoint, size * sizeof(float));
}

}  // namespace gufo::models::glm5_next::rocm
