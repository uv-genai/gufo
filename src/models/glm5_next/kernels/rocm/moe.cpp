#include "src/models/glm5_next/kernels/rocm/moe.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
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

/// Sigmoid kernel.
__global__ void SigmoidKernel(const float* __restrict__ x,
                              float* __restrict__ y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = 1.0f / (1.0f + expf(-x[i]));
  }
}

/// Add bias for selection: sel_scores = probs + bias.
__global__ void AddBiasKernel(const float* __restrict__ probs,
                              const float* __restrict__ bias,
                              float* __restrict__ sel_scores, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    sel_scores[i] = probs[i] + bias[i];
  }
}

/// MatVec: y[m] = W[m, n] @ x[n] (row-major W).
__global__ void MatVecKernel(const float* __restrict__ W,
                             const float* __restrict__ x, float* __restrict__ y,
                             int m, int n) {
  // One workgroup per row; threads split the dot and sum partials in thread
  // order (a serial per-row chain is latency-bound; a tree reduction can flip
  // MoE top-k selection through rounding).
  __shared__ float red[256];
  const int row = blockIdx.x;
  if (row >= m)
    return;
  const float* wrow = W + static_cast<std::size_t>(row) * n;
  float acc = 0.0f;
  for (int j = static_cast<int>(threadIdx.x); j < n;
       j += static_cast<int>(blockDim.x))
    acc += wrow[j] * x[j];
  red[threadIdx.x] = acc;
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = 0.0f;
    for (int i = 0; i < static_cast<int>(blockDim.x); ++i)
      s += red[i];
    y[row] = s;
  }
}

/// Clamped SwiGLU kernel: silu(min(gate, limit)) * clamp(up, ±limit).
__global__ void SwigluClampKernel(const float* __restrict__ gate,
                                  const float* __restrict__ up,
                                  float* __restrict__ out, int n, float limit) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float g = fminf(gate[i], limit);
    const float u = fmaxf(fminf(up[i], limit), -limit);
    const float sig = 1.0f / (1.0f + expf(-g));
    out[i] = g * sig * u;
  }
}

/// Accumulate weighted expert output.
__global__ void AccumulateKernel(float* __restrict__ acc,
                                 const float* __restrict__ expert_out,
                                 float weight, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    acc[i] += weight * expert_out[i];
  }
}

/// Add shared expert output.
__global__ void AddKernel(float* __restrict__ out,
                          const float* __restrict__ shexp, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] += shexp[i];
  }
}

}  // namespace

std::size_t MoeScratchFloats(const MoeGpuGeometry& geom) {
  return 3ULL * geom.num_experts + 3ULL * geom.expert_ff +
         3ULL * geom.hidden_size + 3ULL * geom.shared_ff;
}

bool MoeForwardGpu(const MoeGpuGeometry& geom, const MoeGpuWeights& w,
                   const float* token_input, float* out, float* scratch,
                   hipStream_t stream) {
  const std::uint32_t H = geom.hidden_size;
  const std::uint32_t E = geom.num_experts;
  const std::uint32_t I = geom.expert_ff;
  const std::uint32_t SI = geom.shared_ff;
  const std::uint32_t k = geom.top_k;
  const float limit = geom.swiglu_limit;

  // Device scratch (caller-owned when provided; bump-allocated).
  float* owned = nullptr;
  if (scratch == nullptr) {
    HIP_CHECK(hipMalloc(&owned, MoeScratchFloats(geom) * sizeof(float)));
    scratch = owned;
  }
  float* sp = scratch;
  auto take = [&sp](std::size_t n) {
    float* r = sp;
    sp += n;
    return r;
  };
  float* logits = take(E);
  float* probs = take(E);
  float* sel_scores = take(E);
  float* gate = take(I);
  float* up = take(I);
  float* h = take(I);
  float* expert_out = take(H);
  float* acc = take(H);
  float* gate_s = take(SI);
  float* up_s = take(SI);
  float* h_s = take(SI);
  float* shexp = take(H);

  const int block = 256;

  // 1. Router: logits = gate_inp @ x.
  {
    hipLaunchKernelGGL(MatVecKernel, dim3(E), dim3(256), 0, stream, w.gate_inp,
                       token_input, logits, E, H);
  }

  // 2. probs = sigmoid(logits).
  {
    const int grid = (E + block - 1) / block;
    hipLaunchKernelGGL(SigmoidKernel, dim3(grid), dim3(block), 0, stream,
                       logits, probs, E);
  }

  // 3. sel_scores = probs + bias.
  {
    const int grid = (E + block - 1) / block;
    hipLaunchKernelGGL(AddBiasKernel, dim3(grid), dim3(block), 0, stream, probs,
                       w.exp_probs_b, sel_scores, E);
  }

  // 4. Top-k selection (on host for simplicity — would be on GPU in
  // production). Copy sel_scores back to host for selection.
  std::vector<float> sel_scores_host(E);
  HIP_CHECK(hipMemcpy(sel_scores_host.data(), sel_scores, E * sizeof(float),
                      hipMemcpyDeviceToHost));
  std::vector<float> probs_host(E);
  HIP_CHECK(hipMemcpy(probs_host.data(), probs, E * sizeof(float),
                      hipMemcpyDeviceToHost));

  // Top-k descending.
  std::vector<std::pair<float, std::uint32_t>> indexed(E);
  for (std::uint32_t i = 0; i < E; ++i) {
    indexed[i] = {-sel_scores_host[i], i};
  }
  std::stable_sort(indexed.begin(), indexed.end());

  // 5. Weights from unbiased probs, renorm + scale.
  std::vector<float> weights(k);
  float w_sum = 0.0f;
  for (std::uint32_t i = 0; i < k; ++i) {
    weights[i] = probs_host[indexed[i].second];
    w_sum += weights[i];
  }
  if (geom.expert_weights_norm) {
    const float inv = 1.0f / fmaxf(w_sum, geom.renorm_clamp);
    for (auto& v : weights)
      v *= inv;
  }
  for (auto& v : weights)
    v *= geom.expert_weights_scale;

  // 6. Zero accumulator.
  HIP_CHECK(hipMemset(acc, 0, H * sizeof(float)));

  // Byte stride between expert matrices of a quantized 3D tensor.
  const std::size_t gate_stride = gpu::QuantMatrixBytes(I, H, w.gate_exps_type);
  const std::size_t up_stride = gpu::QuantMatrixBytes(I, H, w.up_exps_type);
  const std::size_t down_stride = gpu::QuantMatrixBytes(H, I, w.down_exps_type);
  if (gate_stride == 0 || up_stride == 0 || down_stride == 0) {
    return false;
  }

  // 7. Routed experts.
  for (std::uint32_t idx = 0; idx < k; ++idx) {
    const std::uint32_t e = indexed[idx].second;
    const float weight = weights[idx];

    // gate = gate_exps[e] @ x  [I] (quantized GEMM).
    {
      const auto* gW = reinterpret_cast<const void*>(
          reinterpret_cast<const std::uint8_t*>(w.gate_exps) +
          static_cast<std::size_t>(e) * gate_stride);
      if (!gpu::QuantGemm(gW, token_input, gate, I, H, w.gate_exps_type,
                          stream))
        return false;
    }
    // up = up_exps[e] @ x  [I] (quantized GEMM).
    {
      const auto* uW = reinterpret_cast<const void*>(
          reinterpret_cast<const std::uint8_t*>(w.up_exps) +
          static_cast<std::size_t>(e) * up_stride);
      if (!gpu::QuantGemm(uW, token_input, up, I, H, w.up_exps_type, stream))
        return false;
    }
    // h = swiglu_clamp(gate, up).
    {
      const int grid = (I + block - 1) / block;
      hipLaunchKernelGGL(SwigluClampKernel, dim3(grid), dim3(block), 0, stream,
                         gate, up, h, I, limit);
    }
    // expert_out = down_exps[e] @ h  [H] (quantized GEMM).
    {
      const auto* dW = reinterpret_cast<const void*>(
          reinterpret_cast<const std::uint8_t*>(w.down_exps) +
          static_cast<std::size_t>(e) * down_stride);
      if (!gpu::QuantGemm(dW, h, expert_out, H, I, w.down_exps_type, stream))
        return false;
    }
    // acc += weight * expert_out.
    {
      const int grid = (H + block - 1) / block;
      hipLaunchKernelGGL(AccumulateKernel, dim3(grid), dim3(block), 0, stream,
                         acc, expert_out, weight, H);
    }
  }

  // 8. Shared expert.
  {
    const int grid = (SI + block - 1) / block;
    hipLaunchKernelGGL(MatVecKernel, dim3(SI), dim3(256), 0, stream,
                       w.gate_shexp, token_input, gate_s, SI, H);
    hipLaunchKernelGGL(MatVecKernel, dim3(SI), dim3(256), 0, stream, w.up_shexp,
                       token_input, up_s, SI, H);
    hipLaunchKernelGGL(SwigluClampKernel, dim3(grid), dim3(block), 0, stream,
                       gate_s, up_s, h_s, SI, limit);
    hipLaunchKernelGGL(MatVecKernel, dim3(H), dim3(256), 0, stream,
                       w.down_shexp, h_s, shexp, H, SI);
  }

  // 9. out = acc + shexp.
  {
    HIP_CHECK(hipMemcpy(out, acc, H * sizeof(float), hipMemcpyDeviceToDevice));
    const int grid = (H + block - 1) / block;
    hipLaunchKernelGGL(AddKernel, dim3(grid), dim3(block), 0, stream, out,
                       shexp, H);
  }

  if (owned)
    hipFree(owned);
  return true;
}

std::size_t DenseScratchFloats(std::uint32_t inter) {
  return 3ULL * inter;
}

bool DenseFfnForwardGpu(std::uint32_t hidden, std::uint32_t inter, float limit,
                        const float* gate, const float* up, const float* down,
                        int gate_type, int up_type, int down_type,
                        const float* token_input, float* out, float* scratch,
                        hipStream_t stream) {
  float* owned = nullptr;
  if (scratch == nullptr) {
    HIP_CHECK(hipMalloc(&owned, DenseScratchFloats(inter) * sizeof(float)));
    scratch = owned;
  }
  float* g = scratch;
  float* u = scratch + inter;
  float* h = scratch + 2 * inter;

  const int block = 256;

  // gate = gate @ x, up = up @ x (quantized on-the-fly).
  if (!gpu::QuantGemm(gate, token_input, g, inter, hidden, gate_type, stream) ||
      !gpu::QuantGemm(up, token_input, u, inter, hidden, up_type, stream)) {
    if (owned)
      hipFree(owned);
    return false;
  }
  {
    const int grid = (inter + block - 1) / block;
    hipLaunchKernelGGL(SwigluClampKernel, dim3(grid), dim3(block), 0, stream, g,
                       u, h, inter, limit);
  }

  // out = down @ h.
  const bool ok =
      gpu::QuantGemm(down, h, out, hidden, inter, down_type, stream);

  if (owned)
    hipFree(owned);
  return ok;
}

}  // namespace gufo::models::glm5_next::rocm
