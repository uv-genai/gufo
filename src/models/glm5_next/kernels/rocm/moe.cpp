#include "src/models/glm5_next/kernels/rocm/moe.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "src/models/glm5_next/quant_gemv.hpp"
#include "strata/kernels/glm_ffn.hpp"
#include "strata/kernels/router_sigmoid.hpp"

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

/// y[r] = sum_s w[s] * x[s * n + r]  (weighted combine of expert outputs).
__global__ void WeightedSumKernel(const float* __restrict__ x,
                                  const float* __restrict__ w,
                                  float* __restrict__ y, int n, int k) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= n)
    return;
  float s = 0.0f;
  for (int i = 0; i < k; ++i)
    s += w[i] * x[static_cast<std::size_t>(i) * n + r];
  y[r] = s;
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
  const std::size_t k = geom.top_k;
  return 3ULL * geom.num_experts + 2ULL * k + 3ULL * k * geom.expert_ff +
         k * geom.hidden_size + 2ULL * geom.hidden_size + 3ULL * geom.shared_ff;
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
  float* ids_f = take(k);  // device int32 ids (reused as floats)
  float* w_dev = take(k);  // device expert weights
  float* gate_all = take(k * I);
  float* up_all = take(k * I);
  float* h_all = take(k * I);
  float* down_all = take(k * H);
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

  // 2-5. Maya router_sigmoid (copied): top-k ids + renormed weights.
  std::vector<std::uint32_t> expert_ids(k);
  std::vector<float> weights(k);
  {
    int* d_ids = nullptr;
    float* d_rw = nullptr;
    if (hipMalloc(&d_ids, k * sizeof(int)) != hipSuccess)
      return false;
    if (hipMalloc(&d_rw, k * sizeof(float)) != hipSuccess) {
      hipFree(d_ids);
      return false;
    }
    strata::kernels::router_sigmoid(
        logits, w.exp_probs_b, 1, (int)E, (int)k, geom.expert_weights_scale,
        geom.expert_weights_norm, d_ids, d_rw, stream);
    std::vector<int> ids_host(k);
    HIP_CHECK(hipMemcpyAsync(ids_host.data(), d_ids, k * sizeof(int),
                             hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipMemcpyAsync(weights.data(), d_rw, k * sizeof(float),
                             hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    hipFree(d_ids);
    hipFree(d_rw);
    for (std::uint32_t i = 0; i < k; ++i)
      expert_ids[i] = static_cast<std::uint32_t>(ids_host[i]);
  }

  // 6. Zero accumulator.
  HIP_CHECK(hipMemset(acc, 0, H * sizeof(float)));

  // Byte stride between expert matrices of a quantized 3D tensor.
  const std::size_t gate_stride = gpu::QuantMatrixBytes(I, H, w.gate_exps_type);
  const std::size_t up_stride = gpu::QuantMatrixBytes(I, H, w.up_exps_type);
  const std::size_t down_stride = gpu::QuantMatrixBytes(H, I, w.down_exps_type);
  if (gate_stride == 0 || up_stride == 0 || down_stride == 0) {
    return false;
  }

  // 7. Routed experts (batched): gate/up for every expert in one call each,
  // one swiglu over all slots, one grouped down (per-slot activations), and
  // one weighted sum. The per-expert loop cost ~48 launches per layer.
  auto* ids_dev = reinterpret_cast<std::int32_t*>(ids_f);
  // Persistent host staging, copied on the caller's stream: the copies must
  // be stream-ordered with the kernels (a default-stream hipMemcpy races
  // with them), and the source must outlive the async copy (function-locals
  // died while the stream was still reading them).
  static std::vector<std::int32_t> s_ids_host;
  static std::vector<float> s_weights_host;
  s_ids_host.resize(k);
  s_weights_host = weights;
  for (std::uint32_t i = 0; i < k; ++i)
    s_ids_host[i] = static_cast<std::int32_t>(expert_ids[i]);
  HIP_CHECK(hipMemcpyAsync(ids_dev, s_ids_host.data(), k * sizeof(std::int32_t),
                           hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(w_dev, s_weights_host.data(), k * sizeof(float),
                           hipMemcpyHostToDevice, stream));
  if (!gpu::QuantGemmMoe(w.gate_exps, token_input, ids_dev, gate_all, I, H, E,
                         k, w.gate_exps_type, stream))
    return false;
  if (!gpu::QuantGemmMoe(w.up_exps, token_input, ids_dev, up_all, I, H, E, k,
                         w.up_exps_type, stream))
    return false;
  {
    const int total = static_cast<int>(k) * I;
    const int grid = (total + block - 1) / block;
    strata::kernels::glm_swiglu_clamp(gate_all, up_all, limit, total, h_all,
                                      stream);
  }
  if (!gpu::QuantGemmMoe(w.down_exps, h_all, ids_dev, down_all, H, I, E, k,
                         w.down_exps_type, stream, H))
    return false;
  {
    const int grid = (H + block - 1) / block;
    hipLaunchKernelGGL(WeightedSumKernel, dim3(grid), dim3(block), 0, stream,
                       down_all, w_dev, acc, H, static_cast<int>(k));
  }

  // 8. Shared expert.
  {
    const int grid = (SI + block - 1) / block;
    hipLaunchKernelGGL(MatVecKernel, dim3(SI), dim3(256), 0, stream,
                       w.gate_shexp, token_input, gate_s, SI, H);
    hipLaunchKernelGGL(MatVecKernel, dim3(SI), dim3(256), 0, stream, w.up_shexp,
                       token_input, up_s, SI, H);
    strata::kernels::glm_swiglu_clamp(gate_s, up_s, limit, SI, h_s, stream);
    hipLaunchKernelGGL(MatVecKernel, dim3(H), dim3(256), 0, stream,
                       w.down_shexp, h_s, shexp, H, SI);
  }

  // 9. out = acc + shexp.
  {
    HIP_CHECK(hipMemcpyAsync(out, acc, H * sizeof(float),
                             hipMemcpyDeviceToDevice, stream));
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
    strata::kernels::glm_swiglu_clamp(g, u, limit, inter, h, stream);
  }

  // out = down @ h.
  const bool ok =
      gpu::QuantGemm(down, h, out, hidden, inter, down_type, stream);

  if (owned)
    hipFree(owned);
  return ok;
}

}  // namespace gufo::models::glm5_next::rocm
