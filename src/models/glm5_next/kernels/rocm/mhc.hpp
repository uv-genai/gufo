#ifndef GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_MHC_HPP_
#define GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_MHC_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::glm5_next::rocm {

/// mHC (manifold-constrained Hyper-Connections) GPU kernel interface.
///
/// Math (matches CPU oracle in mhc_reference.cpp):
///   Read half:
///     1. Joint inverse-RMS over all hc streams (NOT per-stream)
///     2. mixes = hc_fn @ flat_norm  (24-row GEMV)
///     3. pre = sigmoid(mixes[0:hc]*scale[0] + base[0:hc]) + eps
///     4. post = 2*sigmoid(mixes[hc:2hc]*scale[1] + base[hc:2hc])
///     5. comb[dst,src] = sinkhorn(mixes[2hc+dst+hc*src]*scale[2] + base[...])
///        NOTE: src-major row order (dst + hc*src)
///     6. mixed = sum_s pre_s * R_s
///   Sinkhorn (per token, per layer):
///     softmax over dst, +eps, dst-norm, then (iters-1)×[src-norm, dst-norm]
///   Write half:
///     out_dst = block_out * post_dst + sum_src comb[dst,src] * residual_src

struct MhcGpuGeometry {
  std::uint32_t hidden_size;
  std::uint32_t hc_count;
  std::uint32_t hc_mix_dim;
  std::uint32_t sinkhorn_iters;
  float eps;
  float norm_eps;
};

/// Device pointers for mHC weights (one mixer: attn or ffn).
struct MhcGpuWeights {
  const float* fn;     ///< [hc_mix_dim, hc_count * hidden_size]
  const float* scale;  ///< [3]
  const float* base;   ///< [2*hc_count + hc_count*hc_count]
};

/// Run mHC read half on GPU.
/// residual: [hidden, hc, T] on device (stream-major layout).
/// mixed: [hidden, T] on device (output).
/// post: [hc, T] on device (output).
/// comb: [hc*hc*T] on device (output, dst-major per token).
/// Returns true on success.
/// Device scratch floats MhcPreGpu needs for T tokens.
[[nodiscard]] std::size_t MhcScratchFloats(const MhcGpuGeometry& geom,
                                           std::uint32_t seq_len);

/// scratch: MhcScratchFloats() floats of device memory, or nullptr to
/// allocate internally (slow — tests only).
bool MhcPreGpu(const MhcGpuGeometry& geom, const MhcGpuWeights& w,
               const float* residual, float* mixed, float* post, float* comb,
               std::uint32_t seq_len, float* scratch, hipStream_t stream);

/// Run mHC write half on GPU.
/// block_out: [hidden, T] on device.
/// residual: [hidden, hc, T] on device.
/// post: [hc, T] on device.
/// comb: [hc*hc*T] on device.
/// out: [hidden, hc, T] on device (output).
/// Returns true on success.
bool MhcPostGpu(const MhcGpuGeometry& geom, const float* block_out,
                const float* residual, const float* post, const float* comb,
                float* out, std::uint32_t seq_len, hipStream_t stream);

}  // namespace gufo::models::glm5_next::rocm

#endif  // GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_MHC_HPP_
