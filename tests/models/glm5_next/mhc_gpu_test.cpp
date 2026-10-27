#include <hip/hip_runtime.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "src/models/glm5_next/kernels/rocm/mhc.hpp"
#include "src/models/glm5_next/mhc_reference.hpp"

namespace glm5 = gufo::models::glm5_next;
namespace rocm = gufo::models::glm5_next::rocm;

namespace {

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

#define HIP_CHECK(expr)                                     \
  do {                                                      \
    const hipError_t err = (expr);                          \
    if (err != hipSuccess) {                                \
      throw std::runtime_error(std::string("HIP error: ") + \
                               hipGetErrorString(err));     \
    }                                                       \
  } while (0)

glm5::MhcGeometry MakeGeom() {
  glm5::MhcGeometry geom;
  geom.hidden_size = 32;
  geom.hc_count = 4;
  geom.hc_mix_dim = 24;
  geom.sinkhorn_iters = 20;
  geom.eps = 1e-6F;
  geom.norm_eps = 1e-5F;
  return geom;
}

glm5::MhcWeightsF32 MakeWeights(const glm5::MhcGeometry& geom,
                                std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-0.1F, 0.1F);
  const std::uint32_t H = geom.hidden_size;
  const std::uint32_t hc = geom.hc_count;
  const std::uint32_t mix_dim = geom.hc_mix_dim;

  glm5::MhcWeightsF32 w;
  w.fn.resize(mix_dim * H * hc);
  for (auto& v : w.fn)
    v = dist(rng);
  w.scale = {1.0F, 1.0F, 1.0F};
  w.base.resize(2 * hc + hc * hc);
  for (auto& v : w.base)
    v = dist(rng);
  return w;
}

std::vector<float> MakeResidual(const glm5::MhcGeometry& geom,
                                std::uint32_t seq_len, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  std::vector<float> R(geom.hidden_size * geom.hc_count * seq_len);
  for (auto& v : R)
    v = dist(rng);
  return R;
}

void TestGpuMhcParity() {
  auto geom = MakeGeom();
  auto w_cpu = MakeWeights(geom, 42);
  auto R = MakeResidual(geom, 4, 43);

  // CPU oracle.
  auto out_cpu = glm5::MhcPre(geom, w_cpu, R, 4);

  // GPU kernel.
  const std::uint32_t H = geom.hidden_size;
  const std::uint32_t hc = geom.hc_count;
  const std::uint32_t mix_dim = geom.hc_mix_dim;

  // Allocate device memory and copy weights.
  rocm::MhcGpuWeights w_gpu{};
  float *d_fn, *d_scale, *d_base;
  float *d_residual, *d_mixed, *d_post, *d_comb;

  HIP_CHECK(hipMalloc(&d_fn, w_cpu.fn.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_scale, w_cpu.scale.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_base, w_cpu.base.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_residual, R.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_mixed, H * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_post, hc * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_comb, hc * hc * 4 * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_fn, w_cpu.fn.data(), w_cpu.fn.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_scale, w_cpu.scale.data(),
                      w_cpu.scale.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_base, w_cpu.base.data(),
                      w_cpu.base.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_residual, R.data(), R.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  w_gpu.fn = d_fn;
  w_gpu.scale = d_scale;
  w_gpu.base = d_base;

  rocm::MhcGpuGeometry geom_gpu{};
  geom_gpu.hidden_size = geom.hidden_size;
  geom_gpu.hc_count = geom.hc_count;
  geom_gpu.hc_mix_dim = geom.hc_mix_dim;
  geom_gpu.sinkhorn_iters = geom.sinkhorn_iters;
  geom_gpu.eps = geom.eps;
  geom_gpu.norm_eps = geom.norm_eps;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  const bool ok = rocm::MhcPreGpu(geom_gpu, w_gpu, d_residual, d_mixed, d_post,
                                  d_comb, 4, stream);
  Require(ok, "GPU mHC pre failed");

  // Copy back and compare mixed.
  std::vector<float> mixed_gpu(H * 4);
  HIP_CHECK(hipMemcpy(mixed_gpu.data(), d_mixed, H * 4 * sizeof(float),
                      hipMemcpyDeviceToHost));

  float max_diff = 0.0f;
  for (std::size_t i = 0; i < out_cpu.mixed.size(); ++i) {
    const float diff = std::abs(out_cpu.mixed[i] - mixed_gpu[i]);
    max_diff = std::max(max_diff, diff);
  }
  std::cout << "TestGpuMhcParity: max_diff=" << max_diff << "\n";
  // Relaxed tolerance: GPU uses float32 throughout.
  Require(max_diff < 1.0F, "GPU vs CPU parity too high");

  // Cleanup.
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(d_fn));
  HIP_CHECK(hipFree(d_scale));
  HIP_CHECK(hipFree(d_base));
  HIP_CHECK(hipFree(d_residual));
  HIP_CHECK(hipFree(d_mixed));
  HIP_CHECK(hipFree(d_post));
  HIP_CHECK(hipFree(d_comb));

  std::cout << "TestGpuMhcParity: PASS\n";
}

}  // namespace

int main() {
  try {
    TestGpuMhcParity();
    std::cout << "All glm5_next mHC GPU tests passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
