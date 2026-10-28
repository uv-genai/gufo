#include <hip/hip_runtime.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "src/models/glm5_next/kernels/rocm/moe.hpp"
#include "src/models/glm5_next/moe_reference.hpp"

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

glm5::MoeGeometry MakeGeom() {
  glm5::MoeGeometry geom;
  geom.hidden_size = 32;
  geom.num_experts = 16;
  geom.top_k = 4;
  geom.expert_ff = 8;
  geom.shared_ff = 8;
  geom.expert_weights_scale = 2.5F;
  geom.expert_weights_norm = true;
  geom.swiglu_limit = 10.0F;
  geom.renorm_clamp = 6.103515625e-5F;
  return geom;
}

glm5::MoeWeightsF32 MakeWeights(const glm5::MoeGeometry& geom,
                                std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-0.1F, 0.1F);
  const std::uint32_t H = geom.hidden_size;
  const std::uint32_t E = geom.num_experts;
  const std::uint32_t I = geom.expert_ff;
  const std::uint32_t SI = geom.shared_ff;

  glm5::MoeWeightsF32 w;
  w.gate_inp.resize(E * H);
  for (auto& v : w.gate_inp)
    v = dist(rng);
  w.exp_probs_b.resize(E);
  for (auto& v : w.exp_probs_b)
    v = dist(rng);
  w.gate_exps.resize(static_cast<std::size_t>(E) * H * I);
  w.up_exps.resize(static_cast<std::size_t>(E) * H * I);
  w.down_exps.resize(static_cast<std::size_t>(E) * I * H);
  for (auto& v : w.gate_exps)
    v = dist(rng);
  for (auto& v : w.up_exps)
    v = dist(rng);
  for (auto& v : w.down_exps)
    v = dist(rng);
  w.gate_shexp.resize(H * SI);
  w.up_shexp.resize(H * SI);
  w.down_shexp.resize(SI * H);
  for (auto& v : w.gate_shexp)
    v = dist(rng);
  for (auto& v : w.up_shexp)
    v = dist(rng);
  for (auto& v : w.down_shexp)
    v = dist(rng);
  return w;
}

std::vector<float> MakeInput(const glm5::MoeGeometry& geom,
                             std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  std::vector<float> x(geom.hidden_size);
  for (auto& v : x)
    v = dist(rng);
  return x;
}

void TestGpuMoeParity() {
  auto geom = MakeGeom();
  auto w_cpu = MakeWeights(geom, 42);
  auto x = MakeInput(geom, 43);

  // CPU oracle.
  auto out_cpu = glm5::MoEForward(geom, w_cpu, x);

  // GPU kernel.
  const std::uint32_t H = geom.hidden_size;
  const std::uint32_t E = geom.num_experts;
  const std::uint32_t I = geom.expert_ff;
  const std::uint32_t SI = geom.shared_ff;

  // Allocate device memory and copy weights.
  rocm::MoeGpuWeights w_gpu{};
  float *d_gate_inp, *d_exp_probs_b;
  float *d_gate_exps, *d_up_exps, *d_down_exps;
  float *d_gate_shexp, *d_up_shexp, *d_down_shexp;
  float *d_input, *d_out;

  HIP_CHECK(hipMalloc(&d_gate_inp, w_cpu.gate_inp.size() * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&d_exp_probs_b, w_cpu.exp_probs_b.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate_exps, w_cpu.gate_exps.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_up_exps, w_cpu.up_exps.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_down_exps, w_cpu.down_exps.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate_shexp, w_cpu.gate_shexp.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_up_shexp, w_cpu.up_shexp.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_down_shexp, w_cpu.down_shexp.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_input, x.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, H * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_gate_inp, w_cpu.gate_inp.data(),
                      w_cpu.gate_inp.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_exp_probs_b, w_cpu.exp_probs_b.data(),
                      w_cpu.exp_probs_b.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate_exps, w_cpu.gate_exps.data(),
                      w_cpu.gate_exps.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_up_exps, w_cpu.up_exps.data(),
                      w_cpu.up_exps.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_down_exps, w_cpu.down_exps.data(),
                      w_cpu.down_exps.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate_shexp, w_cpu.gate_shexp.data(),
                      w_cpu.gate_shexp.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_up_shexp, w_cpu.up_shexp.data(),
                      w_cpu.up_shexp.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_down_shexp, w_cpu.down_shexp.data(),
                      w_cpu.down_shexp.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_input, x.data(), x.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  w_gpu.gate_inp = d_gate_inp;
  w_gpu.exp_probs_b = d_exp_probs_b;
  w_gpu.gate_exps = d_gate_exps;
  w_gpu.up_exps = d_up_exps;
  w_gpu.down_exps = d_down_exps;
  w_gpu.gate_shexp = d_gate_shexp;
  w_gpu.up_shexp = d_up_shexp;
  w_gpu.down_shexp = d_down_shexp;

  rocm::MoeGpuGeometry geom_gpu{};
  geom_gpu.hidden_size = geom.hidden_size;
  geom_gpu.num_experts = geom.num_experts;
  geom_gpu.top_k = geom.top_k;
  geom_gpu.expert_ff = geom.expert_ff;
  geom_gpu.shared_ff = geom.shared_ff;
  geom_gpu.expert_weights_scale = geom.expert_weights_scale;
  geom_gpu.expert_weights_norm = geom.expert_weights_norm;
  geom_gpu.swiglu_limit = geom.swiglu_limit;
  geom_gpu.renorm_clamp = geom.renorm_clamp;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  const bool ok =
      rocm::MoeForwardGpu(geom_gpu, w_gpu, d_input, d_out, nullptr, stream);
  Require(ok, "GPU MoE forward failed");

  // Copy back and compare.
  std::vector<float> out_gpu(H);
  HIP_CHECK(hipMemcpy(out_gpu.data(), d_out, H * sizeof(float),
                      hipMemcpyDeviceToHost));

  float max_diff = 0.0f;
  for (std::size_t i = 0; i < out_cpu.size(); ++i) {
    const float diff = std::abs(out_cpu[i] - out_gpu[i]);
    max_diff = std::max(max_diff, diff);
  }
  std::cout << "TestGpuMoeParity: max_diff=" << max_diff << "\n";
  Require(max_diff < 1e-3F, "GPU vs CPU parity too high");

  // Cleanup.
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(d_gate_inp));
  HIP_CHECK(hipFree(d_exp_probs_b));
  HIP_CHECK(hipFree(d_gate_exps));
  HIP_CHECK(hipFree(d_up_exps));
  HIP_CHECK(hipFree(d_down_exps));
  HIP_CHECK(hipFree(d_gate_shexp));
  HIP_CHECK(hipFree(d_up_shexp));
  HIP_CHECK(hipFree(d_down_shexp));
  HIP_CHECK(hipFree(d_input));
  HIP_CHECK(hipFree(d_out));

  std::cout << "TestGpuMoeParity: PASS\n";
}

}  // namespace

int main() {
  try {
    TestGpuMoeParity();
    std::cout << "All glm5_next MoE GPU tests passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
