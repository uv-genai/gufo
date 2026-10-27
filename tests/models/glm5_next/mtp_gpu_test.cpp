#include <hip/hip_runtime.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "src/models/glm5_next/kernels/rocm/mtp.hpp"
#include "src/models/glm5_next/mtp_reference.hpp"

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

glm5::MtpGeometry MakeGeom() {
  glm5::MtpGeometry geom;
  geom.hidden_size = 64;
  geom.vocab_size = 128;
  geom.norm_eps = 1e-5F;
  geom.fixed_draft_width = 4;
  return geom;
}

glm5::MtpWeightsF32 MakeWeights(const glm5::MtpGeometry& geom,
                                std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-0.1F, 0.1F);
  const std::uint32_t H = geom.hidden_size;

  glm5::MtpWeightsF32 w;
  w.enorm.resize(H);
  w.hnorm.resize(H);
  w.eh_proj.resize(H * 2 * H);
  w.shared_head_norm.resize(H);
  w.output_head.resize(static_cast<std::size_t>(geom.vocab_size) * H);

  for (auto& v : w.enorm)
    v = dist(rng) + 1.0F;
  for (auto& v : w.hnorm)
    v = dist(rng) + 1.0F;
  for (auto& v : w.eh_proj)
    v = dist(rng);
  for (auto& v : w.shared_head_norm)
    v = dist(rng) + 1.0F;
  for (auto& v : w.output_head)
    v = dist(rng);
  return w;
}

std::vector<float> MakeVec(std::uint32_t n, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  std::vector<float> v(n);
  for (auto& x : v)
    x = dist(rng);
  return v;
}

void TestGpuMtpParity() {
  auto geom = MakeGeom();
  auto w_cpu = MakeWeights(geom, 42);
  auto emb = MakeVec(geom.hidden_size, 43);
  auto hid = MakeVec(geom.hidden_size, 44);

  // CPU oracle.
  auto logits_cpu = glm5::MtpDraft(geom, w_cpu, emb, hid);

  // GPU kernel.
  const std::uint32_t H = geom.hidden_size;

  // Allocate device memory and copy weights.
  rocm::MtpGpuWeights w_gpu{};
  float *d_enorm, *d_hnorm, *d_eh_proj, *d_shared_head_norm, *d_output_head;
  float *d_emb, *d_hid, *d_logits;

  HIP_CHECK(hipMalloc(&d_enorm, w_cpu.enorm.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_hnorm, w_cpu.hnorm.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_eh_proj, w_cpu.eh_proj.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_shared_head_norm,
                      w_cpu.shared_head_norm.size() * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&d_output_head, w_cpu.output_head.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_emb, emb.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_hid, hid.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_logits, geom.vocab_size * sizeof(float)));

  auto copy = [&](auto& dst, const auto& src) {
    HIP_CHECK(hipMemcpy(dst, src.data(), src.size() * sizeof(float),
                        hipMemcpyHostToDevice));
  };
  copy(d_enorm, w_cpu.enorm);
  copy(d_hnorm, w_cpu.hnorm);
  copy(d_eh_proj, w_cpu.eh_proj);
  copy(d_shared_head_norm, w_cpu.shared_head_norm);
  copy(d_output_head, w_cpu.output_head);
  copy(d_emb, emb);
  copy(d_hid, hid);

  w_gpu.enorm = d_enorm;
  w_gpu.hnorm = d_hnorm;
  w_gpu.eh_proj = d_eh_proj;
  w_gpu.shared_head_norm = d_shared_head_norm;
  w_gpu.output_head = d_output_head;

  rocm::MtpGpuGeometry geom_gpu{};
  geom_gpu.hidden_size = geom.hidden_size;
  geom_gpu.vocab_size = geom.vocab_size;
  geom_gpu.norm_eps = geom.norm_eps;
  geom_gpu.fixed_draft_width = geom.fixed_draft_width;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  const bool ok =
      rocm::MtpDraftGpu(geom_gpu, w_gpu, d_emb, d_hid, d_logits, stream);
  Require(ok, "GPU MTP draft failed");

  // Copy back and compare.
  std::vector<float> logits_gpu(geom.vocab_size);
  HIP_CHECK(hipMemcpy(logits_gpu.data(), d_logits,
                      geom.vocab_size * sizeof(float), hipMemcpyDeviceToHost));

  float max_diff = 0.0f;
  for (std::size_t i = 0; i < logits_cpu.size(); ++i) {
    const float diff = std::abs(logits_cpu[i] - logits_gpu[i]);
    max_diff = std::max(max_diff, diff);
  }
  std::cout << "TestGpuMtpParity: max_diff=" << max_diff << "\n";
  Require(max_diff < 1e-3F, "GPU vs CPU parity too high");

  // Cleanup.
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(d_enorm));
  HIP_CHECK(hipFree(d_hnorm));
  HIP_CHECK(hipFree(d_eh_proj));
  HIP_CHECK(hipFree(d_shared_head_norm));
  HIP_CHECK(hipFree(d_output_head));
  HIP_CHECK(hipFree(d_emb));
  HIP_CHECK(hipFree(d_hid));
  HIP_CHECK(hipFree(d_logits));

  std::cout << "TestGpuMtpParity: PASS\n";
}

void TestGpuMtpVerify() {
  Require(rocm::MtpVerify(42, 42), "verify should accept match");
  Require(!rocm::MtpVerify(42, 43), "verify should reject mismatch");
  std::cout << "TestGpuMtpVerify: PASS\n";
}

void TestGpuMtpRollback() {
  const std::size_t n = 1024;
  std::vector<float> state(n), checkpoint(n);
  for (std::size_t i = 0; i < n; ++i) {
    state[i] = static_cast<float>(i);
    checkpoint[i] = static_cast<float>(i * 2);
  }
  rocm::MtpRollbackKdaState(state.data(), checkpoint.data(), n);
  for (std::size_t i = 0; i < n; ++i) {
    Require(state[i] == checkpoint[i], "rollback failed");
  }
  std::cout << "TestGpuMtpRollback: PASS\n";
}

}  // namespace

int main() {
  try {
    TestGpuMtpParity();
    TestGpuMtpVerify();
    TestGpuMtpRollback();
    std::cout << "All glm5_next MTP GPU tests passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
