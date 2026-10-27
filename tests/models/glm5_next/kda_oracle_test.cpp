#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "src/models/glm5_next/reference.hpp"

namespace glm5 = gufo::models::glm5_next;

namespace {

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

// Deterministic PRNG for reproducible test weights.
std::mt19937 MakeRng(std::uint32_t seed) {
  return std::mt19937(seed);
}

float RandUniform(std::mt19937& rng, float lo, float hi) {
  return std::uniform_real_distribution<float>(lo, hi)(rng);
}

// Build a small synthetic KDA model for testing.
glm5::KdaWeightsF32 MakeSyntheticWeights(const glm5::KdaGeometry& geom,
                                         std::uint32_t seed) {
  const std::uint32_t H = geom.hidden_size;
  const std::uint32_t nh = geom.num_heads;
  const std::uint32_t dk = geom.head_dim;
  const std::uint32_t inner = nh * dk;
  const std::uint32_t K = geom.conv_kernel;

  auto rng = MakeRng(seed);
  glm5::KdaWeightsF32 w;
  w.w_q.resize(H * inner);
  w.w_k.resize(H * inner);
  w.w_v.resize(H * inner);
  w.w_out.resize(H * inner);
  for (auto& v : w.w_q)
    v = RandUniform(rng, -0.1F, 0.1F);
  for (auto& v : w.w_k)
    v = RandUniform(rng, -0.1F, 0.1F);
  for (auto& v : w.w_v)
    v = RandUniform(rng, -0.1F, 0.1F);
  for (auto& v : w.w_out)
    v = RandUniform(rng, -0.1F, 0.1F);

  w.conv_q.resize(K * inner);
  w.conv_k.resize(K * inner);
  w.conv_v.resize(K * inner);
  for (auto& v : w.conv_q)
    v = RandUniform(rng, -0.5F, 0.5F);
  for (auto& v : w.conv_k)
    v = RandUniform(rng, -0.5F, 0.5F);
  for (auto& v : w.conv_v)
    v = RandUniform(rng, -0.5F, 0.5F);

  w.f_a.resize(H * dk);
  w.f_b.resize(dk * inner);
  w.dt_bias.resize(inner);
  for (auto& v : w.f_a)
    v = RandUniform(rng, -0.1F, 0.1F);
  for (auto& v : w.f_b)
    v = RandUniform(rng, -0.1F, 0.1F);
  for (auto& v : w.dt_bias)
    v = RandUniform(rng, -1.0F, 1.0F);

  w.ssm_a.resize(nh);
  // ssm_a holds -exp(A_log); use values in (-10, -1) range.
  for (auto& v : w.ssm_a)
    v = -std::exp(RandUniform(rng, 0.0F, 2.3F));

  w.ssm_beta.resize(H * nh);
  for (auto& v : w.ssm_beta)
    v = RandUniform(rng, -0.1F, 0.1F);

  w.g_a.resize(H * dk);
  w.g_b.resize(dk * inner);
  for (auto& v : w.g_a)
    v = RandUniform(rng, -0.1F, 0.1F);
  for (auto& v : w.g_b)
    v = RandUniform(rng, -0.1F, 0.1F);

  w.ssm_norm.resize(dk);
  for (auto& v : w.ssm_norm)
    v = RandUniform(rng, 0.5F, 1.5F);

  return w;
}

// Small geometry for fast tests.
glm5::KdaGeometry MakeSmallGeom() {
  glm5::KdaGeometry geom;
  geom.hidden_size = 64;
  geom.num_heads = 4;
  geom.head_dim = 16;
  geom.conv_kernel = 4;
  geom.gate_lower_bound = -5.0F;
  geom.norm_eps = 1e-5F;
  return geom;
}

std::vector<float> MakeInput(const glm5::KdaGeometry& geom,
                             std::uint32_t seq_len, std::uint32_t seed) {
  auto rng = MakeRng(seed);
  std::vector<float> input(geom.hidden_size * seq_len);
  for (auto& v : input)
    v = RandUniform(rng, -1.0F, 1.0F);
  return input;
}

void TestPrefillBasic() {
  auto geom = MakeSmallGeom();
  auto w = MakeSyntheticWeights(geom, 42);
  auto input = MakeInput(geom, 8, 43);
  auto output = glm5::KdaPrefill(geom, w, input, 8);
  Require(output.size() == geom.hidden_size * 8, "wrong output size");
  // All outputs should be finite.
  for (float v : output) {
    Require(std::isfinite(v), "non-finite output");
  }
  std::cout << "TestPrefillBasic: PASS\n";
}

void TestDeterminism() {
  auto geom = MakeSmallGeom();
  auto w = MakeSyntheticWeights(geom, 42);
  auto input = MakeInput(geom, 16, 43);

  auto out1 = glm5::KdaPrefill(geom, w, input, 16);
  auto out2 = glm5::KdaPrefill(geom, w, input, 16);

  const float max_diff = glm5::MaxAbsDiff(out1, out2);
  Require(max_diff == 0.0F, "prefill not deterministic");
  std::cout << "TestDeterminism: PASS (max_diff=" << max_diff << ")\n";
}

void TestDecodeMatchesPrefill() {
  // Decode one token at a time should match the last token of prefill
  // when state is properly accumulated. For a fresh state with one token,
  // decode output should match prefill T=1.
  auto geom = MakeSmallGeom();
  auto w = MakeSyntheticWeights(geom, 42);
  auto input = MakeInput(geom, 1, 43);

  auto prefill_out = glm5::KdaPrefill(geom, w, input, 1);

  std::vector<float> state(geom.num_heads * geom.head_dim * geom.head_dim,
                           0.0F);
  auto decode_out = glm5::KdaDecode(geom, w, input, state);

  Require(decode_out.size() == geom.hidden_size, "wrong decode output size");
  // Prefill and decode should be very close (same math, T=1).
  const float max_diff = glm5::MaxAbsDiff(prefill_out, decode_out);
  Require(max_diff < 1e-4F, "decode != prefill for T=1");
  std::cout << "TestDecodeMatchesPrefill: PASS (max_diff=" << max_diff << ")\n";
}

void TestStateDeterminism() {
  // Same tokens -> same state after decode, independent of call order
  // if we rebuild state from scratch each time.
  auto geom = MakeSmallGeom();
  auto w = MakeSyntheticWeights(geom, 42);
  auto input = MakeInput(geom, 4, 43);

  // Decode 4 tokens one at a time.
  std::vector<float> state1(geom.num_heads * geom.head_dim * geom.head_dim,
                            0.0F);
  std::vector<float> out1;
  for (std::uint32_t t = 0; t < 4; ++t) {
    auto o = glm5::KdaDecode(
        geom, w,
        std::span<const float>(input.data() + t * geom.hidden_size,
                               geom.hidden_size),
        state1);
    out1.insert(out1.end(), o.begin(), o.end());
  }

  // Repeat.
  std::vector<float> state2(geom.num_heads * geom.head_dim * geom.head_dim,
                            0.0F);
  std::vector<float> out2;
  for (std::uint32_t t = 0; t < 4; ++t) {
    auto o = glm5::KdaDecode(
        geom, w,
        std::span<const float>(input.data() + t * geom.hidden_size,
                               geom.hidden_size),
        state2);
    out2.insert(out2.end(), o.begin(), o.end());
  }

  const float out_diff = glm5::MaxAbsDiff(out1, out2);
  const float state_diff = glm5::MaxAbsDiff(state1, state2);
  Require(out_diff == 0.0F, "decode outputs not deterministic");
  Require(state_diff == 0.0F, "state not deterministic");
  std::cout << "TestStateDeterminism: PASS\n";
}

void TestKLDivergence() {
  // KL(p || p) = 0.
  std::vector<float> p = {0.25F, 0.25F, 0.25F, 0.25F};
  float kl = glm5::KLDivergence(p, p);
  Require(std::abs(kl) < 1e-6F, "KL(p||p) != 0");

  // KL(p || q) > 0 for p != q.
  std::vector<float> q = {0.1F, 0.2F, 0.3F, 0.4F};
  kl = glm5::KLDivergence(p, q);
  Require(kl > 0, "KL(p||q) should be > 0");
  std::cout << "TestKLDivergence: PASS\n";
}

void TestMaxAbsDiff() {
  std::vector<float> a = {1.0F, 2.0F, 3.0F};
  std::vector<float> b = {1.0F, 2.5F, 2.0F};
  Require(glm5::MaxAbsDiff(a, b) == 1.0F, "wrong max abs diff");
  std::cout << "TestMaxAbsDiff: PASS\n";
}

void TestGateLowerBound() {
  // With deep-positive gate args, g1 should saturate at gate_lower_bound.
  auto geom = MakeSmallGeom();
  geom.gate_lower_bound = -5.0F;
  auto w = MakeSyntheticWeights(geom, 42);
  // Make dt_bias very large positive so gate arg is deeply positive.
  for (auto& v : w.dt_bias)
    v = 100.0F;
  auto input = MakeInput(geom, 4, 43);
  auto output = glm5::KdaPrefill(geom, w, input, 4);
  for (float v : output) {
    Require(std::isfinite(v), "non-finite output with saturated gate");
  }
  std::cout << "TestGateLowerBound: PASS\n";
}

}  // namespace

int main() {
  try {
    TestPrefillBasic();
    TestDeterminism();
    TestDecodeMatchesPrefill();
    TestStateDeterminism();
    TestKLDivergence();
    TestMaxAbsDiff();
    TestGateLowerBound();
    std::cout << "All glm5_next KDA oracle tests passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
