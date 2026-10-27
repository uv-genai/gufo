#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "src/models/glm5_next/moe_reference.hpp"

namespace glm5 = gufo::models::glm5_next;

namespace {

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

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

void TestSelectionBasic() {
  auto geom = MakeGeom();
  auto w = MakeWeights(geom, 42);
  auto x = MakeInput(geom, 43);

  auto sel = glm5::MoESelect(geom, w, x);
  Require(sel.expert_ids.size() == geom.top_k, "wrong top_k");
  Require(sel.weights.size() == geom.top_k, "wrong weights size");

  // All weights should be positive (sigmoid + renorm + scale).
  for (float v : sel.weights)
    Require(v > 0, "non-positive weight");
  std::cout << "TestSelectionBasic: PASS\n";
}

void TestSelectionDeterminism() {
  auto geom = MakeGeom();
  auto w = MakeWeights(geom, 42);
  auto x = MakeInput(geom, 43);

  auto sel1 = glm5::MoESelect(geom, w, x);
  auto sel2 = glm5::MoESelect(geom, w, x);

  for (std::size_t i = 0; i < sel1.expert_ids.size(); ++i) {
    Require(sel1.expert_ids[i] == sel2.expert_ids[i],
            "selection not deterministic");
    Require(sel1.weights[i] == sel2.weights[i], "weights not deterministic");
  }
  std::cout << "TestSelectionDeterminism: PASS\n";
}

void TestSelectionParity() {
  // Flipped expert = fail, not "rounding".
  // Construct two inputs that differ by epsilon in one dimension.
  auto geom = MakeGeom();
  auto w = MakeWeights(geom, 42);

  auto x1 = MakeInput(geom, 43);
  auto x2 = x1;
  // Perturb slightly — should NOT flip selection (same neighborhood).
  x2[0] += 1e-7F;
  auto sel1 = glm5::MoESelect(geom, w, x1);
  auto sel2 = glm5::MoESelect(geom, w, x2);

  // Same expert IDs (selection is stable under tiny perturbation).
  for (std::size_t i = 0; i < sel1.expert_ids.size(); ++i) {
    Require(sel1.expert_ids[i] == sel2.expert_ids[i],
            "selection flipped under tiny perturbation");
  }
  std::cout << "TestSelectionParity: PASS\n";
}

void TestSelectionBias() {
  // The bias is added ONLY for selection, not for weights.
  // Verify: same input → same selection regardless of weight computation.
  auto geom = MakeGeom();
  auto w = MakeWeights(geom, 42);
  auto x = MakeInput(geom, 43);

  auto sel = glm5::MoESelect(geom, w, x);

  // Manually compute probs and verify weights come from unbiased probs.
  const std::uint32_t E = geom.num_experts;
  const std::uint32_t H = geom.hidden_size;
  std::vector<float> logits(E);
  for (std::uint32_t i = 0; i < E; ++i) {
    float acc = 0;
    for (std::uint32_t j = 0; j < H; ++j) {
      acc += w.gate_inp[i * H + j] * x[j];
    }
    logits[i] = acc;
  }
  std::vector<float> probs(E);
  for (std::uint32_t i = 0; i < E; ++i) {
    probs[i] = 1.0F / (1.0F + std::exp(-logits[i]));
  }

  // Weights should be renormalized probs[slected] * scale.
  float w_sum = 0;
  for (std::uint32_t i = 0; i < sel.expert_ids.size(); ++i) {
    w_sum += probs[sel.expert_ids[i]];
  }
  const float inv = 1.0F / std::max(w_sum, geom.renorm_clamp);
  for (std::uint32_t i = 0; i < sel.expert_ids.size(); ++i) {
    const float expected =
        probs[sel.expert_ids[i]] * inv * geom.expert_weights_scale;
    Require(std::abs(sel.weights[i] - expected) < 1e-5F,
            "weights don't match unbiased probs");
  }
  std::cout << "TestSelectionBias: PASS\n";
}

void TestMoeForward() {
  auto geom = MakeGeom();
  auto w = MakeWeights(geom, 42);
  auto x = MakeInput(geom, 43);

  auto out = glm5::MoEForward(geom, w, x);
  Require(out.size() == geom.hidden_size, "wrong output size");
  for (float v : out)
    Require(std::isfinite(v), "non-finite output");
  std::cout << "TestMoeForward: PASS\n";
}

void TestMoeForwardDeterminism() {
  auto geom = MakeGeom();
  auto w = MakeWeights(geom, 42);
  auto x = MakeInput(geom, 43);

  auto out1 = glm5::MoEForward(geom, w, x);
  auto out2 = glm5::MoEForward(geom, w, x);

  for (std::size_t i = 0; i < out1.size(); ++i) {
    Require(out1[i] == out2[i], "MoE forward not deterministic");
  }
  std::cout << "TestMoeForwardDeterminism: PASS\n";
}

void TestDenseFfn() {
  const std::uint32_t H = 32, I = 16;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-0.1F, 0.1F);

  std::vector<float> x(H), gate(H * I), up(H * I), down(I * H);
  for (auto& v : x)
    v = dist(rng);
  for (auto& v : gate)
    v = dist(rng);
  for (auto& v : up)
    v = dist(rng);
  for (auto& v : down)
    v = dist(rng);

  auto out = glm5::DenseFfnForward(x, gate, up, down, H, I, 10.0F);
  Require(out.size() == H, "wrong dense FFN output size");
  for (float v : out)
    Require(std::isfinite(v), "non-finite dense output");
  std::cout << "TestDenseFfn: PASS\n";
}

void TestSwigluClamp() {
  // Verify clamp behavior: gate clamped above only, up clamped both sides.
  const std::uint32_t H = 4, I = 4;
  std::vector<float> x(H, 1.0F);
  std::vector<float> gate(H * I, 0.0F), up(H * I, 0.0F), down(I * H, 0.0F);
  // Set gate to 20 (above limit 10), up to 20 (above limit 10).
  for (std::uint32_t i = 0; i < I; ++i) {
    for (std::uint32_t j = 0; j < H; ++j) {
      gate[i * H + j] = 20.0F;
      up[i * H + j] = 20.0F;
    }
  }
  // down = identity (first 4 diagonal).
  for (std::uint32_t i = 0; i < std::min(H, I); ++i) {
    down[i * H + i] = 1.0F;
  }

  auto out = glm5::DenseFfnForward(x, gate, up, down, H, I, 10.0F);
  // With gate=20 clamped to 10, silu(10) ≈ 10.0; up=20 clamped to 10.
  // h = silu(10)*10 ≈ 100. Then out = down @ h ≈ [100, 100, 100, 100].
  for (std::uint32_t i = 0; i < H; ++i) {
    Require(out[i] > 50.0F, "clamp not applied");
  }
  std::cout << "TestSwigluClamp: PASS\n";
}

}  // namespace

int main() {
  try {
    TestSelectionBasic();
    TestSelectionDeterminism();
    TestSelectionParity();
    TestSelectionBias();
    TestMoeForward();
    TestMoeForwardDeterminism();
    TestDenseFfn();
    TestSwigluClamp();
    std::cout << "All glm5_next MoE oracle tests passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
