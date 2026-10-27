#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "src/models/glm5_next/mtp_reference.hpp"

namespace glm5 = gufo::models::glm5_next;

namespace {

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

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

void TestDraftBasic() {
  auto geom = MakeGeom();
  auto w = MakeWeights(geom, 42);
  auto emb = MakeVec(geom.hidden_size, 43);
  auto hid = MakeVec(geom.hidden_size, 44);

  auto logits = glm5::MtpDraft(geom, w, emb, hid);
  Require(logits.size() == geom.vocab_size, "wrong logits size");
  for (float v : logits)
    Require(std::isfinite(v), "non-finite logits");
  std::cout << "TestDraftBasic: PASS\n";
}

void TestDraftDeterminism() {
  auto geom = MakeGeom();
  auto w = MakeWeights(geom, 42);
  auto emb = MakeVec(geom.hidden_size, 43);
  auto hid = MakeVec(geom.hidden_size, 44);

  auto l1 = glm5::MtpDraft(geom, w, emb, hid);
  auto l2 = glm5::MtpDraft(geom, w, emb, hid);

  for (std::size_t i = 0; i < l1.size(); ++i) {
    Require(l1[i] == l2[i], "draft not deterministic");
  }
  std::cout << "TestDraftDeterminism: PASS\n";
}

void TestVerify() {
  Require(glm5::MtpVerify(42, 42), "verify should accept match");
  Require(!glm5::MtpVerify(42, 43), "verify should reject mismatch");
  std::cout << "TestVerify: PASS\n";
}

void TestRollback() {
  const std::size_t state_size = 1024;
  std::vector<float> state(state_size);
  std::vector<float> checkpoint(state_size);

  // Fill with initial values.
  for (std::size_t i = 0; i < state_size; ++i) {
    state[i] = static_cast<float>(i);
    checkpoint[i] = static_cast<float>(i * 2);
  }

  // Mutate state (simulating forward pass).
  for (auto& v : state)
    v += 100.0F;

  // Rollback to checkpoint.
  glm5::MtpRollbackKdaState(state, checkpoint);

  for (std::size_t i = 0; i < state_size; ++i) {
    Require(state[i] == checkpoint[i], "rollback failed");
  }
  std::cout << "TestRollback: PASS\n";
}

void TestFixedDraftWidth() {
  auto geom = MakeGeom();
  Require(geom.fixed_draft_width == 4, "wrong fixed draft width");
  std::cout << "TestFixedDraftWidth: PASS\n";
}

void TestEhProjShape() {
  auto geom = MakeGeom();
  auto w = MakeWeights(geom, 42);
  // eh_proj should be [H, 2*H].
  Require(w.eh_proj.size() == geom.hidden_size * 2 * geom.hidden_size,
          "wrong eh_proj shape");
  std::cout << "TestEhProjShape: PASS\n";
}

}  // namespace

int main() {
  try {
    TestDraftBasic();
    TestDraftDeterminism();
    TestVerify();
    TestRollback();
    TestFixedDraftWidth();
    TestEhProjShape();
    std::cout << "All glm5_next MTP oracle tests passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
