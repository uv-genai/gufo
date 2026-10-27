#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "src/models/glm5_next/mhc_reference.hpp"

namespace glm5 = gufo::models::glm5_next;

namespace {

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

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

void TestSinkhornBasic() {
  auto geom = MakeGeom();
  // 4x4 raw matrix.
  std::vector<float> raw = {
      1.0F, 2.0F,  3.0F,  4.0F,  5.0F,  6.0F,  7.0F,  8.0F,
      9.0F, 10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F,
  };
  auto result = glm5::MhcSinkhorn(geom, raw);
  Require(result.size() == 16, "wrong size");

  // After Sinkhorn: each dst row sums to ~1 (dst-stochastic).
  for (std::uint32_t d = 0; d < 4; ++d) {
    float sum = 0;
    for (std::uint32_t s = 0; s < 4; ++s)
      sum += result[d * 4 + s];
    Require(std::abs(sum - 1.0F) < 0.01F, "dst row not stochastic");
  }
  std::cout << "TestSinkhornBasic: PASS\n";
}

void TestSinkhornDeterminism() {
  auto geom = MakeGeom();
  std::vector<float> raw(16);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-5.0F, 5.0F);
  for (auto& v : raw)
    v = dist(rng);

  auto r1 = glm5::MhcSinkhorn(geom, raw);
  auto r2 = glm5::MhcSinkhorn(geom, raw);

  for (std::size_t i = 0; i < 16; ++i) {
    Require(r1[i] == r2[i], "Sinkhorn not deterministic");
  }
  std::cout << "TestSinkhornDeterminism: PASS\n";
}

void TestPreBasic() {
  auto geom = MakeGeom();
  auto w = MakeWeights(geom, 42);
  auto R = MakeResidual(geom, 4, 43);

  auto out = glm5::MhcPre(geom, w, R, 4);
  Require(out.mixed.size() == geom.hidden_size * 4, "wrong mixed size");
  Require(out.post.size() == geom.hc_count * 4, "wrong post size");
  Require(out.comb.size() == geom.hc_count * geom.hc_count * 4,
          "wrong comb size");

  for (float v : out.mixed)
    Require(std::isfinite(v), "non-finite mixed");
  for (float v : out.post)
    Require(std::isfinite(v), "non-finite post");
  for (float v : out.comb)
    Require(std::isfinite(v), "non-finite comb");
  std::cout << "TestPreBasic: PASS\n";
}

void TestPreDeterminism() {
  auto geom = MakeGeom();
  auto w = MakeWeights(geom, 42);
  auto R = MakeResidual(geom, 8, 43);

  auto out1 = glm5::MhcPre(geom, w, R, 8);
  auto out2 = glm5::MhcPre(geom, w, R, 8);

  for (std::size_t i = 0; i < out1.mixed.size(); ++i)
    Require(out1.mixed[i] == out2.mixed[i], "mixed not deterministic");
  for (std::size_t i = 0; i < out1.post.size(); ++i)
    Require(out1.post[i] == out2.post[i], "post not deterministic");
  for (std::size_t i = 0; i < out1.comb.size(); ++i)
    Require(out1.comb[i] == out2.comb[i], "comb not deterministic");
  std::cout << "TestPreDeterminism: PASS\n";
}

void TestPostBasic() {
  auto geom = MakeGeom();
  auto w = MakeWeights(geom, 42);
  auto R = MakeResidual(geom, 2, 43);
  auto pre = glm5::MhcPre(geom, w, R, 2);

  std::vector<float> block_out(geom.hidden_size * 2);
  std::mt19937 rng(44);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  for (auto& v : block_out)
    v = dist(rng);

  auto out = glm5::MhcPost(geom, block_out, R, pre.post, pre.comb, 2);
  Require(out.size() == geom.hidden_size * geom.hc_count * 2,
          "wrong post output size");
  for (float v : out)
    Require(std::isfinite(v), "non-finite post output");
  std::cout << "TestPostBasic: PASS\n";
}

void TestPostDeterminism() {
  auto geom = MakeGeom();
  auto w = MakeWeights(geom, 42);
  auto R = MakeResidual(geom, 2, 43);
  auto pre = glm5::MhcPre(geom, w, R, 2);

  std::vector<float> block_out(geom.hidden_size * 2);
  std::mt19937 rng(44);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  for (auto& v : block_out)
    v = dist(rng);

  auto out1 = glm5::MhcPost(geom, block_out, R, pre.post, pre.comb, 2);
  auto out2 = glm5::MhcPost(geom, block_out, R, pre.post, pre.comb, 2);

  for (std::size_t i = 0; i < out1.size(); ++i)
    Require(out1[i] == out2[i], "post not deterministic");
  std::cout << "TestPostDeterminism: PASS\n";
}

void TestSinkhornIters() {
  // More iterations should converge to doubly-stochastic.
  auto geom = MakeGeom();
  geom.sinkhorn_iters = 20;
  std::vector<float> raw(16);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-5.0F, 5.0F);
  for (auto& v : raw)
    v = dist(rng);

  auto result = glm5::MhcSinkhorn(geom, raw);

  // Check both row and column sums are close to 1.
  for (std::uint32_t d = 0; d < 4; ++d) {
    float row_sum = 0;
    for (std::uint32_t s = 0; s < 4; ++s)
      row_sum += result[d * 4 + s];
    Require(std::abs(row_sum - 1.0F) < 0.01F, "row sum != 1");
  }
  for (std::uint32_t s = 0; s < 4; ++s) {
    float col_sum = 0;
    for (std::uint32_t d = 0; d < 4; ++d)
      col_sum += result[d * 4 + s];
    Require(std::abs(col_sum - 1.0F) < 0.05F, "col sum != 1");
  }
  std::cout << "TestSinkhornIters: PASS\n";
}

}  // namespace

int main() {
  try {
    TestSinkhornBasic();
    TestSinkhornDeterminism();
    TestPreBasic();
    TestPreDeterminism();
    TestPostBasic();
    TestPostDeterminism();
    TestSinkhornIters();
    std::cout << "All glm5_next mHC oracle tests passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
