#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "src/models/glm5_next/mla_reference.hpp"

namespace glm5 = gufo::models::glm5_next;

namespace {

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

glm5::MlaGeometry MakeGeom() {
  glm5::MlaGeometry geom;
  geom.hidden_size = 64;
  geom.num_heads = 8;
  geom.qk_nope = 16;
  geom.v_head_dim = 16;
  geom.q_lora_rank = 32;
  geom.kv_lora_rank = 24;
  geom.norm_eps = 1e-5F;
  geom.idx_n_head = 4;
  geom.idx_key_dim = 16;
  geom.idx_top_k = 16;
  geom.idx_kpool = 4;
  geom.idx_kpool_select_tail = true;
  return geom;
}

glm5::MlaWeightsF32 MakeWeights(const glm5::MlaGeometry& geom,
                                std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-0.1F, 0.1F);
  const std::uint32_t H = geom.hidden_size;
  const std::uint32_t nh = geom.num_heads;
  const std::uint32_t qd = geom.qk_nope;
  const std::uint32_t vd = geom.v_head_dim;
  const std::uint32_t qlora = geom.q_lora_rank;
  const std::uint32_t kv = geom.kv_lora_rank;
  const std::uint32_t qh = geom.idx_n_head;
  const std::uint32_t ikd = geom.idx_key_dim;

  glm5::MlaWeightsF32 w;
  w.q_a.resize(qlora * H);
  w.q_a_norm.resize(qlora);
  w.q_b.resize(nh * qd * qlora);
  w.kv_a_mqa.resize(kv * H);
  w.kv_a_norm.resize(kv);
  w.k_b.resize(static_cast<std::size_t>(nh) * qd * kv);
  w.v_b.resize(static_cast<std::size_t>(nh) * vd * kv);
  w.output.resize(H * nh * vd);

  for (auto& v : w.q_a)
    v = dist(rng);
  for (auto& v : w.q_a_norm)
    v = dist(rng) + 1.0F;
  for (auto& v : w.q_b)
    v = dist(rng);
  for (auto& v : w.kv_a_mqa)
    v = dist(rng);
  for (auto& v : w.kv_a_norm)
    v = dist(rng) + 1.0F;
  for (auto& v : w.k_b)
    v = dist(rng);
  for (auto& v : w.v_b)
    v = dist(rng);
  for (auto& v : w.output)
    v = dist(rng);

  w.idx_q_b.resize(qh * ikd * qlora);
  w.idx_k.resize(ikd * H);
  w.idx_k_norm_w.resize(ikd);
  w.idx_k_norm_b.resize(ikd);
  w.idx_proj.resize(qh * H);
  w.idx_ape.resize(ikd * geom.idx_kpool);
  w.idx_gate.resize(ikd * H);

  for (auto& v : w.idx_q_b)
    v = dist(rng);
  for (auto& v : w.idx_k)
    v = dist(rng);
  for (auto& v : w.idx_k_norm_w)
    v = dist(rng) + 1.0F;
  for (auto& v : w.idx_k_norm_b)
    v = dist(rng);
  for (auto& v : w.idx_proj)
    v = dist(rng);
  for (auto& v : w.idx_ape)
    v = dist(rng);
  for (auto& v : w.idx_gate)
    v = dist(rng);
  return w;
}

std::vector<float> MakeInput(const glm5::MlaGeometry& geom,
                             std::uint32_t seq_len, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  std::vector<float> cur(geom.hidden_size * seq_len);
  for (auto& v : cur)
    v = dist(rng);
  return cur;
}

void TestDsaSelectionBasic() {
  auto geom = MakeGeom();
  auto w = MakeWeights(geom, 42);
  auto cur = MakeInput(geom, 16, 43);

  // Query at t=15 (last token) — should see pools 0-3 (16/4=4 pools).
  auto sel = glm5::DsaSelectPools(geom, w, cur, 16, 15);
  Require(!sel.cells.empty(), "no cells selected");
  std::cout << "TestDsaSelectionBasic: PASS (" << sel.cells.size()
            << " cells)\n";
}

void TestDsaSelectionDeterminism() {
  auto geom = MakeGeom();
  auto w = MakeWeights(geom, 42);
  auto cur = MakeInput(geom, 16, 43);

  auto sel1 = glm5::DsaSelectPools(geom, w, cur, 16, 15);
  auto sel2 = glm5::DsaSelectPools(geom, w, cur, 16, 15);

  Require(sel1.cells.size() == sel2.cells.size(), "cell count differs");
  for (std::size_t i = 0; i < sel1.cells.size(); ++i) {
    Require(sel1.cells[i] == sel2.cells[i], "cells not deterministic");
  }
  std::cout << "TestDsaSelectionDeterminism: PASS\n";
}

void TestDsaEarlyQuery() {
  auto geom = MakeGeom();
  auto w = MakeWeights(geom, 42);
  auto cur = MakeInput(geom, 16, 43);

  // Query at t=2 (before first pool completes) — tail only.
  auto sel = glm5::DsaSelectPools(geom, w, cur, 16, 2);
  // With kpool_select_tail, should have tail cells.
  Require(!sel.cells.empty(), "no tail cells for early query");
  std::cout << "TestDsaEarlyQuery: PASS\n";
}

void TestAttentionTokenBasic() {
  auto geom = MakeGeom();
  auto w = MakeWeights(geom, 42);
  auto cur = MakeInput(geom, 8, 43);

  auto out = glm5::DsaAttentionToken(geom, w, cur, 8, 7);
  Require(out.size() == geom.num_heads * geom.v_head_dim, "wrong output size");
  for (float v : out)
    Require(std::isfinite(v), "non-finite output");
  std::cout << "TestAttentionTokenBasic: PASS\n";
}

void TestAttentionTokenDeterminism() {
  auto geom = MakeGeom();
  auto w = MakeWeights(geom, 42);
  auto cur = MakeInput(geom, 8, 43);

  auto out1 = glm5::DsaAttentionToken(geom, w, cur, 8, 7);
  auto out2 = glm5::DsaAttentionToken(geom, w, cur, 8, 7);

  for (std::size_t i = 0; i < out1.size(); ++i) {
    Require(out1[i] == out2[i], "attention not deterministic");
  }
  std::cout << "TestAttentionTokenDeterminism: PASS\n";
}

void TestDsaLayerBasic() {
  auto geom = MakeGeom();
  auto w = MakeWeights(geom, 42);
  auto cur = MakeInput(geom, 8, 43);

  auto out = glm5::DsaLayer(geom, w, cur, 8);
  Require(out.size() == geom.hidden_size * 8, "wrong output size");
  for (float v : out)
    Require(std::isfinite(v), "non-finite layer output");
  std::cout << "TestDsaLayerBasic: PASS\n";
}

void TestDsaLayerDeterminism() {
  auto geom = MakeGeom();
  auto w = MakeWeights(geom, 42);
  auto cur = MakeInput(geom, 8, 43);

  auto out1 = glm5::DsaLayer(geom, w, cur, 8);
  auto out2 = glm5::DsaLayer(geom, w, cur, 8);

  for (std::size_t i = 0; i < out1.size(); ++i) {
    Require(out1[i] == out2[i], "DSA layer not deterministic");
  }
  std::cout << "TestDsaLayerDeterminism: PASS\n";
}

void TestExpandedKv() {
  // Verify expanded-KV path: latent kv_lora → per-head K/V via k_b/v_b.
  auto geom = MakeGeom();
  auto w = MakeWeights(geom, 42);
  auto cur = MakeInput(geom, 4, 43);

  // For T=1, attention should still work (tail-only selection).
  auto out = glm5::DsaAttentionToken(geom, w, cur, 4, 3);
  Require(out.size() == geom.num_heads * geom.v_head_dim,
          "wrong expanded-KV output size");
  for (float v : out)
    Require(std::isfinite(v), "non-finite expanded-KV");
  std::cout << "TestExpandedKv: PASS\n";
}

}  // namespace

int main() {
  try {
    TestDsaSelectionBasic();
    TestDsaSelectionDeterminism();
    TestDsaEarlyQuery();
    TestAttentionTokenBasic();
    TestAttentionTokenDeterminism();
    TestDsaLayerBasic();
    TestDsaLayerDeterminism();
    TestExpandedKv();
    std::cout << "All glm5_next MLA/DSA oracle tests passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
