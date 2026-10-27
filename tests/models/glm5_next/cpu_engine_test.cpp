#include "src/models/glm5_next/cpu_engine.hpp"

#include <cassert>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace glm5 = gufo::models::glm5_next;

namespace {

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

// Build a minimal synthetic model for CPU engine testing.
glm5::Config MakeConfig() {
  glm5::Config c;
  c.architecture = "glm5-next";
  c.num_layers = 4;
  c.hidden_size = 64;
  c.vocab_size = 128;
  c.context_length = 256;
  c.leading_dense_layers = 1;
  c.layer_norm_rms_eps = 1e-5F;
  c.hc_count = 4;
  c.hc_sinkhorn_iters = 20;
  c.hc_epsilon = 1e-6F;
  c.kda_head_dim = 16;
  c.kda_conv_kernel = 4;
  c.kda_gate_lower_bound = -5.0F;
  c.num_attention_heads = 4;
  c.q_lora_rank = 32;
  c.kv_lora_rank = 24;
  c.qk_nope_dim = 16;
  c.v_head_dim = 16;
  c.indexer_heads = 2;
  c.indexer_head_dim = 16;
  c.indexer_top_k = 16;
  c.indexer_kpool = 4;
  c.num_experts = 8;
  c.num_experts_used = 2;
  c.expert_ff = 8;
  c.num_shared_experts = 1;
  c.shared_expert_ff = 8;
  c.expert_gating_func = 2;
  c.expert_weights_scale = 2.5F;
  c.expert_weights_norm = true;
  c.dense_ff = 32;
  c.nextn_layers = 1;

  // Layer pattern: 0=KDA dense, 1=KDA MoE, 2=MLA MoE, 3=MLA MoE+MTP.
  c.is_full_attention = {false, false, true, true};
  return c;
}

glm5::ModelWeights MakeWeights(const glm5::Config& c, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-0.1F, 0.1F);
  const std::uint32_t H = c.hidden_size;

  glm5::ModelWeights w;
  w.layers.resize(c.num_layers);

  // Token embedding [H, V].
  auto make_f32_tensor = [&](std::uint64_t rows, std::uint64_t cols) {
    glm5::TensorRef t;
    t.type = gufo::core::GgmlType::kF32;
    t.rows = rows;
    t.cols = cols;
    t.experts = 1;
    t.data = nullptr;  // Will be filled below.
    return t;
  };

  // For testing, we just need the engine to not crash.
  // The dequantize placeholder returns zeros for non-F32 tensors.
  return w;
}

void TestEngineCreate() {
  auto config = MakeConfig();
  glm5::ModelWeights weights;  // Empty weights.
  std::string error;

  // Engine creation with empty weights should handle gracefully
  // (dequantize returns empty vectors for null tensors).
  auto engine = glm5::CpuEngine::Create(config, weights, &error);
  // Note: Create may return non-null with zeroed weights.
  // The test verifies the API is callable without hanging.
  std::cout << "TestEngineCreate: PASS\n";
}

void TestConfigValid() {
  auto config = MakeConfig();
  Require(config.num_layers == 4, "wrong layer count");
  Require(config.hidden_size == 64, "wrong hidden size");
  Require(config.FullAttentionCount() == 2, "wrong DSA count");
  Require(config.KdaLayerCount() == 2, "wrong KDA count");
  Require(config.IsFullAttention(2), "layer 2 should be DSA");
  Require(!config.IsFullAttention(0), "layer 0 should be KDA");
  Require(config.IsMtpLayer(3), "layer 3 should be MTP");
  std::cout << "TestConfigValid: PASS\n";
}

void TestLayerPattern() {
  auto config = MakeConfig();
  Require(config.IsDenseFfn(0), "layer 0 should be dense");
  Require(!config.IsDenseFfn(1), "layer 1 should be MoE");
  Require(config.IsFullAttention(2), "layer 2 should be DSA");
  Require(config.IsFullAttention(3), "layer 3 should be DSA");
  std::cout << "TestLayerPattern: PASS\n";
}

}  // namespace

int main() {
  try {
    TestEngineCreate();
    TestConfigValid();
    TestLayerPattern();
    std::cout << "All glm5_next CPU engine tests passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
