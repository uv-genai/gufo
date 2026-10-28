// End-to-end GPU engine test: GpuEngine vs CpuEngine on synthetic F32 weights,
// multi-token, with NO synchronizing between layer/token work — the production
// call pattern. Kernel-math tests (kda_gpu, moe_gpu, ...) call one op and
// sync; this is the test that catches integration bugs: workspace reuse
// across sequential launches, buffer lifetimes, stream ordering, cache state.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/glm5_next/cpu_engine.hpp"
#include "src/models/glm5_next/gpu_engine.hpp"

namespace glm5 = gufo::models::glm5_next;

namespace {

void Require(bool condition, const std::string& message) {
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

// Host arena for every synthetic tensor (kept alive for engine lifetimes).
struct Tensors {
  std::vector<std::vector<float>> store;
  glm5::TensorRef Add(std::uint64_t cols, std::uint64_t rows,
                      std::uint64_t experts, std::mt19937& rng,
                      const char* name) {
    store.emplace_back(cols * rows * experts);
    auto& v = store.back();
    std::uniform_real_distribution<float> dist(-0.02F, 0.02F);
    for (auto& x : v)
      x = dist(rng);
    glm5::TensorRef t;
    t.data = v.data();
    t.type = gufo::core::GgmlType::kF32;
    t.cols = cols;
    t.rows = rows;
    t.experts = experts;
    t.name = name;
    return t;
  }
};

glm5::Config MakeConfig() {
  glm5::Config c;
  c.architecture = "glm5-next";
  c.hidden_size = 64;
  c.vocab_size = 128;
  c.context_length = 256;
  c.layer_norm_rms_eps = 1e-5F;
  c.layer_norm_eps = 1e-6F;
  const char* bisect = std::getenv("E2E_BISECT");
  if (bisect && std::string(bisect) == "dense") {
    c.leading_dense_layers = 1;
    c.num_layers = 1;
  } else if (bisect && std::string(bisect) == "kda") {
    c.leading_dense_layers = 0;
    c.num_layers = 1;
    c.num_experts = 8;
    c.num_experts_used = 2;
  } else if (bisect && std::string(bisect) == "dsa") {
    c.leading_dense_layers = 0;
    c.num_layers = 1;
  } else {
    c.leading_dense_layers = 2;
    c.num_layers = 8;
  }
  c.hc_count = 4;
  c.hc_sinkhorn_iters = 20;
  c.hc_epsilon = 1e-6F;
  c.kda_head_dim = 8;
  c.kda_conv_kernel = 4;
  c.kda_gate_lower_bound = -5.0F;
  c.num_attention_heads = 4;
  c.q_lora_rank = 16;
  c.kv_lora_rank = 16;
  c.qk_nope_dim = 8;
  c.v_head_dim = 8;
  c.indexer_heads = 2;
  c.indexer_head_dim = 8;
  c.indexer_top_k = 8;
  c.indexer_kpool = 2;
  c.num_experts = 8;
  c.num_experts_used = 2;
  c.expert_ff = 16;
  c.num_shared_experts = 1;
  c.shared_expert_ff = 16;
  c.expert_gating_func = 2;
  c.expert_weights_scale = 2.5F;
  c.expert_weights_norm = true;
  c.dense_ff = 32;
  c.nextn_layers = 0;
  c.is_full_attention.assign(c.num_layers, 0);
  if (c.num_layers >= 8) {
    c.is_full_attention[6] = 1;
    c.is_full_attention[7] = 1;
  } else if (c.num_layers == 1 && std::getenv("E2E_BISECT") &&
             std::string(std::getenv("E2E_BISECT")) == "dsa") {
    c.is_full_attention[0] = 1;
  }
  return c;
}

glm5::ModelWeights MakeWeights(const glm5::Config& c, Tensors& t,
                               std::uint32_t seed) {
  std::mt19937 rng(seed);
  glm5::ModelWeights w;
  const std::uint64_t H = c.hidden_size;
  const std::uint64_t V = c.vocab_size;
  const std::uint64_t E = c.num_experts;
  const std::uint64_t I = c.expert_ff;
  const std::uint64_t SI = c.shared_expert_ff;
  const std::uint64_t D = c.dense_ff;
  const std::uint64_t nh = c.num_attention_heads;
  const std::uint64_t kd = c.kda_head_dim;
  const std::uint64_t inner = nh * kd;
  const std::uint64_t qlora = c.q_lora_rank;
  const std::uint64_t kv = c.kv_lora_rank;
  const std::uint64_t qd = c.qk_nope_dim;
  const std::uint64_t vd = c.v_head_dim;
  const std::uint64_t qh = c.indexer_heads;
  const std::uint64_t ikd = c.indexer_head_dim;
  const std::uint64_t kpool = c.indexer_kpool;

  w.token_embd = t.Add(H, V, 1, rng, "token_embd.weight");
  w.output = t.Add(H, V, 1, rng, "output.weight");
  w.output_norm = t.Add(1, H, 1, rng, "output_norm.weight");

  w.layers.resize(c.num_layers);
  for (std::uint32_t i = 0; i < c.num_layers; ++i) {
    auto& L = w.layers[i];
    L.full_attention = c.IsFullAttention(i);
    L.dense_ffn = c.IsDenseFfn(i);
    L.attn_norm = t.Add(1, H, 1, rng, "attn_norm.weight");
    L.ffn_norm = t.Add(1, H, 1, rng, "ffn_norm.weight");

    // mHC (mix_dim 24 = (2 + hc) * hc with hc = 4).
    L.hc.attn_fn = t.Add(H * c.hc_count, 24, 1, rng, "hc_attn_fn.weight");
    L.hc.attn_scale = t.Add(1, 3, 1, rng, "hc_attn_scale.weight");
    L.hc.attn_base = t.Add(1, 24, 1, rng, "hc_attn_base.weight");
    L.hc.ffn_fn = t.Add(H * c.hc_count, 24, 1, rng, "hc_ffn_fn.weight");
    L.hc.ffn_scale = t.Add(1, 3, 1, rng, "hc_ffn_scale.weight");
    L.hc.ffn_base = t.Add(1, 24, 1, rng, "hc_ffn_base.weight");

    // KDA mixer (bound even on DSA layers; the engine skips them there).
    L.kda.attn_q = t.Add(H, inner, 1, rng, "attn_q.weight");
    L.kda.attn_k = t.Add(H, inner, 1, rng, "attn_k.weight");
    L.kda.attn_v = t.Add(H, inner, 1, rng, "attn_v.weight");
    L.kda.attn_out = t.Add(inner, H, 1, rng, "attn_output.weight");
    L.kda.conv1d_q =
        t.Add(1, c.kda_conv_kernel, inner, rng, "ssm_conv1d_q.weight");
    L.kda.conv1d_k =
        t.Add(1, c.kda_conv_kernel, inner, rng, "ssm_conv1d_k.weight");
    L.kda.conv1d_v =
        t.Add(1, c.kda_conv_kernel, inner, rng, "ssm_conv1d_v.weight");
    L.kda.a = t.Add(1, nh, 1, rng, "ssm_a");
    L.kda.dt_bias = t.Add(1, inner, 1, rng, "ssm_dt.bias");
    L.kda.f_a = t.Add(H, kd, 1, rng, "ssm_f_a.weight");
    L.kda.f_b = t.Add(kd, inner, 1, rng, "ssm_f_b.weight");
    L.kda.beta = t.Add(H, nh, 1, rng, "ssm_beta.weight");
    L.kda.g_a = t.Add(H, kd, 1, rng, "ssm_g_a.weight");
    L.kda.g_b = t.Add(kd, inner, 1, rng, "ssm_g_b.weight");
    L.kda.norm = t.Add(1, kd, 1, rng, "ssm_norm.weight");

    // MLA + DSA indexer.
    L.mla.q_a = t.Add(H, qlora, 1, rng, "attn_q_a.weight");
    L.mla.q_a_norm = t.Add(1, qlora, 1, rng, "attn_q_a_norm.weight");
    L.mla.q_b = t.Add(qlora, nh * qd, 1, rng, "attn_q_b.weight");
    L.mla.kv_a_mqa = t.Add(H, kv, 1, rng, "attn_kv_a_mqa.weight");
    L.mla.kv_a_norm = t.Add(1, kv, 1, rng, "attn_kv_a_norm.weight");
    L.mla.k_b = t.Add(qd, kv, nh, rng, "attn_k_b.weight");
    L.mla.v_b = t.Add(kv, vd, nh, rng, "attn_v_b.weight");
    L.mla.output = t.Add(nh * vd, H, 1, rng, "attn_output.weight");
    L.indexer.q_b = t.Add(qlora, qh * ikd, 1, rng, "indexer.attn_q_b.weight");
    L.indexer.k = t.Add(H, ikd, 1, rng, "indexer.attn_k.weight");
    L.indexer.k_norm_weight = t.Add(1, ikd, 1, rng, "indexer.k_norm.weight");
    L.indexer.k_norm_bias = t.Add(1, ikd, 1, rng, "indexer.k_norm.bias");
    L.indexer.proj = t.Add(H, qh, 1, rng, "indexer.proj.weight");
    L.indexer.compressor_ape =
        t.Add(kpool, ikd, 1, rng, "indexer_compressor_ape.weight");
    L.indexer.compressor_gate =
        t.Add(H, ikd, 1, rng, "indexer_compressor_gate.weight");

    // MoE.
    L.moe.gate_inp = t.Add(H, E, 1, rng, "ffn_gate_inp.weight");
    L.moe.exp_probs_b = t.Add(1, E, 1, rng, "exp_probs_b.bias");
    L.moe.gate_exps = t.Add(H, I, E, rng, "ffn_gate_exps.weight");
    L.moe.up_exps = t.Add(H, I, E, rng, "ffn_up_exps.weight");
    L.moe.down_exps = t.Add(I, H, E, rng, "ffn_down_exps.weight");
    L.moe.gate_shexp = t.Add(H, SI, 1, rng, "ffn_gate_shexp.weight");
    L.moe.up_shexp = t.Add(H, SI, 1, rng, "ffn_up_shexp.weight");
    L.moe.down_shexp = t.Add(SI, H, 1, rng, "ffn_down_shexp.weight");

    // Dense FFN.
    L.dense.gate = t.Add(H, D, 1, rng, "ffn_gate.weight");
    L.dense.up = t.Add(H, D, 1, rng, "ffn_up.weight");
    L.dense.down = t.Add(D, H, 1, rng, "ffn_down.weight");
  }
  return w;
}

void CheckClose(const std::vector<float>& a, const std::vector<float>& b,
                const std::string& what) {
  Require(a.size() == b.size(), what + ": size mismatch");
  double max_err = 0.0;
  std::size_t worst = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double e = std::abs(static_cast<double>(a[i]) - b[i]);
    if (e > max_err) {
      max_err = e;
      worst = i;
    }
  }
  if (max_err >= 2e-3) {
    std::string dump = what + ": max_err=" + std::to_string(max_err) + "\n";
    for (std::size_t i = 0; i < a.size(); ++i) {
      const double e = std::abs(static_cast<double>(a[i]) - b[i]);
      if (e >= 2e-3) {
        dump += "  [" + std::to_string(i) + "] gpu=" + std::to_string(a[i]) +
                " cpu=" + std::to_string(b[i]) + "\n";
      }
    }
    throw std::runtime_error(dump);
  }
  std::cout << what << ": PASS (max_err " << max_err << ")\n";
}

}  // namespace

int main() {
  try {
    const glm5::Config cfg = MakeConfig();
    Tensors tensors;
    glm5::ModelWeights weights = MakeWeights(cfg, tensors, 7);

    std::string err;
    auto cpu = glm5::CpuEngine::Create(cfg, weights, &err);
    Require(cpu != nullptr, "CpuEngine::Create: " + err);
    auto gpu = glm5::GpuEngine::Create(cfg, weights, &err);
    Require(gpu != nullptr, "GpuEngine::Create: " + err);

    const std::vector<std::uint32_t> prompt =
        (std::getenv("E2E_ONE") ? std::vector<std::uint32_t>{3}
                                : std::vector<std::uint32_t>{3, 11, 5, 7});

    // 1. Prefill: hidden after the prompt (production order: no syncs inside).
    const auto hidden_cpu = cpu->Prefill(prompt);
    const auto hidden_gpu = gpu->Prefill(prompt);
    CheckClose(hidden_gpu, hidden_cpu, "prefill hidden");

    // 2. Decode the same hidden -> same token.
    const auto tok_cpu = cpu->Decode(hidden_cpu);
    const auto tok_gpu = gpu->Decode(hidden_gpu);
    Require(tok_cpu == tok_gpu,
            "first token differs: gpu=" + std::to_string(tok_gpu) +
                " cpu=" + std::to_string(tok_cpu));
    std::cout << "first token: PASS (" << tok_gpu << ")\n";

    // 3. Multi-token continuation: GPU Step() must track CPU Prefill() on the
    //    growing prefix — this exercises the KDA/DSA caches and every shared
    //    scratch across many sequential async layer launches.
    std::vector<std::uint32_t> seq = prompt;
    std::uint32_t tok = tok_gpu;
    for (int step = 0; step < 6; ++step) {
      const auto h_gpu = gpu->Step(tok);
      seq.push_back(tok);
      const auto h_cpu = cpu->Prefill(seq);
      CheckClose(h_gpu, h_cpu, "step " + std::to_string(step) + " hidden");
      const auto t_cpu = cpu->Decode(h_cpu);
      const auto t_gpu = gpu->Decode(h_gpu);
      Require(t_cpu == t_gpu,
              "step " + std::to_string(step) + " token differs: gpu=" +
                  std::to_string(t_gpu) + " cpu=" + std::to_string(t_cpu));
      tok = t_gpu;
    }
    std::cout << "ALL PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
