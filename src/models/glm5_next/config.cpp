#include "src/models/glm5_next/config.hpp"

#include <algorithm>
#include <cmath>
#include <string_view>
#include <utility>
#include <variant>

namespace gufo::models::glm5_next {
namespace {

constexpr std::string_view kArchitecture = "glm5-next";

struct Reader {
  const core::GgufReader& gguf;
  std::string* error;
  bool ok{true};

  void Fail(std::string message) {
    if (ok && error != nullptr)
      *error = std::move(message);
    ok = false;
  }

  std::uint32_t U32(std::string_view key, bool required = true,
                    std::uint32_t fallback = 0) {
    const auto value = gguf.GetMetadataUint32(key);
    if (value.has_value())
      return *value;
    if (required || gguf.FindMetadata(key) != nullptr)
      Fail("missing or invalid GGUF integer " + std::string(key));
    return fallback;
  }

  std::uint64_t U64(std::string_view key, bool required = true,
                    std::uint64_t fallback = 0) {
    const auto value = gguf.GetMetadataUint64(key);
    if (value.has_value())
      return *value;
    if (required || gguf.FindMetadata(key) != nullptr)
      Fail("missing or invalid GGUF integer " + std::string(key));
    return fallback;
  }

  float F32(std::string_view key, float fallback) {
    const auto value = gguf.GetMetadataFloat32(key);
    if (!value || !std::isfinite(*value))
      Fail("missing or invalid GGUF float " + std::string(key));
    return value.value_or(fallback);
  }

  bool Bool(std::string_view key, bool fallback) {
    const auto value = gguf.GetMetadataBool(key);
    if (!value.has_value()) {
      Fail("missing or invalid GGUF bool " + std::string(key));
      return fallback;
    }
    return *value;
  }

  /// Integer arrays of any width; GGUF writers pick the narrowest type.
  std::vector<std::uint64_t> U64Array(std::string_view key,
                                      bool required = true) {
    const auto* meta = gguf.FindMetadata(key);
    std::vector<std::uint64_t> out;
    if (meta == nullptr) {
      if (required)
        Fail("missing GGUF array " + std::string(key));
      return out;
    }
    if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
      out = *u;
    } else if (const auto* s =
                   std::get_if<std::vector<std::int64_t>>(&meta->value)) {
      for (auto v : *s) {
        if (v < 0) {
          Fail("negative GGUF array value " + std::string(key));
          return {};
        }
        out.push_back(static_cast<std::uint64_t>(v));
      }
    } else {
      Fail("GGUF array must contain integers: " + std::string(key));
    }
    return out;
  }

  std::vector<float> F32Array(std::string_view key, bool required = true) {
    const auto* meta = gguf.FindMetadata(key);
    std::vector<float> out;
    if (meta == nullptr) {
      if (required)
        Fail("missing GGUF array " + std::string(key));
      return out;
    }
    if (const auto* f = std::get_if<std::vector<double>>(&meta->value)) {
      for (auto v : *f) {
        if (!std::isfinite(v)) {
          Fail("non-finite GGUF array value " + std::string(key));
          return {};
        }
        out.push_back(static_cast<float>(v));
      }
    } else {
      Fail("GGUF array must contain floats: " + std::string(key));
    }
    return out;
  }
};

}  // namespace

std::optional<Config> Config::FromGguf(const core::GgufReader& gguf,
                                       std::string* error_msg) {
  Reader r{gguf, error_msg};
  const auto arch = gguf.GetMetadataString("general.architecture");
  if (arch != kArchitecture) {
    r.Fail("GGUF architecture is " + std::string(arch.value_or("<none>")) +
           ", expected glm5-next");
    return std::nullopt;
  }

  Config c;
  c.architecture = std::string(kArchitecture);
  const std::string_view p = "glm5-next.";
  auto key = [&](std::string_view suffix) {
    return std::string(p) + std::string(suffix);
  };

  // Core geometry.
  c.num_layers = r.U32(key("block_count"));
  c.hidden_size = r.U32(key("embedding_length"));
  c.vocab_size = r.U32(key("vocab_size"));
  c.context_length =
      static_cast<std::uint32_t>(r.U64(key("context_length"), true, 0));
  c.leading_dense_layers = r.U32(key("leading_dense_block_count"), false, 0);
  c.layer_norm_rms_eps = r.F32(key("attention.layer_norm_rms_epsilon"), 1e-5F);
  c.layer_norm_eps = r.F32(key("attention.layer_norm_epsilon"), 1e-6F);

  // mHC.
  c.hc_count = r.U32(key("hyper_connection.count"));
  c.hc_sinkhorn_iters = r.U32(key("hyper_connection.sinkhorn_iterations"));
  c.hc_epsilon = r.F32(key("hyper_connection.epsilon"), 1e-6F);

  // KDA.
  c.kda_head_dim = r.U32(key("kda.head_dim"));
  c.kda_conv_kernel = r.U32(key("ssm.conv_kernel"));
  c.kda_gate_lower_bound = r.F32(key("kda.gate_lower_bound"), -5.0F);

  // Full attention / MLA / DSA.
  c.num_attention_heads = r.U32(key("attention.head_count"));
  c.q_lora_rank = r.U32(key("attention.q_lora_rank"));
  c.kv_lora_rank = r.U32(key("attention.kv_lora_rank"));
  c.qk_nope_dim = r.U32(key("attention.key_length_mla"));
  c.v_head_dim = r.U32(key("attention.value_length_mla"));
  c.indexer_heads = r.U32(key("attention.indexer.head_count"));
  c.indexer_head_dim = r.U32(key("attention.indexer.key_length"));
  c.indexer_top_k = r.U32(key("attention.indexer.top_k"));
  c.indexer_kpool = r.U32(key("attention.indexer.kpool"));

  // MoE.
  c.num_experts = r.U32(key("expert_count"));
  c.num_experts_used = r.U32(key("expert_used_count"));
  c.expert_ff = r.U32(key("expert_feed_forward_length"));
  c.num_shared_experts = r.U32(key("expert_shared_count"), false, 1);
  c.shared_expert_ff =
      r.U32(key("expert_shared_feed_forward_length"), false, c.expert_ff);
  c.expert_gating_func = r.U32(key("expert_gating_func"), false, 2);
  c.expert_weights_scale = r.F32(key("expert_weights_scale"), 2.5F);
  c.expert_weights_norm =
      gguf.GetMetadataBool(key("expert_weights_norm")).value_or(true);

  // Dense FFN.
  c.dense_ff = r.U32(key("feed_forward_length"));

  // MTP.
  c.nextn_layers = r.U32(key("nextn_predict_layers"), false, 0);

  // SwiGLU clamps (optional arrays; default 10.0).
  c.swiglu_clamp_exp = r.F32Array(key("swiglu_clamp_exp"), false);
  c.swiglu_clamp_shexp = r.F32Array(key("swiglu_clamp_shexp"), false);

  // Per-layer attention type from head_count_kv pattern.
  {
    const auto kv_pattern = r.U64Array(key("attention.head_count_kv"));
    if (kv_pattern.size() != c.num_layers) {
      r.Fail("attention.head_count_kv must have one entry per block");
    } else {
      c.is_full_attention.resize(c.num_layers);
      for (std::size_t i = 0; i < kv_pattern.size(); ++i)
        c.is_full_attention[i] = kv_pattern[i] > 0;
    }
  }

  if (!r.ok)
    return std::nullopt;

  // Sanity checks.
  if (c.num_layers == 0 || c.num_layers > kMaxLayers) {
    r.Fail("block_count out of range");
    return std::nullopt;
  }
  if (c.hidden_size == 0 || c.vocab_size == 0 || c.num_attention_heads == 0) {
    r.Fail("core geometry must be positive");
    return std::nullopt;
  }
  if (c.num_experts > kMaxExperts || c.num_experts_used == 0) {
    r.Fail("expert_count or expert_used_count out of range");
    return std::nullopt;
  }
  if (c.num_experts_used > c.num_experts) {
    r.Fail("expert_used_count exceeds expert_count");
    return std::nullopt;
  }
  if (c.expert_gating_func != 2) {
    r.Fail("only sigmoid expert gating (func=2) is supported");
    return std::nullopt;
  }
  if (c.hc_count < 2) {
    r.Fail("hyper_connection.count must be at least 2");
    return std::nullopt;
  }
  return c;
}

std::uint64_t Config::EstimateResidentBytes() const noexcept {
  // Per-element byte factors by quant type (approximate, for budgeting).
  constexpr float kQ6K = 210.0F / 256;  // 0.82 B/elem
  constexpr float kIQ2 = 66.0F / 256;   // 0.26 B/elem
  constexpr float kIQ3 = 98.0F / 256;   // 0.38 B/elem
  constexpr float kF32 = 4.0F;
  constexpr float kQ8 = 34.0F / 32;  // 1.06 B/elem

  const std::uint64_t H = hidden_size;
  const std::uint64_t V = vocab_size;
  const std::uint64_t E = num_experts;
  const std::uint64_t I = expert_ff;
  const std::uint64_t D = dense_ff;
  const std::uint64_t nh = num_attention_heads;
  const std::uint64_t qd = qk_nope_dim;
  const std::uint64_t vd = v_head_dim;
  const std::uint64_t qlora = q_lora_rank;
  const std::uint64_t kvlora = kv_lora_rank;

  std::uint64_t bytes = 0;

  // Embeddings + output head (Q6_K).
  bytes += static_cast<std::uint64_t>(2.0F * H * V * kQ6K);

  // Dense layers (0 .. leading_dense-1).
  for (std::uint32_t i = 0; i < leading_dense_layers && i < num_layers; ++i) {
    if (IsFullAttention(i)) {
      // MLA: q_a, q_b, kv_a, k_b, v_b, output.
      bytes += static_cast<std::uint64_t>(H * qlora * kQ6K);         // q_a
      bytes += static_cast<std::uint64_t>(qlora * nh * qd * kQ6K);   // q_b
      bytes += static_cast<std::uint64_t>(H * kvlora * kQ6K);        // kv_a
      bytes += static_cast<std::uint64_t>(kvlora * nh * qd * kQ6K);  // k_b
      bytes += static_cast<std::uint64_t>(kvlora * nh * vd * kQ6K);  // v_b
      bytes += static_cast<std::uint64_t>(nh * qd * H * kQ6K);       // out
    } else {
      // KDA: q, k, v, output (Q6_K).
      bytes +=
          static_cast<std::uint64_t>(4.0F * H * 2 * nh * kda_head_dim * kQ6K);
    }
    // Dense FFN: gate, up, down.
    bytes += static_cast<std::uint64_t>(3.0F * H * D * kQ6K);
  }

  // MoE layers.
  const std::uint32_t moe_start = std::min(leading_dense_layers, num_layers);
  for (std::uint32_t i = moe_start; i < num_layers; ++i) {
    if (IsFullAttention(i)) {
      bytes += static_cast<std::uint64_t>(H * qlora * kQ6K);
      bytes += static_cast<std::uint64_t>(qlora * nh * qd * kQ6K);
      bytes += static_cast<std::uint64_t>(H * kvlora * kQ6K);
      bytes += static_cast<std::uint64_t>(kvlora * nh * qd * kQ6K);
      bytes += static_cast<std::uint64_t>(kvlora * nh * vd * kQ6K);
      bytes += static_cast<std::uint64_t>(nh * qd * H * kQ6K);
    } else {
      bytes +=
          static_cast<std::uint64_t>(4.0F * H * 2 * nh * kda_head_dim * kQ6K);
    }
    // MoE experts: gate + up (IQ2_XXS) + down (IQ2_S/IQ3_XXS avg).
    bytes += static_cast<std::uint64_t>(2.0F * H * I * E * kIQ2);  // gate+up
    bytes += static_cast<std::uint64_t>(I * H * E * kIQ3);         // down
    // Shared expert (Q6_K).
    bytes += static_cast<std::uint64_t>(3.0F * H * shared_expert_ff * kQ6K);
    // Router (F32).
    bytes += static_cast<std::uint64_t>(H * E * kF32);
  }

  // mHC per layer (all but MTP block): 2 mixers × (fn + base + scale).
  const std::uint32_t mhc_layers = num_layers > 0 ? num_layers - 1 : 0;
  // fn: H*hc_count x hc_count*something — approximate as 16384*24*4 per mixer.
  bytes += static_cast<std::uint64_t>(mhc_layers * 2.0F * 16384 * 24 * kF32);

  // MTP head (eh_proj Q6_K + norms F32).
  if (nextn_layers > 0) {
    bytes += static_cast<std::uint64_t>(2.0F * H * H * kQ6K);  // eh_proj
    bytes += static_cast<std::uint64_t>(4.0F * H * kF32);      // norms
  }

  return bytes;
}

}  // namespace gufo::models::glm5_next
