#include "src/models/glm5_next/weights.hpp"

#include <string>
#include <utility>

namespace gufo::models::glm5_next {
namespace {

// ggml block geometry for the quant types this model uses.
// (block_elements, bytes_per_block) — mirrors Maya's gguf_reader.
struct BlockGeom {
  std::uint32_t elems;
  std::uint32_t bytes;
};

constexpr BlockGeom GetBlockGeom(core::GgmlType type) noexcept {
  switch (type) {
    case core::GgmlType::kF32:
      return {1, 4};
    case core::GgmlType::kF16:
      return {1, 2};
    case core::GgmlType::kBF16:
      return {1, 2};
    case core::GgmlType::kQ4_0:
      return {32, 18};
    case core::GgmlType::kQ5_0:
      return {32, 22};
    case core::GgmlType::kQ8_0:
      return {32, 34};
    case core::GgmlType::kQ2_K:
      return {256, 84};
    case core::GgmlType::kQ3_K:
      return {256, 110};
    case core::GgmlType::kQ4_K:
      return {256, 144};
    case core::GgmlType::kQ5_K:
      return {256, 176};
    case core::GgmlType::kQ6_K:
      return {256, 210};
    case core::GgmlType::kQ8_K:
      return {256, 292};
    case core::GgmlType::kIQ2_XXS:
      return {256, 66};
    case core::GgmlType::kIQ2_XS:
      return {256, 74};
    case core::GgmlType::kIQ2_S:
      return {256, 82};
    case core::GgmlType::kIQ3_XXS:
      return {256, 98};
    case core::GgmlType::kIQ1_S:
      return {256, 50};
    case core::GgmlType::kIQ1_M:
      return {256, 56};
    case core::GgmlType::kIQ3_S:
      return {256, 110};
    case core::GgmlType::kIQ4_XS:
      return {256, 136};
    case core::GgmlType::kIQ4_NL:
      return {32, 18};
    default:
      return {0, 0};
  }
}

/// Bind one GGUF tensor to a TensorRef, verifying its shape.
/// Returns false and sets error on missing/shape-mismatch.
bool Bind(const core::GgufReader& gguf, const Config& config,
          const std::string& name, std::uint64_t expect_cols,
          std::uint64_t expect_rows, std::uint64_t expect_experts,
          TensorRef* out, std::string* error_msg) {
  const auto* t = gguf.FindTensor(name);
  if (t == nullptr) {
    if (error_msg)
      *error_msg = "missing tensor " + name;
    return false;
  }
  const auto dim = [&](std::size_t i) -> std::uint64_t {
    return i < t->dimensions.size() ? t->dimensions[i] : 1;
  };
  const std::uint64_t got_cols = dim(0);
  const std::uint64_t got_rows = dim(1);
  const std::uint64_t got_experts = dim(2);
  if (got_cols != expect_cols || got_rows != expect_rows ||
      (expect_experts > 1 && got_experts != expect_experts)) {
    if (error_msg) {
      *error_msg = "tensor " + name + " has shape [" +
                   std::to_string(got_cols) + ", " + std::to_string(got_rows) +
                   ", " + std::to_string(got_experts) + "], expected [" +
                   std::to_string(expect_cols) + ", " +
                   std::to_string(expect_rows) + ", " +
                   std::to_string(expect_experts) + "]";
    }
    return false;
  }
  out->data = t->data;
  out->type = t->type;
  out->cols = got_cols;
  out->rows = got_rows;
  out->experts = expect_experts > 1 ? got_experts : 1;
  out->file_offset = t->offset;
  out->shard = 0;
  out->name = t->name;
  return true;
}

/// Bind a tensor that is optional (empty if absent).
bool BindOptional(const core::GgufReader& gguf, const std::string& name,
                  std::uint64_t expect_cols, std::uint64_t expect_rows,
                  TensorRef* out, std::string* error_msg) {
  const auto* t = gguf.FindTensor(name);
  if (t == nullptr) {
    out->data = nullptr;
    return true;
  }
  return Bind(gguf, {}, name, expect_cols, expect_rows, 1, out, error_msg);
}

}  // namespace

std::size_t TensorRowBytes(core::GgmlType type, std::uint64_t cols) noexcept {
  const auto geom = GetBlockGeom(type);
  if (geom.elems == 0)
    return 0;
  return (cols / geom.elems) * geom.bytes;
}

std::size_t TensorRef::RowBytes() const noexcept {
  return TensorRowBytes(type, cols);
}

std::uint64_t ModelWeights::ResidentBytes() const noexcept {
  std::uint64_t total = 0;
  auto add = [&](const TensorRef& t) {
    if (!t.empty())
      total += t.SizeBytes();
  };
  add(token_embd);
  add(output);
  add(output_norm);
  for (const auto& layer : layers) {
    add(layer.attn_norm);
    add(layer.ffn_norm);
    add(layer.hc.attn_base);
    add(layer.hc.attn_fn);
    add(layer.hc.attn_scale);
    add(layer.hc.ffn_base);
    add(layer.hc.ffn_fn);
    add(layer.hc.ffn_scale);
    add(layer.kda.a);
    add(layer.kda.norm);
    add(layer.kda.conv1d_q);
    add(layer.kda.conv1d_k);
    add(layer.kda.conv1d_v);
    add(layer.kda.dt_bias);
    add(layer.kda.beta);
    add(layer.kda.f_a);
    add(layer.kda.f_b);
    add(layer.kda.g_a);
    add(layer.kda.g_b);
    add(layer.mla.q_a);
    add(layer.mla.q_a_norm);
    add(layer.mla.q_b);
    add(layer.mla.kv_a_mqa);
    add(layer.mla.kv_a_norm);
    add(layer.mla.k_b);
    add(layer.mla.v_b);
    add(layer.mla.output);
    add(layer.indexer.q_b);
    add(layer.indexer.k);
    add(layer.indexer.k_norm_weight);
    add(layer.indexer.k_norm_bias);
    add(layer.indexer.proj);
    add(layer.indexer.compressor_ape);
    add(layer.indexer.compressor_gate);
    add(layer.moe.gate_inp);
    add(layer.moe.exp_probs_b);
    add(layer.moe.gate_exps);
    add(layer.moe.up_exps);
    add(layer.moe.down_exps);
    add(layer.moe.gate_shexp);
    add(layer.moe.up_shexp);
    add(layer.moe.down_shexp);
    add(layer.dense.gate);
    add(layer.dense.up);
    add(layer.dense.down);
    add(layer.mtp_weights.enorm);
    add(layer.mtp_weights.hnorm);
    add(layer.mtp_weights.eh_proj);
    add(layer.mtp_weights.shared_head_norm);
  }
  return total;
}

bool LoadWeights(const core::GgufReader& gguf, const Config& config,
                 ModelWeights* out, std::string* error_msg) {
  if (out == nullptr) {
    if (error_msg)
      *error_msg = "null output pointer";
    return false;
  }
  *out = {};
  const std::uint32_t H = config.hidden_size;
  const std::uint32_t V = config.vocab_size;
  const std::uint32_t E = config.num_experts;
  const std::uint32_t I = config.expert_ff;
  const std::uint32_t D = config.dense_ff;
  const std::uint32_t nh = config.num_attention_heads;
  const std::uint32_t qd = config.qk_nope_dim;
  const std::uint32_t vd = config.v_head_dim;
  const std::uint32_t qlora = config.q_lora_rank;
  const std::uint32_t kvlora = config.kv_lora_rank;
  const std::uint32_t kd = config.kda_head_dim;

  // Embeddings / output.
  if (!Bind(gguf, config, "token_embd.weight", H, V, 1, &out->token_embd,
            error_msg) ||
      !Bind(gguf, config, "output.weight", H, V, 1, &out->output, error_msg) ||
      !Bind(gguf, config, "output_norm.weight", H, 1, 1, &out->output_norm,
            error_msg)) {
    return false;
  }

  out->layers.resize(config.num_layers);

  for (std::uint32_t i = 0; i < config.num_layers; ++i) {
    auto& L = out->layers[i];
    L.full_attention = config.IsFullAttention(i);
    L.dense_ffn = config.IsDenseFfn(i);
    L.mtp = config.IsMtpLayer(i);
    const std::string prefix = "blk." + std::to_string(i) + ".";

    // Norms (present on every layer).
    if (!Bind(gguf, config, prefix + "attn_norm.weight", H, 1, 1, &L.attn_norm,
              error_msg) ||
        !Bind(gguf, config, prefix + "ffn_norm.weight", H, 1, 1, &L.ffn_norm,
              error_msg)) {
      return false;
    }

    // mHC (all layers except MTP block).
    if (!L.mtp) {
      if (!Bind(gguf, config, prefix + "hc_attn_base.weight", 24, 1, 1,
                &L.hc.attn_base, error_msg) ||
          !Bind(gguf, config, prefix + "hc_attn_fn.weight", 16384, 24, 1,
                &L.hc.attn_fn, error_msg) ||
          !Bind(gguf, config, prefix + "hc_attn_scale.weight", 3, 1, 1,
                &L.hc.attn_scale, error_msg) ||
          !Bind(gguf, config, prefix + "hc_ffn_base.weight", 24, 1, 1,
                &L.hc.ffn_base, error_msg) ||
          !Bind(gguf, config, prefix + "hc_ffn_fn.weight", 16384, 24, 1,
                &L.hc.ffn_fn, error_msg) ||
          !Bind(gguf, config, prefix + "hc_ffn_scale.weight", 3, 1, 1,
                &L.hc.ffn_scale, error_msg)) {
        return false;
      }
    }

    // Attention mixer.
    if (L.full_attention) {
      // MLA + DSA.
      if (!Bind(gguf, config, prefix + "attn_q_a.weight", H, qlora, 1,
                &L.mla.q_a, error_msg) ||
          !Bind(gguf, config, prefix + "attn_q_a_norm.weight", qlora, 1, 1,
                &L.mla.q_a_norm, error_msg) ||
          !Bind(gguf, config, prefix + "attn_q_b.weight", qlora, nh * qd, 1,
                &L.mla.q_b, error_msg) ||
          !Bind(gguf, config, prefix + "attn_kv_a_mqa.weight", H, kvlora, 1,
                &L.mla.kv_a_mqa, error_msg) ||
          !Bind(gguf, config, prefix + "attn_kv_a_norm.weight", kvlora, 1, 1,
                &L.mla.kv_a_norm, error_msg) ||
          !Bind(gguf, config, prefix + "attn_k_b.weight", qd, kvlora, nh,
                &L.mla.k_b, error_msg) ||
          !Bind(gguf, config, prefix + "attn_v_b.weight", kvlora, qd, nh,
                &L.mla.v_b, error_msg) ||
          !Bind(gguf, config, prefix + "attn_output.weight", nh * qd, H, 1,
                &L.mla.output, error_msg)) {
        return false;
      }
      // DSA indexer.
      if (!Bind(gguf, config, prefix + "indexer.attn_q_b.weight", qlora, H, 1,
                &L.indexer.q_b, error_msg) ||
          !Bind(gguf, config, prefix + "indexer.attn_k.weight", H,
                config.indexer_head_dim, 1, &L.indexer.k, error_msg) ||
          !Bind(gguf, config, prefix + "indexer.k_norm.weight",
                config.indexer_head_dim, 1, 1, &L.indexer.k_norm_weight,
                error_msg) ||
          !Bind(gguf, config, prefix + "indexer.k_norm.bias",
                config.indexer_head_dim, 1, 1, &L.indexer.k_norm_bias,
                error_msg) ||
          !Bind(gguf, config, prefix + "indexer.proj.weight", H,
                config.indexer_heads, 1, &L.indexer.proj, error_msg) ||
          !Bind(gguf, config, prefix + "indexer_compressor_ape.weight",
                config.indexer_head_dim, config.indexer_kpool, 1,
                &L.indexer.compressor_ape, error_msg) ||
          !Bind(gguf, config, prefix + "indexer_compressor_gate.weight", H,
                config.indexer_head_dim, 1, &L.indexer.compressor_gate,
                error_msg)) {
        return false;
      }
    } else {
      // KDA.
      const std::uint32_t inner = nh * kd;  // 64*128=8192 per stream.
      if (!Bind(gguf, config, prefix + "ssm_a", nh, 1, 1, &L.kda.a,
                error_msg) ||
          !Bind(gguf, config, prefix + "ssm_norm.weight", kd, 1, 1, &L.kda.norm,
                error_msg) ||
          !Bind(gguf, config, prefix + "ssm_conv1d_q.weight",
                config.kda_conv_kernel, 1, inner, &L.kda.conv1d_q, error_msg) ||
          !Bind(gguf, config, prefix + "ssm_conv1d_k.weight",
                config.kda_conv_kernel, 1, inner, &L.kda.conv1d_k, error_msg) ||
          !Bind(gguf, config, prefix + "ssm_conv1d_v.weight",
                config.kda_conv_kernel, 1, inner, &L.kda.conv1d_v, error_msg) ||
          !Bind(gguf, config, prefix + "ssm_dt.bias", inner, 1, 1,
                &L.kda.dt_bias, error_msg) ||
          !Bind(gguf, config, prefix + "ssm_beta.weight", H, nh, 1, &L.kda.beta,
                error_msg) ||
          !Bind(gguf, config, prefix + "ssm_f_a.weight", H, kd, 1, &L.kda.f_a,
                error_msg) ||
          !Bind(gguf, config, prefix + "ssm_f_b.weight", kd, inner, 1,
                &L.kda.f_b, error_msg) ||
          !Bind(gguf, config, prefix + "ssm_g_a.weight", H, kd, 1, &L.kda.g_a,
                error_msg) ||
          !Bind(gguf, config, prefix + "ssm_g_b.weight", kd, inner, 1,
                &L.kda.g_b, error_msg) ||
          !Bind(gguf, config, prefix + "attn_q.weight", H, inner, 1,
                &L.kda.attn_q, error_msg) ||
          !Bind(gguf, config, prefix + "attn_k.weight", H, inner, 1,
                &L.kda.attn_k, error_msg) ||
          !Bind(gguf, config, prefix + "attn_v.weight", H, inner, 1,
                &L.kda.attn_v, error_msg) ||
          !Bind(gguf, config, prefix + "attn_output.weight", inner, H, 1,
                &L.kda.attn_out, error_msg)) {
        return false;
      }
    }

    // FFN.
    if (L.dense_ffn) {
      if (!Bind(gguf, config, prefix + "ffn_gate.weight", H, D, 1,
                &L.dense.gate, error_msg) ||
          !Bind(gguf, config, prefix + "ffn_up.weight", H, D, 1, &L.dense.up,
                error_msg) ||
          !Bind(gguf, config, prefix + "ffn_down.weight", D, H, 1,
                &L.dense.down, error_msg)) {
        return false;
      }
    } else {
      // MoE.
      if (!Bind(gguf, config, prefix + "ffn_gate_inp.weight", H, E, 1,
                &L.moe.gate_inp, error_msg) ||
          !Bind(gguf, config, prefix + "exp_probs_b.bias", E, 1, 1,
                &L.moe.exp_probs_b, error_msg) ||
          !Bind(gguf, config, prefix + "ffn_gate_exps.weight", H, I, E,
                &L.moe.gate_exps, error_msg) ||
          !Bind(gguf, config, prefix + "ffn_up_exps.weight", H, I, E,
                &L.moe.up_exps, error_msg) ||
          !Bind(gguf, config, prefix + "ffn_down_exps.weight", I, H, E,
                &L.moe.down_exps, error_msg) ||
          !Bind(gguf, config, prefix + "ffn_gate_shexp.weight", H,
                config.shared_expert_ff, 1, &L.moe.gate_shexp, error_msg) ||
          !Bind(gguf, config, prefix + "ffn_up_shexp.weight", H,
                config.shared_expert_ff, 1, &L.moe.up_shexp, error_msg) ||
          !Bind(gguf, config, prefix + "ffn_down_shexp.weight",
                config.shared_expert_ff, H, 1, &L.moe.down_shexp, error_msg)) {
        return false;
      }
    }

    // MTP weights on the last block.
    if (L.mtp) {
      if (!Bind(gguf, config, prefix + "nextn.enorm.weight", H, 1, 1,
                &L.mtp_weights.enorm, error_msg) ||
          !Bind(gguf, config, prefix + "nextn.hnorm.weight", H, 1, 1,
                &L.mtp_weights.hnorm, error_msg) ||
          !Bind(gguf, config, prefix + "nextn.eh_proj.weight", 2 * H, H, 1,
                &L.mtp_weights.eh_proj, error_msg) ||
          !Bind(gguf, config, prefix + "nextn.shared_head_norm.weight", H, 1, 1,
                &L.mtp_weights.shared_head_norm, error_msg)) {
        return false;
      }
    }
  }

  return true;
}

}  // namespace gufo::models::glm5_next
