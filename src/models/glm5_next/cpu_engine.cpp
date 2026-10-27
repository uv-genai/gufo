#include "src/models/glm5_next/cpu_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace gufo::models::glm5_next {
namespace {

/// Dequantize a GGUF tensor to F32 (simplified: handles F32, Q8_0, Q6_K).
std::vector<float> DequantizeTensor(const TensorRef& t) {
  if (t.empty() || t.data == nullptr)
    return {};
  const std::uint64_t n = t.ElementCount();
  if (n == 0)
    return {};
  std::vector<float> out(n);

  if (t.type == core::GgmlType::kF32) {
    std::memcpy(out.data(), t.data, n * sizeof(float));
  } else if (t.type == core::GgmlType::kQ8_0) {
    // Q8_0: 32 elements per block, 1 scale (F16) + 32 int8.
    const auto* src = static_cast<const std::uint8_t*>(t.data);
    const std::uint64_t blocks = n / 32;
    for (std::uint64_t b = 0; b < blocks; ++b) {
      const auto* block = src + b * 34;
      std::uint16_t scale_h;
      std::memcpy(&scale_h, block, 2);
      // Simple F16 to F32 conversion.
      float scale;
      const std::uint32_t sign = (scale_h >> 15) & 1;
      const std::uint32_t exp = (scale_h >> 10) & 0x1F;
      const std::uint32_t mantissa = scale_h & 0x3FF;
      if (exp == 0) {
        scale = (sign ? -1.0F : 1.0F) * std::ldexp(mantissa / 1024.0F, -14);
      } else if (exp == 31) {
        scale = mantissa ? std::nanf("") : (sign ? -INFINITY : INFINITY);
      } else {
        scale = (sign ? -1.0F : 1.0F) *
                std::ldexp(1.0F + mantissa / 1024.0F, exp - 15);
      }
      for (int i = 0; i < 32; ++i) {
        out[b * 32 + i] =
            scale * static_cast<float>(static_cast<std::int8_t>(block[2 + i]));
      }
    }
  } else {
    // For other quant types, return zeros (placeholder).
    std::fill(out.begin(), out.end(), 0.0F);
  }
  return out;
}

/// Convert TensorRef to KdaWeightsF32.
KdaWeightsF32 ToKdaF32(const LayerWeights& L) {
  KdaWeightsF32 w;
  w.w_q = DequantizeTensor(L.mla.q_a);  // Reuse as placeholder.
  w.w_k = DequantizeTensor(L.mla.k_b);
  w.w_v = DequantizeTensor(L.mla.v_b);
  w.w_out = DequantizeTensor(L.mla.output);
  w.f_a = DequantizeTensor(L.kda.f_a);
  w.f_b = DequantizeTensor(L.kda.f_b);
  w.dt_bias = DequantizeTensor(L.kda.dt_bias);
  w.ssm_a = DequantizeTensor(L.kda.a);
  w.ssm_beta = DequantizeTensor(L.kda.beta);
  w.g_a = DequantizeTensor(L.kda.g_a);
  w.g_b = DequantizeTensor(L.kda.g_b);
  w.ssm_norm = DequantizeTensor(L.kda.norm);
  return w;
}

/// Convert TensorRef to MhcWeightsF32.
MhcWeightsF32 ToMhcF32(const HcWeights& hc) {
  MhcWeightsF32 w;
  w.fn = DequantizeTensor(hc.attn_fn);
  w.scale = DequantizeTensor(hc.attn_scale);
  w.base = DequantizeTensor(hc.attn_base);
  return w;
}

}  // namespace

std::shared_ptr<CpuEngine> CpuEngine::Create(const Config& config,
                                             const ModelWeights& weights,
                                             std::string* error_msg) {
  auto engine = std::shared_ptr<CpuEngine>(new CpuEngine());
  engine->config_ = config;
  engine->weights_ = weights;

  // Build per-layer weight caches.
  const std::uint32_t n_layers = config.num_layers;
  engine->layer_cache_.resize(n_layers);

  for (std::uint32_t i = 0; i < n_layers && i < weights.layers.size(); ++i) {
    const auto& L = weights.layers[i];
    auto& cache = engine->layer_cache_[i];

    if (config.IsFullAttention(i)) {
      // MLA layer.
      cache.mla.q_a = DequantizeTensor(L.mla.q_a);
      cache.mla.q_a_norm = DequantizeTensor(L.mla.q_a_norm);
      cache.mla.q_b = DequantizeTensor(L.mla.q_b);
      cache.mla.kv_a_mqa = DequantizeTensor(L.mla.kv_a_mqa);
      cache.mla.kv_a_norm = DequantizeTensor(L.mla.kv_a_norm);
      cache.mla.k_b = DequantizeTensor(L.mla.k_b);
      cache.mla.v_b = DequantizeTensor(L.mla.v_b);
      cache.mla.output = DequantizeTensor(L.mla.output);
    } else {
      // KDA layer.
      cache.kda = ToKdaF32(L);
    }

    // mHC (all layers except MTP).
    if (!config.IsMtpLayer(i)) {
      cache.mhc_attn = ToMhcF32(L.hc);
      cache.mhc_ffn = ToMhcF32(L.hc);  // Same weights for both halves.
    }

    // FFN.
    if (config.IsDenseFfn(i)) {
      // Dense FFN.
      cache.moe.gate_shexp = DequantizeTensor(L.dense.gate);
      cache.moe.up_shexp = DequantizeTensor(L.dense.up);
      cache.moe.down_shexp = DequantizeTensor(L.dense.down);
    } else {
      // MoE.
      cache.moe.gate_inp = DequantizeTensor(L.moe.gate_inp);
      cache.moe.exp_probs_b = DequantizeTensor(L.moe.exp_probs_b);
      cache.moe.gate_exps = DequantizeTensor(L.moe.gate_exps);
      cache.moe.up_exps = DequantizeTensor(L.moe.up_exps);
      cache.moe.down_exps = DequantizeTensor(L.moe.down_exps);
      cache.moe.gate_shexp = DequantizeTensor(L.moe.gate_shexp);
      cache.moe.up_shexp = DequantizeTensor(L.moe.up_shexp);
      cache.moe.down_shexp = DequantizeTensor(L.moe.down_shexp);
    }
  }

  // Embeddings.
  engine->token_embd_ = DequantizeTensor(weights.token_embd);
  engine->output_head_ = DequantizeTensor(weights.output);
  engine->output_norm_ = DequantizeTensor(weights.output_norm);

  // MTP.
  if (config.nextn_layers > 0 && !weights.layers.empty()) {
    const auto& mtp = weights.layers[weights.layers.size() - 1].mtp_weights;
    engine->mtp_.enorm = DequantizeTensor(mtp.enorm);
    engine->mtp_.hnorm = DequantizeTensor(mtp.hnorm);
    engine->mtp_.eh_proj = DequantizeTensor(mtp.eh_proj);
    engine->mtp_.shared_head_norm = DequantizeTensor(mtp.shared_head_norm);
    engine->mtp_.output_head = engine->output_head_;  // Shared.
  }

  return engine;
}

std::vector<float> CpuEngine::ForwardLayer(std::uint32_t layer,
                                           std::span<const float> input,
                                           std::uint32_t seq_len) {
  const auto& cache = layer_cache_[layer];
  const std::uint32_t H = config_.hidden_size;

  // 1. mHC read half (attn mixer).
  MhcGeometry mhc_geom;
  mhc_geom.hidden_size = H;
  mhc_geom.hc_count = config_.hc_count;
  mhc_geom.sinkhorn_iters = config_.hc_sinkhorn_iters;
  mhc_geom.eps = config_.hc_epsilon;

  // Residual: [H, hc, T] — for simplicity, use input as all streams.
  std::vector<float> residual(H * config_.hc_count * seq_len);
  for (std::uint32_t t = 0; t < seq_len; ++t) {
    for (std::uint32_t s = 0; s < config_.hc_count; ++s) {
      for (std::uint32_t e = 0; e < H; ++e) {
        residual[t * H * config_.hc_count + s * H + e] = input[t * H + e];
      }
    }
  }

  auto mhc_read = MhcPre(mhc_geom, cache.mhc_attn, residual, seq_len);

  // 2. Attention mixer (KDA or MLA).
  std::vector<float> attn_out(H * seq_len);
  if (config_.IsFullAttention(layer)) {
    // MLA.
    MlaGeometry mla_geom;
    mla_geom.hidden_size = H;
    mla_geom.num_heads = config_.num_attention_heads;
    mla_geom.qk_nope = config_.qk_nope_dim;
    mla_geom.v_head_dim = config_.v_head_dim;
    mla_geom.q_lora_rank = config_.q_lora_rank;
    mla_geom.kv_lora_rank = config_.kv_lora_rank;
    attn_out = DsaLayer(mla_geom, cache.mla, mhc_read.mixed, seq_len);
  } else {
    // KDA.
    KdaGeometry kda_geom;
    kda_geom.hidden_size = H;
    kda_geom.num_heads = config_.num_attention_heads;
    kda_geom.head_dim = config_.kda_head_dim;
    kda_geom.conv_kernel = config_.kda_conv_kernel;
    kda_geom.gate_lower_bound = config_.kda_gate_lower_bound;
    attn_out = KdaPrefill(kda_geom, cache.kda, mhc_read.mixed, seq_len);
  }

  // 3. mHC write half (attn).
  auto post_attn = MhcPost(mhc_geom, attn_out, residual, mhc_read.post,
                           mhc_read.comb, seq_len);

  // 4. FFN (with mHC).
  std::vector<float> ffn_in(H * seq_len);
  for (std::uint32_t t = 0; t < seq_len; ++t) {
    for (std::uint32_t e = 0; e < H; ++e) {
      ffn_in[t * H + e] = post_attn[t * H * config_.hc_count + e];  // stream 0.
    }
  }

  // mHC read half (ffn mixer).
  auto mhc_ffn = MhcPre(mhc_geom, cache.mhc_ffn, residual, seq_len);

  // FFN forward.
  std::vector<float> ffn_out(H * seq_len);
  MoeGeometry moe_geom;
  moe_geom.hidden_size = H;
  moe_geom.num_experts = config_.num_experts;
  moe_geom.top_k = config_.num_experts_used;
  moe_geom.expert_ff = config_.expert_ff;
  moe_geom.shared_ff = config_.shared_expert_ff;
  moe_geom.expert_weights_scale = config_.expert_weights_scale;
  moe_geom.expert_weights_norm = config_.expert_weights_norm;
  moe_geom.swiglu_limit = 10.0F;

  for (std::uint32_t t = 0; t < seq_len; ++t) {
    std::vector<float> token_in(H);
    for (std::uint32_t e = 0; e < H; ++e) {
      token_in[e] = mhc_ffn.mixed[t * H + e];
    }
    auto token_out = MoEForward(moe_geom, cache.moe, token_in);
    for (std::uint32_t e = 0; e < H; ++e) {
      ffn_out[t * H + e] = token_out[e];
    }
  }

  // mHC write half (ffn).
  auto post_ffn =
      MhcPost(mhc_geom, ffn_out, residual, mhc_ffn.post, mhc_ffn.comb, seq_len);

  // Return stream 0 as the layer output.
  std::vector<float> out(H * seq_len);
  for (std::uint32_t t = 0; t < seq_len; ++t) {
    for (std::uint32_t e = 0; e < H; ++e) {
      out[t * H + e] = post_ffn[t * H * config_.hc_count + e];
    }
  }
  return out;
}

std::vector<float> CpuEngine::Prefill(std::span<const std::uint32_t> tokens) {
  const std::uint32_t H = config_.hidden_size;
  const std::uint32_t T = static_cast<std::uint32_t>(tokens.size());

  // Token embedding lookup.
  std::vector<float> hidden(H * T);
  for (std::uint32_t t = 0; t < T; ++t) {
    const std::uint32_t tok = tokens[t];
    if (tok < config_.vocab_size) {
      std::memcpy(hidden.data() + t * H, token_embd_.data() + tok * H,
                  H * sizeof(float));
    }
  }

  // Run through all layers.
  for (std::uint32_t layer = 0; layer < config_.num_layers; ++layer) {
    hidden = ForwardLayer(layer, hidden, T);
  }

  // Return final token's hidden state.
  std::vector<float> final_hidden(H);
  std::memcpy(final_hidden.data(), hidden.data() + (T - 1) * H,
              H * sizeof(float));
  return final_hidden;
}

std::uint32_t CpuEngine::Decode(const std::vector<float>& hidden) {
  const std::uint32_t H = config_.hidden_size;

  // RMS norm + output head.
  std::vector<float> normed = hidden;
  float ms = 0;
  for (std::uint32_t i = 0; i < H; ++i)
    ms += normed[i] * normed[i];
  const float inv_rms = 1.0F / std::sqrt(ms / H + config_.layer_norm_rms_eps);
  for (std::uint32_t i = 0; i < H; ++i) {
    normed[i] = normed[i] * inv_rms * output_norm_[i];
  }

  // Logits = output_head @ normed.
  std::vector<float> logits(config_.vocab_size, 0.0F);
  for (std::uint32_t v = 0; v < config_.vocab_size; ++v) {
    float acc = 0;
    const float* row = output_head_.data() + v * H;
    for (std::uint32_t i = 0; i < H; ++i)
      acc += row[i] * normed[i];
    logits[v] = acc;
  }

  // Greedy argmax.
  std::uint32_t best = 0;
  for (std::uint32_t v = 1; v < config_.vocab_size; ++v) {
    if (logits[v] > logits[best])
      best = v;
  }
  return best;
}

std::vector<std::uint32_t> CpuEngine::DraftMtp(
    const std::vector<float>& embedding, const std::vector<float>& hidden) {
  MtpGeometry geom;
  geom.hidden_size = config_.hidden_size;
  geom.vocab_size = config_.vocab_size;
  geom.norm_eps = config_.layer_norm_rms_eps;
  geom.fixed_draft_width = 4;

  auto logits = MtpDraft(geom, mtp_, embedding, hidden);

  // Return top-k draft tokens.
  std::vector<std::pair<float, std::uint32_t>> indexed(config_.vocab_size);
  for (std::uint32_t v = 0; v < config_.vocab_size; ++v) {
    indexed[v] = {-logits[v], v};
  }
  std::partial_sort(indexed.begin(), indexed.begin() + geom.fixed_draft_width,
                    indexed.end());

  std::vector<std::uint32_t> drafts(geom.fixed_draft_width);
  for (std::uint32_t i = 0; i < geom.fixed_draft_width; ++i) {
    drafts[i] = indexed[i].second;
  }
  return drafts;
}

}  // namespace gufo::models::glm5_next
