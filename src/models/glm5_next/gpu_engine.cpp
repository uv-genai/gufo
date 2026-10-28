#include "src/models/glm5_next/gpu_engine.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "src/models/glm5_next/gpu_kernels.hpp"
#include "src/models/glm5_next/kernels/rocm/kda.hpp"
#include "src/models/glm5_next/kernels/rocm/mhc.hpp"
#include "src/models/glm5_next/kernels/rocm/moe.hpp"
#include "src/models/glm5_next/kernels/rocm/mtp.hpp"
#include "src/models/glm5_next/quant_gemv.hpp"

namespace gufo::models::glm5_next {
namespace {

constexpr std::uint64_t kArenaAlign = 256;

std::uint64_t Align256(std::uint64_t n) {
  return (n + kArenaAlign - 1) & ~(kArenaAlign - 1);
}

/// Dequantize a GGUF tensor to F32 (handles F32, F16, Q8_0; others → zeros).
void DequantToHost(const TensorRef& t, std::vector<float>& out) {
  const std::uint64_t n = t.ElementCount();
  out.resize(n);
  if (t.empty() || t.data == nullptr || n == 0) {
    std::fill(out.begin(), out.end(), 0.0F);
    return;
  }
  if (t.type == core::GgmlType::kF32) {
    std::memcpy(out.data(), t.data, n * sizeof(float));
    return;
  }
  if (t.type == core::GgmlType::kF16) {
    const auto* src = static_cast<const std::uint16_t*>(t.data);
    for (std::uint64_t i = 0; i < n; ++i) {
      const std::uint16_t h = src[i];
      const std::uint32_t sign = (h >> 15) & 1;
      const std::uint32_t exp = (h >> 10) & 0x1F;
      const std::uint32_t mantissa = h & 0x3FF;
      if (exp == 0)
        out[i] = (sign ? -1.0F : 1.0F) * std::ldexp(mantissa / 1024.0F, -14);
      else if (exp == 31)
        out[i] = mantissa ? std::nanf("") : (sign ? -INFINITY : INFINITY);
      else
        out[i] = (sign ? -1.0F : 1.0F) *
                 std::ldexp(1.0F + mantissa / 1024.0F, exp - 15);
    }
    return;
  }
  if (t.type == core::GgmlType::kQ8_0) {
    const auto* src = static_cast<const std::uint8_t*>(t.data);
    const std::uint64_t blocks = n / 32;
    for (std::uint64_t b = 0; b < blocks; ++b) {
      const auto* blk = src + b * 34;
      std::uint16_t scale_h;
      std::memcpy(&scale_h, blk, 2);
      const std::uint32_t sign = (scale_h >> 15) & 1;
      const std::uint32_t exp = (scale_h >> 10) & 0x1F;
      const std::uint32_t mantissa = scale_h & 0x3FF;
      float scale;
      if (exp == 0)
        scale = (sign ? -1.0F : 1.0F) * std::ldexp(mantissa / 1024.0F, -14);
      else if (exp == 31)
        scale = mantissa ? std::nanf("") : (sign ? -INFINITY : INFINITY);
      else
        scale = (sign ? -1.0F : 1.0F) *
                std::ldexp(1.0F + mantissa / 1024.0F, exp - 15);
      for (int i = 0; i < 32; ++i)
        out[b * 32 + i] =
            scale * static_cast<float>(static_cast<std::int8_t>(blk[2 + i]));
    }
    return;
  }
  // Unsupported type — zeros.
  std::fill(out.begin(), out.end(), 0.0F);
}

}  // namespace

GpuEngine::~GpuEngine() {
  hipFree(w_arena_);
  hipFree(state_);
  hipFree(sc_);
  hipFree(output_head_f32_);
  if (stream_)
    hipStreamDestroy(stream_);
}

std::shared_ptr<GpuEngine> GpuEngine::Create(const Config& config,
                                             const ModelWeights& weights,
                                             std::string* error_msg) {
  setvbuf(stderr, nullptr, _IONBF, 0);  // Unbuffered stderr for debugging.
  fprintf(stderr, "[glm5_gpu] Create start, layers=%u\n", config.num_layers);
  auto engine = std::shared_ptr<GpuEngine>(new GpuEngine());
  engine->config_ = config;
  engine->H_ = config.hidden_size;
  engine->hc_ = config.hc_count;
  engine->nh_ = config.num_attention_heads;
  engine->dk_ = config.kda_head_dim;
  engine->inner_ = engine->nh_ * engine->dk_;

  if (hipStreamCreateWithFlags(&engine->stream_, hipStreamNonBlocking) !=
      hipSuccess) {
    if (error_msg)
      *error_msg = "hipStreamCreate failed";
    return nullptr;
  }

  // ---- Weight arena: upload weights. Small weights (norms, biases) are
  // dequantized to F32. Large quantized weights (experts, attention) are
  // uploaded as-is (kernels handle quantized data or dequantize on-the-fly).
  // This keeps the arena within GPU memory (~90GB for Maya-S).
  std::uint64_t arena_bytes = 0;

  auto measure_raw = [&](const TensorRef& t) -> std::uint64_t {
    if (t.empty() || t.data == nullptr)
      return 0;
    return Align256(t.SizeBytes());  // Raw quantized bytes.
  };

  // Only measure quantized tensors for the arena. F32 tensors get
  // separate allocations (see upload lambda).
  arena_bytes += measure_raw(weights.token_embd);
  arena_bytes += measure_raw(weights.output);
  arena_bytes += measure_raw(weights.output_norm);

  const std::uint32_t n_layers = config.num_layers;
  for (std::uint32_t i = 0; i < n_layers && i < weights.layers.size(); ++i) {
    const auto& W = weights.layers[i];
    for (const auto* t :
         {&W.kda.attn_q,   &W.kda.attn_k,      &W.kda.attn_v,
          &W.kda.attn_out, &W.kda.conv1d_q,    &W.kda.conv1d_k,
          &W.kda.conv1d_v, &W.kda.a,           &W.kda.dt_bias,
          &W.kda.f_a,      &W.kda.f_b,         &W.kda.beta,
          &W.kda.g_a,      &W.kda.g_b,         &W.kda.norm,
          &W.hc.attn_fn,   &W.hc.attn_scale,   &W.hc.attn_base,
          &W.moe.gate_inp, &W.moe.exp_probs_b, &W.moe.gate_exps,
          &W.moe.up_exps,  &W.moe.down_exps,   &W.moe.gate_shexp,
          &W.moe.up_shexp, &W.moe.down_shexp,  &W.attn_norm,
          &W.ffn_norm,     &W.mla.q_a,         &W.mla.q_a_norm,
          &W.mla.q_b,      &W.mla.kv_a_mqa}) {
      arena_bytes += measure_raw(*t);
    }
  }

  if (arena_bytes == 0) {
    if (error_msg)
      *error_msg = "no weights to upload";
    return nullptr;
  }
  fprintf(stderr, "[glm5_gpu] allocating weight arena: %lu MiB (%lu bytes)\n",
          (unsigned long)(arena_bytes / (1024 * 1024)),
          (unsigned long)arena_bytes);
  {
    std::size_t free_b = 0, total_b = 0;
    hipMemGetInfo(&free_b, &total_b);
    fprintf(stderr, "[glm5_gpu] GPU memory: free=%lu MiB total=%lu MiB\n",
            (unsigned long)(free_b / (1024 * 1024)),
            (unsigned long)(total_b / (1024 * 1024)));
  }
  const hipError_t alloc_err = hipMalloc(&engine->w_arena_, arena_bytes);
  fprintf(stderr, "[glm5_gpu] arena alloc result: %d (%s)\n", (int)alloc_err,
          hipGetErrorString(alloc_err));
  if (alloc_err != hipSuccess) {
    if (error_msg)
      *error_msg = "weight arena alloc failed";
    return nullptr;
  }
  engine->w_arena_bytes_ = arena_bytes;

  // Upload tensors directly (raw bytes for quantized, dequantized for F32).
  std::uint64_t arena_off = 0;
  std::vector<float> tmp;

  auto upload = [&](const TensorRef& t, bool force_f32 = false) -> float* {
    if (t.empty() || t.data == nullptr)
      return nullptr;
    const std::uint64_t bytes = t.SizeBytes();
    if (bytes == 0)
      return nullptr;
    fprintf(stderr, "[glm5_gpu] upload %s bytes=%lu force_f32=%d\n",
            std::string(t.name).c_str(), (unsigned long)bytes, (int)force_f32);
    float* dev = nullptr;

    if (force_f32 || t.type == core::GgmlType::kF32 ||
        t.type == core::GgmlType::kF16 || t.type == core::GgmlType::kBF16) {
      // Upload as F32 (dequantize if needed).
      std::vector<float> f32;
      DequantToHost(t, f32);
      const std::uint64_t f32_bytes = f32.size() * sizeof(float);
      fprintf(stderr, "[glm5_gpu]   dequantized to %lu floats\n",
              (unsigned long)f32.size());
      hipMalloc(&dev, f32_bytes);
      hipMemcpy(dev, f32.data(), f32_bytes, hipMemcpyHostToDevice);
      arena_off += Align256(f32_bytes);
    } else {
      // Upload raw quantized bytes — kernels dequantize on-the-fly.
      dev = reinterpret_cast<float*>(reinterpret_cast<char*>(engine->w_arena_) +
                                     arena_off);
      hipMemcpy(dev, t.data, bytes, hipMemcpyHostToDevice);
      arena_off += Align256(bytes);
    }
    return dev;
  };

  engine->token_embd_ = upload(weights.token_embd);
  fprintf(stderr, "[glm5_gpu] token_embd done\n");
  engine->output_head_ = upload(weights.output);
  fprintf(stderr, "[glm5_gpu] output_head done\n");
  engine->output_norm_ = upload(weights.output_norm);
  fprintf(stderr, "[glm5_gpu] output_norm done\n");

  std::vector<float*> layer_ptr(n_layers * 32, nullptr);
  for (std::uint32_t i = 0; i < n_layers && i < weights.layers.size(); ++i) {
    const auto& W = weights.layers[i];
    std::size_t base = i * 32;
    // KDA attention weights need F32 for the kernel.
    layer_ptr[base + 0] = upload(W.kda.attn_q);    // quantized (Q6_K)
    layer_ptr[base + 1] = upload(W.kda.attn_k);    // quantized (Q6_K)
    layer_ptr[base + 2] = upload(W.kda.attn_v);    // quantized (Q6_K)
    layer_ptr[base + 3] = upload(W.kda.attn_out);  // quantized (Q6_K)
    // KDA mixer weights (small, F32 is fine).
    layer_ptr[base + 4] = upload(W.kda.conv1d_q, true);
    layer_ptr[base + 5] = upload(W.kda.conv1d_k, true);
    layer_ptr[base + 6] = upload(W.kda.conv1d_v, true);
    layer_ptr[base + 7] = upload(W.kda.a, true);
    layer_ptr[base + 8] = upload(W.kda.dt_bias, true);
    layer_ptr[base + 9] = upload(W.kda.f_a, true);
    layer_ptr[base + 10] = upload(W.kda.f_b, true);
    layer_ptr[base + 11] = upload(W.kda.beta, true);
    layer_ptr[base + 12] = upload(W.kda.g_a, true);
    layer_ptr[base + 13] = upload(W.kda.g_b, true);
    layer_ptr[base + 14] = upload(W.kda.norm, true);
    // mHC weights (small, F32).
    layer_ptr[base + 15] = upload(W.hc.attn_fn, true);
    layer_ptr[base + 16] = upload(W.hc.attn_scale, true);
    layer_ptr[base + 17] = upload(W.hc.attn_base, true);
    // MoE weights — keep quantized (large).
    layer_ptr[base + 18] = upload(W.moe.gate_inp, true);  // F32 router
    layer_ptr[base + 19] = upload(W.moe.exp_probs_b, true);
    layer_ptr[base + 20] = upload(W.moe.gate_exps);  // quantized
    layer_ptr[base + 21] = upload(W.moe.up_exps);    // quantized
    layer_ptr[base + 22] = upload(W.moe.down_exps);  // quantized
    layer_ptr[base + 23] = upload(W.moe.gate_shexp, true);
    layer_ptr[base + 24] = upload(W.moe.up_shexp, true);
    layer_ptr[base + 25] = upload(W.moe.down_shexp, true);
    layer_ptr[base + 26] = upload(W.attn_norm, true);
    layer_ptr[base + 27] = upload(W.ffn_norm, true);
    layer_ptr[base + 28] = upload(W.mla.q_a, true);
    layer_ptr[base + 29] = upload(W.mla.q_a_norm, true);
    layer_ptr[base + 30] = upload(W.mla.q_b, true);
    layer_ptr[base + 31] = upload(W.mla.kv_a_mqa, true);
  }
  engine->w_ptrs_ = std::move(layer_ptr);

  fprintf(stderr,
          "[glm5_gpu] arena=%p token_embd=%p output_head=%p output_norm=%p\n",
          (void*)engine->w_arena_, (void*)engine->token_embd_,
          (void*)engine->output_head_, (void*)engine->output_norm_);
  fprintf(stderr, "[glm5_gpu] w_ptrs[0]=%p w_ptrs[1]=%p w_ptrs[2]=%p\n",
          (void*)engine->w_ptrs_[0], (void*)engine->w_ptrs_[1],
          (void*)engine->w_ptrs_[2]);

  // ---- State arena: residual + KDA state + DSA buffers.
  const std::uint64_t hc_dim =
      static_cast<std::uint64_t>(engine->hc_) * engine->H_;
  std::uint64_t state_floats = 2 * hc_dim;  // R and hc_post ping-pong.

  engine->kda_S_off_.resize(n_layers, -1);
  engine->kda_conv_off_.resize(n_layers, -1);
  engine->dsa_lat_off_.resize(n_layers, -1);

  for (std::uint32_t il = 0; il < n_layers; ++il) {
    if (config.IsFullAttention(il)) {
      engine->dsa_lat_off_[il] = state_floats;
      state_floats += static_cast<std::uint64_t>(config.kv_lora_rank) * 1024;
    } else {
      engine->kda_S_off_[il] = state_floats;
      state_floats += static_cast<std::uint64_t>(engine->inner_) * engine->dk_;
      engine->kda_conv_off_[il] = state_floats;
      state_floats += static_cast<std::uint64_t>(3) * engine->inner_ *
                      (config.kda_conv_kernel - 1);
    }
  }

  engine->state_bytes_ = state_floats * sizeof(float);
  fprintf(stderr, "[glm5_gpu] state arena: %lu MiB\n",
          (unsigned long)(engine->state_bytes_ / (1024 * 1024)));
  if (hipMalloc(&engine->state_, engine->state_bytes_) != hipSuccess) {
    if (error_msg)
      *error_msg = "state arena alloc failed";
    return nullptr;
  }
  hipMemset(engine->state_, 0, engine->state_bytes_);
  fprintf(stderr, "[glm5_gpu] state arena done\n");

  // ---- Scratch arena: activations for one token (Maya sc_ pattern).
  const std::uint64_t sc_floats = std::max<std::uint64_t>(
      config.vocab_size + 64,
      32 * engine->H_ + 8 * engine->inner_ + engine->nh_ * engine->dk_ * 8);
  engine->sc_bytes_ = sc_floats * sizeof(float);
  fprintf(stderr, "[glm5_gpu] scratch arena: %lu MiB\n",
          (unsigned long)(engine->sc_bytes_ / (1024 * 1024)));
  if (hipMalloc(&engine->sc_, engine->sc_bytes_) != hipSuccess) {
    if (error_msg)
      *error_msg = "scratch arena alloc failed";
    return nullptr;
  }
  fprintf(stderr, "[glm5_gpu] scratch arena done\n");

  // ---- Dequantize output head to F32 (needed for GEMM).
  {
    fprintf(stderr, "[glm5_gpu] dequantizing output head...\n");
    std::vector<float> f32;
    DequantToHost(weights.output, f32);
    fprintf(stderr, "[glm5_gpu] output head: %lu floats\n",
            (unsigned long)f32.size());
    if (hipMalloc(&engine->output_head_f32_, f32.size() * sizeof(float)) !=
        hipSuccess) {
      if (error_msg)
        *error_msg = "output_head F32 alloc failed";
      return nullptr;
    }
    hipMemcpy(engine->output_head_f32_, f32.data(), f32.size() * sizeof(float),
              hipMemcpyHostToDevice);
    fprintf(stderr, "[glm5_gpu] output head done\n");
  }

  fprintf(stderr, "[glm5_gpu] Create complete\n");
  return engine;
}

bool GpuEngine::StepLayers(std::uint32_t pos, hipStream_t stream) {
  // Maya step_layers: for each layer, hc_pre → attn → hc_post → hc_pre → ffn →
  // hc_post.
  float* R = state_;          // residual at state arena head.
  const std::uint32_t T = 1;  // one token at a time.

  for (std::uint32_t il = 0; il < config_.num_layers; ++il) {
    const std::size_t base = il * 32;

    // 1. mHC read (attn mixer).
    rocm::MhcGpuGeometry mhc_geom;
    mhc_geom.hidden_size = H_;
    mhc_geom.hc_count = hc_;
    mhc_geom.hc_mix_dim = 24;
    mhc_geom.sinkhorn_iters = config_.hc_sinkhorn_iters;
    mhc_geom.eps = config_.hc_epsilon;
    mhc_geom.norm_eps = config_.layer_norm_rms_eps;

    rocm::MhcGpuWeights mhc_w;
    mhc_w.fn = w_ptrs_[base + 15];
    mhc_w.scale = w_ptrs_[base + 16];
    mhc_w.base = w_ptrs_[base + 17];

    float* mixed = sc_;
    float* post = sc_ + H_;
    float* comb = post + hc_;
    float* attn_out = comb + hc_ * hc_;

    if (!rocm::MhcPreGpu(mhc_geom, mhc_w, R, mixed, post, comb, T, stream))
      return false;

    // 2. Attention (KDA or DSA).
    if (config_.IsFullAttention(il)) {
      // MLA/DSA — simplified for now (copy mixed to attn_out).
      hipMemcpy(attn_out, mixed, H_ * sizeof(float), hipMemcpyDeviceToDevice);
    } else {
      // KDA mixer.
      rocm::KdaGpuGeometry kda_geom;
      kda_geom.hidden_size = H_;
      kda_geom.num_heads = nh_;
      kda_geom.head_dim = dk_;
      kda_geom.conv_kernel = config_.kda_conv_kernel;
      kda_geom.gate_lower_bound = config_.kda_gate_lower_bound;
      kda_geom.norm_eps = config_.layer_norm_rms_eps;

      rocm::KdaGpuWeights kda_w;
      kda_w.w_q = w_ptrs_[base + 0];
      kda_w.w_k = w_ptrs_[base + 1];
      kda_w.w_v = w_ptrs_[base + 2];
      kda_w.w_out = w_ptrs_[base + 3];
      // Quant types: attention weights are Q6_K(14).
      kda_w.w_q_type = 14;
      kda_w.w_k_type = 14;
      kda_w.w_v_type = 14;
      kda_w.w_out_type = 14;
      kda_w.conv_q = w_ptrs_[base + 4];
      kda_w.conv_k = w_ptrs_[base + 5];
      kda_w.conv_v = w_ptrs_[base + 6];
      kda_w.ssm_a = w_ptrs_[base + 7];
      kda_w.dt_bias = w_ptrs_[base + 8];
      kda_w.f_a = w_ptrs_[base + 9];
      kda_w.f_b = w_ptrs_[base + 10];
      kda_w.ssm_beta = w_ptrs_[base + 11];
      kda_w.g_a = w_ptrs_[base + 12];
      kda_w.g_b = w_ptrs_[base + 13];
      kda_w.ssm_norm = w_ptrs_[base + 14];

      float* state_ptr =
          kda_S_off_[il] >= 0 ? state_ + kda_S_off_[il] : nullptr;
      if (!rocm::KdaPrefillGpu(kda_geom, kda_w, mixed, state_ptr, attn_out, T,
                               stream))
        return false;
    }

    // 3. mHC write (attn).
    float* R_new = state_ + hc_ * H_;
    if (!rocm::MhcPostGpu(mhc_geom, attn_out, R, post, comb, R_new, T, stream))
      return false;
    R = R_new;

    // 4. mHC read (FFN mixer).
    if (!rocm::MhcPreGpu(mhc_geom, mhc_w, R, mixed, post, comb, T, stream))
      return false;

    // 5. FFN (MoE).
    rocm::MoeGpuGeometry moe_geom;
    moe_geom.hidden_size = H_;
    moe_geom.num_experts = config_.num_experts;
    moe_geom.top_k = config_.num_experts_used;
    moe_geom.expert_ff = config_.expert_ff;
    moe_geom.shared_ff = config_.shared_expert_ff;
    moe_geom.expert_weights_scale = config_.expert_weights_scale;
    moe_geom.expert_weights_norm = config_.expert_weights_norm;
    moe_geom.swiglu_limit = 10.0f;
    moe_geom.renorm_clamp = 6.103515625e-5f;

    rocm::MoeGpuWeights moe_w;
    moe_w.gate_inp = w_ptrs_[base + 18];
    moe_w.exp_probs_b = w_ptrs_[base + 19];
    moe_w.gate_exps = w_ptrs_[base + 20];
    moe_w.up_exps = w_ptrs_[base + 21];
    moe_w.down_exps = w_ptrs_[base + 22];
    moe_w.gate_shexp = w_ptrs_[base + 23];
    moe_w.up_shexp = w_ptrs_[base + 24];
    moe_w.down_shexp = w_ptrs_[base + 25];
    // Quant types: gate/up=IQ2_XXS(16), down=IQ3_XXS(18), shared=Q6_K(14).
    moe_w.gate_exps_type = 16;
    moe_w.up_exps_type = 16;
    moe_w.down_exps_type = 18;

    float* ffn_out = sc_ + H_ + hc_ + hc_ * hc_ + H_;
    if (!rocm::MoeForwardGpu(moe_geom, moe_w, mixed, ffn_out, stream))
      return false;

    // 6. mHC write (FFN).
    R_new = (R == state_) ? state_ + hc_ * H_ : state_;
    if (!rocm::MhcPostGpu(mhc_geom, ffn_out, R, post, comb, R_new, T, stream))
      return false;
    R = R_new;
  }

  return true;
}

std::vector<float> GpuEngine::Prefill(std::span<const std::uint32_t> tokens) {
  const std::uint32_t T = static_cast<std::uint32_t>(tokens.size());
  if (T == 0)
    return std::vector<float>(H_, 0.0f);

  // For each token: embed → step_layers.
  for (std::uint32_t t = 0; t < T; ++t) {
    // Embedding lookup: copy row from token_embd to residual.
    float* R = state_;
    const std::uint32_t tok = tokens[t];
    if (token_embd_ && tok < config_.vocab_size) {
      hipMemcpy(R, token_embd_ + static_cast<std::size_t>(tok) * H_,
                H_ * sizeof(float), hipMemcpyDeviceToDevice);
      // Initialize all hc streams.
      for (std::uint32_t s = 1; s < hc_; ++s) {
        hipMemcpy(R + s * H_, R, H_ * sizeof(float), hipMemcpyDeviceToDevice);
      }
    }

    // Run layers.
    if (!StepLayers(t, stream_)) {
      return std::vector<float>(H_, 0.0f);
    }
  }

  // Final RMS norm + output head + argmax.
  // R is now at the last position in the residual arena.
  float* final_hidden = state_;  // Simplified: use first stream.
  gpu::LaunchRmsNorm(final_hidden, output_norm_, H_, config_.layer_norm_rms_eps,
                     stream_);
  gpu::LaunchMatVec(output_head_f32_, final_hidden, sc_, config_.vocab_size, H_,
                    stream_);
  int token_id = 0;
  gpu::LaunchArgmax(sc_, config_.vocab_size, &token_id, stream_);

  std::vector<float> hidden(H_);
  hipMemcpy(hidden.data(), final_hidden, H_ * sizeof(float),
            hipMemcpyDeviceToHost);
  return hidden;
}

std::uint32_t GpuEngine::Decode(const std::vector<float>& hidden) {
  // Copy hidden to state arena, run output head.
  hipMemcpy(state_, hidden.data(), H_ * sizeof(float), hipMemcpyHostToDevice);
  gpu::LaunchRmsNorm(state_, output_norm_, H_, config_.layer_norm_rms_eps,
                     stream_);
  gpu::LaunchMatVec(output_head_f32_, state_, sc_, config_.vocab_size, H_,
                    stream_);
  int token_id = 0;
  gpu::LaunchArgmax(sc_, config_.vocab_size, &token_id, stream_);
  return static_cast<std::uint32_t>(token_id);
}

}  // namespace gufo::models::glm5_next
