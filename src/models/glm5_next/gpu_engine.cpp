#include "src/models/glm5_next/gpu_engine.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <tuple>
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

/// Per-layer weight-slot layout (parallel w_ptrs_/w_types_ arrays).
enum Slot : int {
  kKdaQ = 0,
  kKdaK,
  kKdaV,
  kKdaOut,
  kKdaConvQ,
  kKdaConvK,
  kKdaConvV,
  kKdaA,
  kKdaDtBias,
  kKdaFa,
  kKdaFb,
  kKdaBeta,
  kKdaGa,
  kKdaGb,
  kKdaNorm,
  kHcAttnFn,
  kHcAttnScale,
  kHcAttnBase,
  kMoeGateInp,
  kMoeExpProbsB,
  kMoeGateExps,
  kMoeUpExps,
  kMoeDownExps,
  kMoeGateShexp,
  kMoeUpShexp,
  kMoeDownShexp,
  kAttnNorm,
  kFfnNorm,
  kMlaQa,
  kMlaQaNorm,
  kMlaQb,
  kMlaKvAMqa,
  kMlaKvANorm,
  kMlaKb,
  kMlaVb,
  kMlaOut,
  kIdxQb,
  kIdxK,
  kIdxKNormW,
  kIdxKNormB,
  kIdxProj,
  kIdxApe,
  kIdxGate,
  kDenseGate,
  kDenseUp,
  kDenseDown,
  kHcFfnFn,
  kHcFfnScale,
  kHcFfnBase,
  kSlotCount,
};

/// Decode-position cap for DSA latent caches (Maya's default context).
constexpr std::uint32_t kMaxDsaPos = 32768;

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
  for (auto& st : mla_state_) {
    // Caches and scratch are one allocation each (see Create).
    hipFree(st.latents);
    hipFree(st.scratch);
  }
  hipFree(ws_);
  hipFree(w_arena_);
  hipFree(state_);
  hipFree(sc_);
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

  // ---- Weight arena: quantized weights upload as raw bytes and are
  // dequantized on-the-fly by the GPU kernels; small weights (norms,
  // biases, routers) upload as F32. This keeps the arena within GPU memory
  // (~90GB for Maya-S).

  // Tensors uploaded as raw quantized bytes share one arena allocation;
  // tensors uploaded as F32 (small weights, dense packs) get their own
  // allocation and do not consume arena space.
  auto is_f32_upload = [](const TensorRef& t, bool force_f32) {
    return force_f32 || t.type == core::GgmlType::kF32 ||
           t.type == core::GgmlType::kF16 || t.type == core::GgmlType::kBF16;
  };

  struct SlotSpec {
    const TensorRef* t;
    bool force_f32;
  };
  std::vector<SlotSpec> flat;
  flat.push_back({&weights.token_embd, false});
  flat.push_back({&weights.output, false});
  flat.push_back({&weights.output_norm, true});
  const std::uint32_t n_layers = config.num_layers;
  for (std::uint32_t i = 0; i < n_layers && i < weights.layers.size(); ++i) {
    const auto& W = weights.layers[i];
    // KDA attention weights (quantized) + small F32 mixer weights,
    // mHC, MoE (experts quantized, router/shared F32), norms, MLA,
    // dense FFN (leading layers only; quantized).
    // Per-layer slot order must match the Slot enum. Quantized tensors upload
    // raw (kernels dequantize on the fly); small F32 tensors upload dense.
    const SlotSpec specs[kSlotCount] = {
        {&W.kda.attn_q, false},               // kKdaQ
        {&W.kda.attn_k, false},               // kKdaK
        {&W.kda.attn_v, false},               // kKdaV
        {&W.kda.attn_out, false},             // kKdaOut
        {&W.kda.conv1d_q, true},              // kKdaConvQ
        {&W.kda.conv1d_k, true},              // kKdaConvK
        {&W.kda.conv1d_v, true},              // kKdaConvV
        {&W.kda.a, true},                     // kKdaA
        {&W.kda.dt_bias, true},               // kKdaDtBias
        {&W.kda.f_a, true},                   // kKdaFa
        {&W.kda.f_b, true},                   // kKdaFb
        {&W.kda.beta, true},                  // kKdaBeta
        {&W.kda.g_a, true},                   // kKdaGa
        {&W.kda.g_b, true},                   // kKdaGb
        {&W.kda.norm, true},                  // kKdaNorm
        {&W.hc.attn_fn, true},                // kHcAttnFn
        {&W.hc.attn_scale, true},             // kHcAttnScale
        {&W.hc.attn_base, true},              // kHcAttnBase
        {&W.moe.gate_inp, true},              // kMoeGateInp
        {&W.moe.exp_probs_b, true},           // kMoeExpProbsB
        {&W.moe.gate_exps, false},            // kMoeGateExps
        {&W.moe.up_exps, false},              // kMoeUpExps
        {&W.moe.down_exps, false},            // kMoeDownExps
        {&W.moe.gate_shexp, true},            // kMoeGateShexp
        {&W.moe.up_shexp, true},              // kMoeUpShexp
        {&W.moe.down_shexp, true},            // kMoeDownShexp
        {&W.attn_norm, true},                 // kAttnNorm
        {&W.ffn_norm, true},                  // kFfnNorm
        {&W.mla.q_a, false},                  // kMlaQa
        {&W.mla.q_a_norm, true},              // kMlaQaNorm
        {&W.mla.q_b, false},                  // kMlaQb
        {&W.mla.kv_a_mqa, false},             // kMlaKvAMqa
        {&W.mla.kv_a_norm, true},             // kMlaKvANorm
        {&W.mla.k_b, false},                  // kMlaKb
        {&W.mla.v_b, false},                  // kMlaVb
        {&W.mla.output, false},               // kMlaOut
        {&W.indexer.q_b, false},              // kIdxQb
        {&W.indexer.k, false},                // kIdxK
        {&W.indexer.k_norm_weight, true},     // kIdxKNormW
        {&W.indexer.k_norm_bias, true},       // kIdxKNormB
        {&W.indexer.proj, false},             // kIdxProj
        {&W.indexer.compressor_ape, true},    // kIdxApe
        {&W.indexer.compressor_gate, false},  // kIdxGate
        {&W.dense.gate, false},               // kDenseGate
        {&W.dense.up, false},                 // kDenseUp
        {&W.dense.down, false},               // kDenseDown
        {&W.hc.ffn_fn, true},                 // kHcFfnFn
        {&W.hc.ffn_scale, true},              // kHcFfnScale
        {&W.hc.ffn_base, true},               // kHcFfnBase
    };
    flat.insert(flat.end(), std::begin(specs), std::end(specs));
  }

  std::uint64_t arena_bytes = 0;
  for (const SlotSpec& s : flat) {
    if (s.t->empty() || s.t->data == nullptr ||
        is_f32_upload(*s.t, s.force_f32))
      continue;
    arena_bytes += Align256(s.t->SizeBytes());
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

  // Upload tensors (raw quantized bytes into the arena; F32 from dense
  // or forced dequant into individual allocations). The returned type is the
  // device-side ggml type (0 = F32) — what the GPU kernels must dispatch on.
  //
  // Copies go through a pinned staging buffer: direct pageable H2D copies
  // from the GGUF file mapping into the huge arena stall for tens of seconds
  // at a time, while mmap->pinned memcpy plus pinned DMA stays at disk speed.
  constexpr std::uint64_t kStagingBytes = 256ULL << 20;
  void* staging = nullptr;
  if (hipHostMalloc(&staging, kStagingBytes, hipHostMallocDefault) !=
      hipSuccess) {
    if (error_msg)
      *error_msg = "pinned staging alloc failed";
    return nullptr;
  }
  std::uint64_t arena_off = 0;

  auto upload_bytes = [&](float* dev, const void* src, std::uint64_t bytes) {
    const auto* s = static_cast<const char*>(src);
    auto* d = reinterpret_cast<char*>(dev);
    std::uint64_t off = 0;
    while (off < bytes) {
      const std::uint64_t chunk =
          std::min<std::uint64_t>(kStagingBytes, bytes - off);
      std::memcpy(staging, s + off, chunk);
      hipMemcpy(d + off, staging, chunk, hipMemcpyHostToDevice);
      off += chunk;
    }
  };

  auto upload = [&](const TensorRef& t,
                    bool force_f32) -> std::pair<float*, int> {
    if (t.empty() || t.data == nullptr) {
      fprintf(stderr, "[glm5_gpu] upload %s EMPTY\n",
              std::string(t.name).c_str());
      return {nullptr, 0};
    }
    const std::uint64_t bytes = t.SizeBytes();
    if (bytes == 0) {
      // Non-empty tensor with unknown block geometry is a wiring bug — never
      // silently skip it (the kernels would run with a null/garbage pointer).
      fprintf(stderr, "[glm5_gpu] upload %s UNSUPPORTED TYPE %d\n",
              std::string(t.name).c_str(), static_cast<int>(t.type));
      return {nullptr, -1};
    }
    fprintf(stderr, "[glm5_gpu] upload %s bytes=%lu force_f32=%d\n",
            std::string(t.name).c_str(), (unsigned long)bytes, (int)force_f32);
    float* dev = nullptr;

    if (is_f32_upload(t, force_f32)) {
      // Upload as F32 (dequantize if needed). Own allocation: no arena space.
      const auto t0 = std::chrono::steady_clock::now();
      std::vector<float> f32;
      DequantToHost(t, f32);
      const auto t1 = std::chrono::steady_clock::now();
      const std::uint64_t f32_bytes = f32.size() * sizeof(float);
      hipMalloc(&dev, f32_bytes);
      const auto t2 = std::chrono::steady_clock::now();
      upload_bytes(dev, f32.data(), f32_bytes);
      const auto t3 = std::chrono::steady_clock::now();
      auto ms = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(b - a)
            .count();
      };
      fprintf(stderr,
              "[glm5_gpu]   f32 dev=%p deq=%ldms malloc=%ldms copy=%ldms\n",
              (void*)dev, ms(t0, t1), ms(t1, t2), ms(t2, t3));
      return {dev, 0};
    }
    // Raw quantized bytes — kernels dequantize on-the-fly.
    const auto t0 = std::chrono::steady_clock::now();
    dev = reinterpret_cast<float*>(reinterpret_cast<char*>(engine->w_arena_) +
                                   arena_off);
    upload_bytes(dev, t.data, bytes);
    const auto t1 = std::chrono::steady_clock::now();
    arena_off += Align256(bytes);
    fprintf(
        stderr, "[glm5_gpu]   raw dev=%p type=%d copy=%ldms\n", (void*)dev,
        static_cast<int>(t.type),
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
    return {dev, static_cast<int>(t.type)};
  };

  std::vector<float*> flat_dev(flat.size(), nullptr);
  std::vector<int> flat_type(flat.size(), 0);
  for (std::size_t i = 0; i < flat.size(); ++i) {
    std::tie(flat_dev[i], flat_type[i]) = upload(*flat[i].t, flat[i].force_f32);
    if (flat_type[i] == -1) {
      if (error_msg)
        *error_msg = "unsupported quant type in weight tensor";
      return nullptr;
    }
  }
  engine->token_embd_ = flat_dev[0];
  engine->token_embd_type_ = flat_type[0];
  engine->output_head_ = flat_dev[1];
  engine->output_head_type_ = flat_type[1];
  engine->output_norm_ = flat_dev[2];
  fprintf(stderr,
          "[glm5_gpu] arena=%p token_embd=%p(output type %d) output_head=%p"
          " (type %d) output_norm=%p\n",
          (void*)engine->w_arena_, (void*)engine->token_embd_,
          engine->token_embd_type_, (void*)engine->output_head_,
          engine->output_head_type_, (void*)engine->output_norm_);

  std::vector<float*> layer_ptr(static_cast<std::size_t>(n_layers) * kSlotCount,
                                nullptr);
  std::vector<int> layer_type(static_cast<std::size_t>(n_layers) * kSlotCount,
                              0);
  for (std::uint32_t i = 0; i < n_layers && i < weights.layers.size(); ++i) {
    for (int k = 0; k < kSlotCount; ++k) {
      const std::size_t src = 3 + static_cast<std::size_t>(i) * kSlotCount + k;
      layer_ptr[static_cast<std::size_t>(i) * kSlotCount + k] = flat_dev[src];
      layer_type[static_cast<std::size_t>(i) * kSlotCount + k] = flat_type[src];
    }
  }
  engine->w_ptrs_ = std::move(layer_ptr);
  engine->w_types_ = std::move(layer_type);
  hipHostFree(staging);

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

  // ---- Shared kernel workspace (one allocation reused by every kernel call).
  {
    const std::uint64_t kda_ws = rocm::KdaScratchFloats(
        {engine->H_, engine->nh_, engine->dk_, 0, 0.0F, 0.0F}, 1);
    rocm::MhcGpuGeometry mg{};
    mg.hidden_size = engine->H_;
    mg.hc_count = engine->hc_;
    mg.hc_mix_dim = 24;
    const std::uint64_t mhc_ws = rocm::MhcScratchFloats(mg, 1);
    rocm::MoeGpuGeometry og{};
    og.hidden_size = engine->H_;
    og.num_experts = config.num_experts;
    og.expert_ff = config.expert_ff;
    og.shared_ff = config.shared_expert_ff;
    const std::uint64_t moe_ws = rocm::MoeScratchFloats(og);
    const std::uint64_t dense_ws = rocm::DenseScratchFloats(config.dense_ff);
    engine->ws_bytes_ =
        std::max({kda_ws, mhc_ws, moe_ws, dense_ws}) * sizeof(float);
    if (hipMalloc(&engine->ws_, engine->ws_bytes_) != hipSuccess) {
      if (error_msg)
        *error_msg = "kernel workspace alloc failed";
      return nullptr;
    }
  }
  fprintf(stderr, "[glm5_gpu] kernel workspace: %lu MiB\n",
          (unsigned long)(engine->ws_bytes_ / (1024 * 1024)));

  // ---- Per-layer DSA state: latent / indexer-key caches + kernel scratch.
  engine->mla_geom_.hidden_size = engine->H_;
  engine->mla_geom_.num_heads = config.num_attention_heads;
  engine->mla_geom_.qk_nope = config.qk_nope_dim;
  engine->mla_geom_.v_head_dim = config.v_head_dim;
  engine->mla_geom_.q_lora_rank = config.q_lora_rank;
  engine->mla_geom_.kv_lora_rank = config.kv_lora_rank;
  engine->mla_geom_.norm_eps = config.layer_norm_rms_eps;
  engine->mla_geom_.idx_n_head = config.indexer_heads;
  engine->mla_geom_.idx_key_dim = config.indexer_head_dim;
  engine->mla_geom_.idx_top_k = config.indexer_top_k;
  engine->mla_geom_.idx_kpool = config.indexer_kpool;
  engine->mla_geom_.idx_kpool_select_tail = true;
  engine->mla_state_.assign(n_layers, rocm::MlaGpuState{});
  const std::uint32_t max_pos =
      std::min<std::uint32_t>(config.context_length, kMaxDsaPos);
  for (std::uint32_t i = 0; i < n_layers; ++i) {
    if (!config.IsFullAttention(i))
      continue;
    auto& st = engine->mla_state_[i];
    st.max_pos = max_pos;
    const std::uint64_t cache_bytes =
        rocm::MlaCacheBytes(max_pos, engine->mla_geom_);
    void* caches = nullptr;
    if (hipMalloc(&caches, cache_bytes) != hipSuccess) {
      if (error_msg)
        *error_msg = "DSA cache alloc failed";
      return nullptr;
    }
    hipMemset(caches, 0, cache_bytes);
    st.latents = static_cast<float*>(caches);
    st.idx_ik = reinterpret_cast<float*>(static_cast<char*>(caches) +
                                         static_cast<std::size_t>(max_pos) *
                                             sizeof(float) *
                                             engine->mla_geom_.kv_lora_rank);
    st.idx_ig = reinterpret_cast<float*>(
        static_cast<char*>(caches) +
        static_cast<std::size_t>(max_pos) * sizeof(float) *
            (engine->mla_geom_.kv_lora_rank + engine->mla_geom_.idx_key_dim));
    const std::uint64_t scratch_bytes =
        rocm::MlaStateFloats(max_pos, engine->mla_geom_) * sizeof(float);
    if (hipMalloc(&st.scratch, scratch_bytes) != hipSuccess) {
      if (error_msg)
        *error_msg = "DSA scratch alloc failed";
      return nullptr;
    }
  }
  fprintf(stderr, "[glm5_gpu] DSA state done (max_pos=%u)\n", max_pos);

  fprintf(stderr, "[glm5_gpu] Create complete\n");
  return engine;
}

bool GpuEngine::StepLayers(std::uint32_t pos, hipStream_t stream) {
  // Maya step_layers: for each layer, hc_pre → attn → hc_post → hc_pre → ffn →
  // hc_post.
  float* R = state_;          // residual at state arena head.
  const std::uint32_t T = 1;  // one token at a time.

  for (std::uint32_t il = 0; il < config_.num_layers; ++il) {
    // The NextN draft block is drafter-only (Maya runs the trunk without it).
    if (config_.IsMtpLayer(il))
      continue;
    const std::size_t base = static_cast<std::size_t>(il) * kSlotCount;

    // 1. mHC read (attn mixer).
    rocm::MhcGpuGeometry mhc_geom;
    mhc_geom.hidden_size = H_;
    mhc_geom.hc_count = hc_;
    mhc_geom.hc_mix_dim = 24;
    mhc_geom.sinkhorn_iters = config_.hc_sinkhorn_iters;
    mhc_geom.eps = config_.hc_epsilon;
    mhc_geom.norm_eps = config_.layer_norm_rms_eps;

    rocm::MhcGpuWeights mhc_attn_w;
    mhc_attn_w.fn = w_ptrs_[base + kHcAttnFn];
    mhc_attn_w.scale = w_ptrs_[base + kHcAttnScale];
    mhc_attn_w.base = w_ptrs_[base + kHcAttnBase];

    float* mixed = sc_;
    float* post = sc_ + H_;
    float* comb = post + hc_;
    float* attn_out = comb + hc_ * hc_;

    if (!rocm::MhcPreGpu(mhc_geom, mhc_attn_w, R, mixed, post, comb, T, ws_,
                         stream))
      return false;

    // 2. Attention (KDA mixer or NoPE-MLA + DSA indexer).
    if (config_.IsFullAttention(il)) {
      rocm::MlaGpuWeights mla_w;
      mla_w.q_a = w_ptrs_[base + kMlaQa];
      mla_w.q_a_type = w_types_[base + kMlaQa];
      mla_w.q_a_norm = w_ptrs_[base + kMlaQaNorm];
      mla_w.q_b = w_ptrs_[base + kMlaQb];
      mla_w.q_b_type = w_types_[base + kMlaQb];
      mla_w.kv_a_mqa = w_ptrs_[base + kMlaKvAMqa];
      mla_w.kv_a_mqa_type = w_types_[base + kMlaKvAMqa];
      mla_w.kv_a_norm = w_ptrs_[base + kMlaKvANorm];
      mla_w.k_b = w_ptrs_[base + kMlaKb];
      mla_w.k_b_type = w_types_[base + kMlaKb];
      mla_w.v_b = w_ptrs_[base + kMlaVb];
      mla_w.v_b_type = w_types_[base + kMlaVb];
      mla_w.output = w_ptrs_[base + kMlaOut];
      mla_w.output_type = w_types_[base + kMlaOut];
      mla_w.idx_q_b = w_ptrs_[base + kIdxQb];
      mla_w.idx_q_b_type = w_types_[base + kIdxQb];
      mla_w.idx_k = w_ptrs_[base + kIdxK];
      mla_w.idx_k_type = w_types_[base + kIdxK];
      mla_w.idx_k_norm_w = w_ptrs_[base + kIdxKNormW];
      mla_w.idx_k_norm_b = w_ptrs_[base + kIdxKNormB];
      mla_w.idx_proj = w_ptrs_[base + kIdxProj];
      mla_w.idx_proj_type = w_types_[base + kIdxProj];
      mla_w.idx_ape = w_ptrs_[base + kIdxApe];
      mla_w.idx_gate = w_ptrs_[base + kIdxGate];
      mla_w.idx_gate_type = w_types_[base + kIdxGate];

      if (!rocm::MlaStepGpu(mla_geom_, mla_w, mla_state_[il], mixed, attn_out,
                            pos, stream))
        return false;
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
      kda_w.w_q = w_ptrs_[base + kKdaQ];
      kda_w.w_k = w_ptrs_[base + kKdaK];
      kda_w.w_v = w_ptrs_[base + kKdaV];
      kda_w.w_out = w_ptrs_[base + kKdaOut];
      // Quant types come from the loaded tensors, not hardcoded constants.
      kda_w.w_q_type = w_types_[base + kKdaQ];
      kda_w.w_k_type = w_types_[base + kKdaK];
      kda_w.w_v_type = w_types_[base + kKdaV];
      kda_w.w_out_type = w_types_[base + kKdaOut];
      kda_w.conv_q = w_ptrs_[base + kKdaConvQ];
      kda_w.conv_k = w_ptrs_[base + kKdaConvK];
      kda_w.conv_v = w_ptrs_[base + kKdaConvV];
      kda_w.ssm_a = w_ptrs_[base + kKdaA];
      kda_w.dt_bias = w_ptrs_[base + kKdaDtBias];
      kda_w.f_a = w_ptrs_[base + kKdaFa];
      kda_w.f_b = w_ptrs_[base + kKdaFb];
      kda_w.ssm_beta = w_ptrs_[base + kKdaBeta];
      kda_w.g_a = w_ptrs_[base + kKdaGa];
      kda_w.g_b = w_ptrs_[base + kKdaGb];
      kda_w.ssm_norm = w_ptrs_[base + kKdaNorm];

      float* state_ptr =
          kda_S_off_[il] >= 0 ? state_ + kda_S_off_[il] : nullptr;
      if (!rocm::KdaPrefillGpu(kda_geom, kda_w, mixed, state_ptr, attn_out, T,
                               ws_, stream))
        return false;
    }

    // 3. mHC write (attn).
    float* R_new = state_ + hc_ * H_;
    if (!rocm::MhcPostGpu(mhc_geom, attn_out, R, post, comb, R_new, T, stream))
      return false;
    R = R_new;

    // 4. mHC read (FFN mixer) — uses the FFN mixer weights.
    rocm::MhcGpuWeights mhc_ffn_w;
    mhc_ffn_w.fn = w_ptrs_[base + kHcFfnFn];
    mhc_ffn_w.scale = w_ptrs_[base + kHcFfnScale];
    mhc_ffn_w.base = w_ptrs_[base + kHcFfnBase];
    if (!rocm::MhcPreGpu(mhc_geom, mhc_ffn_w, R, mixed, post, comb, T, ws_,
                         stream))
      return false;

    // 5. FFN (dense for leading layers, MoE otherwise).
    float* ffn_out = sc_ + H_ + hc_ + hc_ * hc_ + H_;
    if (config_.IsDenseFfn(il)) {
      if (!rocm::DenseFfnForwardGpu(
              H_, config_.dense_ff, 10.0f, w_ptrs_[base + kDenseGate],
              w_ptrs_[base + kDenseUp], w_ptrs_[base + kDenseDown],
              w_types_[base + kDenseGate], w_types_[base + kDenseUp],
              w_types_[base + kDenseDown], mixed, ffn_out, ws_, stream))
        return false;
    } else {
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
      moe_w.gate_inp = w_ptrs_[base + kMoeGateInp];
      moe_w.exp_probs_b = w_ptrs_[base + kMoeExpProbsB];
      moe_w.gate_exps = w_ptrs_[base + kMoeGateExps];
      moe_w.up_exps = w_ptrs_[base + kMoeUpExps];
      moe_w.down_exps = w_ptrs_[base + kMoeDownExps];
      moe_w.gate_shexp = w_ptrs_[base + kMoeGateShexp];
      moe_w.up_shexp = w_ptrs_[base + kMoeUpShexp];
      moe_w.down_shexp = w_ptrs_[base + kMoeDownShexp];
      // Quant types come from the loaded tensors (Maya-S: IQ2_XXS gate/up,
      // IQ2_S/IQ3_XXS/Q3_K down — but never hardcoded here).
      moe_w.gate_exps_type = w_types_[base + kMoeGateExps];
      moe_w.up_exps_type = w_types_[base + kMoeUpExps];
      moe_w.down_exps_type = w_types_[base + kMoeDownExps];

      if (!rocm::MoeForwardGpu(moe_geom, moe_w, mixed, ffn_out, ws_, stream))
        return false;
    }

    // 6. mHC write (FFN).
    R_new = (R == state_) ? state_ + hc_ * H_ : state_;
    if (!rocm::MhcPostGpu(mhc_geom, ffn_out, R, post, comb, R_new, T, stream))
      return false;
    R = R_new;
  }

  return true;
}

std::vector<float> GpuEngine::RunToken(std::uint32_t token, std::uint32_t pos) {
  // Embedding lookup: dequantize row `token` of token_embd to residual,
  // then copy to every hyper-connection stream.
  float* R = state_;
  if (token_embd_ && token < config_.vocab_size) {
    if (token_embd_type_ == 0) {
      hipMemcpy(R, token_embd_ + static_cast<std::size_t>(token) * H_,
                H_ * sizeof(float), hipMemcpyDeviceToDevice);
    } else if (!gpu::QuantEmbedLookup(token_embd_, R, H_, token,
                                      token_embd_type_, stream_)) {
      return std::vector<float>(H_, 0.0F);
    }
    for (std::uint32_t s = 1; s < hc_; ++s) {
      hipMemcpy(R + s * H_, R, H_ * sizeof(float), hipMemcpyDeviceToDevice);
    }
  }
  if (!StepLayers(pos, stream_)) {
    return std::vector<float>(H_, 0.0F);
  }
  // After an even number of mHC writes (2 per layer) the residual is back at
  // the head of the state arena.
  std::vector<float> hidden(H_);
  hipMemcpy(hidden.data(), state_, H_ * sizeof(float), hipMemcpyDeviceToHost);
  return hidden;
}

std::vector<float> GpuEngine::Prefill(std::span<const std::uint32_t> tokens) {
  if (tokens.empty())
    return std::vector<float>(H_, 0.0F);
  std::vector<float> hidden;
  for (std::uint32_t t = 0; t < tokens.size(); ++t) {
    hidden = RunToken(tokens[t], t);
    if (hidden.empty())
      return std::vector<float>(H_, 0.0F);
  }
  pos_ = static_cast<std::uint32_t>(tokens.size());
  return hidden;
}

std::vector<float> GpuEngine::Step(std::uint32_t token) {
  return RunToken(token, pos_++);
}

std::uint32_t GpuEngine::Decode(const std::vector<float>& hidden) {
  // RMS norm + output head + greedy argmax. The head stays quantized on the
  // GPU (QuantGemm dequantizes on-the-fly) — no F32 copy of [vocab, hidden].
  hipMemcpy(state_, hidden.data(), H_ * sizeof(float), hipMemcpyHostToDevice);
  gpu::LaunchRmsNorm(state_, output_norm_, H_, config_.layer_norm_rms_eps,
                     stream_);
  if (!gpu::QuantGemm(output_head_, state_, sc_, config_.vocab_size, H_,
                      output_head_type_, stream_)) {
    fprintf(stderr, "[glm5_gpu] output head GEMM failed (type %d)\n",
            output_head_type_);
    return 0;
  }
  int token_id = 0;
  gpu::LaunchArgmax(sc_, config_.vocab_size, &token_id, stream_);
  return static_cast<std::uint32_t>(token_id);
}

}  // namespace gufo::models::glm5_next
