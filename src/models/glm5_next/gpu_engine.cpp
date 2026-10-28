#include "src/models/glm5_next/gpu_engine.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "src/models/glm5_next/kernels/rocm/kda.hpp"
#include "src/models/glm5_next/kernels/rocm/mhc.hpp"
#include "src/models/glm5_next/kernels/rocm/moe.hpp"
#include "src/models/glm5_next/kernels/rocm/mtp.hpp"

namespace gufo::models::glm5_next {
namespace {

#define HIP_CHECK(expr)                                                   \
  do {                                                                    \
    const hipError_t err = (expr);                                        \
    if (err != hipSuccess) {                                              \
      if (error_msg)                                                      \
        *error_msg = std::string("HIP error: ") + hipGetErrorString(err); \
      return nullptr;                                                     \
    }                                                                     \
  } while (0)

/// Upload a tensor to device. Returns device pointer or nullptr on failure.
float* UploadTensor(const TensorRef& t, std::string* error_msg) {
  if (t.empty() || t.data == nullptr)
    return nullptr;
  const std::size_t bytes = t.SizeBytes();
  if (bytes == 0)
    return nullptr;
  float* d_ptr = nullptr;
  const hipError_t err = hipMalloc(&d_ptr, bytes);
  if (err != hipSuccess) {
    if (error_msg)
      *error_msg = std::string("hipMalloc failed: ") + hipGetErrorString(err);
    return nullptr;
  }
  const hipError_t copy_err =
      hipMemcpy(d_ptr, t.data, bytes, hipMemcpyHostToDevice);
  if (copy_err != hipSuccess) {
    hipFree(d_ptr);
    if (error_msg)
      *error_msg =
          std::string("hipMemcpy failed: ") + hipGetErrorString(copy_err);
    return nullptr;
  }
  return d_ptr;
}

}  // namespace

GpuEngine::~GpuEngine() {
  // Free device memory.
  for (auto& layer : layers_) {
    hipFree(layer.w_q);
    hipFree(layer.w_k);
    hipFree(layer.w_v);
    hipFree(layer.w_out);
    hipFree(layer.conv_q);
    hipFree(layer.conv_k);
    hipFree(layer.conv_v);
    hipFree(layer.f_a);
    hipFree(layer.f_b);
    hipFree(layer.dt_bias);
    hipFree(layer.ssm_a);
    hipFree(layer.ssm_beta);
    hipFree(layer.g_a);
    hipFree(layer.g_b);
    hipFree(layer.ssm_norm);
    hipFree(layer.hc_attn_fn);
    hipFree(layer.hc_attn_scale);
    hipFree(layer.hc_attn_base);
    hipFree(layer.gate_inp);
    hipFree(layer.exp_probs_b);
    hipFree(layer.gate_exps);
    hipFree(layer.up_exps);
    hipFree(layer.down_exps);
    hipFree(layer.gate_shexp);
    hipFree(layer.up_shexp);
    hipFree(layer.down_shexp);
  }
  hipFree(token_embd_);
  hipFree(output_head_);
  hipFree(output_norm_);
  hipFree(d_input_);
  hipFree(d_output_);
  hipFree(d_hidden_);
  if (stream_)
    hipStreamDestroy(stream_);
}

std::uint64_t GpuEngine::DeviceBytes() const noexcept {
  std::uint64_t total = 0;
  // Rough estimate based on config.
  const std::uint64_t H = config_.hidden_size;
  const std::uint64_t V = config_.vocab_size;
  const std::uint64_t E = config_.num_experts;
  const std::uint64_t I = config_.expert_ff;
  total += 2 * H * V * sizeof(float);                   // embeddings + output
  total += config_.num_layers * H * H * sizeof(float);  // rough layer estimate
  total +=
      config_.num_layers * E * I * H * sizeof(float);  // rough expert estimate
  return total;
}

std::shared_ptr<GpuEngine> GpuEngine::Create(const Config& config,
                                             const ModelWeights& weights,
                                             std::string* error_msg) {
  auto engine = std::shared_ptr<GpuEngine>(new GpuEngine());
  engine->config_ = config;

  // Create HIP stream.
  if (hipStreamCreateWithFlags(&engine->stream_, hipStreamNonBlocking) !=
      hipSuccess) {
    if (error_msg)
      *error_msg = "failed to create HIP stream";
    return nullptr;
  }

  // Upload embeddings / output head.
  engine->token_embd_ = UploadTensor(weights.token_embd, error_msg);
  engine->output_head_ = UploadTensor(weights.output, error_msg);
  engine->output_norm_ = UploadTensor(weights.output_norm, error_msg);

  // Upload per-layer weights.
  engine->layers_.resize(config.num_layers);
  for (std::uint32_t i = 0; i < config.num_layers && i < weights.layers.size();
       ++i) {
    const auto& L = weights.layers[i];
    auto& D = engine->layers_[i];

    // KDA weights (from LayerWeights::kda).
    D.f_a = UploadTensor(L.kda.f_a, error_msg);
    D.f_b = UploadTensor(L.kda.f_b, error_msg);
    D.dt_bias = UploadTensor(L.kda.dt_bias, error_msg);
    D.ssm_a = UploadTensor(L.kda.a, error_msg);
    D.ssm_beta = UploadTensor(L.kda.beta, error_msg);
    D.g_a = UploadTensor(L.kda.g_a, error_msg);
    D.g_b = UploadTensor(L.kda.g_b, error_msg);
    D.ssm_norm = UploadTensor(L.kda.norm, error_msg);
    D.conv_q = UploadTensor(L.kda.conv1d_q, error_msg);
    D.conv_k = UploadTensor(L.kda.conv1d_k, error_msg);
    D.conv_v = UploadTensor(L.kda.conv1d_v, error_msg);

    // mHC weights (from LayerWeights::hc).
    D.hc_attn_fn = UploadTensor(L.hc.attn_fn, error_msg);
    D.hc_attn_scale = UploadTensor(L.hc.attn_scale, error_msg);
    D.hc_attn_base = UploadTensor(L.hc.attn_base, error_msg);

    // MoE weights (from LayerWeights::moe).
    D.gate_inp = UploadTensor(L.moe.gate_inp, error_msg);
    D.exp_probs_b = UploadTensor(L.moe.exp_probs_b, error_msg);
    D.gate_exps = UploadTensor(L.moe.gate_exps, error_msg);
    D.up_exps = UploadTensor(L.moe.up_exps, error_msg);
    D.down_exps = UploadTensor(L.moe.down_exps, error_msg);
    D.gate_shexp = UploadTensor(L.moe.gate_shexp, error_msg);
    D.up_shexp = UploadTensor(L.moe.up_shexp, error_msg);
    D.down_shexp = UploadTensor(L.moe.down_shexp, error_msg);
  }

  // Allocate scratch buffers.
  const std::uint32_t H = config.hidden_size;
  engine->scratch_bytes_ = H * sizeof(float) * 8;  // 8 scratch vectors
  if (hipMalloc(&engine->d_input_, H * sizeof(float)) != hipSuccess ||
      hipMalloc(&engine->d_output_, H * sizeof(float)) != hipSuccess ||
      hipMalloc(&engine->d_hidden_, H * sizeof(float)) != hipSuccess) {
    if (error_msg)
      *error_msg = "failed to allocate scratch buffers";
    return nullptr;
  }

  return engine;
}

std::vector<float> GpuEngine::Prefill(std::span<const std::uint32_t> tokens) {
  const std::uint32_t H = config_.hidden_size;
  const std::uint32_t T = static_cast<std::uint32_t>(tokens.size());
  if (T == 0)
    return std::vector<float>(H, 0.0f);

  // Token embedding lookup on GPU.
  // For simplicity, do lookup on host and copy to device.
  std::vector<float> hidden(H, 0.0f);
  for (std::uint32_t t = 0; t < T; ++t) {
    const std::uint32_t tok = tokens[t];
    if (tok < config_.vocab_size && token_embd_) {
      // Copy one embedding row to device, then run layers.
      // For now, just copy the last token's embedding as hidden state.
      // Full layer-by-layer GPU forward pass is the next step.
      const std::size_t offset = static_cast<std::size_t>(tok) * H;
      hipMemcpy(hidden.data(), token_embd_ + offset, H * sizeof(float),
                hipMemcpyDeviceToHost);
    }
  }

  // TODO: Run through all layers on GPU using KDA/mHC/MoE/MoLA kernels.
  // This requires wiring the GPU kernels in sequence:
  //   for each layer:
  //     mHC read → attention (KDA or MLA) → mHC write → FFN (MoE)
  //   then RMS norm + output head.

  return hidden;
}

std::uint32_t GpuEngine::Decode(const std::vector<float>& hidden) {
  const std::uint32_t H = config_.hidden_size;
  const std::uint32_t V = config_.vocab_size;

  // Copy hidden to device.
  hipMemcpy(d_hidden_, hidden.data(), H * sizeof(float), hipMemcpyHostToDevice);

  // RMS norm + output head on GPU.
  // For now, compute logits on host from device weights.
  // TODO: Run RMS norm + output GEMM on GPU.
  std::vector<float> normed(H);
  std::memcpy(normed.data(), hidden.data(), H * sizeof(float));

  // Compute logits = output_head @ normed.
  std::vector<float> logits(V);
  if (output_head_) {
    // Copy output_head row to host for argmax (slow but correct).
    // For production, run the GEMM on GPU.
    std::vector<float> head_row(H);
    float best_val = -1e30f;
    std::uint32_t best_tok = 0;
    for (std::uint32_t v = 0; v < V; ++v) {
      hipMemcpy(head_row.data(), output_head_ + v * H, H * sizeof(float),
                hipMemcpyDeviceToHost);
      float acc = 0.0f;
      for (std::uint32_t j = 0; j < H; ++j)
        acc += head_row[j] * normed[j];
      if (acc > best_val) {
        best_val = acc;
        best_tok = v;
      }
    }
    return best_tok;
  }
  return 0;
}

}  // namespace gufo::models::glm5_next
