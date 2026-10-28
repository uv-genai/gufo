#ifndef GUFO_MODELS_GLM5_NEXT_WEIGHTS_HPP_
#define GUFO_MODELS_GLM5_NEXT_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/glm5_next/config.hpp"

namespace gufo::models::glm5_next {

/// Non-owning view of one GGUF tensor. `rows` x `cols` follows the GGUF
/// convention: cols (ne[0]) is the contiguous reduction dimension, rows
/// (ne[1]) the output dimension, and `experts` (ne[2]) stacks whole matrices.
struct TensorRef {
  const void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint64_t cols{0};
  std::uint64_t rows{1};
  std::uint64_t experts{1};
  std::uint64_t file_offset{0};
  std::uint32_t shard{0};
  std::string_view name;

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  [[nodiscard]] std::uint64_t ElementCount() const noexcept {
    return cols * rows * experts;
  }
  [[nodiscard]] std::size_t RowBytes() const noexcept;
  [[nodiscard]] std::size_t SizeBytes() const noexcept {
    return RowBytes() * rows * experts;
  }
  [[nodiscard]] const std::uint8_t* Expert(std::uint64_t e) const noexcept {
    return static_cast<const std::uint8_t*>(data) + RowBytes() * rows * e;
  }
};

/// mHC (hyper-connection) mixer weights per layer.
/// The GGUF packs six F32 tensors per layer: base, fn, scale for attn and ffn.
struct HcWeights {
  TensorRef attn_base;   ///< [24] F32
  TensorRef attn_fn;     ///< [16384, 24] F32
  TensorRef attn_scale;  ///< [3] F32
  TensorRef ffn_base;    ///< [24] F32
  TensorRef ffn_fn;      ///< [16384, 24] F32
  TensorRef ffn_scale;   ///< [3] F32
};

/// KDA mixer weights (linear-attention layers).
struct KdaWeights {
  TensorRef a;         ///< [64] F32 (log-space decay)
  TensorRef norm;      ///< [128] F32
  TensorRef conv1d_q;  ///< [4, 1, 8192] F32
  TensorRef conv1d_k;  ///< [4, 1, 8192] F32
  TensorRef conv1d_v;  ///< [4, 1, 8192] F32
  TensorRef dt_bias;   ///< [8192] F32
  TensorRef beta;      ///< [4096, 64] Q8_0
  TensorRef f_a;       ///< [4096, 128] Q8_0
  TensorRef f_b;       ///< [128, 8192] Q8_0
  TensorRef g_a;       ///< [4096, 128] Q8_0
  TensorRef g_b;       ///< [128, 8192] Q8_0
  // Q/K/V projections (attention).
  TensorRef attn_q;    ///< [4096, 8192] Q6_K
  TensorRef attn_k;    ///< [4096, 8192] Q6_K
  TensorRef attn_v;    ///< [4096, 8192] Q6_K
  TensorRef attn_out;  ///< [8192, 4096] Q6_K
};

/// MLA attention weights (full-attention / DSA layers).
struct MlaWeights {
  TensorRef q_a;        ///< [4096, 1536] Q6_K
  TensorRef q_a_norm;   ///< [1536] F32
  TensorRef q_b;        ///< [1536, 16384] Q6_K
  TensorRef kv_a_mqa;   ///< [4096, 512] Q6_K
  TensorRef kv_a_norm;  ///< [512] F32
  TensorRef k_b;        ///< [256, 512, 64] Q6_K (packed per head)
  TensorRef v_b;        ///< [512, 256, 64] Q6_K
  TensorRef output;     ///< [16384, 4096] Q6_K
};

/// DSA indexer weights (present on DSA layers only).
struct DsaIndexerWeights {
  TensorRef q_b;              ///< [1536, 4096] Q8_0
  TensorRef k;                ///< [4096, 128] Q8_0
  TensorRef k_norm_weight;    ///< [128] F32
  TensorRef k_norm_bias;      ///< [128] F32
  TensorRef proj;             ///< [4096, 32] Q8_0
  TensorRef compressor_ape;   ///< [128, 4] F32
  TensorRef compressor_gate;  ///< [4096, 128] Q8_0
};

/// MoE weights for one layer.
struct MoeWeights {
  TensorRef gate_inp;     ///< [4096, 288] F32 (router)
  TensorRef exp_probs_b;  ///< [288] F32 (noaux_tc bias)
  TensorRef gate_exps;    ///< [4096, 2048, 288] IQ2_XXS
  TensorRef up_exps;      ///< [4096, 2048, 288] IQ2_XXS
  TensorRef down_exps;    ///< [2048, 4096, 288] IQ2_S/IQ3_XXS
  TensorRef gate_shexp;   ///< [4096, 2048] Q6_K
  TensorRef up_shexp;     ///< [4096, 2048] Q6_K
  TensorRef down_shexp;   ///< [2048, 4096] Q6_K
};

/// Dense FFN weights (leading layers only).
struct DenseFfnWeights {
  TensorRef gate;  ///< [4096, 12288] Q6_K
  TensorRef up;    ///< [4096, 12288] Q6_K
  TensorRef down;  ///< [12288, 4096] Q6_K
};

/// NextN MTP weights (on the last block).
struct MtpWeights {
  TensorRef enorm;             ///< [4096] F32
  TensorRef hnorm;             ///< [4096] F32
  TensorRef eh_proj;           ///< [8192, 4096] Q6_K ([embed|hidden] packed)
  TensorRef shared_head_norm;  ///< [4096] F32
};

/// All weights for one transformer block.
struct LayerWeights {
  bool full_attention{false};
  bool dense_ffn{false};
  bool mtp{false};

  TensorRef attn_norm;  ///< [4096] F32
  TensorRef ffn_norm;   ///< [4096] F32

  HcWeights hc;
  KdaWeights kda;
  MlaWeights mla;
  DsaIndexerWeights indexer;
  MoeWeights moe;
  DenseFfnWeights dense;
  MtpWeights mtp_weights;
};

/// Complete model weights, bound to a mapped GGUF.
struct ModelWeights {
  TensorRef token_embd;   ///< [4096, 154880] Q6_K
  TensorRef output;       ///< [4096, 154880] Q6_K
  TensorRef output_norm;  ///< [4096] F32

  std::vector<LayerWeights> layers;

  /// Total resident bytes across all bound tensors.
  [[nodiscard]] std::uint64_t ResidentBytes() const noexcept;
};

/// Load and bind all tensor references from a GGUF artifact.
/// Returns false on any missing/shape-mismatched tensor.
[[nodiscard]] bool LoadWeights(const core::GgufReader& gguf,
                               const Config& config, ModelWeights* out,
                               std::string* error_msg = nullptr);

/// Per-quant-type byte size of one row of `cols` elements.
[[nodiscard]] std::size_t TensorRowBytes(core::GgmlType type,
                                         std::uint64_t cols) noexcept;

}  // namespace gufo::models::glm5_next

#endif  // GUFO_MODELS_GLM5_NEXT_WEIGHTS_HPP_
