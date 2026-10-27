#include "src/models/glm5_next/config.hpp"

#include <unistd.h>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <variant>
#include <vector>

#include "src/models/glm5_next/weights.hpp"

namespace glm5 = gufo::models::glm5_next;

namespace {

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

// Build a minimal metadata-only GGUF (no tensor data) for config parsing tests.
using Value = std::variant<std::uint32_t, std::uint64_t, float, bool,
                           std::string, std::vector<std::uint64_t>>;

void WriteU32(std::ostream& out, std::uint32_t v) {
  out.write(reinterpret_cast<const char*>(&v), 4);
}
void WriteU64(std::ostream& out, std::uint64_t v) {
  out.write(reinterpret_cast<const char*>(&v), 8);
}
void WriteF32(std::ostream& out, float v) {
  out.write(reinterpret_cast<const char*>(&v), 4);
}
void WriteString(std::ostream& out, const std::string& s) {
  WriteU64(out, s.size());
  out.write(s.data(), static_cast<std::streamsize>(s.size()));
}

void WriteMetaEntry(std::ostream& out, const std::string& key,
                    const Value& val) {
  WriteString(out, key);
  if (const auto* u = std::get_if<std::uint32_t>(&val)) {
    WriteU32(out, 4);  // kUint32
    WriteU32(out, *u);
  } else if (const auto* u64 = std::get_if<std::uint64_t>(&val)) {
    WriteU32(out, 10);  // kUint64
    WriteU64(out, *u64);
  } else if (const auto* f = std::get_if<float>(&val)) {
    WriteU32(out, 6);  // kFloat32
    WriteF32(out, *f);
  } else if (const auto* b = std::get_if<bool>(&val)) {
    WriteU32(out, 7);  // kBool
    out.put(*b ? 1 : 0);
  } else if (const auto* s = std::get_if<std::string>(&val)) {
    WriteU32(out, 8);  // kString
    WriteString(out, *s);
  } else if (const auto* arr = std::get_if<std::vector<std::uint64_t>>(&val)) {
    WriteU32(out, 9);   // kArray
    WriteU32(out, 10);  // element type = uint64
    WriteU64(out, arr->size());
    for (auto v : *arr)
      WriteU64(out, v);
  }
}

void WriteGguf(const std::string& path,
               const std::map<std::string, Value>& meta) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  WriteU32(out, 0x46554747);  // GGUF magic
  WriteU32(out, 3);           // version
  WriteU64(out, 0);           // tensor count
  WriteU64(out, meta.size());
  for (const auto& [k, v] : meta)
    WriteMetaEntry(out, k, v);
  // Pad to 32-byte alignment.
  while (out.tellp() % 32)
    out.put(0);
}

const std::map<std::string, Value> kValidMeta = {
    {"general.architecture", std::string("glm5-next")},
    {"glm5-next.block_count", std::uint32_t{46}},
    {"glm5-next.embedding_length", std::uint32_t{4096}},
    {"glm5-next.vocab_size", std::uint32_t{154880}},
    {"glm5-next.context_length", std::uint64_t{1048576}},
    {"glm5-next.leading_dense_block_count", std::uint32_t{3}},
    {"glm5-next.attention.layer_norm_rms_epsilon", 1e-5f},
    {"glm5-next.attention.layer_norm_epsilon", 1e-6f},
    {"glm5-next.hyper_connection.count", std::uint32_t{4}},
    {"glm5-next.hyper_connection.sinkhorn_iterations", std::uint32_t{20}},
    {"glm5-next.hyper_connection.epsilon", 1e-6f},
    {"glm5-next.kda.head_dim", std::uint32_t{128}},
    {"glm5-next.ssm.conv_kernel", std::uint32_t{4}},
    {"glm5-next.kda.gate_lower_bound", -5.0f},
    {"glm5-next.attention.head_count", std::uint32_t{64}},
    {"glm5-next.attention.q_lora_rank", std::uint32_t{1536}},
    {"glm5-next.attention.kv_lora_rank", std::uint32_t{512}},
    {"glm5-next.attention.key_length_mla", std::uint32_t{256}},
    {"glm5-next.attention.value_length_mla", std::uint32_t{256}},
    {"glm5-next.attention.indexer.head_count", std::uint32_t{32}},
    {"glm5-next.attention.indexer.key_length", std::uint32_t{128}},
    {"glm5-next.attention.indexer.top_k", std::uint32_t{2048}},
    {"glm5-next.attention.indexer.kpool", std::uint32_t{4}},
    {"glm5-next.expert_count", std::uint32_t{288}},
    {"glm5-next.expert_used_count", std::uint32_t{8}},
    {"glm5-next.expert_feed_forward_length", std::uint32_t{2048}},
    {"glm5-next.expert_shared_count", std::uint32_t{1}},
    {"glm5-next.expert_shared_feed_forward_length", std::uint32_t{2048}},
    {"glm5-next.expert_gating_func", std::uint32_t{2}},
    {"glm5-next.expert_weights_scale", 2.5f},
    {"glm5-next.expert_weights_norm", true},
    {"glm5-next.feed_forward_length", std::uint32_t{12288}},
    {"glm5-next.nextn_predict_layers", std::uint32_t{1}},
    {"glm5-next.attention.head_count_kv",
     std::vector<std::uint64_t>{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
                                0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
                                0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1}},
};

std::optional<glm5::Config> Parse(const std::map<std::string, Value>& meta) {
  char tmpl[] = "/tmp/gufo-glm5-config-XXXXXX";
  const int fd = mkstemp(tmpl);
  assert(fd >= 0);
  ::close(fd);
  WriteGguf(tmpl, meta);
  std::string err;
  auto reader = gufo::core::GgufReader::OpenFile(tmpl, &err);
  std::optional<glm5::Config> result;
  if (reader)
    result = glm5::Config::FromGguf(*reader, &err);
  ::unlink(tmpl);
  return result;
}

void TestValidConfig() {
  auto c = Parse(kValidMeta);
  Require(c.has_value(), "valid metadata rejected");
  Require(c->num_layers == 46, "wrong block_count");
  Require(c->hidden_size == 4096, "wrong hidden_size");
  Require(c->vocab_size == 154880, "wrong vocab_size");
  Require(c->context_length == 1048576, "wrong context_length");
  Require(c->leading_dense_layers == 3, "wrong leading_dense_layers");
  Require(c->hc_count == 4, "wrong hc_count");
  Require(c->hc_sinkhorn_iters == 20, "wrong sinkhorn_iters");
  Require(c->kda_head_dim == 128, "wrong kda_head_dim");
  Require(c->kda_conv_kernel == 4, "wrong kda_conv_kernel");
  Require(c->kda_gate_lower_bound == -5.0f, "wrong gate_lower_bound");
  Require(c->num_attention_heads == 64, "wrong num_attention_heads");
  Require(c->q_lora_rank == 1536, "wrong q_lora_rank");
  Require(c->kv_lora_rank == 512, "wrong kv_lora_rank");
  Require(c->qk_nope_dim == 256, "wrong qk_nope_dim");
  Require(c->v_head_dim == 256, "wrong v_head_dim");
  Require(c->indexer_heads == 32, "wrong indexer_heads");
  Require(c->indexer_top_k == 2048, "wrong indexer_top_k");
  Require(c->indexer_kpool == 4, "wrong indexer_kpool");
  Require(c->num_experts == 288, "wrong num_experts");
  Require(c->num_experts_used == 8, "wrong num_experts_used");
  Require(c->expert_ff == 2048, "wrong expert_ff");
  Require(c->num_shared_experts == 1, "wrong num_shared_experts");
  Require(c->shared_expert_ff == 2048, "wrong shared_expert_ff");
  Require(c->expert_gating_func == 2, "wrong gating_func");
  Require(c->expert_weights_scale == 2.5f, "wrong expert_weights_scale");
  Require(c->expert_weights_norm, "wrong expert_weights_norm");
  Require(c->dense_ff == 12288, "wrong dense_ff");
  Require(c->nextn_layers == 1, "wrong nextn_layers");

  // Layer cadence.
  Require(c->is_full_attention.size() == 46, "wrong head_count_kv size");
  Require(c->FullAttentionCount() == 12, "wrong DSA layer count");
  Require(c->KdaLayerCount() == 34, "wrong KDA layer count");
  Require(c->IsFullAttention(3), "layer 3 should be DSA");
  Require(!c->IsFullAttention(0), "layer 0 should be KDA");
  Require(c->IsFullAttention(45), "layer 45 should be DSA");
  Require(c->IsDenseFfn(0), "layer 0 should be dense FFN");
  Require(c->IsDenseFfn(2), "layer 2 should be dense FFN");
  Require(!c->IsDenseFfn(3), "layer 3 should be MoE");
  Require(c->IsMtpLayer(45), "layer 45 should be MTP");

  std::cout << "TestValidConfig: PASS\n";
}

void TestWrongArchitecture() {
  auto meta = kValidMeta;
  meta["general.architecture"] = std::string("qwen4exp");
  auto c = Parse(meta);
  Require(!c.has_value(), "wrong architecture accepted");
  std::cout << "TestWrongArchitecture: PASS\n";
}

void TestMissingRequired() {
  {
    auto meta = kValidMeta;
    meta.erase("glm5-next.block_count");
    Require(!Parse(meta).has_value(), "missing block_count accepted");
  }
  {
    auto meta = kValidMeta;
    meta.erase("glm5-next.embedding_length");
    Require(!Parse(meta).has_value(), "missing embedding_length accepted");
  }
  {
    auto meta = kValidMeta;
    meta.erase("glm5-next.expert_count");
    Require(!Parse(meta).has_value(), "missing expert_count accepted");
  }
  {
    auto meta = kValidMeta;
    meta.erase("glm5-next.attention.head_count_kv");
    Require(!Parse(meta).has_value(), "missing head_count_kv accepted");
  }
  std::cout << "TestMissingRequired: PASS\n";
}

void TestInvalidValues() {
  {
    auto meta = kValidMeta;
    meta["glm5-next.expert_gating_func"] = std::uint32_t{1};  // softmax
    Require(!Parse(meta).has_value(), "softmax gating accepted");
  }
  {
    auto meta = kValidMeta;
    meta["glm5-next.expert_used_count"] = std::uint32_t{999};
    Require(!Parse(meta).has_value(), "expert_used_count > expert_count");
  }
  {
    auto meta = kValidMeta;
    meta["glm5-next.hyper_connection.count"] = std::uint32_t{1};
    Require(!Parse(meta).has_value(), "hc_count=1 accepted");
  }
  std::cout << "TestInvalidValues: PASS\n";
}

void TestResidentEstimate() {
  auto c = Parse(kValidMeta);
  Require(c.has_value(), "valid metadata rejected");
  const auto bytes = c->EstimateResidentBytes();
  // Maya-S v2 is ~90 GiB of tensors. The estimate should be within 2x.
  constexpr std::uint64_t kGiB = 1ULL << 30;
  Require(bytes > 40 * kGiB, "estimate too low");
  Require(bytes < 200 * kGiB, "estimate too high");
  std::cout << "TestResidentEstimate: PASS (" << bytes / kGiB << " GiB)\n";
}

void TestTensorRowBytes() {
  // Q6_K: 256 elements = 210 bytes.
  Require(glm5::TensorRowBytes(gufo::core::GgmlType::kQ6_K, 256) == 210,
          "Q6_K row bytes wrong");
  Require(glm5::TensorRowBytes(gufo::core::GgmlType::kQ6_K, 512) == 420,
          "Q6_K 512-elem row bytes wrong");
  // F32: 4 bytes/elem.
  Require(glm5::TensorRowBytes(gufo::core::GgmlType::kF32, 128) == 512,
          "F32 row bytes wrong");
  // IQ2_XXS: 256 elements = 66 bytes.
  Require(glm5::TensorRowBytes(gufo::core::GgmlType::kIQ2_XXS, 256) == 66,
          "IQ2_XXS row bytes wrong");
  std::cout << "TestTensorRowBytes: PASS\n";
}

}  // namespace

int main() {
  try {
    TestValidConfig();
    TestWrongArchitecture();
    TestMissingRequired();
    TestInvalidValues();
    TestResidentEstimate();
    TestTensorRowBytes();
    std::cout << "All glm5_next config tests passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
