// GPU quantized-GEMM parity tests for GLM-5.3-Flash.
//
// Every supported ggml storage type is compared against the host reference
// (gufo::quant::Dot / Dequantize — the llama.cpp-accurate dequant paths), plus
// hand-computed golden blocks that pin the IQ2_XXS semantics and the
// codebook tables.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/quant/ggml_gemm.hpp"
#include "src/core/quant/iq_grids.hpp"
#include "src/models/glm5_next/quant_gemv.hpp"

namespace gpu = gufo::models::glm5_next::gpu;
namespace quant = gufo::quant;

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

// ---- random block builders -------------------------------------------------

// F16 bits for a value in ~[0.06, 2): normal (so host and device F16 decode
// agree bit-for-bit) with a small magnitude so FP32 accumulation of 32 blocks
// stays well within tolerance.
std::uint16_t RandF16Bits(std::mt19937& rng) {
  const std::uint32_t sign = (rng() & 1U) << 15U;
  const std::uint32_t exp = 11U + (rng() % 5U);  // 2^-4 .. 2^0
  const std::uint32_t mant = rng() % 1024U;
  return static_cast<std::uint16_t>(sign | (exp << 10U) | mant);
}

std::size_t BlockBytes(int type) {
  switch (type) {
    case 0:
      return 1024;
    case 1:
      return 512;
    case 8:
      return 34;
    case 10:
      return 84;
    case 11:
      return 110;
    case 14:
      return 210;
    case 16:
      return 66;
    case 18:
      return 98;
    case 22:
      return 82;
    default:
      return 0;
  }
}

std::size_t BlockElems(int type) {
  return type == 8 ? 32 : 256;
}

void FillRandomBlock(int type, std::mt19937& rng, std::uint8_t* out) {
  const std::size_t bytes = BlockBytes(type);
  for (std::size_t i = 0; i < bytes; ++i)
    out[i] = static_cast<std::uint8_t>(rng());
  if (type == 0) {
    auto* f = reinterpret_cast<float*>(out);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    for (int i = 0; i < 256; ++i)
      f[i] = dist(rng);
    return;
  }
  if (type == 1) {
    auto* h = reinterpret_cast<std::uint16_t*>(out);
    for (int i = 0; i < 256; ++i)
      h[i] = RandF16Bits(rng);
    return;
  }
  // F16 scale placement follows the block layout (Q6_K/Q3_K/Q2_K carry d
  // last; Q2_K also has dmin after d).
  const std::uint16_t d = RandF16Bits(rng);
  const std::size_t d_off = (type == 14)   ? 208
                            : (type == 11) ? 108
                            : (type == 10) ? 80
                                           : 0;
  std::memcpy(out + d_off, &d, 2);
  if (type == 10) {
    const std::uint16_t dmin = RandF16Bits(rng);
    std::memcpy(out + 82, &dmin, 2);
  }
  if (type == 8) {
    for (int i = 0; i < 32; ++i)
      out[2 + i] = static_cast<std::uint8_t>(rng());
  }
}

// ---- tests -----------------------------------------------------------------

void TestIqTables() {
  // Spot values checked against llama.cpp ggml-common.h.
  Require(quant::kIq2xxsGrid[0] == 0x0808080808080808ULL, "iq2xxs_grid[0]");
  Require(quant::kIq2xxsGrid[1] == 0x080808080808082bULL, "iq2xxs_grid[1]");
  Require(quant::kIq3xxsGrid[0] == 0x04040404ULL, "iq3xxs_grid[0]");
  // ksigns_iq2xs: v with bit 7 set iff v has odd parity.
  Require(quant::IqSigns(0) == 0, "IqSigns(0)");
  Require(quant::IqSigns(1) == 129, "IqSigns(1)");
  Require(quant::IqSigns(3) == 3, "IqSigns(3)");
  Require(quant::IqSigns(7) == 135, "IqSigns(7)");
  Require(quant::IqSigns(127) == 255, "IqSigns(127)");
  std::printf("ok  iq tables\n");
}

// Hand-computed golden: one IQ2_XXS block with code 0 everywhere (grid value
// 8 per element). Group 0's scale nibble is 15 -> db = (0.5+15)*0.25 = 3.875
// -> value 31.0; the other groups carry scale 0 -> db = 0.125 -> value 1.0.
// No sign bits are set anywhere.
void TestIq2XxsGolden() {
  std::uint8_t blk[66] = {0};
  blk[0] = 0x00;
  blk[1] = 0x3C;      // F16 1.0
  blk[2 + 7] = 0xF0;  // group 0 second uint32 = 0xF0000000: scale nibble 15.
  std::vector<float> out(256);
  quant::DequantizeIQ2_XXS(blk, out.data(), 256);
  for (int i = 0; i < 256; ++i) {
    const float want = (i < 32) ? 31.0F : 1.0F;
    Require(std::abs(out[i] - want) < 1e-6F, "iq2 golden host [" +
                                                 std::to_string(i) +
                                                 "]=" + std::to_string(out[i]));
  }

  std::uint8_t* d_blk = nullptr;
  HIP_CHECK(hipMalloc(&d_blk, sizeof(blk)));
  HIP_CHECK(hipMemcpy(d_blk, blk, sizeof(blk), hipMemcpyHostToDevice));
  float* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, 256 * sizeof(float)));
  Require(gpu::QuantEmbedLookup(d_blk, d_out, 256, 0, 16, nullptr),
          "QuantEmbedLookup launch");
  std::vector<float> got(256);
  HIP_CHECK(
      hipMemcpy(got.data(), d_out, 256 * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_blk));
  for (int i = 0; i < 256; ++i) {
    const float want = (i < 32) ? 31.0F : 1.0F;
    Require(
        std::abs(got[i] - want) < 1e-6F,
        "iq2 golden gpu [" + std::to_string(i) + "]=" + std::to_string(got[i]));
  }
  std::printf("ok  iq2_xxs golden block\n");
}

void TestQuantGemmParity(int type, int m, int n, std::uint32_t seed) {
  std::mt19937 rng(seed);
  const std::size_t blk_bytes = BlockBytes(type);
  const std::size_t blk_elems = BlockElems(type);
  const std::size_t n_blocks = n / blk_elems;
  std::vector<std::uint8_t> w(static_cast<std::size_t>(m) * n_blocks *
                              blk_bytes);
  for (int r = 0; r < m; ++r) {
    for (std::size_t b = 0; b < n_blocks; ++b) {
      FillRandomBlock(
          type, rng,
          w.data() + (static_cast<std::size_t>(r) * n_blocks + b) * blk_bytes);
    }
  }
  std::vector<float> x(n);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  for (auto& v : x)
    v = dist(rng);

  std::vector<float> ref(m);
  for (int r = 0; r < m; ++r) {
    const void* row =
        w.data() + static_cast<std::size_t>(r) * n_blocks * blk_bytes;
    ref[r] = quant::Dot(static_cast<gufo::core::GgmlType>(type), row,
                        std::span<const float>(x), n);
  }

  void* d_w = nullptr;
  float *d_x = nullptr, *d_y = nullptr;
  HIP_CHECK(hipMalloc(&d_w, w.size()));
  HIP_CHECK(hipMemcpy(d_w, w.data(), w.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&d_x, n * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_x, x.data(), n * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&d_y, m * sizeof(float)));
  Require(gpu::QuantGemm(d_w, d_x, d_y, m, n, type, nullptr), "QuantGemm");
  std::vector<float> got(m);
  HIP_CHECK(
      hipMemcpy(got.data(), d_y, m * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_y));

  for (int r = 0; r < m; ++r) {
    const float tol = 1e-3F * (1.0F + std::abs(ref[r]));
    Require(std::abs(got[r] - ref[r]) <= tol,
            "type " + std::to_string(type) + " m=" + std::to_string(m) + " n=" +
                std::to_string(n) + " row " + std::to_string(r) + ": got " +
                std::to_string(got[r]) + " want " + std::to_string(ref[r]));
  }
  std::printf("ok  QuantGemm type=%d m=%d n=%d\n", type, m, n);
}

void TestQuantGemmMoeParity(int type, std::uint32_t seed) {
  std::mt19937 rng(seed);
  const int E = 3, m = 5, n = 512, top_k = 2;
  const std::size_t blk_bytes = BlockBytes(type);
  const std::size_t blk_elems = BlockElems(type);
  const std::size_t n_blocks = n / blk_elems;
  std::vector<std::uint8_t> w(static_cast<std::size_t>(E) * m * n_blocks *
                              blk_bytes);
  for (std::size_t i = 0; i < w.size(); ++i)
    w[i] = static_cast<std::uint8_t>(rng());
  // Rebuild with valid F16 scales per block.
  for (int e = 0; e < E; ++e) {
    for (int r = 0; r < m; ++r) {
      for (std::size_t b = 0; b < n_blocks; ++b) {
        FillRandomBlock(
            type, rng,
            w.data() +
                (((static_cast<std::size_t>(e) * m + r) * n_blocks) + b) *
                    blk_bytes);
      }
    }
  }
  std::vector<float> x(n);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  for (auto& v : x)
    v = dist(rng);
  const std::vector<std::int32_t> ids = {2, 0};

  std::vector<float> ref(top_k * m);
  const std::size_t expert_bytes = gpu::QuantMatrixBytes(m, n, type);
  for (int slot = 0; slot < top_k; ++slot) {
    for (int r = 0; r < m; ++r) {
      const void* row = w.data() + ids[slot] * expert_bytes +
                        static_cast<std::size_t>(r) * n_blocks * blk_bytes;
      ref[slot * m + r] = quant::Dot(static_cast<gufo::core::GgmlType>(type),
                                     row, std::span<const float>(x), n);
    }
  }

  void* d_w = nullptr;
  float *d_x = nullptr, *d_y = nullptr;
  std::int32_t* d_ids = nullptr;
  HIP_CHECK(hipMalloc(&d_w, w.size()));
  HIP_CHECK(hipMemcpy(d_w, w.data(), w.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&d_x, n * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_x, x.data(), n * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&d_ids, top_k * sizeof(std::int32_t)));
  HIP_CHECK(hipMemcpy(d_ids, ids.data(), top_k * sizeof(std::int32_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&d_y, top_k * m * sizeof(float)));
  Require(
      gpu::QuantGemmMoe(d_w, d_x, d_ids, d_y, m, n, E, top_k, type, nullptr),
      "QuantGemmMoe");
  std::vector<float> got(top_k * m);
  HIP_CHECK(hipMemcpy(got.data(), d_y, top_k * m * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_ids));
  HIP_CHECK(hipFree(d_y));

  for (int i = 0; i < top_k * m; ++i) {
    const float tol = 1e-3F * (1.0F + std::abs(ref[i]));
    Require(std::abs(got[i] - ref[i]) <= tol,
            "moe type " + std::to_string(type) + " idx " + std::to_string(i) +
                ": got " + std::to_string(got[i]) + " want " +
                std::to_string(ref[i]));
  }
  std::printf("ok  QuantGemmMoe type=%d\n", type);
}

void TestEmbedLookupParity(int type, std::uint32_t seed) {
  std::mt19937 rng(seed);
  const int vocab = 6, hidden = 512, token = 4;
  const std::size_t blk_bytes = BlockBytes(type);
  const std::size_t blk_elems = BlockElems(type);
  const std::size_t n_blocks = hidden / blk_elems;
  std::vector<std::uint8_t> w(static_cast<std::size_t>(vocab) * n_blocks *
                              blk_bytes);
  for (int r = 0; r < vocab; ++r) {
    for (std::size_t b = 0; b < n_blocks; ++b) {
      FillRandomBlock(
          type, rng,
          w.data() + (static_cast<std::size_t>(r) * n_blocks + b) * blk_bytes);
    }
  }
  std::vector<float> ref(hidden);
  quant::Dequantize(
      static_cast<gufo::core::GgmlType>(type),
      w.data() + static_cast<std::size_t>(token) * n_blocks * blk_bytes,
      ref.data(), hidden);

  void* d_w = nullptr;
  float* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_w, w.size()));
  HIP_CHECK(hipMemcpy(d_w, w.data(), w.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&d_out, hidden * sizeof(float)));
  Require(gpu::QuantEmbedLookup(d_w, d_out, hidden, token, type, nullptr),
          "QuantEmbedLookup");
  std::vector<float> got(hidden);
  HIP_CHECK(hipMemcpy(got.data(), d_out, hidden * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_out));

  for (int i = 0; i < hidden; ++i) {
    Require(std::abs(got[i] - ref[i]) <= 1e-5F * (1.0F + std::abs(ref[i])),
            "embed type " + std::to_string(type) + " [" + std::to_string(i) +
                "]: got " + std::to_string(got[i]) + " want " +
                std::to_string(ref[i]));
  }
  std::printf("ok  QuantEmbedLookup type=%d\n", type);
}

}  // namespace

int main() {
  try {
    TestIqTables();
    TestIq2XxsGolden();
    for (int type : {0, 1, 8, 10, 11, 14, 16, 18, 22}) {
      Require(gpu::QuantGemmSupported(type), "supported");
      TestQuantGemmParity(type, 5, 512, 100 + type);
      TestQuantGemmParity(type, 100, 1024, 200 + type);
    }
    TestQuantGemmParity(16, 70, 4096, 7);  // tpr=16, rpw=4
    TestQuantGemmParity(14, 3, 256, 8);    // single block per row
    TestQuantGemmMoeParity(16, 11);
    TestQuantGemmMoeParity(14, 12);
    TestQuantGemmMoeParity(18, 13);
    TestQuantGemmMoeParity(10, 14);
    TestQuantGemmMoeParity(11, 15);
    TestQuantGemmMoeParity(22, 16);
    TestEmbedLookupParity(16, 21);
    TestEmbedLookupParity(18, 22);
    TestEmbedLookupParity(14, 23);
    TestEmbedLookupParity(0, 24);
    TestEmbedLookupParity(22, 25);
    TestEmbedLookupParity(11, 26);
    TestEmbedLookupParity(10, 27);
    std::printf("ALL PASS\n");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
