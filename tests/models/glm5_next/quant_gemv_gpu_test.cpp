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

// Host mirror of Q81StoreWarp: d rounded to f16 like the kernel stores it.
struct Q81Group {
  float d;
  std::int8_t qs[32];
};

std::uint16_t F32ToF16(float v) {
  std::uint32_t u;
  std::memcpy(&u, &v, 4);
  const std::uint32_t sign = (u >> 16) & 0x8000U;
  std::int32_t exp = static_cast<std::int32_t>((u >> 23) & 0xFF) - 127 + 15;
  std::uint32_t mant = u & 0x7FFFFFU;
  if (exp <= 0)
    return static_cast<std::uint16_t>(sign);
  if (exp >= 31)
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  // round-to-nearest-even (matches __float2half on device)
  const std::uint32_t rounded = mant + 0xFFFU + ((mant >> 13U) & 1U);
  if ((rounded & 0x800000U) != 0U) {
    ++exp;
    if (exp >= 31)
      return static_cast<std::uint16_t>(sign | 0x7C00U);
  }
  return static_cast<std::uint16_t>(sign | (exp << 10U) |
                                    ((rounded & 0x7FFFFFU) >> 13U));
}

float F16BitsToFloat(std::uint16_t h) {
  const std::uint32_t s = (h >> 15) & 1U;
  const std::uint32_t e = (h >> 10) & 0x1FU;
  const std::uint32_t m = h & 0x3FFU;
  if (e == 0)
    return (s ? -1.0f : 1.0f) * std::ldexp(m / 1024.0f, -14);
  if (e == 31)
    return m ? std::nanf("") : (s ? -INFINITY : INFINITY);
  return (s ? -1.0f : 1.0f) * std::ldexp(1.0f + m / 1024.0f, e - 15);
}

void QuantizeQ81Host(const std::vector<float>& x, std::vector<Q81Group>& out) {
  const std::size_t groups = x.size() / 32;
  out.resize(groups);
  for (std::size_t g = 0; g < groups; ++g) {
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i)
      amax = std::max(amax, std::abs(x[g * 32 + i]));
    const float d = std::min(amax / 127.0f, 65504.0f);
    out[g].d = F16BitsToFloat(F32ToF16(d));
    for (int i = 0; i < 32; ++i) {
      const float v = x[g * 32 + i];
      out[g].qs[i] = amax == 0.0f
                         ? 0
                         : static_cast<std::int8_t>(std::max(
                               -127.0f, std::min(127.0f, std::round(v / d))));
    }
  }
}

// Reference: host dequant (llama.cpp dequantize_row contract) dotted against
// the same Q8_1 activation the GPU kernel builds.
float Q8RefDot(int type, const void* row, std::size_t n,
               const std::vector<Q81Group>& q8) {
  std::vector<float> w(n);
  quant::Dequantize(static_cast<gufo::core::GgmlType>(type), row, w.data(), n);
  float s = 0.0f;
  for (std::size_t i = 0; i < n; ++i)
    s += w[i] * (static_cast<float>(q8[i / 32].qs[i % 32]) * q8[i / 32].d);
  return s;
}

// The IQ paths accumulate integers and apply the scale with integer division
// (llama vecdot / Maya glm_fast.cu — the production rounding). Mirror it here
// or the last-bit truncation shows up as spurious diffs on small sums.
float IqIntDotRef(int type, const void* row, std::size_t n,
                  const std::vector<Q81Group>& q8) {
  const std::uint8_t* w = static_cast<const std::uint8_t*>(row);
  const std::size_t nb = n / 256;
  double total = 0.0;  // accumulate calls in double, scale per call in float
  float acc = 0.0f;
  for (std::size_t kb = 0; kb < nb; ++kb) {
    const std::uint8_t* blk =
        w + kb * static_cast<std::size_t>(BlockBytes(type));
    const Q81Group* xg = &q8[kb * 8];
    for (int iqs = 0; iqs < 16; iqs += 2) {
      const Q81Group& x = xg[iqs / 2];
      int sumi = 0, sumi0 = 0, sumi1 = 0;
      float d_scale = 0.0f;
      if (type == 16) {
        // IQ2_XXS
        std::uint32_t q2, aux32;
        std::memcpy(&q2, blk + 2 + 4 * iqs, 4);
        std::memcpy(&aux32, blk + 2 + 4 * iqs + 4, 4);
        const auto* aux8 = reinterpret_cast<const std::uint8_t*>(&q2);
        for (int k0 = 0; k0 < 8; k0 += 2) {
          const std::uint64_t grid = quant::kIq2xxsGrid[aux8[k0 / 2]];
          const std::uint8_t signs =
              quant::IqSigns((aux32 >> (7 * k0 / 2)) & 0xFFU);
          for (int j = 0; j < 8; ++j) {
            int g = static_cast<std::int8_t>(
                static_cast<std::uint8_t>(grid >> (8 * j)));
            if (signs & (1U << j))
              g = -g;
            sumi += g * x.qs[4 * k0 + j];
          }
        }
        const int ls = static_cast<int>(aux32 >> 27) | 1;
        sumi = sumi * ls / 8;
        d_scale = quant::Fp16ToFloat(
            static_cast<std::uint16_t>(blk[0] | (blk[1] << 8)));
      } else if (type == 18) {
        // IQ3_XXS
        std::uint32_t a0, a1, aux32;
        std::memcpy(&a0, blk + 2 + 4 * iqs, 4);
        std::memcpy(&a1, blk + 2 + 4 * iqs + 4, 4);
        std::memcpy(&aux32, blk + 2 + 64 + 2 * iqs, 4);
        const auto* q3 = reinterpret_cast<const std::uint8_t*>(
            (std::uint64_t)a0 | ((std::uint64_t)a1 << 32));
        std::uint8_t q3b[8];
        std::memcpy(q3b, &a0, 4);
        std::memcpy(q3b + 4, &a1, 4);
        (void)q3;
        for (int l0 = 0; l0 < 8; l0 += 2) {
          const std::uint32_t signs =
              quant::IqSigns((aux32 >> (7 * l0 / 2)) & 0xFFU);
          const int g0 = static_cast<std::int8_t>(
              static_cast<std::uint8_t>(quant::kIq3xxsGrid[q3b[l0 + 0]]));
          const int g1 = static_cast<std::int8_t>(
              static_cast<std::uint8_t>(quant::kIq3xxsGrid[q3b[l0 + 1]]));
          for (int j = 0; j < 4; ++j) {
            int v0 = static_cast<std::int8_t>(static_cast<std::uint8_t>(
                quant::kIq3xxsGrid[q3b[l0 + 0]] >> (8 * j)));
            int v1 = static_cast<std::int8_t>(static_cast<std::uint8_t>(
                quant::kIq3xxsGrid[q3b[l0 + 1]] >> (8 * j)));
            if (signs & (1U << j))
              v0 = -v0;
            if (signs & (1U << (j + 4)))
              v1 = -v1;
            sumi += v0 * x.qs[4 * l0 + j] + v1 * x.qs[4 * l0 + j + 4];
          }
          (void)g0;
          (void)g1;
        }
        const int ls = static_cast<int>(aux32 >> 28);
        sumi = (ls * sumi + sumi / 2) / 2;
        d_scale = quant::Fp16ToFloat(
            static_cast<std::uint16_t>(blk[0] | (blk[1] << 8)));
      } else {
        // IQ2_S
        std::uint32_t qp, sp;
        const int ib = iqs / 2;
        std::memcpy(&qp, blk + 2 + 4 * ib, 4);
        std::memcpy(&sp, blk + 2 + 32 + 4 * ib, 4);
        const auto* codes = reinterpret_cast<const std::uint8_t*>(&qp);
        const auto* sgn = reinterpret_cast<const std::uint8_t*>(&sp);
        const std::uint8_t qh = blk[2 + 64 + ib];
        const std::uint8_t ls0 = blk[2 + 72 + ib] & 0x0FU;
        const std::uint8_t ls1 = blk[2 + 72 + ib] >> 4U;
        for (int l0 = 0; l0 < 8; l0 += 2) {
          const std::uint64_t grid =
              quant::kIq2sGrid[codes[l0 / 2] | ((qh << (8 - l0)) & 0x300U)];
          for (int j = 0; j < 8; ++j) {
            int g = static_cast<std::int8_t>(
                static_cast<std::uint8_t>(grid >> (8 * j)));
            const int sgn_bit = (sgn[l0 / 2] >> j) & 1U;
            if (sgn_bit)
              g = -g;
            const int u = x.qs[4 * l0 + j];
            if (l0 < 4)
              sumi0 += g * u;
            else
              sumi1 += g * u;
          }
        }
        sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
        d_scale = quant::Fp16ToFloat(
            static_cast<std::uint16_t>(blk[0] | (blk[1] << 8)));
      }
      acc += d_scale * x.d * static_cast<float>(sumi);
      (void)total;
    }
  }
  return acc;
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

  std::vector<Q81Group> q8;
  if (type != 0 && type != 1)
    QuantizeQ81Host(x, q8);
  std::vector<float> ref(m);
  for (int r = 0; r < m; ++r) {
    const void* row =
        w.data() + static_cast<std::size_t>(r) * n_blocks * blk_bytes;
    ref[r] = (type == 0 || type == 1)
                 ? quant::Dot(static_cast<gufo::core::GgmlType>(type), row,
                              std::span<const float>(x), n)
             : (type == 16 || type == 18 || type == 22)
                 ? IqIntDotRef(type, row, n, q8)
                 : Q8RefDot(type, row, n, q8);
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

  std::vector<Q81Group> q8;
  QuantizeQ81Host(x, q8);
  std::vector<float> ref(top_k * m);
  const std::size_t expert_bytes = gpu::QuantMatrixBytes(m, n, type);
  for (int slot = 0; slot < top_k; ++slot) {
    for (int r = 0; r < m; ++r) {
      const void* row = w.data() + ids[slot] * expert_bytes +
                        static_cast<std::size_t>(r) * n_blocks * blk_bytes;
      ref[slot * m + r] = (type == 16 || type == 18 || type == 22)
                              ? IqIntDotRef(type, row, n, q8)
                              : Q8RefDot(type, row, n, q8);
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

// Per-slot x (x_group = m): each slot's row block multiplies its own x —
// the MoE down-projection shape. This is what silently read past the Q8
// scratch before x_vecs was fixed.
void TestMoePerSlotXParity(int type, std::uint32_t seed) {
  std::mt19937 rng(seed);
  const int E = 3, m = 5, n = 512, top_k = 2;
  const std::size_t blk_bytes = BlockBytes(type);
  const std::size_t blk_elems = BlockElems(type);
  const std::size_t n_blocks = n / blk_elems;
  std::vector<std::uint8_t> w((std::size_t)E * m * n_blocks * blk_bytes);
  for (int e = 0; e < E; ++e)
    for (int r = 0; r < m; ++r)
      for (std::size_t b = 0; b < n_blocks; ++b)
        FillRandomBlock(
            type, rng,
            w.data() + (((std::size_t)e * m + r) * n_blocks + b) * blk_bytes);
  std::vector<float> x((std::size_t)top_k * n);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  for (auto& v : x)
    v = dist(rng);
  const std::vector<std::int32_t> ids = {1, 2};

  std::vector<float> ref((std::size_t)top_k * m);
  const std::size_t expert_bytes = gpu::QuantMatrixBytes(m, n, type);
  for (int slot = 0; slot < top_k; ++slot) {
    std::vector<float> x_slot(x.begin() + slot * n, x.begin() + (slot + 1) * n);
    std::vector<Q81Group> q8;
    QuantizeQ81Host(x_slot, q8);
    for (int r = 0; r < m; ++r) {
      const void* row = w.data() + ids[slot] * expert_bytes +
                        static_cast<std::size_t>(r) * n_blocks * blk_bytes;
      ref[slot * m + r] = IqIntDotRef(type, row, n, q8);
    }
  }

  void* d_w = nullptr;
  float *d_x = nullptr, *d_y = nullptr;
  std::int32_t* d_ids = nullptr;
  HIP_CHECK(hipMalloc(&d_w, w.size()));
  HIP_CHECK(hipMemcpy(d_w, w.data(), w.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&d_x, x.size() * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_x, x.data(), x.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&d_ids, top_k * sizeof(std::int32_t)));
  HIP_CHECK(hipMemcpy(d_ids, ids.data(), top_k * sizeof(std::int32_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&d_y, top_k * m * sizeof(float)));
  Require(
      gpu::QuantGemmMoe(d_w, d_x, d_ids, d_y, m, n, E, top_k, type, nullptr, m),
      "QuantGemmMoe per-slot x");
  std::vector<float> got(top_k * m);
  HIP_CHECK(hipMemcpy(got.data(), d_y, top_k * m * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_ids));
  HIP_CHECK(hipFree(d_y));
  for (int i = 0; i < top_k * m; ++i) {
    Require(std::abs(got[i] - ref[i]) <= 1e-3F * (1.0F + std::abs(ref[i])),
            "moe-slot type " + std::to_string(type) + " idx " +
                std::to_string(i) + ": got " + std::to_string(got[i]) +
                " want " + std::to_string(ref[i]));
  }
  std::printf("ok  QuantGemmMoe per-slot x type=%d\n", type);
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
    TestMoePerSlotXParity(18, 31);
    TestMoePerSlotXParity(22, 32);
    TestMoePerSlotXParity(16, 33);
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
