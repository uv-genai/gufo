// Phase 1: activation-quantisation gate for the IU4 decode path.
//
// Plan: docs/models/qwen3.8-flash-next/IU4-PLAN.md
//
// The iu4 WMMA is iu4 x iu4 with no mixed iu8 x iu4 instruction on gfx1151, so
// int4 weights force int4 activations. This is the decision point: if int4
// activations cost too much, the kernel is not worth building.
//
// Baseline is the current path - fp16 activations quantised to Q8_1 at
// src/models/qwen38_flash_next/kernels/rocm/batch.cpp:549. This compares
// three activation encodings against an fp16 reference:
//   fp16  reference, no quantisation
//   int8  per-32 fp16 scale (Q8_0/Q8_1 shape) - what runs today
//   int4  per-32 fp16 scale + fp16 min - what the iu4 WMMA would need
// and reports the error each introduces into the dot product.
//
// DATA SOURCE. Accuracy here depends entirely on using activations from a live
// run, not synthetic ones. Pass a raw float32 binary of row-major activations
// with --acts FILE --rows R --cols C. Without --acts the harness uses a
// synthetic heavy-tailed distribution and prints a WARN line; any result from
// that path is an estimate and must not be quoted as measured.
//
// Build: c++ -O2 -std=c++20 tools/bench/iu4_act_quant_gate.cpp -o /tmp/actgate
// Run:   /tmp/actgate [--acts FILE --rows R --cols C] [--seed N]
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string acts_path;
  int rows = 0;
  int cols = 0;
  uint32_t seed = 0x9E3779B9u;
};

// fp16 round-trip, matching what a half-precision activation buffer does.
float ToF16(float f) {
  union { float f; uint32_t u; } u{f};
  const uint32_t sign = (u.u >> 16) & 0x8000u;
  const int32_t exp = static_cast<int32_t>((u.u >> 23) & 0xFFu) - 127 + 15;
  const uint32_t frac = u.u & 0x7FFFFFu;
  if (exp <= 0) return 0.0f * (sign ? -1.0f : 1.0f);
  if (exp >= 31) return (sign ? -1.0f : 1.0f) * 1e30f;
  return std::ldexp(static_cast<float>((frac >> 13) | 0x400u) / 1024.0f,
                    exp - 25) * (sign ? -1.0f : 1.0f);
}

// Per-block quantisation, block = 32 elements, fp16 scale.
// int8: symmetric, no zero point (Q8_0 shape).
// int4: affine, scale + min, so the dequant is scale*q - min (Q4_1 shape).
struct BlockQ {
  float scale;
  float min;
  std::vector<int> q;
};

std::vector<BlockQ> QuantInt8(const std::vector<float>& row, size_t cols) {
  std::vector<BlockQ> out;
  for (size_t b = 0; b < cols; b += 32) {
    const size_t n = std::min<size_t>(32, cols - b);
    float amax = 0.0f;
    for (size_t i = 0; i < n; ++i) amax = std::max(amax, std::fabs(row[b + i]));
    const float scale = ToF16(amax > 0.0f ? amax / 127.0f : 0.0f);
    BlockQ blk{scale, 0.0f, {}};
    blk.q.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      const int v = scale > 0.0f
                        ? static_cast<int>(std::lround(row[b + i] / scale))
                        : 0;
      blk.q.push_back(std::clamp(v, -127, 127));
    }
    out.push_back(std::move(blk));
  }
  return out;
}

std::vector<BlockQ> QuantInt4(const std::vector<float>& row, size_t cols) {
  std::vector<BlockQ> out;
  for (size_t b = 0; b < cols; b += 32) {
    const size_t n = std::min<size_t>(32, cols - b);
    float lo = row[b], hi = row[b];
    for (size_t i = 0; i < n; ++i) {
      lo = std::min(lo, row[b + i]);
      hi = std::max(hi, row[b + i]);
    }
    const float scale = ToF16((hi - lo) > 0.0f ? (hi - lo) / 15.0f : 0.0f);
    BlockQ blk{scale, scale > 0.0f ? lo : 0.0f, {}};
    blk.q.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      const int v = scale > 0.0f
                        ? static_cast<int>(std::lround((row[b + i] - blk.min) /
                                                       scale))
                        : 0;
      blk.q.push_back(std::clamp(v, 0, 15));
    }
    out.push_back(std::move(blk));
  }
  return out;
}

float Dequant(const BlockQ& blk, size_t i, bool affine) {
  return affine ? (blk.scale * static_cast<float>(blk.q[i]) - blk.min)
                : (blk.scale * static_cast<float>(blk.q[i]));
}

// Dot product of a row against a pseudo-random weight row, under one
// activation encoding. weights stay f32 so the only error is the activations.
double DotWith(const std::vector<float>& acts, const std::vector<float>& w,
               size_t cols, const std::vector<BlockQ>* q, bool affine) {
  double acc = 0.0;
  for (size_t b = 0; b < cols; b += 32) {
    const size_t n = std::min<size_t>(32, cols - b);
    for (size_t i = 0; i < n; ++i) {
      const double a =
          q ? Dequant((*q)[b / 32], i, affine)
            : static_cast<double>(acts[b + i]);
      acc += a * static_cast<double>(w[b + i]);
    }
  }
  return acc;
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* what) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", what);
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--acts") opt.acts_path = next("--acts");
    else if (a == "--rows") opt.rows = std::atoi(next("--rows"));
    else if (a == "--cols") opt.cols = std::atoi(next("--cols"));
    else if (a == "--seed") opt.seed = static_cast<uint32_t>(std::stoul(next("--seed")));
    else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }

  std::vector<float> acts;
  bool real = false;
  if (!opt.acts_path.empty()) {
    std::ifstream f(opt.acts_path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", opt.acts_path.c_str()); return 1; }
    f.seekg(0, std::ios::end);
    const size_t bytes = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    const size_t want = static_cast<size_t>(opt.rows) * opt.cols * sizeof(float);
    if (bytes < want) {
      std::fprintf(stderr, "file has %zu bytes, need %zu (rows*cols*4)\n",
                   bytes, want);
      return 1;
    }
    acts.resize(static_cast<size_t>(opt.rows) * opt.cols);
    f.read(reinterpret_cast<char*>(acts.data()), static_cast<std::streamsize>(want));
    real = true;
  } else {
    opt.rows = opt.rows > 0 ? opt.rows : 128;
    opt.cols = opt.cols > 0 ? opt.cols : 2048;
    std::printf("WARN: no --acts file; using SYNTHETIC heavy-tailed "
                "activations. Results are an ESTIMATE, not measured.\n\n");
    std::mt19937 rng(opt.seed);
    std::normal_distribution<float> core(0.0f, 1.0f);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    acts.resize(static_cast<size_t>(opt.rows) * opt.cols);
    for (auto& v : acts) {
      // Heavy tail: 5% of elements blown up, which is what real activations
      // do and what hurts low-bit quantisation most.
      v = core(rng);
      if (u01(rng) < 0.05f) v *= 8.0f;
    }
  }

  // fp16-round the reference so the comparison isolates quantisation, not
  // float-vs-half storage.
  for (auto& v : acts) v = ToF16(v);

  const size_t rows = real ? static_cast<size_t>(opt.rows) : static_cast<size_t>(opt.rows);
  const size_t cols = static_cast<size_t>(opt.cols);

  std::mt19937 wrng(opt.seed ^ 0xA5A5A5A5u);
  std::normal_distribution<float> wd(0.0f, 0.05f);

  double err8 = 0.0, err4 = 0.0, ref_energy = 0.0;
  // Per-dot error normalised by the RMS of the reference over the whole run.
  // A per-dot relative error is meaningless here: with random zero-mean
  // weights the reference dots cluster near zero, so dividing by them explodes.
  double worst8 = 0.0, worst4 = 0.0;
  int worse = 0, total = 0;

  for (size_t r = 0; r < rows; ++r) {
    std::vector<float> w(cols);
    for (auto& v : w) v = wd(wrng);
    std::vector<float> row(acts.begin() + static_cast<long>(r * cols),
                           acts.begin() + static_cast<long>((r + 1) * cols));
    const auto q8 = QuantInt8(row, cols);
    const auto q4 = QuantInt4(row, cols);

    // A small block of output columns so each row contributes many dots.
    for (size_t c = 0; c < 8; ++c) {
      std::vector<float> wc(cols);
      for (auto& v : wc) v = wd(wrng);
      const double ref = DotWith(row, wc, cols, nullptr, false);
      const double d8 = DotWith(row, wc, cols, &q8, false);
      const double d4 = DotWith(row, wc, cols, &q4, true);
      ref_energy += ref * ref;
      err8 += (d8 - ref) * (d8 - ref);
      err4 += (d4 - ref) * (d4 - ref);
      if (std::fabs(d4 - ref) > std::fabs(d8 - ref)) ++worse;
      ++total;
    }
  }

  const double rms8 = std::sqrt(err8 / total);
  const double rms4 = std::sqrt(err4 / total);
  const double rms_ref = std::sqrt(ref_energy / total);
  worst8 = rms8 / rms_ref;
  worst4 = rms4 / rms_ref;

  std::printf("source: %s\n", real ? opt.acts_path.c_str() : "synthetic");
  std::printf("rows=%zu cols=%zu dots=%d\n\n", rows, cols, total);
  std::printf("reference dot RMS: %.6f\n\n", rms_ref);
  std::printf("%-10s %14s %12s\n", "encoding", "RMS err", "rel to ref");
  std::printf("%-10s %14.6f %12.5f\n", "fp16", 0.0, 0.0);
  std::printf("%-10s %14.6f %12.5f\n", "int8", rms8, worst8);
  std::printf("%-10s %14.6f %12.5f\n", "int4", rms4, worst4);
  std::printf("\nint4/int8 RMS error ratio: %.3f  (higher = int4 worse)\n",
              rms4 / (rms8 > 0 ? rms8 : 1e-12));
  std::printf("dots where int4 is worse than int8: %d / %d\n", worse, total);

  if (!real) {
    std::printf("\nVERDICT: ESTIMATE ONLY. Re-run with --acts from a live "
                "capture before deciding.\n");
  }
  return 0;
}
