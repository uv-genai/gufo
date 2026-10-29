// Carbon copy of Maya `src/kernels/cuda/glm_kda.cu` + the KDA block of
// `step_layers` (`src/core/glm_model.cu` ~2243-2278).  HIP-ported with the
// same thread maps, phase order, fast-math, and indexing.  GEMMs go through
// QuantGemm (Maya `gm_gemv` / `native_mmvq`).
//
// Five kernels (names match Maya):
//   conv + conv_slide | gate | recurrence | out_gate
// plus `l2_heads_kernel` from glm_model.cu.

#include "src/models/glm5_next/kernels/rocm/kda.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Maya GLM kernels (copied, HIP via hip_compat) — do not reimplement.
#include "src/models/glm5_next/quant_gemv.hpp"
#include "strata/kernels/glm_kda.hpp"
#include "strata/kernels/native_mmvq.hpp"

namespace gufo::models::glm5_next::rocm {
namespace {

#define HIP_CHECK(expr)            \
  do {                             \
    const hipError_t err = (expr); \
    if (err != hipSuccess) {       \
      return false;                \
    }                              \
  } while (0)

constexpr int KD_THREADS = 256;
constexpr int KD_MAX_D_CONV = 16;  // the slide's register staging

__device__ __forceinline__ float dsigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

// ---- Maya glm_kda.cu (literal) -------------------------------------------

__global__ void glm_kda_conv_kernel(const float* __restrict__ x_proj,
                                    const float* __restrict__ conv_w,
                                    const float* __restrict__ conv_state,
                                    int d_inner, int d_conv, int T,
                                    float* __restrict__ out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = d_inner * T;
  if (idx >= total)
    return;
  const int c = idx % d_inner;  // ggml order: features fastest
  const int t = idx / d_inner;
  const int hist = d_conv - 1;

  float acc = 0.0f;
  for (int tap = 0; tap < d_conv; ++tap) {
    const int src = t - hist + tap;
    const float x = src >= 0
                        ? x_proj[(size_t)src * d_inner + c]
                        : conv_state[(size_t)(src + hist) + (size_t)hist * c];
    acc += conv_w[(size_t)tap + (size_t)d_conv * c] * x;
  }
  out[(size_t)t * d_inner + c] = acc / (1.0f + __expf(-acc));  // silu
}

__global__ void glm_kda_conv_slide_kernel(const float* __restrict__ x_proj,
                                          float* __restrict__ conv_state,
                                          int d_inner, int d_conv, int T) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= d_inner)
    return;
  const int hist = d_conv - 1;
  float nv[KD_MAX_D_CONV];
  for (int j = 0; j < hist; ++j) {
    const int src = T - hist + j;
    nv[j] = src >= 0 ? x_proj[(size_t)src * d_inner + c]
                     : conv_state[(size_t)(src + hist) + (size_t)hist * c];
  }
  for (int j = 0; j < hist; ++j)
    conv_state[(size_t)j + (size_t)hist * c] = nv[j];
}

__global__ void glm_kda_gate_kernel(const float* __restrict__ g,
                                    const float* __restrict__ ssm_a,
                                    float lower_bound, int head_dim, int n_head,
                                    int T, float* __restrict__ g1) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = head_dim * n_head * T;
  if (idx >= total)
    return;
  const int e = idx % head_dim;
  const int h = (idx / head_dim) % n_head;
  const int t = idx / (head_dim * n_head);
  const float v = g[(size_t)(e + head_dim * h) + (size_t)t * head_dim * n_head];
  g1[idx] = lower_bound * dsigmoid(-v * ssm_a[h]);
}

__global__ void glm_kda_recurrence_kernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ g1,
    const float* __restrict__ beta, float* __restrict__ state, int n_head,
    int head_dim, int T, float* __restrict__ out) {
  const int h = blockIdx.x;
  if (h >= n_head)
    return;
  const int dk = head_dim;
  const int tid = threadIdx.x;
  const int nt = blockDim.x;
  const float scale = rsqrtf((float)dk);

  float* S = state + (size_t)h * dk * dk;
  extern __shared__ float smem[];
  float* s_u = smem;
  float* s_delta = smem + dk;

  for (int t = 0; t < T; ++t) {
    const float* qt = q + ((size_t)t * n_head + h) * dk;
    const float* kt = k + ((size_t)t * n_head + h) * dk;
    const float* vt = v + ((size_t)t * n_head + h) * dk;
    const float* gt = g1 + ((size_t)t * n_head + h) * dk;
    const float b = beta[(size_t)t * n_head + h];

    for (int i = tid; i < dk; i += nt) {
      const float decay = __expf(gt[i]);
      for (int j = 0; j < dk; ++j)
        S[(size_t)i * dk + j] *= decay;
    }
    __syncthreads();

    for (int j = tid; j < dk; j += nt) {
      float acc = 0.0f;
      for (int i = 0; i < dk; ++i)
        acc += S[(size_t)i * dk + j] * kt[i];
      s_u[j] = acc;
    }
    __syncthreads();

    for (int j = tid; j < dk; j += nt)
      s_delta[j] = b * (vt[j] - s_u[j]);
    __syncthreads();

    for (int idx = tid; idx < dk * dk; idx += nt) {
      const int i = idx / dk, j = idx % dk;
      S[(size_t)i * dk + j] += kt[i] * s_delta[j];
    }
    __syncthreads();

    for (int j = tid; j < dk; j += nt) {
      float acc = 0.0f;
      for (int i = 0; i < dk; ++i)
        acc += S[(size_t)i * dk + j] * qt[i];
      out[((size_t)t * n_head + h) * dk + j] = acc * scale;
    }
    __syncthreads();
  }
}

__global__ void glm_kda_out_gate_kernel(const float* __restrict__ o,
                                        const float* __restrict__ norm_w,
                                        const float* __restrict__ g2,
                                        int head_dim, int n_head, int T,
                                        float eps, float* __restrict__ gated) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_head * T)
    return;
  const int h = idx % n_head;
  const int t = idx / n_head;
  const float* ot = o + ((size_t)t * n_head + h) * head_dim;
  const float* gt = g2 + ((size_t)t * n_head + h) * head_dim;

  float ss = 0.0f;
  for (int e = 0; e < head_dim; ++e)
    ss += ot[e] * ot[e];
  const float inv = rsqrtf(ss / (float)head_dim + eps);
  for (int e = 0; e < head_dim; ++e)
    gated[((size_t)t * n_head + h) * head_dim + e] =
        ot[e] * inv * norm_w[e] * dsigmoid(gt[e]);
}

// ---- Maya glm_fast.cu kda_rec_block (engine_kind=glm-fast decode) ----
__global__ void kda_rec_fast_kernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ g1,
    const float* __restrict__ beta_raw, float* __restrict__ state,
    const float* __restrict__ g2, const float* __restrict__ norm_w, float eps,
    float* __restrict__ out, int n_head) {
  constexpr int HD = 128;
  const int h = blockIdx.x;
  if (h >= n_head)
    return;
  const int t = threadIdx.x, j = t & (HD - 1), half = t >> 7;
  __shared__ float sq[HD], sk[HD], sdec[HD], sdel[HD];
  __shared__ float sacc[2][HD];
  __shared__ float sred[32];
  if (t < HD) {
    sq[t] = q[h * HD + t];
    sk[t] = k[h * HD + t];
    sdec[t] = __expf(g1[h * HD + t]);
  }
  const float b = 1.0f / (1.0f + expf(-beta_raw[h]));
  __syncthreads();
  float* Sh = state + (size_t)h * HD * HD;
  float reg[64];
  float A = 0.0f;
#pragma unroll
  for (int ii = 0; ii < 64; ++ii) {
    const int i = half * 64 + ii;
    const float s = Sh[i * HD + j] * sdec[i];
    reg[ii] = s;
    A += s * sk[i];
  }
  sacc[half][j] = A;
  __syncthreads();
  if (half == 0)
    sdel[j] = b * (v[h * HD + j] - (sacc[0][j] + sacc[1][j]));
  __syncthreads();
  const float dl = sdel[j];
  float O = 0.0f;
#pragma unroll
  for (int ii = 0; ii < 64; ++ii) {
    const int i = half * 64 + ii;
    const float s2 = reg[ii] + sk[i] * dl;
    Sh[i * HD + j] = s2;
    O += s2 * sq[i];
  }
  __syncthreads();
  sacc[half][j] = O;
  __syncthreads();
  const float o = (sacc[0][j] + sacc[1][j]) * rsqrtf((float)HD);
  float ss = half == 0 ? o * o : 0.0f;
  {
    const int lane_id = t & 31, w = t >> 5, nw = 256 >> 5;
    float vv = ss;
#pragma unroll
    for (int o2 = 16; o2 > 0; o2 >>= 1)
      vv += __shfl_xor_sync(0xffffffffull, vv, o2);
    __syncthreads();
    if (lane_id == 0)
      sred[w] = vv;
    __syncthreads();
    vv = 0.0f;
    for (int i = 0; i < nw; ++i)
      vv += sred[i];
    ss = vv;
  }
  const float inv = rsqrtf(ss / (float)HD + eps);
  if (half == 0) {
    out[h * HD + j] = o * inv * norm_w[j] * dsigmoid(g2[h * HD + j]);
  }
}

// ---- Maya glm_model.cu l2_heads / sigmoid / axpy (literal) ----------------

__global__ void l2_heads_kernel(const float* __restrict__ x,
                                float* __restrict__ y, int dk, int n_head,
                                int T) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_head * T)
    return;
  const int h = idx % n_head;
  const int t = idx / n_head;
  const float* xt = x + ((size_t)t * n_head + h) * dk;
  float ss = 0.0f;
  for (int e = 0; e < dk; ++e)
    ss += xt[e] * xt[e];
  const float inv = rsqrtf(ss + 1e-6f);
  float* yt = y + ((size_t)t * n_head + h) * dk;
  for (int e = 0; e < dk; ++e)
    yt[e] = xt[e] * inv;
}

__global__ void sigmoid_kernel(float* __restrict__ x, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  x[i] = 1.0f / (1.0f + expf(-x[i]));
}

__global__ void axpy_kernel(float* __restrict__ acc, float alpha,
                            const float* __restrict__ v, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  acc[i] += alpha * v[i];
}

// ---- optional intermediate dumps (GLM_CB_DIR) ----------------------------
// Files are tagged with a global call counter so T=1-per-token GpuEngine
// dumps do not overwrite each other.  Call 0 = first token.

int g_dump_call = -1;  // pre-incremented in KdaPrefillGpu; first call is 0

void DumpF32(const char* name, const float* dev, std::size_t n,
             hipStream_t stream) {
  const char* dir = std::getenv("GLM_CB_DIR");
  if (dir == nullptr || dir[0] == '\0')
    return;
  std::vector<float> host(n);
  if (hipMemcpyAsync(host.data(), dev, n * sizeof(float), hipMemcpyDeviceToHost,
                     stream) != hipSuccess)
    return;
  hipStreamSynchronize(stream);
  const std::string path = std::string(dir) + "/" + name + "-" +
                           std::to_string(g_dump_call) + ".f32";
  FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr)
    return;
  std::fwrite(host.data(), sizeof(float), n, f);
  std::fclose(f);
}

}  // namespace

// Maya gm_gemv(W, x, bias, alpha, y, out, in) → native_mmvq(..., n_in=in,
// n_out=out). This wrapper takes (out, in) like QuantGemm so call sites stay
// (rows, cols).
inline bool MayaGemv(const void* W, const float* x, void* q81, float* y,
                     int out, int in, int ggml_type, hipStream_t stream) {
  if (ggml_type == 14) {  // Q6_K
    strata::kernels::native_q6_k_f32(W, x, q81, y, in, out, 1, stream);
    return true;
  }
  if (ggml_type == 8) {  // Q8_0
    strata::kernels::native_q8_0_f32(W, x, q81, y, in, out, 1, stream);
    return true;
  }
  // F32 / other: QuantGemm fallback (tests use F32).
  return gpu::QuantGemm(W, x, y, out, in, ggml_type, stream);
}

// ---- Maya host wrappers (call the copied glm_kda.cu) ---------------------

void GlmKdaConv(const float* x_proj, const float* conv_w, float* conv_state,
                int d_inner, int d_conv, int T, float* out,
                hipStream_t stream) {
  strata::kernels::glm_kda_conv(x_proj, conv_w, conv_state, d_inner, d_conv, T,
                                out, stream);
}

void GlmKdaGate(const float* g, const float* ssm_a, float lower_bound,
                int head_dim, int n_head, int T, float* g1,
                hipStream_t stream) {
  strata::kernels::glm_kda_gate(g, ssm_a, lower_bound, head_dim, n_head, T, g1,
                                stream);
}

void GlmKdaRecurrence(const float* q, const float* k, const float* v,
                      const float* g1, const float* beta, float* state,
                      int n_head, int head_dim, int T, float* out,
                      hipStream_t stream) {
  strata::kernels::glm_kda_recurrence(q, k, v, g1, beta, state, n_head,
                                      head_dim, T, out, stream);
}

void GlmKdaOutGate(const float* o, const float* norm_w, const float* g2,
                   int head_dim, int n_head, int T, float eps, float* gated,
                   hipStream_t stream) {
  strata::kernels::glm_kda_out_gate(o, norm_w, g2, head_dim, n_head, T, eps,
                                    gated, stream);
}

// ---- KdaPrefillGpu: step_layers KDA block, same order ---------------------

std::size_t KdaScratchFloats(const KdaGpuGeometry& geom,
                             std::uint32_t seq_len) {
  const std::uint32_t inner = geom.num_heads * geom.head_dim;
  // proj x3 | conv x3 | g | g1 | beta | scan | g2 | gated | fa | fb | ga | gb
  return 14ULL * inner * seq_len + 2ULL * geom.num_heads * seq_len +
         4ULL * geom.head_dim * seq_len;
}

bool KdaPrefillGpu(const KdaGpuGeometry& geom, const KdaGpuWeights& w,
                   const float* input, float* state, float* conv_state,
                   float* out, std::uint32_t seq_len, float* scratch,
                   hipStream_t stream) {
  const std::uint32_t H = geom.hidden_size;
  const std::uint32_t nh = geom.num_heads;
  const std::uint32_t dk = geom.head_dim;
  const std::uint32_t inner = nh * dk;
  const std::uint32_t K = geom.conv_kernel;
  const float lb = geom.gate_lower_bound;
  const int T = static_cast<int>(seq_len);
  if (T <= 0)
    return true;
  if (std::getenv("GLM_CB_DIR") != nullptr)
    ++g_dump_call;

  float* owned = nullptr;
  if (scratch == nullptr) {
    HIP_CHECK(hipMalloc(&owned, KdaScratchFloats(geom, T) * sizeof(float)));
    scratch = owned;
  }
  float* sp = scratch;
  auto take = [&sp](std::size_t n) {
    float* r = sp;
    sp += n;
    return r;
  };

  const std::size_t inner_T = (std::size_t)inner * T;
  float* proj_q = take(inner_T);
  float* proj_k = take(inner_T);
  float* proj_v = take(inner_T);
  float* q_c = take(inner_T);
  float* k_c = take(inner_T);
  float* v_c = take(inner_T);
  float* g1 = take(inner_T);
  float* beta = take((std::size_t)nh * T);
  float* scan = take(inner_T);
  float* g2 = take(inner_T);
  float* gated = take(inner_T);
  float* fa = take((std::size_t)dk * T);
  float* fb = take(inner_T);  // also carries g after +dt_bias
  float* ga = take((std::size_t)dk * T);
  float* gb = take(inner_T);

  const int block = KD_THREADS;
  const int grid_inner_T = (int)((inner_T + block - 1) / block);

  // Q8_1 activation scratch (Maya gm_gemv) — one buffer, max n_in = inner.
  const std::size_t q81_bytes =
      strata::kernels::native_q8_1_bytes((int)inner, 1);
  std::vector<std::uint8_t> q81_host(q81_bytes);
  void* q81 = nullptr;
  HIP_CHECK(hipMalloc(&q81, q81_bytes));

  // ---- step_layers KDA, same order ----
  // for si in q,k,v: gm_gemv(attn_*) -> proj; glm_kda_conv -> conv[si]
  const float* wqkv[3] = {w.w_q, w.w_k, w.w_v};
  const int wqkv_t[3] = {w.w_q_type, w.w_k_type, w.w_v_type};
  const float* conv_w[3] = {w.conv_q, w.conv_k, w.conv_v};
  float* proj[3] = {proj_q, proj_k, proj_v};
  float* conv_out[3] = {q_c, k_c, v_c};
  const std::size_t conv_stride = (std::size_t)inner * (K - 1);
  for (int si = 0; si < 3; ++si) {
    for (int t = 0; t < T; ++t) {
      if (!MayaGemv(wqkv[si], input + (std::size_t)t * H, q81,
                    proj[si] + (std::size_t)t * inner, (int)inner, (int)H,
                    wqkv_t[si], stream)) {
        hipFree(q81);
        return false;
      }
    }
    float* cst = nullptr;
    if (conv_state != nullptr)
      cst = conv_state + (std::size_t)si * conv_stride;
    GlmKdaConv(proj[si], conv_w[si], cst, (int)inner, (int)K, T, conv_out[si],
               stream);
    if (si == 0)
      DumpF32("kda_proj_q", proj[si], inner_T, stream);
    if (si == 0)
      DumpF32("kda_conv_q", conv_out[si], inner_T, stream);
  }

  // l2_heads on q and k only
  {
    const int total = (int)(nh * T);
    hipLaunchKernelGGL(
        l2_heads_kernel, dim3((total + KD_THREADS - 1) / KD_THREADS),
        dim3(KD_THREADS), 0, stream, q_c, q_c, (int)dk, (int)nh, T);
    hipLaunchKernelGGL(
        l2_heads_kernel, dim3((total + KD_THREADS - 1) / KD_THREADS),
        dim3(KD_THREADS), 0, stream, k_c, k_c, (int)dk, (int)nh, T);
  }
  DumpF32("kda_q", q_c, inner_T, stream);
  DumpF32("kda_k", k_c, inner_T, stream);
  DumpF32("kda_v", v_c, inner_T, stream);

  // g = f_b @ (f_a @ x) + dt_bias; g1 = lb * sigmoid(-g * A)
  for (int t = 0; t < T; ++t) {
    const float* x = input + (std::size_t)t * H;
    if (!MayaGemv(w.f_a, x, q81, fa + (std::size_t)t * dk, (int)dk, (int)H,
                  w.f_a_type, stream) ||
        !MayaGemv(w.f_b, fa + (std::size_t)t * dk, q81,
                  fb + (std::size_t)t * inner, (int)inner, (int)dk, w.f_b_type,
                  stream)) {
      hipFree(q81);
      return false;
    }
  }
  hipLaunchKernelGGL(axpy_kernel, dim3(grid_inner_T), dim3(block), 0, stream,
                     fb, 1.0f, w.dt_bias, (int)inner_T);
  DumpF32("kda_fa", fa, (std::size_t)dk * T, stream);
  DumpF32("kda_gpre", fb, inner_T, stream);
  // Maya writes the bias-added vector as `g` for the gate; reuse fb in place.
  hipLaunchKernelGGL(
      glm_kda_gate_kernel, dim3((int)((inner_T + KD_THREADS - 1) / KD_THREADS)),
      dim3(KD_THREADS), 0, stream, fb, w.ssm_a, lb, (int)dk, (int)nh, T, g1);
  DumpF32("kda_g1", g1, inner_T, stream);

  // beta raw → sigmoid (Maya glm_kda_recurrence takes sigmoid'd beta).
  for (int t = 0; t < T; ++t) {
    if (!MayaGemv(w.ssm_beta, input + (std::size_t)t * H, q81,
                  beta + (std::size_t)t * nh, (int)nh, (int)H, w.beta_type,
                  stream)) {
      hipFree(q81);
      return false;
    }
  }
  hipLaunchKernelGGL(sigmoid_kernel, dim3((nh * T + block - 1) / block),
                     dim3(block), 0, stream, beta, (int)(nh * T));
  DumpF32("kda_beta", beta, (std::size_t)nh * T, stream);

  // g2 = g_b @ (g_a @ x)
  for (int t = 0; t < T; ++t) {
    const float* x = input + (std::size_t)t * H;
    if (!MayaGemv(w.g_a, x, q81, ga + (std::size_t)t * dk, (int)dk, (int)H,
                  w.g_a_type, stream) ||
        !MayaGemv(w.g_b, ga + (std::size_t)t * dk, q81,
                  gb + (std::size_t)t * inner, (int)inner, (int)dk, w.g_b_type,
                  stream)) {
      hipFree(q81);
      return false;
    }
  }
  DumpF32("kda_g2", gb, inner_T, stream);

  // Maya glm_kda: recurrence then out_gate (step_layers order).
  GlmKdaRecurrence(q_c, k_c, v_c, g1, beta, state, (int)nh, (int)dk, T, scan,
                   stream);
  DumpF32("kda_scan", scan, inner_T, stream);
  GlmKdaOutGate(scan, w.ssm_norm, gb, (int)dk, (int)nh, T, geom.norm_eps, gated,
                stream);
  DumpF32("kda_gated", gated, inner_T, stream);

  for (int t = 0; t < T; ++t) {
    if (!MayaGemv(w.w_out, gated + (std::size_t)t * inner, q81,
                  out + (std::size_t)t * H, (int)H, (int)inner, w.w_out_type,
                  stream)) {
      hipFree(q81);
      return false;
    }
  }

  hipFree(q81);
  if (owned)
    hipFree(owned);
  return true;
}

bool KdaDecodeGpu(const KdaGpuGeometry& geom, const KdaGpuWeights& w,
                  const float* token_input, float* state, float* conv_state,
                  float* out, float* scratch, hipStream_t stream) {
  return KdaPrefillGpu(geom, w, token_input, state, conv_state, out, 1, scratch,
                       stream);
}

}  // namespace gufo::models::glm5_next::rocm
