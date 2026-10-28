#ifndef GUFO_MODELS_GLM5_NEXT_GPU_KERNELS_H_
#define GUFO_MODELS_GLM5_NEXT_GPU_KERNELS_H_

#include <hip/hip_runtime.h>

namespace gufo::models::glm5_next::gpu {

/// Launch RMS norm on one vector on GPU.
void LaunchRmsNorm(float* x, const float* gamma, int n, float eps,
                   hipStream_t stream);

/// Launch MatVec: y[m] = W[m, n] @ x[n].
void LaunchMatVec(const float* W, const float* x, float* y, int m, int n,
                  hipStream_t stream);

/// Launch embedding lookup: out[hidden] = embd[token * hidden + :].
void LaunchEmbedLookup(const float* embd, float* out, int hidden, int token,
                       hipStream_t stream);

/// Launch argmax: returns token id via host pointer.
void LaunchArgmax(const float* logits, int n, int* host_result,
                  hipStream_t stream);

}  // namespace gufo::models::glm5_next::gpu

#endif  // GUFO_MODELS_GLM5_NEXT_GPU_KERNELS_H_
