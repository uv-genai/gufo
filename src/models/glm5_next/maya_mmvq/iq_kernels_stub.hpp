// Stub for Maya `iq_kernels.hpp` — only what `native_mmvq.cu` references.
// IQ formats go through iq_mmvq; KDA needs Q6_K/Q8_0 only.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

inline size_t iq_row_bytes(int, int64_t) {
  return 0;
}

inline void iq_mmvq(int, const void*, const void*, float*, int, int, int,
                    void*) {}

}  // namespace strata::kernels
