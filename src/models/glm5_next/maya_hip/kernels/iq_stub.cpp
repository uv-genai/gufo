#include "strata/kernels/iq_kernels.hpp"
namespace strata::kernels {
bool iq_supported(int) noexcept {
  return false;
}
size_t iq_row_bytes(int, int64_t) noexcept {
  return 0;
}
void quantize_q8_1_rows(const float*, int64_t, int64_t, void*, void*) {}
void iq_mmvq(int, const void*, const void*, float*, int, int, int, void*) {}
void iq_dequant_f16(int, const void*, int64_t, uint16_t*, void*) {}
void iq_dequant_f32(int, const void*, int64_t, float*, void*) {}
void iq_embed_rows(int, const void*, size_t, const int32_t*, int64_t, int64_t,
                   float*, void*) {}
void iq_dequant_gu_f16(int, const void*, const void*, int64_t, int64_t,
                       uint16_t*, void*) {}
bool gemm_iq_f16_grouped(int, const void*, size_t, int, int, const void*, int,
                         const int32_t*, int, int, float*, int, void*,
                         const int32_t*, const GroupedWeightTable*, bool) {
  return false;
}
bool gemm_iq_f16_grouped_supported(int) {
  return false;
}
NativeExpertLayout native_expert_layout(int, int, int64_t, int64_t) {
  return {};
}
size_t native_expert_scratch_bytes(int64_t, int64_t) {
  return 0;
}
void native_expert_grouped(const NativeExpertLayout&, const unsigned long long*,
                           const int32_t*, const int32_t*, const int32_t*,
                           const int32_t*, int64_t, int64_t, const void*, void*,
                           float*, void*, float) {}
}  // namespace strata::kernels
