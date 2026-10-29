// Stubs for optional MMQ prefill / ggml-backend symbols not compiled here.
// CPU native experts come from native_expert.cpp (STRATA_NATIVE_EXPERTS).
#include <stddef.h>
#include <stdint.h>

namespace strata {
namespace prefill {
namespace mmq {
struct Product;
struct Context {
  void run_glu(const Product& p, float limit, void* stream);
};
void Context::run_glu(const Product&, float, void*) {}
bool glu_supported(int, long) { return false; }
bool to_f16_supported(int) { return false; }
size_t q8_block_bytes() { return 0; }
void* pool_slab(int, void*, unsigned long) { return nullptr; }
void to_f16(int, const void*, unsigned short*, long, void*) {}
}  // namespace mmq
}  // namespace prefill
namespace kernels {
namespace cpu {
bool kq_type_ok(int) { return false; }
int kq_act_bytes(int, int) { return 0; }
bool kq_avx512_ok() { return false; }
void kq_pack_act(const void* const*, int, int, void*) {}
void kq_rows_packed(int, const unsigned char*, unsigned long, int, const void*,
                    int, float* const*, int, int) {}
}  // namespace cpu
}  // namespace kernels
}  // namespace strata

extern "C" {
void ggml_critical_section_start(void) {}
void ggml_critical_section_end(void) {}
void ggml_backend_tensor_set(void* t, const void* s, size_t o, size_t n) {
  (void)t; (void)s; (void)o; (void)n;
}
void ggml_backend_tensor_memset(void* t, int v, size_t o, size_t n) {
  (void)t; (void)v; (void)o; (void)n;
}
int ggml_backend_buffer_get_usage(void* b) { (void)b; return 0; }
}

// ggml-cpu type traits: enough for load checks (GPU path does the dots).
extern "C" {
const void* ggml_get_type_traits_cpu(int);
const void* ggml_get_type_traits_cpu(int) { return nullptr; }
}

extern "C" void ggml_cpu_init(void) {}
