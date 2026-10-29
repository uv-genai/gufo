# Maya GLM-5 engine, carbon copy (hip_compat + same .cu sources).
# Isolated: only gufo's glm5 prompt/serve link this.

set(MAYA_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

add_library(gufo_maya_engine STATIC
  # core (composed engine)
  src/core/layer.cpp
  src/core/session.cpp
  src/core/expert_source.cpp
  src/core/remote_experts.cpp
  src/core/expert_cache.cpp
  src/core/native_head.cpp
  src/core/native_dense.cpp
  src/core/verify.cpp
  src/core/mtp.cpp
  src/core/glm_model.cu
  src/core/glm_fast_path.cu
  src/core/glm_prefill.cu
  src/core/glm_multi.cu
  src/core/glm_rows.cu
  src/core/glm_window.cu
  src/core/glm_dflash.cu
  src/core/glm_mtp.cu
  src/core/glm_mtp_rows.cu
  # core runtime
  src/core/device.cu
  src/core/pinned.cu
  src/core/graph.cpp
  src/core/weights.cpp
  src/core/layout.cpp
  # artifact
  src/artifact/gguf_reader.cpp
  # platform
  src/platform/memory.cpp
  src/platform/direct_file.cpp
  # cpu kernels — exactly Maya strata_kernels_cpu
  src/kernels/cpu/expert.cpp
  src/kernels/cpu/pool.cpp
  src/kernels/cpu/expert_layout.cpp
  src/kernels/cpu/q2_avx2.cpp
  src/kernels/cpu/exl3_ref.cpp
  src/kernels/cpu/exl3_cpu.cpp
  src/kernels/cpu/native_expert.cpp
  src/kernels/cpu/iq_avx2.cpp
  src/kernels/cpu/iq_avx512.cpp
  src/kernels/cpu/iq_avx512_one.cpp
  src/kernels/ngram.cpp
  src/ngram/ple_reader.cpp
  # prefill
  src/prefill/gemm.cu
  src/prefill/kernels.cu
  src/prefill/prefill.cpp
  # moe_mmq/ggml_cuda_host: MMQ prefill (needs full ggml-cuda HIP); fallback path OK
  # ggml type traits / row sizes
  third_party/llama.cpp/ggml/src/ggml.c
  stubs_missing.cpp
  third_party/llama.cpp/ggml/src/ggml-quants.c
  third_party/llama.cpp/ggml/src/ggml-alloc.c
  third_party/llama.cpp/ggml/src/gguf.cpp
  # gufo glue
  gufo_maya_api.cpp
)

# All Maya GLM kernels (CUDA-shaped, compiled HIP via hip_compat).
file(GLOB MAYA_KER_CU ${MAYA_ROOT}/src/kernels/cuda/*.cu)
target_sources(gufo_maya_engine PRIVATE ${MAYA_KER_CU})
set_source_files_properties(${MAYA_KER_CU}
  src/core/glm_model.cu src/core/glm_fast_path.cu src/core/glm_prefill.cu
  src/core/glm_multi.cu src/core/glm_rows.cu src/core/glm_window.cu
  src/core/glm_dflash.cu src/core/glm_mtp.cu src/core/glm_mtp_rows.cu
  src/core/device.cu src/core/pinned.cu
  src/prefill/gemm.cu src/prefill/kernels.cu
  PROPERTIES LANGUAGE HIP)

target_include_directories(gufo_maya_engine PUBLIC
  ${MAYA_ROOT}/include/strata/hip_compat
  ${MAYA_ROOT}/include
  ${MAYA_ROOT}/third_party/llama.cpp/ggml/include
  ${MAYA_ROOT}/third_party/llama.cpp/ggml/src
  ${MAYA_ROOT}/third_party/llama.cpp/ggml/src/ggml-cuda
  ${PROJECT_SOURCE_DIR}/src
)
# Maya HIP build: every CXX/HIP TU gets hip_compat/cuda_runtime.h force-included
# (cmake/hip_backend.cmake).  No raw CUDA toolkit.
# Maya HIP build force-includes hip_compat for HIP TUs.  CXX finds the shim
# via -I hip_compat when a file includes <cuda_runtime.h>.
foreach(_lang HIP)
  target_compile_options(gufo_maya_engine PUBLIC
    "$<$<COMPILE_LANGUAGE:${_lang}>:-include>"
    "$<$<COMPILE_LANGUAGE:${_lang}>:${MAYA_ROOT}/include/strata/hip_compat/cuda_runtime.h>")
endforeach()
target_compile_definitions(gufo_maya_engine PUBLIC
  STRATA_ENABLE_HIP=1 STRATA_USE_HIP=1 STRATA_HIP=1 STRATA_HAS_WMMA=1
  STRATA_VERSION="0.1.27" STRATA_HIP_ARCHS="gfx1151"
  GGML_USE_HIP=1 STRATA_NATIVE_EXPERTS=1)
target_link_libraries(gufo_maya_engine PUBLIC hip::host Threads::Threads)
# Maya: only the CPU kernel files get AVX flags (CMakeLists expert.cpp / q2_avx2.cpp)
set_source_files_properties(src/kernels/cpu/expert.cpp PROPERTIES
  COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma;-mbmi2;-mbmi")
set_source_files_properties(src/kernels/cpu/q2_avx2.cpp PROPERTIES
  COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
set_source_files_properties(src/kernels/cpu/kq_avx512.cpp src/kernels/cpu/native_expert.cpp
  src/kernels/cpu/iq_avx2.cpp
  src/kernels/cpu/iq_avx512.cpp
  src/kernels/cpu/iq_avx512_one.cpp
  src/kernels/cpu/expert.cpp src/kernels/cpu/iq_avx512.cpp src/kernels/cpu/iq_avx512_one.cpp
  PROPERTIES
  COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma;-mbmi2;-mbmi")
set_source_files_properties(src/kernels/cpu/iq_avx2.cpp src/kernels/cpu/q2_avx2.cpp
  PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
set_source_files_properties(third_party/llama.cpp/ggml/src/ggml.c
  PROPERTIES LANGUAGE C)
set_source_files_properties(stubs_missing.cpp gufo_maya_api.cpp
  PROPERTIES LANGUAGE CXX)
if(TARGET roc::hipblas)
  target_link_libraries(gufo_maya_engine PUBLIC roc::hipblas)
endif()
