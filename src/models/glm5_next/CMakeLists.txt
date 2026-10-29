# GLM-5.3-Flash (glm5-next) model.
# CPU oracles + GPU kernels for KDA/mHC/MoE/MLA/MTP.
# GLM-5.3 inference itself is Maya's engine (maya_engine/) — carbon copy.

if(ENGINE_ENABLE_HIP)
  find_package(Threads REQUIRED)
  add_subdirectory(maya_engine)
endif()

# GPU kernels (ROCm).
if(ENGINE_ENABLE_HIP)
  # Maya GLM kernels (HIP build via strata/hip_compat — sources copied, not rewritten).
  add_library(gufo_maya_hip STATIC
    maya_hip/kernels/glm_kda.cu
    maya_hip/kernels/glm_dsa.cu
    maya_hip/kernels/glm_hc.cu
    maya_hip/kernels/glm_ffn.cu
    maya_hip/kernels/router_sigmoid.cu
    maya_hip/kernels/elementwise.cu
    maya_hip/kernels/native_mmvq.cu
    maya_hip/kernels/iq_stub.cpp)
  set_source_files_properties(
    maya_hip/kernels/glm_kda.cu
    maya_hip/kernels/glm_dsa.cu
    maya_hip/kernels/glm_hc.cu
    maya_hip/kernels/glm_ffn.cu
    maya_hip/kernels/router_sigmoid.cu
    maya_hip/kernels/elementwise.cu
    maya_hip/kernels/native_mmvq.cu
    PROPERTIES LANGUAGE HIP)
  # hip_compat first so <cuda_runtime.h> is Maya's shim; then "strata/kernels/*".
  target_include_directories(gufo_maya_hip PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/maya_hip/strata/hip_compat
    ${CMAKE_CURRENT_SOURCE_DIR}/maya_hip
    ${CMAKE_CURRENT_SOURCE_DIR}/maya_hip/strata/kernels)
  target_compile_definitions(gufo_maya_hip PUBLIC STRATA_HIP=1)
  target_link_libraries(gufo_maya_hip PUBLIC hip::host)

  add_library(gufo_glm5_next_kernels STATIC
    kernels/rocm/kda.cpp
    kernels/rocm/mla.cpp
    kernels/rocm/mhc.cpp
    kernels/rocm/moe.cpp
    kernels/rocm/mtp.cpp)
  set_source_files_properties(kernels/rocm/kda.cpp PROPERTIES LANGUAGE HIP)
  set_source_files_properties(kernels/rocm/mla.cpp PROPERTIES LANGUAGE HIP)
  set_source_files_properties(kernels/rocm/mhc.cpp PROPERTIES LANGUAGE HIP)
  set_source_files_properties(kernels/rocm/moe.cpp PROPERTIES LANGUAGE HIP)
  set_source_files_properties(kernels/rocm/mtp.cpp PROPERTIES LANGUAGE HIP)
  target_link_libraries(gufo_glm5_next_kernels PUBLIC gufo_core gufo_maya_hip hip::host)
  target_include_directories(gufo_glm5_next_kernels PUBLIC
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/models/glm5_next)
endif()

add_library(gufo_glm5_next STATIC
  chat_template.cpp
  config.cpp
  cpu_engine.cpp
  engine.cpp
  gpu_engine.cpp
  mhc_reference.cpp
  mla_reference.cpp
  moe_reference.cpp
  mtp_reference.cpp
  reference.cpp
  weights.cpp)

if(ENGINE_ENABLE_HIP)
  target_sources(gufo_glm5_next PRIVATE gpu_kernels.hip.cpp quant_gemv.hip.cpp)
  set_source_files_properties(gpu_kernels.hip.cpp PROPERTIES LANGUAGE HIP)
  set_source_files_properties(quant_gemv.hip.cpp PROPERTIES LANGUAGE HIP)
  target_link_libraries(gufo_glm5_next PUBLIC hip::host)
endif()

target_include_directories(gufo_glm5_next PUBLIC
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/src/models/glm5_next)

target_link_libraries(gufo_glm5_next PUBLIC
  gufo_core)

if(ENGINE_ENABLE_HIP)
  target_link_libraries(gufo_glm5_next PUBLIC gufo_glm5_next_kernels)
endif()
