# GLM-5.3-Flash (glm5-next) model.
# CPU oracles + GPU kernels for KDA/mHC/MoE/MLA/MTP.

# GPU kernels (ROCm).
if(ENGINE_ENABLE_HIP)
  add_library(gufo_glm5_next_kernels STATIC
    kernels/rocm/kda.cpp
    kernels/rocm/mhc.cpp)
  set_source_files_properties(kernels/rocm/kda.cpp PROPERTIES LANGUAGE HIP)
  set_source_files_properties(kernels/rocm/mhc.cpp PROPERTIES LANGUAGE HIP)
  target_link_libraries(gufo_glm5_next_kernels PUBLIC gufo_core hip::host)
  target_include_directories(gufo_glm5_next_kernels PUBLIC
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/models/glm5_next)
endif()

add_library(gufo_glm5_next STATIC
  chat_template.cpp
  config.cpp
  cpu_engine.cpp
  engine.cpp
  mhc_reference.cpp
  mla_reference.cpp
  moe_reference.cpp
  mtp_reference.cpp
  reference.cpp
  weights.cpp)

target_include_directories(gufo_glm5_next PUBLIC
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/src/models/glm5_next)

target_link_libraries(gufo_glm5_next PUBLIC
  gufo_core)
