# GLM-5.3-Flash (glm5-next) model skeleton.
# Phase 1: config + GGUF load + weights binding. No GPU kernels yet.

add_library(gufo_glm5_next STATIC
  config.cpp
  engine.cpp
  weights.cpp)

target_include_directories(gufo_glm5_next PUBLIC
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/src/models/glm5_next)

target_link_libraries(gufo_glm5_next PUBLIC
  gufo_core)
