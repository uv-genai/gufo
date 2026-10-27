# GLM-5.3-Flash skeleton tests (CPU only in Phase 1).
add_custom_target(glm5_next_tests)

function(add_glm5_next_test name library labels)
  set(target glm5_next_${name}_test)
  add_executable(${target} ${name}_test.cpp)
  target_link_libraries(${target} PRIVATE ${library})
  add_test(NAME glm5_next.${name} COMMAND ${target})
  set_tests_properties(glm5_next.${name} PROPERTIES
    LABELS "glm5_next;${labels}")
  add_dependencies(glm5_next_tests ${target})
endfunction()

add_glm5_next_test(config gufo_glm5_next "cpu")
add_glm5_next_test(chat_template gufo_glm5_next "cpu")
add_glm5_next_test(kda_oracle gufo_glm5_next "cpu")
