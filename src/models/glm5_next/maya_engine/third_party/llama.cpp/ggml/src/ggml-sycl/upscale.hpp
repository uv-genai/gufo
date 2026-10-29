#pragma once

#include "common.hpp"
#include "dpct/helper.hpp"

#include <sycl/sycl.hpp>

#define SYCL_UPSCALE_BLOCK_SIZE 256

void ggml_sycl_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
