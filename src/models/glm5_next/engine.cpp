#include "src/models/glm5_next/engine.hpp"

#include <utility>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::glm5_next {

std::shared_ptr<Model> Model::Load(const std::string& model_path,
                                   const ModelOptions& options,
                                   std::string* error_msg) {
  std::string local_error;
  auto reader = core::GgufReader::OpenFile(model_path, &local_error);
  if (reader == nullptr) {
    if (error_msg)
      *error_msg = "failed to open GGUF: " + local_error;
    return nullptr;
  }

  auto config = Config::FromGguf(*reader, &local_error);
  if (!config) {
    if (error_msg)
      *error_msg = "config parse failed: " + local_error;
    return nullptr;
  }

  if (options.max_context > 0 && options.max_context < config->context_length) {
    // Truncation is allowed; we just record it.
    config->context_length = options.max_context;
  }

  auto model = std::shared_ptr<Model>(new Model());
  model->config_ = std::move(*config);

  if (!LoadWeights(*reader, model->config_, &model->weights_, &local_error)) {
    if (error_msg)
      *error_msg = "weight bind failed: " + local_error;
    return nullptr;
  }

  return model;
}

}  // namespace gufo::models::glm5_next
