#ifndef GUFO_MODELS_GLM5_NEXT_ENGINE_HPP_
#define GUFO_MODELS_GLM5_NEXT_ENGINE_HPP_

#include <cstdint>
#include <memory>
#include <string>

#include "src/models/glm5_next/config.hpp"
#include "src/models/glm5_next/weights.hpp"

namespace gufo::models::glm5_next {

struct ModelOptions {
  std::uint32_t max_context{0};  ///< 0 = use GGUF native.
};

/// Loaded GLM-5.3-Flash model. Phase 1 skeleton: loads config + weights,
/// reports geometry, and estimates resident bytes. Decode is not yet wired.
class Model {
public:
  virtual ~Model() = default;

  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  Model(Model&&) = delete;
  Model& operator=(Model&&) = delete;

  /// Load a `glm5-next` GGUF artifact. Returns nullptr on failure.
  [[nodiscard]] static std::shared_ptr<Model> Load(
      const std::string& model_path, const ModelOptions& options,
      std::string* error_msg = nullptr);

  [[nodiscard]] const Config& GetConfig() const noexcept { return config_; }
  [[nodiscard]] std::uint64_t ResidentBytes() const noexcept {
    return weights_.ResidentBytes();
  }

protected:
  Model() = default;

  Config config_;
  ModelWeights weights_;
};

}  // namespace gufo::models::glm5_next

#endif  // GUFO_MODELS_GLM5_NEXT_ENGINE_HPP_
