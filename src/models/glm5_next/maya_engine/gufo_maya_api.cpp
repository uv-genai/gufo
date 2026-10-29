#include "src/models/glm5_next/maya_engine/gufo_maya_api.hpp"

#include "strata/core/glm_model.hpp"

namespace gufo::models::glm5_next::maya {

struct Impl {
  strata::core::Glm5Model model;
  std::int32_t last_prompt = -1;
};

void Destroy(Impl* p) {
  delete p;
}

std::unique_ptr<Impl, void (*)(Impl*)> Load(const std::string& pack_dir,
                                            std::string* err) {
  std::unique_ptr<Impl, void (*)(Impl*)> impl(new Impl(), Destroy);
  if (!impl->model.load_pack_env(pack_dir, 512, *err)) {
    err->insert(0, "maya Glm5Model load: ");
    return {nullptr, Destroy};
  }
  return impl;
}

bool Prefill(Impl& m, const std::vector<std::int32_t>& tokens,
             std::string* err) {
  if (tokens.empty()) {
    *err = "empty prompt";
    return false;
  }
  m.last_prompt = tokens.back();
  // Maya generate.cpp batched path: all but last via prefill, last is `next`
  // (NextN caches), then forward(last) yields the first logits.
  if (tokens.size() >= 2 && m.model.fast()) {
    std::vector<std::int32_t> head(tokens.begin(), tokens.end() - 1);
    if (!m.model.prefill(head, *err, tokens.back()) && !err->empty()) {
      return false;
    }
    return true;
  }
  std::vector<float> lg;
  return m.model.forward(tokens, lg, *err);
}

bool GenerateGreedy(Impl& m, int max_new, std::vector<std::int32_t>* out,
                    std::string* err) {
  constexpr std::int32_t kEos = 154820;
  if (m.last_prompt < 0) {
    *err = "GenerateGreedy: Prefill not called";
    return false;
  }
  std::vector<float> logits;
  std::int32_t feed = m.last_prompt;
  for (int i = 0; i < max_new; ++i) {
    if (!m.model.forward({feed}, logits, *err)) {
      return false;
    }
    if (logits.empty()) {
      *err = "GenerateGreedy: empty logits";
      return false;
    }
    std::int32_t best = 0;
    float bestv = logits[0];
    for (std::size_t j = 1; j < logits.size(); ++j) {
      if (logits[j] > bestv) {
        bestv = logits[j];
        best = static_cast<std::int32_t>(j);
      }
    }
    out->push_back(best);
    if (best == kEos) {
      break;
    }
    feed = best;
  }
  return true;
}

}  // namespace gufo::models::glm5_next::maya
