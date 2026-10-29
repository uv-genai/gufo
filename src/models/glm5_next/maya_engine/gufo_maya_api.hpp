// Carbon-copy wrapper around Maya Glm5Model (vendored under this directory).
// gufo glm5 prompt/serve call THIS — not gufo's own kernels — so tokens match
// Maya.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gufo::models::glm5_next::maya {

struct Impl;

/// Load the Maya pack (directory with native_experts.txt + shards).
[[nodiscard]] std::unique_ptr<Impl, void (*)(Impl*)> Load(
    const std::string& pack_dir, std::string* err);

/// Batched prompt prefill (Maya Glm5Model::prefill).  `tokens` are the framed
/// prompt ids.  Returns the last prompt token's id (needed as `next`).
bool Prefill(Impl& m, const std::vector<std::int32_t>& tokens,
             std::string* err);

/// Greedy generate up to max_new tokens (EOS 154820 stops).  Appends to out.
bool GenerateGreedy(Impl& m, int max_new, std::vector<std::int32_t>* out,
                    std::string* err);

}  // namespace gufo::models::glm5_next::maya
