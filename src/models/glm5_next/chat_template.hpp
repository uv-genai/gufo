#ifndef GUFO_MODELS_GLM5_NEXT_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_GLM5_NEXT_CHAT_TEMPLATE_HPP_

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/reasoning.hpp"

namespace gufo::models::glm5_next {

/// Chat message roles supported by the GLM-5 chat template.
struct ChatMessage {
  struct ToolArgument {
    std::string name;
    std::string value;
    bool is_string{true};
  };

  struct ToolCall {
    std::string name;
    std::vector<ToolArgument> arguments;
    std::string id;
  };

  std::string role;  ///< "system", "user", "assistant", "tool"
  std::string content;
  std::string reasoning_content;
  std::vector<ToolCall> tool_calls;
  std::string tool_call_id;
};

struct ChatTool {
  std::string name;
  std::string description;
  std::string parameters_json{"{}"};
  /// Complete function definition (JSON), including optional fields.
  std::string definition_json;
};

struct ChatTemplateOptions {
  ReasoningEffort reasoning_effort{ReasoningEffort::kMax};
  bool clear_thinking{false};
  bool tools_present{false};
};

/// Renders the GLM-5.3-Flash conversation format:
/// `[gMASK]<sop>` + `<|system|>` / `<|user|>` / `<|assistant|>` framing,
/// with reasoning-effort and tool JSON macros from the artifact's template.
[[nodiscard]] std::string RenderChat(std::span<const ChatMessage> messages,
                                     const ChatTemplateOptions& options = {});
[[nodiscard]] std::string RenderChat(std::span<const ChatMessage> messages,
                                     std::span<const ChatTool> tools,
                                     const ChatTemplateOptions& options = {});

/// Validates that the GGUF's embedded chat template matches the pinned
/// GLM-5 format this renderer implements.
[[nodiscard]] bool ValidateGgufTemplate(const core::GgufReader& reader,
                                        std::string* error_msg = nullptr);

/// The control-token prefix every GLM-5 prompt begins with.
[[nodiscard]] constexpr std::string_view ChatPrefix() noexcept {
  return "[gMASK]<sop>";
}

[[nodiscard]] constexpr std::string_view ChatTemplateId() noexcept {
  return "glm5-flash-compiled-v1";
}

/// Special token IDs (from Maya-S v2 GGUF metadata).
struct SpecialTokenIds {
  std::uint32_t gmask{151936};  ///< [gMASK] (placeholder; loaded from GGUF)
  std::uint32_t sop{151937};    ///< <sop>
  std::uint32_t eom{154829};    ///< <|eom|>
  std::uint32_t eos{154820};    ///< <|eos|>
  std::uint32_t eot{154827};    ///< <|eot|>
  std::uint32_t image{154854};  ///< image placeholder
  std::uint32_t video{154855};  ///< video placeholder
};

/// Load special token IDs from GGUF metadata.
[[nodiscard]] SpecialTokenIds LoadSpecialTokenIds(
    const core::GgufReader& reader) noexcept;

}  // namespace gufo::models::glm5_next

#endif  // GUFO_MODELS_GLM5_NEXT_CHAT_TEMPLATE_HPP_
