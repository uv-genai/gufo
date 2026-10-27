#include "src/models/glm5_next/chat_template.hpp"

#include <algorithm>
#include <sstream>

namespace gufo::models::glm5_next {
namespace {

/// JSON-escape a string for tool definition embedding.
std::string JsonEscape(std::string_view s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (const char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out;
}

/// Emit one tool definition as compact JSON (the `tool_to_json` macro).
std::string ToolToJson(const ChatTool& tool) {
  if (!tool.definition_json.empty() && tool.definition_json != "{}") {
    return tool.definition_json;
  }
  std::string out = "{\"name\": \"" + JsonEscape(tool.name) + "\"";
  if (!tool.description.empty()) {
    out += ", \"description\": \"" + JsonEscape(tool.description) + "\"";
  }
  out += ", \"parameters\": " +
         (tool.parameters_json.empty() ? "{}" : tool.parameters_json) + "}";
  return out;
}

/// Render the tools system block (`<|system|>\n# Tools\n...`).
std::string RenderToolsBlock(std::span<const ChatTool> tools) {
  std::string out;
  out += "<|system|>\n# Tools\n\n";
  out +=
      "You may call one or more functions to assist with the user query.\n\n";
  out +=
      "You are provided with function signatures within <tools></tools> "
      "XML tags:\n<tools>\n";
  for (const auto& tool : tools) {
    out += ToolToJson(tool);
    out += "\n";
  }
  out += "</tools>\n\n";
  out +=
      "For each function call, output the function name and arguments "
      "within the following XML format:\n";
  out +=
      "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key>"
      "<arg_value>{arg-value-1}</arg_value>...</tool_call>";
  return out;
}

/// Render a tool call as the GLM XML format.
std::string RenderToolCall(const ChatMessage::ToolCall& tc) {
  std::string out = "<tool_call>" + tc.name;
  for (const auto& arg : tc.arguments) {
    out += "<arg_key>" + arg.name + "</arg_key>";
    out += "<arg_value>" + arg.value + "</arg_value>";
  }
  out += "</tool_call>";
  return out;
}

/// Render the reasoning-effort system line (if any).
std::string RenderReasoningEffort(ReasoningEffort effort) {
  switch (effort) {
    case ReasoningEffort::kLow:
      return "<|system|>Reasoning Effort: Low";
    case ReasoningEffort::kHigh:
      return "<|system|>Reasoning Effort: High";
    case ReasoningEffort::kMax:
    default:
      return "<|system|>Reasoning Effort: Max";
  }
}

}  // namespace

std::string RenderChat(std::span<const ChatMessage> messages,
                       const ChatTemplateOptions& options) {
  return RenderChat(messages, std::span<const ChatTool>{}, options);
}

std::string RenderChat(std::span<const ChatMessage> messages,
                       std::span<const ChatTool> tools,
                       const ChatTemplateOptions& options) {
  std::string out;
  out += ChatPrefix();  // "[gMASK]<sop>"

  // Reasoning effort system line (always emitted for Max default).
  out += RenderReasoningEffort(options.reasoning_effort);

  // Tools system block.
  if (!tools.empty() || options.tools_present) {
    out += RenderToolsBlock(tools);
  }

  // Messages.
  for (std::size_t i = 0; i < messages.size(); ++i) {
    const auto& m = messages[i];
    if (m.role == "system") {
      out += "<|system|>" + m.content + "\n";
    } else if (m.role == "user") {
      out += "<|user|>" + m.content + "\n";
    } else if (m.role == "assistant") {
      out += "<|assistant|>";
      if (!m.reasoning_content.empty()) {
        out += "<think>" + m.reasoning_content + "</think>";
      }
      if (!m.content.empty()) {
        out += m.content;
      }
      for (const auto& tc : m.tool_calls) {
        out += RenderToolCall(tc);
      }
      out += "\n";
    } else if (m.role == "tool") {
      // Tool response wrapped in output...output.
      out += "output" + m.content + "output";
    }
  }

  // Generation prompt.
  out += "<|assistant|></think>";
  return out;
}

bool ValidateGgufTemplate(const core::GgufReader& reader,
                          std::string* error_msg) {
  const auto tmpl = reader.GetMetadataString("tokenizer.chat_template");
  if (!tmpl.has_value()) {
    if (error_msg)
      *error_msg = "GGUF missing tokenizer.chat_template";
    return false;
  }
  // The GLM-5 template starts with [gMASK]<sop> and uses <|role|> framing.
  const std::string_view t = *tmpl;
  if (t.substr(0, 12) != "[gMASK]<sop>") {
    if (error_msg)
      *error_msg = "chat_template does not start with [gMASK]<sop>";
    return false;
  }
  if (t.find("<|user|>") == std::string_view::npos) {
    if (error_msg)
      *error_msg = "chat_template missing <|user|> framing";
    return false;
  }
  return true;
}

SpecialTokenIds LoadSpecialTokenIds(const core::GgufReader& reader) noexcept {
  SpecialTokenIds ids;
  if (const auto v = reader.GetMetadataUint32("tokenizer.ggml.bos_token_id"))
    ids.sop = *v;
  if (const auto v = reader.GetMetadataUint32("tokenizer.ggml.eos_token_id"))
    ids.eos = *v;
  if (const auto v = reader.GetMetadataUint32("tokenizer.ggml.eot_token_id"))
    ids.eot = *v;
  if (const auto v = reader.GetMetadataUint32("tokenizer.ggml.eom_token_id"))
    ids.eom = *v;
  return ids;
}

}  // namespace gufo::models::glm5_next
