#include "src/models/glm5_next/chat_template.hpp"

#include <cassert>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

#include "src/models/qwen/tokenizer.hpp"

namespace glm5 = gufo::models::glm5_next;
namespace tok = gufo::tokenization;

namespace {

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

// ---- Chat template golden tests ----

void TestChatPrefix() {
  Require(glm5::ChatPrefix() == "[gMASK]<sop>", "wrong chat prefix");
  std::cout << "TestChatPrefix: PASS\n";
}

void TestSimpleUserAssistant() {
  std::vector<glm5::ChatMessage> msgs = {
      {.role = "user", .content = "Hello"},
  };
  auto result = glm5::RenderChat(msgs);
  // Must start with [gMASK]<sop>
  Require(result.substr(0, 12) == "[gMASK]<sop>", "missing prefix");
  // Must contain <|user|>
  Require(result.find("<|user|>Hello") != std::string::npos,
          "missing user content");
  // Must end with generation prompt
  Require(result.find("<|assistant|></think>") != std::string::npos,
          "missing generation prompt");
  std::cout << "TestSimpleUserAssistant: PASS\n";
}

void TestSystemUser() {
  std::vector<glm5::ChatMessage> msgs = {
      {.role = "system", .content = "You are helpful."},
      {.role = "user", .content = "Hi"},
  };
  auto result = glm5::RenderChat(msgs);
  Require(result.find("<|system|>You are helpful.") != std::string::npos,
          "missing system content");
  Require(result.find("<|user|>Hi") != std::string::npos, "missing user");
  std::cout << "TestSystemUser: PASS\n";
}

void TestReasoningEffort() {
  std::vector<glm5::ChatMessage> msgs = {
      {.role = "user", .content = "Think hard"},
  };
  {
    glm5::ChatTemplateOptions opts;
    opts.reasoning_effort = gufo::ReasoningEffort::kLow;
    auto result = glm5::RenderChat(msgs, opts);
    Require(result.find("Reasoning Effort: Low") != std::string::npos,
            "missing low reasoning effort");
  }
  {
    glm5::ChatTemplateOptions opts;
    opts.reasoning_effort = gufo::ReasoningEffort::kHigh;
    auto result = glm5::RenderChat(msgs, opts);
    Require(result.find("Reasoning Effort: High") != std::string::npos,
            "missing high reasoning effort");
  }
  {
    glm5::ChatTemplateOptions opts;
    opts.reasoning_effort = gufo::ReasoningEffort::kMax;
    auto result = glm5::RenderChat(msgs, opts);
    Require(result.find("Reasoning Effort: Max") != std::string::npos,
            "missing max reasoning effort");
  }
  std::cout << "TestReasoningEffort: PASS\n";
}

void TestToolsRendering() {
  std::vector<glm5::ChatTool> tools = {
      {.name = "get_weather",
       .description = "Get weather for a city",
       .parameters_json =
           "{\"type\": \"object\", \"properties\": {\"city\": {\"type\": "
           "\"string\"}}}"},
  };
  std::vector<glm5::ChatMessage> msgs = {
      {.role = "user", .content = "Weather in Paris?"},
  };
  auto result = glm5::RenderChat(msgs, tools);
  Require(result.find("<tools>") != std::string::npos, "missing <tools>");
  Require(result.find("get_weather") != std::string::npos, "missing tool name");
  Require(result.find("</tools>") != std::string::npos, "missing </tools>");
  Require(
      result.find("You may call one or more functions") != std::string::npos,
      "missing tools instruction");
  std::cout << "TestToolsRendering: PASS\n";
}

void TestToolCallInAssistant() {
  glm5::ChatMessage::ToolCall tc;
  tc.name = "get_weather";
  tc.arguments = {{"city", "Paris", true}};
  tc.id = "call_1";

  std::vector<glm5::ChatMessage> msgs = {
      {.role = "user", .content = "Weather?"},
      {.role = "assistant", .content = "", .tool_calls = {tc}},
      {.role = "tool", .content = "Sunny, 25C", .tool_call_id = "call_1"},
  };
  auto result = glm5::RenderChat(msgs);
  Require(result.find("<tool_call>get_weather") != std::string::npos,
          "missing tool call XML");
  Require(result.find("<arg_key>city</arg_key>") != std::string::npos,
          "missing arg_key");
  Require(result.find("<arg_value>Paris</arg_value>") != std::string::npos,
          "missing arg_value");
  Require(result.find("outputSunny, 25Coutput") != std::string::npos,
          "missing tool response");
  std::cout << "TestToolCallInAssistant: PASS\n";
}

void TestMultiTurn() {
  std::vector<glm5::ChatMessage> msgs = {
      {.role = "user", .content = "Hello"},
      {.role = "assistant", .content = "Hi there!"},
      {.role = "user", .content = "How are you?"},
      {.role = "assistant", .content = "I'm fine."},
  };
  auto result = glm5::RenderChat(msgs);
  Require(result.find("<|user|>Hello") != std::string::npos, "missing turn 1");
  Require(result.find("<|assistant|>Hi there!") != std::string::npos,
          "missing reply 1");
  Require(result.find("<|user|>How are you?") != std::string::npos,
          "missing turn 2");
  Require(result.find("<|assistant|>I'm fine.") != std::string::npos,
          "missing reply 2");
  std::cout << "TestMultiTurn: PASS\n";
}

// ---- Tokenizer GLM4 pre-tokenizer tests ----

// Build a small synthetic vocab for testing tokenizer behavior.
void TestTokenizerBasic() {
  std::vector<std::string> tokens;
  std::vector<std::string> merges;
  // Byte-level tokens for ASCII 0-255.
  for (int i = 0; i < 256; ++i) {
    std::string s;
    if (i >= '!' && i <= '~') {
      s = std::string(1, static_cast<char>(i));
    } else {
      s = "<0x" + std::to_string(i) + ">";
      if (i < 16)
        s = "<0x0" + std::to_string(i) + ">";
    }
    tokens.push_back(s);
  }
  // "hello" as a multi-char token (index 256).
  tokens.push_back("hello");

  auto tokenizer =
      tok::QwenTokenizer::CreateFromVocabulary(tokens, merges, {}, nullptr);
  Require(tokenizer != nullptr, "tokenizer creation failed");

  // Single ASCII chars map to their byte tokens.
  auto ids = tokenizer->Encode("h", {});
  Require(ids.size() == 1, "h should be 1 token");

  // "hello" with no merges and no ignore_merges: 5 byte tokens.
  ids = tokenizer->Encode("hello", {});
  Require(ids.size() == 5, "hello without ignore_merges = 5 byte tokens");
  std::cout << "TestTokenizerBasic: PASS\n";
}

void TestTokenizerMultiChar() {
  // With merges: h+e -> he, he+l -> hel, hel+l -> hell, hell+o -> hello.
  std::vector<std::string> tokens;
  std::vector<std::string> merges = {"h e", "he l", "hel l", "hell o"};
  for (int i = 0; i < 256; ++i) {
    std::string s;
    if (i >= '!' && i <= '~') {
      s = std::string(1, static_cast<char>(i));
    } else {
      s = "<0x" + std::to_string(i) + ">";
      if (i < 16)
        s = "<0x0" + std::to_string(i) + ">";
    }
    tokens.push_back(s);
  }
  tokens.push_back("he");     // 256
  tokens.push_back("hel");    // 257
  tokens.push_back("hell");   // 258
  tokens.push_back("hello");  // 259

  auto tokenizer =
      tok::QwenTokenizer::CreateFromVocabulary(tokens, merges, {}, nullptr);
  Require(tokenizer != nullptr, "tokenizer creation failed");

  auto ids = tokenizer->Encode("hello", {});
  Require(ids.size() == 1, "hello with merges should be 1 token");
  Require(ids[0] == 259, "hello token ID wrong");
  std::cout << "TestTokenizerMultiChar: PASS\n";
}

}  // namespace

int main() {
  try {
    TestChatPrefix();
    TestSimpleUserAssistant();
    TestSystemUser();
    TestReasoningEffort();
    TestToolsRendering();
    TestToolCallInAssistant();
    TestMultiTurn();
    TestTokenizerBasic();
    TestTokenizerMultiChar();
    std::cout << "All glm5_next chat template tests passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
