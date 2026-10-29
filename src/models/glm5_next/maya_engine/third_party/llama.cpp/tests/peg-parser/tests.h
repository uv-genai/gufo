#pragma once

// Common includes for all test files
#include "../testing.h"
#include "chat-peg-parser.h"
#include "json.h"
#include "peg-parser.h"
#include "simple-tokenize.h"

#include <string>
#include <vector>

struct bench_tool_call {
    std::string id;
    std::string name;
    common_json args;
};

// Test function declarations
void test_basic(testing & t);
void test_json_parser(testing & t);
void test_gbnf_generation(testing & t);
void test_unicode(testing & t);
void test_json_serialization(testing & t);
void test_python_dict_parser(testing & t);
