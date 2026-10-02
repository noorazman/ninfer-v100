#pragma once

#include "ninfer/gguf/gguf_parser.h"

#include <filesystem>
#include <functional>
#include <string>

namespace ninfer::gguf {

struct ConvertOptions {
    std::filesystem::path input_gguf;
    std::filesystem::path output_ninfer;
    std::filesystem::path resources_dir; // Optional path containing frontend/ resources
    std::string profile = "groupwise-int";
    int threads = 16;
    std::function<void(std::string_view message, float progress)> progress_callback;
};

// Converts a GGUF model into a valid .ninfer artifact.
// Throws std::runtime_error on error.
void convert_gguf_to_ninfer(const ConvertOptions& options);

} // namespace ninfer::gguf
