#include "gguf/converter.h"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: ninfer-gguf-convert <input.gguf> <output.ninfer> [--profile groupwise-int] [--threads N]\n";
        return 1;
    }

    ninfer::gguf::ConvertOptions options;
    options.input_gguf = argv[1];
    options.output_ninfer = argv[2];
    options.profile = "groupwise-int";
    options.threads = 16;

    for (int i = 3; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if (arg == "--profile" && i + 1 < argc) {
            options.profile = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            options.threads = std::atoi(argv[++i]);
        } else if (arg == "--resources" && i + 1 < argc) {
            options.resources_dir = argv[++i];
        }
    }

    auto start_time = std::chrono::steady_clock::now();
    std::cout << "[GGUF Convert] Converting: " << options.input_gguf.string() << "\n"
              << "              Output:     " << options.output_ninfer.string() << "\n"
              << "              Profile:    " << options.profile << "\n"
              << "              Threads:    " << options.threads << "\n";

    options.progress_callback = [](std::string_view msg, float progress) {
        std::cout << "  [" << std::setw(3) << static_cast<int>(progress * 100.0f) << "%] "
                  << msg << std::endl;
    };

    try {
        ninfer::gguf::convert_gguf_to_ninfer(options);
    } catch (const std::exception& e) {
        std::cerr << "[GGUF Convert] ERROR: " << e.what() << "\n";
        return 1;
    }

    auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
    std::cout << "[GGUF Convert] Done in " << std::fixed << std::setprecision(2) << elapsed << "s.\n";
    return 0;
}
