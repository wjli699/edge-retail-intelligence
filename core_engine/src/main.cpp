#include "core_engine/app.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void print_usage(const char* exe) {
  std::cerr << "Usage: " << exe << " [--config <path>] [--verbose]\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  edge::retail::core::AppConfig cfg;
  cfg.config_path = "configs/default.yaml";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    }
    if (arg == "--verbose" || arg == "-v") {
      cfg.verbose = true;
      continue;
    }
    if (arg == "--config" && i + 1 < argc) {
      cfg.config_path = argv[++i];
      continue;
    }
    std::cerr << "Unknown argument: " << arg << "\n";
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  edge::retail::core::App app(std::move(cfg));
  return app.run();
}
