#include "core_engine/app.hpp"

#include <iostream>

namespace edge::retail::core {

App::App(AppConfig cfg) : cfg_(std::move(cfg)) {}

int App::run() {
  if (cfg_.verbose) {
    std::cerr << "edge_retail_core_engine: config=" << cfg_.config_path << "\n";
  }
  // Phase 1+: load YAML, build DeepStream pipeline, attach pad_probe, emit JSON events.
  std::cout << R"({"event":"core_engine_ready","phase":1,"message":"placeholder — pipeline not linked yet"})"
            << std::endl;
  return 0;
}

}  // namespace edge::retail::core
