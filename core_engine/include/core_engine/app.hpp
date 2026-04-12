#pragma once

#include <string>

namespace edge::retail::core {

/// Application entry for the C++ core engine (DeepStream pipeline to be wired in Phase 1+).
struct AppConfig {
  std::string config_path;
  bool verbose{false};
};

class App {
 public:
  explicit App(AppConfig cfg);
  /// Parse config, initialize subsystems. Returns 0 on success, non-zero on fatal error.
  int run();

 private:
  AppConfig cfg_;
};

}  // namespace edge::retail::core
