#include "core_engine/app.hpp"
#include "core_engine/pipeline.hpp"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>

namespace edge::retail::core {

namespace {
Pipeline* g_pipeline = nullptr;
}

App::App(AppConfig cfg) : cfg_(std::move(cfg)) {}

void App::stop() {
  if (g_pipeline) g_pipeline->stop();
}

bool App::load_config() {
  try {
    YAML::Node root = YAML::LoadFile(cfg_.config_path);

    if (auto srcs = root["sources"]) {
      for (const auto& s : srcs) {
        SourceConfig sc;
        sc.uri = s["uri"].as<std::string>();
        sc.enabled = s["enabled"] ? s["enabled"].as<bool>() : true;
        cfg_.sources.push_back(sc);
      }
    }

    if (auto mdl = root["models"]) {
      cfg_.models.detector_config = mdl["detector"].as<std::string>("");
      cfg_.models.tracker_config  = mdl["tracker"].as<std::string>("");
    }

    if (auto out = root["output"]) {
      cfg_.output.mode = out["mode"].as<std::string>("stdout");
      if (out["file"]) cfg_.output.file_path = out["file"].as<std::string>();
    }

    return true;
  } catch (const YAML::Exception& e) {
    std::cerr << "[app] Config error: " << e.what() << "\n";
    return false;
  }
}

int App::run() {
  if (!load_config()) return 1;

  if (cfg_.verbose) {
    std::cerr << "[app] Sources   : " << cfg_.sources.size() << "\n";
    std::cerr << "[app] Detector  : " << cfg_.models.detector_config << "\n";
    std::cerr << "[app] Tracker   : " << cfg_.models.tracker_config << "\n";
    std::cerr << "[app] Output    : " << cfg_.output.mode << "\n";
  }

  std::ostream* out = &std::cout;
  std::ofstream file_out;
  if (cfg_.output.mode == "file" && !cfg_.output.file_path.empty()) {
    file_out.open(cfg_.output.file_path, std::ios::app);
    if (!file_out) {
      std::cerr << "[app] Cannot open output file: " << cfg_.output.file_path << "\n";
      return 1;
    }
    out = &file_out;
  }

  Pipeline pipeline(cfg_, [out](const std::string& json) {
    *out << json << "\n";
    out->flush();
  });

  g_pipeline = &pipeline;
  int ret = pipeline.run();
  g_pipeline = nullptr;
  return ret;
}

}  // namespace edge::retail::core
