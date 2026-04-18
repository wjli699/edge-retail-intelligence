#pragma once

#include <string>
#include <vector>

namespace edge::retail::core {

struct SourceConfig {
  std::string uri;
  bool enabled{true};
};

struct ModelConfig {
  std::string detector_config;  // path to nvinfer config file (.txt)
  std::string tracker_config;   // path to nvtracker ll-config-file (.yml)
};

struct OutputConfig {
  std::string mode{"zmq"};              // "stdout" | "file" | "zmq"
  std::string file_path;
  std::string endpoint{"tcp://*:5555"}; // ZMQ bind address (mode=zmq)
};

struct AppConfig {
  std::string config_path;
  bool verbose{false};
  std::vector<SourceConfig> sources;
  ModelConfig models;
  OutputConfig output;
};

class App {
 public:
  explicit App(AppConfig cfg);
  int run();
  void stop();

 private:
  bool load_config();
  AppConfig cfg_;
};

}  // namespace edge::retail::core
