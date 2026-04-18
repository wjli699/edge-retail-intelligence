#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace edge::retail::core {

struct BBox {
  float left, top, width, height;
};

struct Detection {
  int class_id;
  std::string label;
  float confidence;
  BBox bbox;
  uint64_t tracking_id{0};
  bool tracked{false};
  // ReID embedding from secondary GIE (Phase 3+). Empty when SGIE not configured.
  // Serialised as a base64-encoded block of little-endian float32 values.
  std::vector<float> embedding;
};

struct FrameEvent {
  int64_t timestamp_ms;
  int source_id;
  uint64_t frame_number;
  std::vector<Detection> detections;
};

// Returns a compact JSON line suitable for stdout / JSONL file.
std::string serialize_frame_event(const FrameEvent& evt);

}  // namespace edge::retail::core
