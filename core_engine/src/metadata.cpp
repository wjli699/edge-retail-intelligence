#include "core_engine/metadata.hpp"

#include <iomanip>
#include <sstream>

namespace edge::retail::core {

static std::string escape_json(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    switch (c) {
      case '"':  out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n";  break;
      case '\r': out += "\\r";  break;
      default:   out += c;
    }
  }
  return out;
}

std::string serialize_frame_event(const FrameEvent& evt) {
  std::ostringstream o;
  o << std::fixed << std::setprecision(4);

  o << "{\"event\":\"frame\""
    << ",\"ts_ms\":"     << evt.timestamp_ms
    << ",\"source_id\":" << evt.source_id
    << ",\"frame\":"     << evt.frame_number
    << ",\"detections\":[";

  for (size_t i = 0; i < evt.detections.size(); ++i) {
    const auto& d = evt.detections[i];
    if (i) o << ',';
    o << "{\"class\":"  << d.class_id
      << ",\"label\":\"" << escape_json(d.label) << "\""
      << ",\"conf\":"   << d.confidence
      << ",\"bbox\":{"
        << "\"l\":"  << d.bbox.left
        << ",\"t\":" << d.bbox.top
        << ",\"w\":" << d.bbox.width
        << ",\"h\":" << d.bbox.height
      << "}";
    if (d.tracked) o << ",\"tid\":" << d.tracking_id;
    o << '}';
  }

  o << "]}";
  return o.str();
}

}  // namespace edge::retail::core
