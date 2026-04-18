#include "core_engine/metadata.hpp"

#include <cstdint>
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

// Encodes raw bytes as base64. Used for the ReID embedding float array so the
// Python consumer can recover it with:
//   np.frombuffer(base64.b64decode(s), dtype=np.float32)
static std::string base64_encode(const uint8_t* data, size_t len) {
  static const char kChars[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve(((len + 2) / 3) * 4);
  for (size_t i = 0; i < len; i += 3) {
    uint32_t b = static_cast<uint32_t>(data[i]) << 16;
    if (i + 1 < len) b |= static_cast<uint32_t>(data[i + 1]) << 8;
    if (i + 2 < len) b |= static_cast<uint32_t>(data[i + 2]);
    out += kChars[(b >> 18) & 63];
    out += kChars[(b >> 12) & 63];
    out += (i + 1 < len) ? kChars[(b >> 6) & 63] : '=';
    out += (i + 2 < len) ? kChars[b & 63]        : '=';
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
    o << "{\"class\":"   << d.class_id
      << ",\"label\":\"" << escape_json(d.label) << "\""
      << ",\"conf\":"    << d.confidence
      << ",\"bbox\":{"
        << "\"l\":"  << d.bbox.left
        << ",\"t\":" << d.bbox.top
        << ",\"w\":" << d.bbox.width
        << ",\"h\":" << d.bbox.height
      << "}";
    if (d.tracked) o << ",\"tid\":" << d.tracking_id;
    if (!d.embedding.empty()) {
      // Encode the float32 array as base64 to keep the JSON line compact.
      const auto* raw = reinterpret_cast<const uint8_t*>(d.embedding.data());
      o << ",\"emb\":\"" << base64_encode(raw, d.embedding.size() * sizeof(float)) << '"';
    }
    o << '}';
  }

  o << "]}";
  return o.str();
}

}  // namespace edge::retail::core
