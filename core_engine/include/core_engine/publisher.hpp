#pragma once

#include <string>

namespace edge::retail::core {

// ZeroMQ PUB socket wrapper.
// Binds on construction, publishes newline-delimited JSON messages.
// All messages are sent with an empty topic prefix so any SUB socket
// subscribed with "" receives everything.
class Publisher {
 public:
  explicit Publisher(const std::string& endpoint);
  ~Publisher();

  // Non-copyable
  Publisher(const Publisher&) = delete;
  Publisher& operator=(const Publisher&) = delete;

  void send(const std::string& msg);

 private:
  void* ctx_{nullptr};
  void* sock_{nullptr};
};

}  // namespace edge::retail::core
