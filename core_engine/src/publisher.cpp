#include "core_engine/publisher.hpp"

#include <zmq.h>

#include <iostream>
#include <stdexcept>

namespace edge::retail::core {

Publisher::Publisher(const std::string& endpoint) {
  ctx_ = zmq_ctx_new();
  if (!ctx_) throw std::runtime_error("[publisher] zmq_ctx_new failed");

  sock_ = zmq_socket(ctx_, ZMQ_PUB);
  if (!sock_) {
    zmq_ctx_destroy(ctx_);
    throw std::runtime_error("[publisher] zmq_socket failed");
  }

  // Outgoing HWM — drop oldest messages if the consumer is slow
  int hwm = 1000;
  zmq_setsockopt(sock_, ZMQ_SNDHWM, &hwm, sizeof(hwm));

  if (zmq_bind(sock_, endpoint.c_str()) != 0) {
    zmq_close(sock_);
    zmq_ctx_destroy(ctx_);
    throw std::runtime_error("[publisher] zmq_bind failed on " + endpoint);
  }

  std::cerr << "[publisher] ZMQ PUB bound to " << endpoint << "\n";
}

Publisher::~Publisher() {
  if (sock_) { zmq_close(sock_);      sock_ = nullptr; }
  if (ctx_)  { zmq_ctx_destroy(ctx_); ctx_  = nullptr; }
}

void Publisher::send(const std::string& msg) {
  // zmq_send is thread-safe per socket only when called from one thread;
  // the pipeline pad probe runs on a GStreamer streaming thread, so this
  // is safe as long as send() is only called from that single thread.
  if (zmq_send(sock_, msg.data(), msg.size(), ZMQ_DONTWAIT) < 0) {
    // EAGAIN means HWM hit and no receiver — silently drop
    if (errno != EAGAIN) {
      std::cerr << "[publisher] zmq_send error: " << zmq_strerror(errno) << "\n";
    }
  }
}

}  // namespace edge::retail::core
