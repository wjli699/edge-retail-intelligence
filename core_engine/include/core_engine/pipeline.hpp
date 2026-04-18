#pragma once

#include "core_engine/app.hpp"

#include <gst/gst.h>

#include <functional>
#include <string>
#include <vector>

namespace edge::retail::core {

using EventCallback = std::function<void(const std::string&)>;

class Pipeline {
 public:
  Pipeline(const AppConfig& cfg, EventCallback on_event);
  ~Pipeline();

  // Builds and runs the pipeline. Blocks until EOS, error, or stop().
  int run();
  void stop();

 private:
  struct SourceBinData {
    int source_id;
    GstElement* streammux;
  };

  bool build();
  void teardown();

  // GStreamer signal/probe callbacks
  static void on_pad_added(GstElement* element, GstPad* pad, gpointer data);
  static GstPadProbeReturn on_buffer_probe(GstPad* pad, GstPadProbeInfo* info, gpointer user_data);
  static gboolean on_bus_message(GstBus* bus, GstMessage* msg, gpointer data);

  const AppConfig& cfg_;
  EventCallback on_event_;

  GstElement* pipeline_{nullptr};
  GstElement* streammux_{nullptr};
  GstElement* pgie_{nullptr};    // nvinfer primary — PeopleNet detection
  GstElement* sgie_{nullptr};    // nvinfer secondary — ReID embeddings (optional)
  GstElement* tracker_{nullptr}; // nvtracker — NvDCF
  GstElement* sink_{nullptr};    // fakesink (headless) OR filesink (osd mode)
  // OSD chain — all null when osd_file is not set
  GstElement* tiler_{nullptr};       // nvmultistreamtiler (N surfaces → 1 tiled frame)
  GstElement* osd_{nullptr};         // nvdsosd
  GstElement* videoconv_{nullptr};   // nvvideoconvert (RGBA NVMM → NV12 NVMM)
  GstElement* encoder_{nullptr};     // nvv4l2h264enc (Jetson HW encoder)
  GstElement* h264parse_{nullptr};   // h264parse (matroskamux needs parsed stream)
  GstElement* muxer_{nullptr};       // matroskamux (MKV container)

  GMainLoop* loop_{nullptr};

  // Per-source callback data; owned by Pipeline, freed in teardown()
  std::vector<SourceBinData*> source_bin_data_;
};

}  // namespace edge::retail::core
