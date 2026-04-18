#include "core_engine/pipeline.hpp"
#include "core_engine/metadata.hpp"

#include <gstnvdsmeta.h>
#include <nvdsmeta.h>

#include <glib.h>
#include <gst/gst.h>

#include <chrono>
#include <iostream>

namespace edge::retail::core {

// PeopleNet v2.x class order: person=0, bag=1, face=2
static const char* kLabels[] = {"person", "bag", "face"};
static constexpr int kNumLabels = 3;

static int64_t now_ms() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

Pipeline::Pipeline(const AppConfig& cfg, EventCallback on_event)
    : cfg_(cfg), on_event_(std::move(on_event)) {}

Pipeline::~Pipeline() { teardown(); }

void Pipeline::teardown() {
  for (auto* sd : source_bin_data_) delete sd;
  source_bin_data_.clear();

  if (pipeline_) {
    gst_element_set_state(pipeline_, GST_STATE_NULL);
    gst_object_unref(pipeline_);
    pipeline_ = nullptr;
  }
  if (loop_) {
    g_main_loop_unref(loop_);
    loop_ = nullptr;
  }
}

// ---------------------------------------------------------------------------
// nvurisrcbin pad-added — connects the decoded video pad to nvstreammux
// ---------------------------------------------------------------------------

void Pipeline::on_pad_added(GstElement* /*element*/, GstPad* pad, gpointer data) {
  auto* sd = static_cast<SourceBinData*>(data);

  // Only handle video pads (vsrc_*); skip audio pads (asrc_*)
  GstCaps* caps = gst_pad_get_current_caps(pad);
  if (!caps) caps = gst_pad_query_caps(pad, nullptr);
  const gchar* mime = gst_structure_get_name(gst_caps_get_structure(caps, 0));
  bool is_video = g_str_has_prefix(mime, "video/");
  gst_caps_unref(caps);
  if (!is_video) return;

  gchar sink_name[32];
  g_snprintf(sink_name, sizeof(sink_name), "sink_%u", sd->source_id);

  // GStreamer 1.16 API (gst_element_request_pad_simple added in 1.20)
  GstPad* mux_sink = gst_element_get_request_pad(sd->streammux, sink_name);
  if (!mux_sink) {
    std::cerr << "[pipeline] Failed to get " << sink_name << " from nvstreammux\n";
    return;
  }
  if (GST_PAD_LINK_FAILED(gst_pad_link(pad, mux_sink))) {
    std::cerr << "[pipeline] Failed to link source " << sd->source_id
              << " → nvstreammux:" << sink_name << "\n";
  } else {
    std::cerr << "[pipeline] Source " << sd->source_id
              << " linked → nvstreammux:" << sink_name << "\n";
  }
  gst_object_unref(mux_sink);
}

// ---------------------------------------------------------------------------
// nvtracker src pad probe — extracts NvDs metadata and emits JSON events
// ---------------------------------------------------------------------------

GstPadProbeReturn Pipeline::on_buffer_probe(GstPad* /*pad*/, GstPadProbeInfo* info,
                                             gpointer user_data) {
  auto* self = static_cast<Pipeline*>(user_data);
  GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);

  NvDsBatchMeta* batch = gst_buffer_get_nvds_batch_meta(buf);
  if (!batch) return GST_PAD_PROBE_OK;

  for (NvDsMetaList* lf = batch->frame_meta_list; lf; lf = lf->next) {
    auto* fm = static_cast<NvDsFrameMeta*>(lf->data);

    FrameEvent evt;
    evt.timestamp_ms = now_ms();
    evt.source_id    = fm->source_id;
    evt.frame_number = fm->frame_num;

    for (NvDsMetaList* lo = fm->obj_meta_list; lo; lo = lo->next) {
      auto* om = static_cast<NvDsObjectMeta*>(lo->data);

      Detection d;
      d.class_id   = om->class_id;
      d.label      = (om->class_id >= 0 && om->class_id < kNumLabels)
                         ? kLabels[om->class_id]
                         : "unknown";
      d.confidence = om->confidence;
      d.bbox       = {om->rect_params.left, om->rect_params.top,
                      om->rect_params.width, om->rect_params.height};
      d.tracking_id = om->object_id;
      d.tracked     = (om->object_id != UNTRACKED_OBJECT_ID);

      evt.detections.push_back(d);
    }

    if (!evt.detections.empty()) {
      self->on_event_(serialize_frame_event(evt));
    }
  }
  return GST_PAD_PROBE_OK;
}

// ---------------------------------------------------------------------------
// Bus watch — handles EOS / errors
// ---------------------------------------------------------------------------

gboolean Pipeline::on_bus_message(GstBus* /*bus*/, GstMessage* msg, gpointer data) {
  auto* self = static_cast<Pipeline*>(data);
  switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS:
      std::cerr << "[pipeline] EOS received\n";
      g_main_loop_quit(self->loop_);
      break;
    case GST_MESSAGE_ERROR: {
      GError* err = nullptr;
      gchar* dbg  = nullptr;
      gst_message_parse_error(msg, &err, &dbg);
      std::cerr << "[pipeline] ERROR: " << err->message
                << (dbg ? std::string(" (") + dbg + ")" : "") << "\n";
      g_error_free(err);
      g_free(dbg);
      g_main_loop_quit(self->loop_);
      break;
    }
    case GST_MESSAGE_WARNING: {
      GError* err = nullptr;
      gchar* dbg  = nullptr;
      gst_message_parse_warning(msg, &err, &dbg);
      std::cerr << "[pipeline] WARNING: " << err->message << "\n";
      g_error_free(err);
      g_free(dbg);
      break;
    }
    default:
      break;
  }
  return TRUE;
}

// ---------------------------------------------------------------------------
// Pipeline construction
//
//   nvurisrcbin(s) --[pad-added]--> nvstreammux
//   nvstreammux --> nvinfer(PeopleNet) --> nvtracker --> nvdsosd --> fakesink
//                                              ^
//                                        pad_probe (emits JSON)
// ---------------------------------------------------------------------------

bool Pipeline::build() {
  gst_init(nullptr, nullptr);
  loop_ = g_main_loop_new(nullptr, FALSE);

  pipeline_  = gst_pipeline_new("edge-retail");
  streammux_ = gst_element_factory_make("nvstreammux", "muxer");
  pgie_      = gst_element_factory_make("nvinfer",     "pgie");
  tracker_   = gst_element_factory_make("nvtracker",   "tracker");
  osd_       = gst_element_factory_make("nvdsosd",     "osd");
  sink_      = gst_element_factory_make("fakesink",    "sink");

  if (!pipeline_ || !streammux_ || !pgie_ || !tracker_ || !osd_ || !sink_) {
    std::cerr << "[pipeline] Failed to create one or more GStreamer elements\n";
    return false;
  }

  // Count enabled sources for batch-size
  guint num_sources = 0;
  for (const auto& s : cfg_.sources)
    if (s.enabled) ++num_sources;

  g_object_set(streammux_,
    "batch-size",           num_sources,
    "width",                (guint)1920,
    "height",               (guint)1080,
    "batched-push-timeout", (gint)4000000,  // 4 s timeout on Jetson RTSP
    "live-source",          (gboolean)TRUE,
    nullptr);

  g_object_set(pgie_, "config-file-path",
               cfg_.models.detector_config.c_str(), nullptr);

  g_object_set(tracker_,
    "ll-lib-file",   "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
    "ll-config-file", cfg_.models.tracker_config.c_str(),
    "tracker-width",  (guint)640,
    "tracker-height", (guint)384,
    "gpu-id",         (guint)0,
    nullptr);

  g_object_set(sink_, "sync", (gboolean)FALSE, nullptr);

  // Add static elements and link the main chain
  gst_bin_add_many(GST_BIN(pipeline_),
                   streammux_, pgie_, tracker_, osd_, sink_, nullptr);
  if (!gst_element_link_many(streammux_, pgie_, tracker_, osd_, sink_, nullptr)) {
    std::cerr << "[pipeline] Failed to link main chain\n";
    return false;
  }

  // Add one nvurisrcbin per enabled source, connect via pad-added signal
  int sid = 0;
  for (const auto& src_cfg : cfg_.sources) {
    if (!src_cfg.enabled) continue;

    GstElement* src_bin = gst_element_factory_make("nvurisrcbin", nullptr);
    if (!src_bin) {
      std::cerr << "[pipeline] Failed to create nvurisrcbin for: " << src_cfg.uri << "\n";
      return false;
    }
    g_object_set(src_bin,
      "uri",    src_cfg.uri.c_str(),
      "gpu-id", (guint)0,
      nullptr);

    gst_bin_add(GST_BIN(pipeline_), src_bin);

    auto* sd = new SourceBinData{sid, streammux_};
    source_bin_data_.push_back(sd);
    g_signal_connect(src_bin, "pad-added", G_CALLBACK(on_pad_added), sd);

    ++sid;
  }

  // Pad probe on tracker src pad — fires every decoded + tracked frame
  GstPad* tracker_src = gst_element_get_static_pad(tracker_, "src");
  gst_pad_add_probe(tracker_src, GST_PAD_PROBE_TYPE_BUFFER,
                    on_buffer_probe, this, nullptr);
  gst_object_unref(tracker_src);

  // Bus watch for EOS / errors
  GstBus* bus = gst_element_get_bus(pipeline_);
  gst_bus_add_watch(bus, on_bus_message, this);
  gst_object_unref(bus);

  return true;
}

// ---------------------------------------------------------------------------
// Public run / stop
// ---------------------------------------------------------------------------

int Pipeline::run() {
  if (!build()) return 1;

  std::cerr << "[pipeline] Starting — source: "
            << (cfg_.sources.empty() ? "(none)" : cfg_.sources[0].uri) << "\n";

  GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    std::cerr << "[pipeline] Failed to set pipeline to PLAYING\n";
    return 1;
  }

  g_main_loop_run(loop_);  // blocks until quit()
  gst_element_set_state(pipeline_, GST_STATE_NULL);
  return 0;
}

void Pipeline::stop() {
  if (loop_) g_main_loop_quit(loop_);
}

}  // namespace edge::retail::core
