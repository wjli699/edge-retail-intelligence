#include "core_engine/pipeline.hpp"
#include "core_engine/metadata.hpp"

#include <gstnvdsmeta.h>
#include <gstnvdsinfer.h>
#include <nvdsmeta.h>
#include <nvdsinfer.h>

#include <glib.h>
#include <gst/gst.h>

#include <chrono>
#include <iostream>

namespace edge::retail::core {

// PeopleNet v2.x class order: person=0, bag=1, face=2
static const char* kLabels[] = {"person", "bag", "face"};
static constexpr int kNumLabels   = 3;
static constexpr guint kReidGieId = 2;  // must match gie-unique-id in SGIE config

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

      // Extract ReID embedding from secondary GIE tensor meta (Phase 3+)
      for (NvDsMetaList* l_um = om->obj_user_meta_list; l_um; l_um = l_um->next) {
        auto* um = static_cast<NvDsUserMeta*>(l_um->data);
        if (um->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META) continue;

        auto* tmeta = static_cast<NvDsInferTensorMeta*>(um->user_meta_data);
        if (tmeta->unique_id != kReidGieId) continue;

        for (guint li = 0; li < tmeta->num_output_layers; ++li) {
          const NvDsInferLayerInfo& linfo = tmeta->output_layers_info[li];
          if (linfo.isInput) continue;
          auto* data = static_cast<float*>(tmeta->out_buf_ptrs_host[li]);
          // Flatten all dims to get total element count
          int n = 1;
          for (unsigned di = 0; di < linfo.inferDims.numDims; ++di)
            n *= linfo.inferDims.d[di];
          d.embedding.assign(data, data + n);
          break;
        }
        break;
      }

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
//
// Without ReID (reid_config empty):
//   nvstreammux → nvinfer(PeopleNet) → nvtracker → [osd chain] → sink
//
// With ReID (reid_config set):
//   nvstreammux → nvinfer(PeopleNet) → nvinfer(ReID/SGIE) → nvtracker → [osd chain] → sink
//                                                                ^
//                                                          pad_probe: extracts
//                                                          detections + embeddings
//
// OSD chain (when output.osd_file is set):
//   nvmultistreamtiler → nvdsosd → nvvideoconvert → nvv4l2h264enc → h264parse → matroskamux → filesink
//   Tiler composites all source surfaces into one frame (1 row × N columns) so
//   nvdsosd draws boxes in the correct spatial region per source.
//   Safe because NvDCF visual features are disabled (useColorNames=0, useHog=0),
//   so the tracker does not pin NVMM surfaces concurrently with nvdsosd.
//
// Without OSD: fakesink (headless, for ZMQ/stdout-only operation).
// ---------------------------------------------------------------------------

bool Pipeline::build() {
  gst_init(nullptr, nullptr);
  loop_ = g_main_loop_new(nullptr, FALSE);

  pipeline_  = gst_pipeline_new("edge-retail");
  streammux_ = gst_element_factory_make("nvstreammux", "muxer");
  pgie_      = gst_element_factory_make("nvinfer",     "pgie");
  tracker_   = gst_element_factory_make("nvtracker",   "tracker");

  // Count enabled sources (needed for tiler columns and streammux batch-size)
  guint num_sources = 0;
  for (const auto& s : cfg_.sources)
    if (s.enabled) ++num_sources;

  const bool osd_enabled = !cfg_.output.osd_file.empty();
  if (osd_enabled) {
    tiler_     = gst_element_factory_make("nvmultistreamtiler", "tiler");
    osd_       = gst_element_factory_make("nvdsosd",            "osd");
    videoconv_ = gst_element_factory_make("nvvideoconvert",     "videoconv");
    encoder_   = gst_element_factory_make("nvv4l2h264enc",      "encoder");
    h264parse_ = gst_element_factory_make("h264parse",          "h264parse");
    muxer_     = gst_element_factory_make("matroskamux",        "muxer_mkv");
    sink_      = gst_element_factory_make("filesink",           "sink");
    if (!tiler_ || !osd_ || !videoconv_ || !encoder_ || !h264parse_ || !muxer_ || !sink_) {
      std::cerr << "[pipeline] Failed to create OSD chain elements\n";
      return false;
    }
    // Tile all sources side-by-side: 1 row × N columns.
    // Each tile is 960×540 (16:9), total output = (960*N) × 540.
    const guint tile_w = 960;
    const guint tile_h = 540;
    g_object_set(tiler_,
      "rows",    (guint)1,
      "columns", (guint)num_sources,
      "width",   (guint)(tile_w * num_sources),
      "height",  (guint)tile_h,
      nullptr);
    // process-mode=0: CPU rendering — avoids GPU/NVMM surface contention on Jetson
    g_object_set(osd_, "process-mode", (gint)0, nullptr);
    g_object_set(encoder_, "bitrate", (guint)4000000, nullptr);
    g_object_set(sink_, "location", cfg_.output.osd_file.c_str(),
                        "sync",     (gboolean)FALSE, nullptr);
    std::cerr << "[pipeline] OSD recording → " << cfg_.output.osd_file << "\n";
  } else {
    sink_ = gst_element_factory_make("fakesink", "sink");
    if (!sink_) {
      std::cerr << "[pipeline] Failed to create fakesink\n";
      return false;
    }
    g_object_set(sink_, "sync", (gboolean)FALSE, nullptr);
  }

  if (!pipeline_ || !streammux_ || !pgie_ || !tracker_) {
    std::cerr << "[pipeline] Failed to create core GStreamer elements\n";
    return false;
  }

  // Optional secondary GIE for ReID embedding extraction
  if (!cfg_.models.reid_config.empty()) {
    sgie_ = gst_element_factory_make("nvinfer", "sgie");
    if (!sgie_) {
      std::cerr << "[pipeline] Failed to create secondary nvinfer (sgie)\n";
      return false;
    }
    g_object_set(sgie_, "config-file-path", cfg_.models.reid_config.c_str(), nullptr);
    std::cerr << "[pipeline] ReID SGIE enabled: " << cfg_.models.reid_config << "\n";
  }

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

  // Add all elements and link the main chain
  if (osd_enabled) {
    // nvvideoconvert must output NV12 NVMM for nvv4l2h264enc
    GstCaps* nv12_caps = gst_caps_from_string(
        "video/x-raw(memory:NVMM), format=NV12");

    if (sgie_) {
      gst_bin_add_many(GST_BIN(pipeline_),
                       streammux_, pgie_, sgie_, tracker_,
                       tiler_, osd_, videoconv_, encoder_, h264parse_, muxer_, sink_, nullptr);
      bool ok = gst_element_link_many(streammux_, pgie_, sgie_, tracker_, tiler_, osd_, nullptr)
             && gst_element_link(osd_, videoconv_)
             && gst_element_link_filtered(videoconv_, encoder_, nv12_caps)
             && gst_element_link_many(encoder_, h264parse_, muxer_, sink_, nullptr);
      if (!ok) {
        std::cerr << "[pipeline] Failed to link main chain (SGIE + OSD)\n";
        gst_caps_unref(nv12_caps);
        return false;
      }
    } else {
      gst_bin_add_many(GST_BIN(pipeline_),
                       streammux_, pgie_, tracker_,
                       tiler_, osd_, videoconv_, encoder_, h264parse_, muxer_, sink_, nullptr);
      bool ok = gst_element_link_many(streammux_, pgie_, tracker_, tiler_, osd_, nullptr)
             && gst_element_link(osd_, videoconv_)
             && gst_element_link_filtered(videoconv_, encoder_, nv12_caps)
             && gst_element_link_many(encoder_, h264parse_, muxer_, sink_, nullptr);
      if (!ok) {
        std::cerr << "[pipeline] Failed to link main chain (OSD)\n";
        gst_caps_unref(nv12_caps);
        return false;
      }
    }
    gst_caps_unref(nv12_caps);
  } else {
    if (sgie_) {
      gst_bin_add_many(GST_BIN(pipeline_),
                       streammux_, pgie_, sgie_, tracker_, sink_, nullptr);
      if (!gst_element_link_many(streammux_, pgie_, sgie_, tracker_, sink_, nullptr)) {
        std::cerr << "[pipeline] Failed to link main chain (SGIE)\n";
        return false;
      }
    } else {
      gst_bin_add_many(GST_BIN(pipeline_),
                       streammux_, pgie_, tracker_, sink_, nullptr);
      if (!gst_element_link_many(streammux_, pgie_, tracker_, sink_, nullptr)) {
        std::cerr << "[pipeline] Failed to link main chain\n";
        return false;
      }
    }
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
