#include "core_engine/pipeline.hpp"
#include "core_engine/metadata.hpp"

#include <gstnvdsmeta.h>
#include <gstnvdsinfer.h>
#include <nvdsmeta.h>
#include <nvdsinfer.h>

#include <glib.h>
#include <gst/gst.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>

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
// ReID gallery helpers
// ---------------------------------------------------------------------------

// 8 visually distinct colours (RGBA, 0–1). Same global ID → same colour on
// every camera tile, making cross-camera identity immediately obvious.
static const NvOSD_ColorParams kIdColors[] = {
    {1.00f, 0.20f, 0.20f, 1.0f},  // 0 red
    {0.20f, 0.90f, 0.20f, 1.0f},  // 1 green
    {0.25f, 0.50f, 1.00f, 1.0f},  // 2 blue
    {1.00f, 0.85f, 0.10f, 1.0f},  // 3 yellow
    {1.00f, 0.50f, 0.00f, 1.0f},  // 4 orange
    {0.75f, 0.20f, 1.00f, 1.0f},  // 5 purple
    {0.10f, 0.95f, 0.95f, 1.0f},  // 6 cyan
    {1.00f, 0.40f, 0.75f, 1.0f},  // 7 pink
};
static constexpr int kNumIdColors = static_cast<int>(sizeof(kIdColors) / sizeof(kIdColors[0]));
static constexpr float kReidThreshold = 0.70f;
static constexpr double kGalleryTTL   = 30.0;  // seconds

static std::string make_key(int src, guint64 tid) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%d:%llu", src, static_cast<unsigned long long>(tid));
  return buf;
}

static double mono_sec() {
  using namespace std::chrono;
  return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

static std::vector<float> l2_normalize(const std::vector<float>& v) {
  float sq = 0.f;
  for (float x : v) sq += x * x;
  float inv = 1.f / (std::sqrt(sq) + 1e-8f);
  std::vector<float> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = v[i] * inv;
  return out;
}

static float cosine(const std::vector<float>& a, const std::vector<float>& b) {
  float dot = 0.f;
  size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i) dot += a[i] * b[i];
  return dot;
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

      // Extract ReID embedding from secondary GIE tensor meta
      for (NvDsMetaList* l_um = om->obj_user_meta_list; l_um; l_um = l_um->next) {
        auto* um = static_cast<NvDsUserMeta*>(l_um->data);
        if (um->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META) continue;
        auto* tmeta = static_cast<NvDsInferTensorMeta*>(um->user_meta_data);
        if (tmeta->unique_id != kReidGieId) continue;
        for (guint li = 0; li < tmeta->num_output_layers; ++li) {
          const NvDsInferLayerInfo& linfo = tmeta->output_layers_info[li];
          if (linfo.isInput) continue;
          auto* data = static_cast<float*>(tmeta->out_buf_ptrs_host[li]);
          int n = 1;
          for (unsigned di = 0; di < linfo.inferDims.numDims; ++di)
            n *= linfo.inferDims.d[di];
          d.embedding.assign(data, data + n);
          break;
        }
        break;
      }

      // ── Cross-camera ReID matching + OSD annotation ──────────────────────
      // For tracked persons with an embedding: look up / assign a global ID
      // that is consistent across both cameras, then paint the bounding box
      // with the per-ID colour and show "ID:N" so the viewer can immediately
      // see that the same colour/number = the same physical person.
      const bool osd_active = !self->cfg_.output.osd_file.empty()
                           || self->cfg_.output.osd_display;
      if (d.tracked && om->class_id == 0 && !d.embedding.empty() && osd_active) {

        const std::string key = make_key(static_cast<int>(fm->source_id), om->object_id);
        const double now      = mono_sec();
        auto emb_norm         = l2_normalize(d.embedding);

        std::lock_guard<std::mutex> lock(self->reid_mutex_);

        // Evict stale gallery entries
        for (auto it = self->reid_gallery_.begin(); it != self->reid_gallery_.end(); ) {
          it = (now - it->second.ts > kGalleryTTL)
               ? self->reid_gallery_.erase(it) : std::next(it);
        }

        int global_id = -1;

        // If this (cam, tid) already has a global ID, reuse it (stable label).
        auto ta = self->track_global_.find(key);
        if (ta != self->track_global_.end()) {
          global_id = ta->second;
        } else {
          // First time we see this track — search other cameras for a match.
          float best_sim = kReidThreshold;
          for (const auto& [gkey, entry] : self->reid_gallery_) {
            // Skip same-camera entries
            if (gkey.substr(0, gkey.find(':')) ==
                std::to_string(static_cast<int>(fm->source_id))) continue;
            float sim = cosine(emb_norm, entry.emb);
            if (sim > best_sim) { best_sim = sim; global_id = entry.global_id; }
          }
          if (global_id == -1) global_id = self->next_global_id_++;
          self->track_global_[key] = global_id;
        }

        // Update gallery with latest normalised embedding
        self->reid_gallery_[key] = {emb_norm, global_id, now};

        // Paint bbox and label
        const NvOSD_ColorParams& col = kIdColors[global_id % kNumIdColors];
        om->rect_params.border_color = col;
        om->rect_params.border_width = 3;

        g_free(om->text_params.display_text);
        om->text_params.display_text = g_strdup_printf("ID:%d", global_id);
        om->text_params.x_offset     = static_cast<unsigned int>(om->rect_params.left);
        om->text_params.y_offset     = static_cast<unsigned int>(
            om->rect_params.top > 18 ? om->rect_params.top - 18 : 0);
        om->text_params.font_params.font_name  = const_cast<char*>("Serif");
        om->text_params.font_params.font_size  = 12;
        om->text_params.font_params.font_color = {1.0f, 1.0f, 1.0f, 1.0f};
        om->text_params.set_bg_clr             = 1;
        om->text_params.text_bg_clr            = {col.red * 0.6f, col.green * 0.6f,
                                                   col.blue * 0.6f, 0.75f};
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
//                                                        ^
//                                                  pad_probe (tracker src)
//
// With ReID (reid_config set):
//   nvstreammux → nvinfer(PeopleNet) → nvtracker → nvinfer(ReID/SGIE) → [osd chain] → sink
//                                                                              ^
//                                                                        pad_probe (sgie src)
//
// Tracker runs BEFORE SGIE so that:
//   (a) SGIE attaches embeddings to NvDsObjectMeta entries that already carry
//       stable tracking IDs — the probe can correlate embedding + tid in one pass.
//   (b) NvDCF does not reconstruct NvDsObjectMeta after the SGIE, which would
//       silently drop the SGIE's NvDsUserMeta tensors before the probe fires.
//
// OSD chain (when output.osd_file or output.osd_display is set):
//   nvmultistreamtiler → nvdsosd → [tee →] nvvideoconvert → … → filesink / nveglglessink
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
    // Re-infer every frame — default (0) infers each tracking ID only once
    // in its lifetime, which gives <5% embedding coverage.
    g_object_set(sgie_, "secondary-reinfer-interval", (guint)1, nullptr);
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

  // ── Build OSD downstream chain ────────────────────────────────────────────
  //
  //  display only : tiler → osd → nveglglessink
  //  file only    : tiler → osd → videoconv → encoder → h264parse → matroskamux → filesink
  //  both         : tiler → osd → tee → queue_file → videoconv → … → filesink
  //                                   → queue_disp → nveglglessink
  //  neither      : fakesink (headless)
  //
  const bool file_enabled    = !cfg_.output.osd_file.empty();
  const bool display_enabled = cfg_.output.osd_display;
  const bool osd_enabled     = file_enabled || display_enabled;

  if (osd_enabled) {
    tiler_ = gst_element_factory_make("nvmultistreamtiler", "tiler");
    osd_   = gst_element_factory_make("nvdsosd",            "osd");
    if (!tiler_ || !osd_) {
      std::cerr << "[pipeline] Failed to create tiler/osd elements\n";
      return false;
    }
    const guint tile_w = 960;
    const guint tile_h = 540;
    g_object_set(tiler_,
      "rows",    (guint)1,
      "columns", (guint)num_sources,
      "width",   (guint)(tile_w * num_sources),
      "height",  (guint)tile_h,
      nullptr);
    g_object_set(osd_, "process-mode", (gint)0, nullptr);

    if (file_enabled) {
      videoconv_ = gst_element_factory_make("nvvideoconvert", "videoconv");
      encoder_   = gst_element_factory_make("nvv4l2h264enc",  "encoder");
      h264parse_ = gst_element_factory_make("h264parse",      "h264parse");
      muxer_     = gst_element_factory_make("matroskamux",    "muxer_mkv");
      sink_      = gst_element_factory_make("filesink",        "filesink");
      if (!videoconv_ || !encoder_ || !h264parse_ || !muxer_ || !sink_) {
        std::cerr << "[pipeline] Failed to create file-recording elements\n";
        return false;
      }
      g_object_set(encoder_, "bitrate", (guint)4000000, nullptr);
      g_object_set(sink_, "location", cfg_.output.osd_file.c_str(),
                          "sync",     (gboolean)FALSE, nullptr);
      std::cerr << "[pipeline] OSD recording → " << cfg_.output.osd_file << "\n";
    }

    if (display_enabled) {
      // nvvideoconvert is required before nveglglessink: the OSD outputs NVMM
      // RGBA which the EGL sink cannot consume directly on Jetson.
      videoconv_disp_ = gst_element_factory_make("nvvideoconvert", "videoconv_disp");
      display_sink_   = gst_element_factory_make("nveglglessink",  "display");
      if (!videoconv_disp_ || !display_sink_) {
        std::cerr << "[pipeline] Failed to create display elements\n";
        return false;
      }
      g_object_set(display_sink_, "sync", (gboolean)FALSE, nullptr);
      std::cerr << "[pipeline] Live display window enabled\n";
    }

    if (file_enabled && display_enabled) {
      // Both branches: tee after OSD, one queue per branch.
      tee_        = gst_element_factory_make("tee",   "tee");
      queue_file_ = gst_element_factory_make("queue", "queue_file");
      queue_disp_ = gst_element_factory_make("queue", "queue_disp");
      if (!tee_ || !queue_file_ || !queue_disp_) {
        std::cerr << "[pipeline] Failed to create tee/queue elements\n";
        return false;
      }
    }
  } else {
    sink_ = gst_element_factory_make("fakesink", "sink");
    if (!sink_) { std::cerr << "[pipeline] Failed to create fakesink\n"; return false; }
    g_object_set(sink_, "sync", (gboolean)FALSE, nullptr);
  }

  // ── Add elements to bin and link ──────────────────────────────────────────
  // Helper: the inference tail before OSD depends on whether SGIE is present.
  GstElement* infer_tail = sgie_ ? sgie_ : tracker_;

  if (!osd_enabled) {
    // Headless path
    if (sgie_) {
      gst_bin_add_many(GST_BIN(pipeline_), streammux_, pgie_, tracker_, sgie_, sink_, nullptr);
      if (!gst_element_link_many(streammux_, pgie_, tracker_, sgie_, sink_, nullptr)) {
        std::cerr << "[pipeline] Failed to link headless chain (SGIE)\n"; return false;
      }
    } else {
      gst_bin_add_many(GST_BIN(pipeline_), streammux_, pgie_, tracker_, sink_, nullptr);
      if (!gst_element_link_many(streammux_, pgie_, tracker_, sink_, nullptr)) {
        std::cerr << "[pipeline] Failed to link headless chain\n"; return false;
      }
    }

  } else if (file_enabled && display_enabled) {
    // File + display via tee
    GstCaps* nv12      = gst_caps_from_string("video/x-raw(memory:NVMM), format=NV12");
    // Force CPU memory for display: nveglglessink cannot copy NVMM surfaces.
    GstCaps* rgba_cpu  = gst_caps_from_string("video/x-raw, format=RGBA");
    gst_bin_add_many(GST_BIN(pipeline_),
                     streammux_, pgie_, tracker_,
                     tiler_, osd_, tee_, queue_file_, queue_disp_,
                     videoconv_, encoder_, h264parse_, muxer_, sink_,
                     videoconv_disp_, display_sink_, nullptr);
    if (sgie_) gst_bin_add(GST_BIN(pipeline_), sgie_);
    bool ok = gst_element_link_many(streammux_, pgie_, tracker_, nullptr);
    if (sgie_) ok = ok && gst_element_link(tracker_, sgie_);
    ok = ok
      && gst_element_link_many(infer_tail, tiler_, osd_, tee_, nullptr)
      && gst_element_link_many(tee_, queue_file_, videoconv_, nullptr)
      && gst_element_link_filtered(videoconv_, encoder_, nv12)
      && gst_element_link_many(encoder_, h264parse_, muxer_, sink_, nullptr)
      && gst_element_link_many(tee_, queue_disp_, videoconv_disp_, nullptr)
      && gst_element_link_filtered(videoconv_disp_, display_sink_, rgba_cpu);
    gst_caps_unref(nv12);
    gst_caps_unref(rgba_cpu);
    if (!ok) { std::cerr << "[pipeline] Failed to link file+display chain\n"; return false; }

  } else if (file_enabled) {
    // File only
    GstCaps* nv12 = gst_caps_from_string("video/x-raw(memory:NVMM), format=NV12");
    gst_bin_add_many(GST_BIN(pipeline_),
                     streammux_, pgie_, tracker_,
                     tiler_, osd_, videoconv_, encoder_, h264parse_, muxer_, sink_, nullptr);
    if (sgie_) gst_bin_add(GST_BIN(pipeline_), sgie_);
    bool ok = gst_element_link_many(streammux_, pgie_, tracker_, nullptr);
    if (sgie_) ok = ok && gst_element_link(tracker_, sgie_);
    ok = ok
      && gst_element_link_many(infer_tail, tiler_, osd_, videoconv_, nullptr)
      && gst_element_link_filtered(videoconv_, encoder_, nv12)
      && gst_element_link_many(encoder_, h264parse_, muxer_, sink_, nullptr);
    gst_caps_unref(nv12);
    if (!ok) { std::cerr << "[pipeline] Failed to link file-only chain\n"; return false; }

  } else {
    // Display only — CPU caps force nvvideoconvert to download from NVMM
    GstCaps* rgba_cpu = gst_caps_from_string("video/x-raw, format=RGBA");
    gst_bin_add_many(GST_BIN(pipeline_),
                     streammux_, pgie_, tracker_,
                     tiler_, osd_, videoconv_disp_, display_sink_, nullptr);
    if (sgie_) gst_bin_add(GST_BIN(pipeline_), sgie_);
    bool ok = gst_element_link_many(streammux_, pgie_, tracker_, nullptr);
    if (sgie_) ok = ok && gst_element_link(tracker_, sgie_);
    ok = ok
      && gst_element_link_many(infer_tail, tiler_, osd_, videoconv_disp_, nullptr)
      && gst_element_link_filtered(videoconv_disp_, display_sink_, rgba_cpu);
    gst_caps_unref(rgba_cpu);
    if (!ok) { std::cerr << "[pipeline] Failed to link display-only chain\n"; return false; }
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

  // Probe on the last inference element's src pad so embeddings are already
  // attached when the callback fires. With SGIE: probe sgie src (tracker ran
  // first, so tids are stable and sgie tensors haven't been overwritten).
  // Without SGIE: probe tracker src as before.
  GstElement* probe_elem = sgie_ ? sgie_ : tracker_;
  GstPad* probe_pad = gst_element_get_static_pad(probe_elem, "src");
  gst_pad_add_probe(probe_pad, GST_PAD_PROBE_TYPE_BUFFER,
                    on_buffer_probe, this, nullptr);
  gst_object_unref(probe_pad);

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
