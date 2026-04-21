# Learning Notes: Building a DeepStream Edge AI Pipeline on Jetson

This document captures the full learning arc of this project — from a blank Jetson board to a working multi-camera, real-time AI pipeline with person detection, tracking, ReID-based cross-camera identity, and annotated video output. It is written as a practical guide for anyone wanting to understand the *why* behind the choices, not just the *what*.

---

## Table of Contents

1. [Environment Setup — Jetson + JetPack + DeepStream](#1-environment-setup)
2. [GStreamer Fundamentals](#2-gstreamer-fundamentals)
3. [DeepStream Plugin Ecosystem](#3-deepstream-plugin-ecosystem)
4. [Building the Pipeline in C++](#4-building-the-pipeline-in-c)
5. [Primary Inference — PeopleNet Detection](#5-primary-inference--peoplenet-detection)
6. [Object Tracking — NvDCF](#6-object-tracking--nvdcf)
7. [Secondary Inference — ReID Embeddings](#7-secondary-inference--reid-embeddings)
8. [Testing AI Models — Evaluation Workflow](#8-testing-ai-models--evaluation-workflow)
9. [OSD and Video Output](#9-osd-and-video-output)
10. [Cross-Camera Identity — C++ Gallery](#10-cross-camera-identity--c-gallery)
11. [Decoupled Messaging — ZeroMQ](#11-decoupled-messaging--zeromq)
12. [Key Pitfalls and Hard-Won Lessons](#12-key-pitfalls-and-hard-won-lessons)
13. [Validation Checklist](#13-validation-checklist)

---

## 1. Environment Setup

### Hardware

**NVIDIA Jetson Orin NX** is a system-on-module for edge inference. Key characteristics:
- Ampere GPU with 16 CUDA cores (smaller than a discrete GPU, bigger than a mobile GPU)
- Shared DRAM between CPU and GPU — called **unified memory**
- Hardware video encoders/decoders (NVDEC/NVENC) built into the SoC
- JetPack ships everything pre-installed: CUDA, cuDNN, TensorRT, GStreamer, DeepStream

Find your JetPack version:
```bash
cat /etc/nv_tegra_release
# R35 (release), REVISION: 3.1  →  JetPack 5.1.1
```

Check TensorRT (critical for picking compatible tools):
```bash
dpkg -l libnvinfer8
# 8.5.2-1+cuda11.4
```

### Software stack

```
JetPack 5.x
 ├── L4T Linux kernel (5.10 with NVIDIA Tegra patches)
 ├── CUDA 11.4
 ├── TensorRT 8.5
 ├── cuDNN 8.6
 ├── GStreamer 1.16.3 (with NVIDIA plugins)
 └── DeepStream 6.2 SDK
      ├── /opt/nvidia/deepstream/deepstream-6.2/
      ├── GStreamer plugins: nvdec, nvinfer, nvtracker, nvdsosd, …
      └── C headers: gstnvdsmeta.h, nvdsinfer.h, …
```

### Installing project dependencies

DeepStream plugins and CUDA already ship with JetPack. Additional dependencies:

```bash
# GStreamer dev headers (for C++ pipeline code)
sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

# YAML config loader
sudo apt install libyaml-cpp-dev

# ZeroMQ (Phase 2 messaging)
sudo apt install libzmq3-dev
pip3 install pyzmq

# Python tooling
pip3 install numpy
```

### CMake and build

The CMakeLists.txt locates DeepStream headers and libraries using `pkg-config` and a path variable:

```cmake
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
# DeepStream headers are at DS_SDK_PATH/sources/includes/
```

Build:
```bash
cd core_engine
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

Override the SDK path if not at the default location:
```bash
cmake -DDS_SDK_PATH=/opt/nvidia/deepstream/deepstream-6.2 ..
```

---

## 2. GStreamer Fundamentals

DeepStream is built entirely on GStreamer. Understanding GStreamer first makes DeepStream much less magic.

### Elements, Pads, Pipelines

A **GStreamer pipeline** is a directed graph of **elements** connected through **pads**.

```
element_A [src pad] ---caps negotiation---> [sink pad] element_B
```

- **Source pad** (`src`): where data flows *out* of an element
- **Sink pad** (`sink`): where data flows *into* an element
- **Caps** (capabilities): the format description of the data on a pad (e.g. `video/x-raw, format=NV12, width=1920, height=1080`)

Creating and linking elements:
```cpp
GstElement* pipeline = gst_pipeline_new("my-pipeline");
GstElement* src      = gst_element_factory_make("videotestsrc", "src");
GstElement* sink     = gst_element_factory_make("fakesink", "sink");

gst_bin_add_many(GST_BIN(pipeline), src, sink, nullptr);
gst_element_link(src, sink);   // negotiates caps automatically
```

Running:
```cpp
GMainLoop* loop = g_main_loop_new(nullptr, FALSE);
gst_element_set_state(pipeline, GST_STATE_PLAYING);
g_main_loop_run(loop);   // blocks; GLib event loop drives the pipeline
```

### Dynamic pads

Some elements emit pads only when they know the stream format (after decoding a header). `nvurisrcbin` works this way — it probes the RTSP stream, then emits a `pad-added` signal with the decoded video pad:

```cpp
g_signal_connect(src_bin, "pad-added", G_CALLBACK(on_pad_added), user_data);

// In the callback:
static void on_pad_added(GstElement*, GstPad* pad, gpointer data) {
    // Request a sink pad from nvstreammux and link
    GstPad* mux_sink = gst_element_get_request_pad(mux, "sink_0");
    gst_pad_link(pad, mux_sink);
    gst_object_unref(mux_sink);
}
```

In GStreamer 1.16 (Jetson JetPack 5.x), the API is `gst_element_get_request_pad`. The simpler `gst_element_request_pad_simple` was only added in 1.20.

### Pad probes — intercepting the stream

A **pad probe** is a callback that fires every time a buffer passes through a pad, without disrupting the pipeline. It is the primary way to read or modify DeepStream metadata:

```cpp
GstPad* pad = gst_element_get_static_pad(element, "src");
gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, my_callback, user_data, nullptr);
gst_object_unref(pad);

// Callback:
GstPadProbeReturn my_callback(GstPad*, GstPadProbeInfo* info, gpointer data) {
    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    NvDsBatchMeta* batch = gst_buffer_get_nvds_batch_meta(buf);
    // … read NvDsFrameMeta, NvDsObjectMeta, …
    return GST_PAD_PROBE_OK;  // pass buffer through unchanged
}
```

### Bus messages

The **bus** carries asynchronous messages (errors, EOS, state changes) from elements to the application:

```cpp
GstBus* bus = gst_element_get_bus(pipeline);
gst_bus_add_watch(bus, on_bus_message, user_data);
gst_object_unref(bus);

static gboolean on_bus_message(GstBus*, GstMessage* msg, gpointer data) {
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:   g_main_loop_quit(loop); break;
        case GST_MESSAGE_ERROR: /* log and quit */ break;
    }
    return TRUE;
}
```

### Caps filters

When linking two elements that could negotiate several formats, a `capsfilter` forces a specific format:

```cpp
GstCaps* nv12 = gst_caps_from_string("video/x-raw(memory:NVMM), format=NV12");
gst_element_link_filtered(src, dst, nv12);
gst_caps_unref(nv12);
```

This is critical when connecting NVMM (GPU memory) elements to CPU-side elements — the caps must explicitly declare `(memory:NVMM)` or its absence.

---

## 3. DeepStream Plugin Ecosystem

DeepStream adds a set of GStreamer plugins optimised for NVIDIA hardware. Each one is a standard GStreamer element.

### Core plugins

| Plugin | GType name | Purpose |
|--------|-----------|---------|
| `nvurisrcbin` | `GstNvUriBinSrc` | Decodes RTSP/file/USB sources using NVDEC; emits decoded NVMM frames |
| `nvstreammux` | `GstNvStreamMux` | Batches frames from multiple sources into `NvDsBatchMeta` |
| `nvinfer` | `GstNvInfer` | Runs TensorRT inference (PGIE or SGIE) |
| `nvtracker` | `GstNvTracker` | Per-object tracking; loads tracker library at runtime |
| `nvmultistreamtiler` | `GstNvMultiStreamTiler` | Composites multi-source frames into a single tiled frame |
| `nvdsosd` | `GstNvDsOsd` | Draws bounding boxes and labels from `NvDsObjectMeta` |
| `nvvideoconvert` | `GstNvVideoConvert` | Format/memory conversion between NVMM formats and CPU |
| `nvv4l2h264enc` | `GstNvV4l2H264Enc` | Jetson hardware H.264 encoder (V4L2 interface to NVENC) |

### NVMM — the zero-copy memory model

DeepStream elements pass video frames through **NVMM** (NVIDIA Media Memory Manager) buffers — GPU/DMA memory that doesn't need CPU round-trips. GStreamer caps with `(memory:NVMM)` indicate NVMM buffers.

Key rule: elements that need CPU access (like `nveglglessink` for display, or `appsink` for Python access) need a format conversion step that downloads from NVMM to system RAM. `nvvideoconvert` handles this, but only if the downstream caps don't include `(memory:NVMM)`.

### Metadata: the `NvDs` struct hierarchy

DeepStream metadata is attached to each `GstBuffer` as a side-channel:

```
GstBuffer
  └── NvDsBatchMeta (gst_buffer_get_nvds_batch_meta)
        └── NvDsFrameMeta[] (frame_meta_list)
              ├── source_id, frame_num, ntp_timestamp
              └── NvDsObjectMeta[] (obj_meta_list)
                    ├── class_id, confidence, rect_params (bbox)
                    ├── object_id (tracking ID from nvtracker)
                    ├── text_params (OSD label — you can overwrite this)
                    ├── rect_params.border_color (OSD bbox color — you can overwrite this)
                    └── obj_user_meta_list
                          └── NvDsInferTensorMeta (SGIE output tensors)
```

The entire pipeline shares these structs by reference — modifying `om->rect_params.border_color` in the pad probe changes what `nvdsosd` draws downstream, without any copy.

---

## 4. Building the Pipeline in C++

### Project structure

```
Pipeline::build()   → creates all GStreamer elements, links them, installs probe
Pipeline::run()     → gst_element_set_state(PLAYING), blocks on g_main_loop_run
Pipeline::stop()    → g_main_loop_quit (called from signal handler)
Pipeline::teardown()→ gst_element_set_state(NULL), gst_object_unref
```

### Element creation pattern

```cpp
GstElement* pgie = gst_element_factory_make("nvinfer", "pgie");
if (!pgie) {
    // Always check — a missing plugin returns nullptr silently
    std::cerr << "Failed to create nvinfer\n";
    return false;
}
// Configure via GObject properties:
g_object_set(pgie, "config-file-path", "path/to/config.txt", nullptr);
```

`g_object_set` uses a null-terminated vararg list of `"property-name", value, ..., nullptr`. Getting the type wrong (e.g., passing a plain `int` where a `guint` is expected) causes silent type mismatch bugs on some compilers. Always cast explicitly: `(guint)1`, `(gboolean)TRUE`.

### The multi-source pattern

`nvstreammux` batches frames from N cameras into one `NvDsBatchMeta`. Each source requires:
1. One `nvurisrcbin` (creates the source element)
2. One `pad-added` signal handler that links the decoded video pad to `streammux.sink_N`
3. `batch-size` set to N on `nvstreammux`

```cpp
g_object_set(streammux_,
    "batch-size",           num_sources,   // N cameras
    "width",                (guint)1920,
    "height",               (guint)1080,
    "batched-push-timeout", (gint)4000000, // 4s — important for live RTSP
    "live-source",          (gboolean)TRUE,
    nullptr);
```

`batched-push-timeout` is critical for RTSP: it tells the mux how long to wait for all sources before pushing a partial batch. Without it, a lagging camera stalls the whole pipeline.

### Linking multiple topologies

The pipeline topology changes depending on config (SGIE enabled? OSD file? Live display?). Use conditional linking rather than always building the full graph:

```cpp
GstElement* infer_tail = sgie_ ? sgie_ : tracker_;  // last inference element

// Then link infer_tail → tiler → osd → …
gst_element_link_many(infer_tail, tiler_, osd_, ...);
```

This pattern makes the pipeline composable without combinatorial link code.

---

## 5. Primary Inference — PeopleNet Detection

### nvinfer config file structure

`nvinfer` is configured entirely through a `.txt` file. Key fields for a detector:

```ini
[property]
gpu-id=0
process-mode=1          # 1=primary (whole frame), 2=secondary (object crops)
network-type=0          # 0=detector, 1=classifier, 100=other (raw output)
gie-unique-id=1         # integer ID; other elements reference this

# Model files
onnx-file=path/to/model.onnx
model-engine-file=path/to/model.engine   # TRT builds this if absent

# Input normalization
net-scale-factor=0.0039215697906911373   # 1/255
model-color-format=0    # 0=RGB, 1=BGR, 2=GRAY

infer-dims=3;544;960    # C;H;W

# Precision
network-mode=2          # 0=FP32, 1=INT8, 2=FP16

batch-size=1
num-detected-classes=3

# Class filtering — removes bag(1) and face(2) before they reach the tracker
filter-out-class-ids=1;2

# Detector-specific
cluster-mode=2          # 2=NMS
output-blob-names=output_bbox/BiasAdd:0;output_cov/Sigmoid:0

[class-attrs-0]
pre-cluster-threshold=0.3   # per-class confidence threshold
topk=20
nms-iou-threshold=0.5
```

### TensorRT engine auto-build

On first run, nvinfer:
1. Checks if `model-engine-file` exists and is newer than `onnx-file`
2. If not, compiles the ONNX to a TRT engine (3–10 min on Jetson Orin NX)
3. Saves the engine to disk; subsequent runs load it in seconds

You can pre-build manually with `trtexec`:
```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=model.onnx \
  --fp16 \
  --saveEngine=model.engine
```

For models with dynamic output shapes (e.g., the ReID model's `[?, ?]` output), add shape constraints:
```bash
--minShapes=input:1x3x256x128 \
--optShapes=input:8x3x256x128 \
--maxShapes=input:8x3x256x128
```

**Engines are not portable.** They encode GPU-specific kernel choices. An engine built on Jetson Orin NX will not load on a desktop RTX GPU. Always rebuild on the target device.

### Model paths are relative to the config file

A subtle but important convention: paths in `.txt` config files are relative to the *config file's directory*, not the working directory. This means if your configs live in `core_engine/configs/` and models in `core_engine/models/`, use:
```ini
onnx-file=../../models/peoplenet/resnet34_peoplenet_int8.onnx
```

### Class filtering before the tracker

PeopleNet detects three classes: person (0), bag (1), face (2). NvDCF assigns tracking IDs from a single global counter across all classes. Without `filter-out-class-ids=1;2` in the PGIE config, bags and faces get tracking IDs and show up in OSD boxes, which is confusing. Filter them before they reach the tracker so they never enter `NvDsObjectMeta`.

---

## 6. Object Tracking — NvDCF

### How nvtracker works

`nvtracker` is a GStreamer plugin that wraps a pluggable tracker library. This project uses `libnvds_nvmultiobjecttracker.so` (ships with DeepStream), which implements the **NvDCF** (NVIDIA Discriminative Correlation Filter) tracker.

The tracker:
1. Receives `NvDsObjectMeta` from the upstream detector for each frame
2. Associates detections with existing tracks using IoU + optional visual similarity
3. Assigns a stable `object_id` (tracking ID) to each track
4. Maintains tracks across frames even when the detector misses an object

Configuration via `ll-config-file` (YAML):
```cpp
g_object_set(tracker_,
    "ll-lib-file",    "/path/to/libnvds_nvmultiobjecttracker.so",
    "ll-config-file", "configs/config_tracker_NvDCF.yml",
    "tracker-width",  (guint)640,
    "tracker-height", (guint)384,
    "gpu-id",         (guint)0,
    nullptr);
```

Tracker resolution (`tracker-width/height`) is where the tracker processes frames. Larger = more accurate association; smaller = faster. 640×384 is a good balance on Orin NX.

### NVMM surface pinning problem

NvDCF's visual tracking mode (`visualTrackerType=1`) pins NVMM surfaces to read pixel data for the correlation filter. This races with `nvdsosd` and other elements that also map the buffer, causing:
```
NvDecGetSurfPinHandle: Surface not registered
```

**Fix**: disable visual tracking by setting `visualTrackerType: 0` in the config. The tracker falls back to pure IoU + Kalman filter, which doesn't touch pixels:

```yaml
VisualTracker:
  visualTrackerType: 0   # DUMMY — no pixel access
```

Cross-camera ReID (via SGIE) handles appearance matching separately, so visual tracking in NvDCF isn't needed.

### Key NvDCF parameters explained

```yaml
TargetManagement:
  maxShadowTrackingAge: 90   # frames to keep a lost track alive before eviction
                             # 90 frames ≈ 3s at 30fps; longer = more ID stability
                             # through occlusions, but more ghost tracks

  probationAge: 2            # frames before a tentative new detection becomes a track
                             # protects against single-frame false positives

StateEstimator:
  processNoiseVar4Vel: 0.15  # Kalman velocity noise; higher = trusts detector more
                             # vs motion prediction; allows sharper direction changes
                             # Default 0.03 is too low for retail scenes

DataAssociator:
  minMatchingScore4Iou: 0.2575  # minimum IoU overlap to associate a detection
                                # with an existing track
```

### Tracking ID instability

With visual features off, the tracker relies entirely on the Kalman filter for motion prediction. When a person is occluded for more than `maxShadowTrackingAge` frames and then reappears, they get a **new tracking ID**. This is a fundamental limitation of IoU-only tracking. The cross-camera ReID gallery handles it gracefully: the new track's embedding matches the same gallery entry, so the same `global_id` is reassigned.

---

## 7. Secondary Inference — ReID Embeddings

### What SGIE does differently from PGIE

```
PGIE (process-mode=1): runs on the full frame → produces NvDsObjectMeta (bboxes)
SGIE (process-mode=2): runs on each object crop from a PGIE → produces tensor output per object
```

DeepStream automatically crops each bbox from the PGIE output to the SGIE input dimensions, runs inference, and attaches the output tensor to `NvDsObjectMeta::obj_user_meta_list`.

### Critical: SGIE must come AFTER the tracker

**The most important ordering rule in this project:**

```
WRONG:  streammux → pgie → sgie → tracker → sink
RIGHT:  streammux → pgie → tracker → sgie → sink
```

Why: NvDCF **reconstructs** `NvDsObjectMeta` entries as it runs (it may merge, split, or re-create track objects). Any `NvDsUserMeta` tensor data attached by the SGIE before tracking is **silently discarded**. The symptom is 0% embedding coverage in the probe even though nvinfer logs show the SGIE running.

### The `operate-on-gie-id` trap

The SGIE config supports `operate-on-gie-id=N` to only process objects that came from a specific PGIE. This sounds correct — but after nvtracker runs, it changes `unique_component_id` on most tracked objects. The field no longer matches the PGIE's ID, so ~95% of person crops are silently skipped.

**Fix**: remove `operate-on-gie-id` entirely. Use `operate-on-class-ids=0` to restrict to person class:

```ini
# operate-on-gie-id intentionally omitted
operate-on-class-ids=0
```

### `secondary-reinfer-interval` — the per-frame vs one-shot setting

This GObject property (not in the `.txt` config) controls how often the SGIE processes each tracking ID:

- `0` (default): infer each `tracking_id` exactly **once in its lifetime** — only the first frame the person appears
- `1`: infer every frame

For ReID matching you need embeddings every frame (or at minimum regularly), so set this in C++:

```cpp
g_object_set(sgie_, "secondary-reinfer-interval", (guint)1, nullptr);
```

Without this fix, embedding coverage is ~5% regardless of all other settings.

### network-type for embedding models

| `network-type` | Meaning |
|---|---|
| `0` | Detector (PGIE) |
| `1` | Classifier — nvinfer writes argmax class to `NvDsClassifierMeta` |
| `100` | Other / custom — nvinfer writes raw tensor to `NvDsInferTensorMeta` |

ReID models output a continuous embedding vector, not class probabilities. Use `network-type=100`. Using `1` causes nvinfer to try to interpret the 256-dim vector as 256 class scores and produce garbage `NvDsClassifierMeta`.

### Reading the embedding in the pad probe

```cpp
for (NvDsMetaList* l_um = om->obj_user_meta_list; l_um; l_um = l_um->next) {
    auto* um = static_cast<NvDsUserMeta*>(l_um->data);
    if (um->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META) continue;

    auto* tmeta = static_cast<NvDsInferTensorMeta*>(um->user_meta_data);
    if (tmeta->unique_id != kReidGieId) continue;  // match SGIE's gie-unique-id

    for (guint li = 0; li < tmeta->num_output_layers; ++li) {
        const NvDsInferLayerInfo& linfo = tmeta->output_layers_info[li];
        if (linfo.isInput) continue;

        auto* data = static_cast<float*>(tmeta->out_buf_ptrs_host[li]);
        int n = 1;
        for (unsigned di = 0; di < linfo.inferDims.numDims; ++di)
            n *= linfo.inferDims.d[di];
        embedding.assign(data, data + n);
    }
}
```

Note: `out_buf_ptrs_host` is the CPU-side copy. `out_buf_ptrs_dev` is the GPU pointer. Always use `host` in the pad probe (which runs on the CPU).

### ImageNet normalization — the silent accuracy killer

The ResNet50 model's ONNX graph starts with a Conv layer — it has **no preprocessing node**. Without explicit normalization in nvinfer config, the model receives raw [0,255] pixel values.

**Effect**: all embeddings collapse. Different people produce cosine similarity of 0.875 — essentially random with no discriminative power.

**Fix**: apply the ImageNet statistics the model was trained with:

```ini
net-scale-factor=0.017352607709750568   # ≈ 1/(255 × 0.226) = 1/57.63
offsets=123.675;116.28;103.53           # ImageNet mean: R;G;B in [0,255]
```

DeepStream applies: `output = net-scale-factor * (pixel - offset)`, which equals the standard ImageNet preprocessing `(pixel/255 - mean) / std`.

How to detect this problem: run `reid_probe.py` and check inter-ID mean cosine similarity. If it's >0.70 for different identities, normalization is likely wrong.

---

## 8. Testing AI Models — Evaluation Workflow

### Phase-by-phase verification

Do not run the full pipeline before each piece works independently.

**Step 1 — Detection only**
```bash
./build/edge_retail_core_engine --config configs/default.yaml --verbose 2>&1 | head -50
```
Look for `[pipeline] Source 0 linked → nvstreammux:sink_0` and no ERROR messages. Set `output.mode: stdout` and check that JSON `frame` events appear with `"class":0` detections.

**Step 2 — Tracking IDs**
In the JSON output, look for `"tid"` field that is stable across frames for the same person. If `"tid"` is `-1` or constantly changing, tracking is not working.

**Step 3 — Embedding coverage (reid_probe.py)**
```bash
cd messaging/zmq
python3 reid_probe.py --endpoint tcp://localhost:5555 --frames 500
```

Expected healthy output:
```
Coverage  : 100.0%          ← every person detection has an embedding
Norm mean : 1.000           ← embeddings are already L2-normalised
Intra-ID mean : 0.78        ← same person across frames is similar
Inter-ID mean : 0.42        ← different people are dissimilar
Gap (intra−inter): 0.36     ← ✓ Embeddings are discriminative
```

Diagnosing problems:
| Symptom | Likely cause |
|---------|-------------|
| Coverage <10% | `secondary-reinfer-interval` not set to 1 |
| Coverage ~5% | `operate-on-gie-id=1` filtering out most tracks |
| Inter-ID mean >0.70 | Missing `offsets` / wrong `net-scale-factor` |
| Inter-ID mean ≈ Intra-ID mean | Model not running (network-type or path wrong) |
| Norm ≠ 1.0 | Model output is not BN-normalised; L2-normalise in code |

**Step 4 — Cross-camera matching**
```bash
python3 reid_matcher.py --endpoint tcp://localhost:5555 --threshold 0.75 --status-interval 30
```
Watch for `reid_link` events. If you see the same person on both cameras in the OSD video with the same `ID:N`, the system is working.

### Evaluating a new model without restarting the pipeline

`reid_probe.py` subscribes to the live ZMQ stream and computes statistics without touching the pipeline. This makes model comparison fast:

1. Deploy new ONNX/engine, restart pipeline
2. Run `reid_probe.py --frames 500` — get stats in ~1 min
3. Check gap ≥ 0.30 and inter-ID mean < 0.55
4. Compare against previous stats

### Understanding cosine similarity for ReID

Cosine similarity measures the angle between two embedding vectors, not their magnitude. It ranges from -1 (opposite) to +1 (identical direction).

For a well-trained ReID model:
- **Same person, different frames**: similarity ~0.75–0.90 (some variation from pose/lighting)
- **Different people**: similarity ~0.35–0.55 (well-separated in embedding space)
- **Random/collapsed model**: all similarities cluster near 0.85–0.95 (no discrimination)

The matching threshold (default 0.75 in the C++ gallery, 0.75 in `reid_matcher.py`) should sit above the inter-ID mean and below the intra-ID mean. If those distributions overlap, the model can't reliably distinguish identities.

---

## 9. OSD and Video Output

### nvmultistreamtiler

When you have N sources, the frames in `NvDsBatchMeta` come from different cameras. `nvmultistreamtiler` composites them into a single frame so `nvdsosd` can draw on a coherent surface:

```cpp
g_object_set(tiler_,
    "rows",    (guint)1,
    "columns", (guint)num_sources,  // side-by-side layout
    "width",   (guint)(960 * num_sources),
    "height",  (guint)540,
    nullptr);
```

Each tile is 960×540 (16:9). With 2 cameras the output frame is 1920×540.

### nvdsosd

`nvdsosd` draws from `NvDsObjectMeta` automatically. You can override before it runs by writing to the metadata in your pad probe:

```cpp
// Change bbox color
om->rect_params.border_color = {1.0f, 0.5f, 0.0f, 1.0f};  // RGBA orange
om->rect_params.border_width = 3;

// Change text label
g_free(om->text_params.display_text);                        // free old allocation
om->text_params.display_text = g_strdup_printf("ID:%d", global_id);
om->text_params.font_params.font_size = 12;
om->text_params.font_params.font_color = {1.0f, 1.0f, 1.0f, 1.0f};  // white
om->text_params.set_bg_clr = 1;
om->text_params.text_bg_clr = {0.6f, 0.3f, 0.0f, 0.75f};  // darker background
```

`display_text` is a `gchar*` owned by GLib. Always `g_free` before reassigning, and always allocate with `g_strdup_printf` (not `strdup` or `new`). Mixing allocators will crash.

`process-mode=0` means CPU rendering — required when `visualTrackerType=0` because there's no NVMM surface pinning race.

### Jetson hardware encoder

```
nvvideoconvert → nvv4l2h264enc → h264parse → matroskamux → filesink
```

- `nvvideoconvert` converts NVMM RGBA (nvdsosd output) to NVMM NV12 (encoder input). The encoder requires a caps filter: `video/x-raw(memory:NVMM), format=NV12`
- `nvv4l2h264enc` is the V4L2 interface to the Jetson NVENC hardware encoder. Bitrate ~4 Mbps is good for annotated 1920×540 video
- `h264parse` is needed because `matroskamux` requires a parsed H.264 stream (with access unit delimiters). Without it, the MKV file won't play
- `matroskamux` (MKV) is preferred over `qtmux`/`mp4mux` because it writes its index progressively. If you Ctrl+C a `qtmux` pipeline, the moov atom is never written and the file is unplayable

### Live display — the NVMM surface problem

`nveglglessink` is the Jetson EGL display sink. It cannot consume NVMM buffers directly:
```
ERROR: nveglglessink cannot handle NVRM surface array
```

The fix: force `nvvideoconvert` to download to CPU RAM by setting CPU caps (no `memory:NVMM`) between it and the display sink:

```cpp
GstCaps* rgba_cpu = gst_caps_from_string("video/x-raw, format=RGBA");
gst_element_link_filtered(videoconv_disp_, display_sink_, rgba_cpu);
gst_caps_unref(rgba_cpu);
```

Without this, even with `nvvideoconvert` in the chain, GStreamer negotiates NVMM-to-NVMM because both elements support it.

### Splitting to file + display simultaneously: tee

`tee` is a standard GStreamer element that duplicates the stream. Each output pad needs its own `queue` to decouple buffer flow and prevent deadlocks:

```
osd → tee → queue_file → videoconv → encoder → … → filesink
           → queue_disp → videoconv_disp → [cpu caps] → nveglglessink
```

```cpp
// Linking tee branches: must use pad request, not gst_element_link
gst_element_link_many(tee_, queue_file_, videoconv_, nullptr);
gst_element_link_many(tee_, queue_disp_, videoconv_disp_, nullptr);
```

### Playback

On Jetson ARM64, VLC 3.0.x crashes on MKV playback due to a GPU acceleration bug. Use `ffplay` instead:
```bash
ffplay /tmp/reid_annotated.mkv
```

---

## 10. Cross-Camera Identity — C++ Gallery

### Problem statement

Per-camera tracking IDs (`tid`) are local: camera 0 has `tid=3` for one person, camera 1 has `tid=7` for the same person. Without a stable shared identity, you can't know they're the same person.

### Gallery design

```cpp
struct GalleryEntry {
    std::vector<float> emb;   // L2-normalised embedding (updated each frame)
    int   global_id;          // stable cross-camera ID
    double ts;                // monotonic timestamp for TTL eviction
};

// key = "source_id:tracking_id"
std::unordered_map<std::string, int>          track_global_;   // frozen assignment
std::unordered_map<std::string, GalleryEntry> reid_gallery_;   // embedding store
```

### Assignment logic (inside the pad probe)

```cpp
const std::string key = make_key(source_id, tracking_id);
auto ta = track_global_.find(key);

if (ta != track_global_.end()) {
    // Known track — reuse its frozen global_id (no flickering)
    global_id = ta->second;
} else {
    // New track — search other cameras for a match
    float best_sim = kReidThreshold;  // 0.70
    for (const auto& [gkey, entry] : reid_gallery_) {
        if (same_camera(gkey, source_id)) continue;
        float sim = cosine(emb_norm, entry.emb);
        if (sim > best_sim) { best_sim = sim; global_id = entry.global_id; }
    }
    if (global_id == -1) global_id = next_global_id_++;  // new person
    track_global_[key] = global_id;  // freeze the assignment
}

// Always update the gallery with the latest embedding
reid_gallery_[key] = {emb_norm, global_id, now};
```

Key design decisions:
- **Freeze on first assignment**: once `track_global_[key]` is set, never change it. This prevents the label from flickering when embeddings vary slightly frame-to-frame.
- **TTL eviction** (`kGalleryTTL = 30s`): remove gallery entries for tracks that have been gone for >30 seconds. This limits memory growth in long sessions and avoids false matches to people who left long ago.
- **Cross-camera only**: skip same-camera gallery entries during search. A person splitting into two same-camera tracks (e.g., due to re-detection after occlusion) should not match itself across the gallery.
- **Mutex**: the probe callback fires on GStreamer's streaming thread. The gallery is shared state — protect it with `std::mutex`.

### Color coding

```cpp
static const NvOSD_ColorParams kIdColors[] = {
    {1.00f, 0.20f, 0.20f, 1.0f},  // red
    {0.20f, 0.90f, 0.20f, 1.0f},  // green
    // …8 colors total
};
const NvOSD_ColorParams& col = kIdColors[global_id % 8];
om->rect_params.border_color = col;
```

8 colors is enough for most retail scenes (rarely more than 8 unique people in frame simultaneously). The modulo wrapping means colors repeat after 8 IDs, but in practice you can distinguish by the `ID:N` label.

---

## 11. Decoupled Messaging — ZeroMQ

### Why ZeroMQ instead of a shared library call

The C++ pipeline (real-time inference) and the business logic layer (matching, alerts, REST API) have different requirements:
- C++: must not be blocked by a slow consumer
- Python/API layer: doesn't need every frame, just events

ZeroMQ PUB/SUB: the publisher sends and forgets. If no subscriber is listening, messages are dropped (not buffered forever). This keeps the inference loop real-time.

### Publisher side (C++)

```cpp
zmq::context_t ctx(1);
zmq::socket_t pub(ctx, ZMQ_PUB);
pub.bind("tcp://*:5555");

// Publish each JSON string
pub.send(zmq::buffer(json_string), zmq::send_flags::dontwait);
// ZMQ_DONTWAIT: drop silently if no consumer; never block the caller
```

The `ZMQ_SNDHWM` (high-water mark, default 1000 messages) is a safety valve. When the outgoing queue fills, new sends are dropped rather than blocking.

### Subscriber side (Python)

```python
import zmq, json

ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.connect("tcp://localhost:5555")
sub.setsockopt_string(zmq.SUBSCRIBE, "")  # subscribe to all topics

while True:
    data = json.loads(sub.recv_string())
    # process frame event…
```

### Event format

```json
{
  "event": "frame",
  "ts_ms": 1713000000000,
  "source_id": 0,
  "frame": 142,
  "detections": [
    {
      "class": 0, "label": "person", "conf": 0.85,
      "bbox": {"l": 312.0, "t": 88.0, "w": 96.0, "h": 260.0},
      "tid": 7,
      "emb": "<base64-encoded float32[256]>"
    }
  ]
}
```

The embedding is base64-encoded to keep the JSON compact and avoid float serialization precision loss. Decode in Python:
```python
import base64, numpy as np
emb = np.frombuffer(base64.b64decode(det["emb"]), dtype=np.float32)
```

---

## 12. Key Pitfalls and Hard-Won Lessons

These are the things that cost the most debugging time. Read these before you start.

### P1 — SGIE must come after nvtracker

NvDCF reconstructs `NvDsObjectMeta`. Any tensor metadata attached before tracking is silently lost.
```
pgie → tracker → sgie    ← CORRECT
pgie → sgie → tracker    ← tensor metadata silently dropped
```

### P2 — `secondary-reinfer-interval` defaults to "infer once per track lifetime"

The default `0` means each tracking ID is processed by the SGIE exactly once — the first frame it appears. Subsequent frames have no embedding. Set to `1` in C++ (it is not in the config file):
```cpp
g_object_set(sgie_, "secondary-reinfer-interval", (guint)1, nullptr);
```

### P3 — `operate-on-gie-id` breaks with NvDCF

After the tracker runs, it changes `unique_component_id` on most objects. Setting `operate-on-gie-id=1` causes ~95% of crops to be silently skipped. Remove the field; filter by class instead (`operate-on-class-ids=0`).

### P4 — Missing ImageNet normalization collapses embeddings

If the ONNX model has no preprocessing node (inspect with Netron), you must add normalization manually in the nvinfer config. For ResNet50 trained on Market-1501:
```ini
net-scale-factor=0.017352607709750568
offsets=123.675;116.28;103.53
```
Without this, all inter-ID cosine similarities cluster at ~0.875 — the model produces no useful discrimination.

### P5 — `nveglglessink` cannot handle NVMM buffers

`nvvideoconvert` alone doesn't fix this — it still negotiates NVMM-to-NVMM. Explicitly force CPU download with:
```cpp
GstCaps* rgba_cpu = gst_caps_from_string("video/x-raw, format=RGBA");
gst_element_link_filtered(videoconv_disp_, display_sink_, rgba_cpu);
```

### P6 — Tee branches need queues to prevent deadlock

Always put a `queue` element after each `tee` output pad. Without it, the two downstream branches run synchronously and can deadlock when one branch (e.g., the encoder) backs up.

### P7 — `display_text` is GLib-managed memory

```cpp
g_free(om->text_params.display_text);               // MUST free before reassigning
om->text_params.display_text = g_strdup_printf("...");  // use GLib allocation
```
Mixing with `strdup`/`new`/`delete` crashes.

### P8 — Engine files encode GPU architecture

An engine built on Jetson Orin NX (Ampere 8.7) will not load on Jetson AGX Xavier (Volta 7.2) or an RTX 3090 (Ampere 8.6). Always rebuild on the target. Gitignore engine files.

### P9 — `g_object_set` type errors are silent

Passing an `int` where `guint` is expected, or a `bool` where `gboolean` is expected, compiles but silently passes the wrong value. Always cast explicitly:
```cpp
g_object_set(element, "property", (guint)value, nullptr);  // explicit cast
```

### P10 — VLC crashes on ARM64 MKV

VLC 3.0.x on Jetson ARM64 crashes on MKV playback. Use `ffplay` instead.

### P11 — `batched-push-timeout` is essential for multi-camera RTSP

Without a timeout, `nvstreammux` waits indefinitely for a frame from all sources before pushing. If one camera lags, the pipeline stalls. Set `batched-push-timeout=4000000` (4 seconds in nanoseconds).

### P12 — OSNet softmax-only checkpoint collapses to useless embeddings

The publicly available torchreid Market-1501 checkpoint was trained with softmax cross-entropy only. Its BatchNorm neck has `running_var ≈ 0.0001`, causing all embeddings to land in a tiny angular cluster (mean pairwise cosine sim 0.956 on random pairs). Always verify with `reid_probe.py` before trusting a new model checkpoint.

---

## 13. Validation Checklist

Use this checklist when setting up the pipeline from scratch or after a significant change.

```
[ ] gst_init runs and pipeline reaches PLAYING state without errors
[ ] Source pads link to nvstreammux (look for "Source 0 linked → nvstreammux:sink_0")
[ ] Frame events appear in ZMQ output with source_id=0, source_id=1
[ ] Detections have class=0 (person) and conf > 0.3
[ ] tracking_id (tid) is stable across consecutive frames for the same person
[ ] Embedding coverage = 100% (all person detections have emb field)
[ ] Embedding norm ≈ 1.0 (L2-normalised)
[ ] Inter-ID cosine sim mean < 0.55 (different people are dissimilar)
[ ] Intra-ID cosine sim mean > 0.65 (same person is similar across frames)
[ ] Gap (intra−inter) ≥ 0.30 (embeddings are discriminative)
[ ] OSD video shows boxes with ID:N labels
[ ] Same physical person has same ID:N and same box color on both camera tiles
[ ] reid_matcher.py produces reid_link events for cross-camera pairs
[ ] MKV file plays back cleanly with ffplay
[ ] Live display window opens and shows annotated video
```

---

## Summary: The Learning Arc

This project covers the complete journey from raw video to cross-camera identity:

```
RTSP streams
    ↓ nvurisrcbin (NVDEC hardware decode, NVMM output)
    ↓ nvstreammux (multi-source batching, NvDsBatchMeta)
    ↓ nvinfer/PGIE (TensorRT detector — PeopleNet, person bboxes)
    ↓ nvtracker/NvDCF (Kalman+IoU tracking, stable per-camera TIDs)
    ↓ nvinfer/SGIE (TensorRT embedding — ResNet50, 256-dim per crop)
    ↓ pad probe (read NvDsInferTensorMeta, gallery lookup, global_id, OSD color)
    ↓ nvmultistreamtiler (compose N tiles into one frame)
    ↓ nvdsosd (draw color-coded boxes + ID labels from metadata)
    ↓ nvv4l2h264enc (Jetson NVENC hardware encode)
    ↓ matroskamux → MKV file  +  nveglglessink → live display window
    ↓ ZeroMQ PUB → Python subscribers (reid_matcher.py, reid_probe.py)
```

The core insight: DeepStream is just GStreamer with a set of NVIDIA-specific elements and a shared metadata structure (`NvDsBatchMeta`). Everything that feels magical — multi-source batching, SGIE crop routing, OSD rendering — is standard GStreamer element graph behavior. Understanding pad probes, caps negotiation, and the `NvDs` struct hierarchy unlocks the whole system.
