# Edge Retail Intelligence Engine

Real-time edge AI for retail analytics: person detection, per-camera tracking, cross-camera re-identification (ReID), and loitering detection.

---

## Architecture

```
[ RTSP Cameras ]
      ↓
C++ Core Engine  (DeepStream 6.2 / GStreamer / TensorRT)
  • Person detection   — PeopleNet ResNet-34 INT8
  • Per-camera tracking — NvDCF correlation-filter tracker
  • Metadata extraction — GStreamer pad probe → JSONL events
      ↓
Message Bus  (Phase 2 — ZeroMQ / Kafka)
      ↓
Application Layer  (Phase 5 — FastAPI)
  • Multi-camera ReID correlation
  • Loitering detection (dwell-time + zone rules)
  • REST API
      ↓
Storage / UI  (Phase 6 — optional)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Video pipeline | C++17, NVIDIA DeepStream 6.2, GStreamer 1.16 |
| Inference | PeopleNet v2.3.3 (ResNet-34), TensorRT INT8 |
| Tracking | NvDCF (libnvds_nvmultiobjecttracker) |
| Config | YAML (yaml-cpp) + INI (nvinfer) |
| Messaging (Phase 2+) | ZeroMQ / Kafka |
| API layer (Phase 5+) | FastAPI, Python 3.8+ |
| Deployment | Docker / Docker Compose |

Target hardware: **NVIDIA Jetson Orin NX** (JetPack 5.x) or x86 + NVIDIA GPU.

---

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| 1 | C++ DeepStream pipeline — detection, tracking, JSONL output | **Done** |
| 2 | ZeroMQ / Kafka messaging | Planned |
| 3 | ReID embedding extraction + cross-camera correlation | Planned |
| 4 | Loitering detection engine (zones, dwell-time thresholds) | Planned |
| 5 | FastAPI control plane (camera management, event query, rules) | Planned |
| 6 | Production hardening — Docker, metrics, storage, UI | Planned |

---

## Phase 1 — Quick Start

### Prerequisites

- NVIDIA DeepStream 6.2 SDK
- CUDA Toolkit + TensorRT (included with JetPack 5.x on Jetson)
- GStreamer 1.16+: `sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev`
- yaml-cpp: `sudo apt install libyaml-cpp-dev`
- NGC CLI or `NGC_API_KEY` env var (for model download)

### 1. Download PeopleNet model

```bash
./scripts/download_peoplenet.sh
```

This places `resnet34_peoplenet_int8.onnx` in `models/peoplenet/`. On first pipeline run, TensorRT auto-compiles it to `resnet34_peoplenet_int8.engine` (~5 min on Jetson Orin NX; cached for all subsequent runs).

### 2. Build

```bash
cd core_engine
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### 3. Configure

Edit `core_engine/configs/default.yaml` to point at your RTSP stream:

```yaml
sources:
  - uri: "rtsp://192.168.9.146:18554/stream3"
    enabled: true
```

### 4. Run

```bash
# Run from core_engine/ so config-relative paths resolve correctly
cd core_engine
./build/edge_retail_core_engine --config configs/default.yaml --verbose
```

To capture events to a file, change `output.mode` in `default.yaml`:

```yaml
output:
  mode: file
  file: /tmp/edge_retail_events.jsonl
```

---

## Output Format

One JSON object per frame containing at least one detection, written to stdout (or a JSONL file):

```json
{
  "event": "frame",
  "ts_ms": 1713000000000,
  "source_id": 0,
  "frame": 142,
  "detections": [
    {
      "class": 0,
      "label": "person",
      "conf": 0.8543,
      "bbox": { "l": 312.0, "t": 88.0, "w": 96.0, "h": 260.0 },
      "tid": 7
    }
  ]
}
```

| Field | Description |
|---|---|
| `ts_ms` | Wall-clock timestamp (Unix ms) |
| `source_id` | Camera / stream index (0-based) |
| `frame` | Frame number within the stream |
| `class` | 0 = person, 1 = bag, 2 = face |
| `conf` | Detection confidence (0–1) |
| `bbox` | Bounding box in pixels: left, top, width, height |
| `tid` | Stable tracking ID assigned by NvDCF (absent if untracked) |

---

## Configuration Reference

### `core_engine/configs/default.yaml`

```yaml
version: 1

sources:
  - uri: "rtsp://<host>:<port>/<path>"   # RTSP, file://, or device index
    enabled: true

models:
  detector: "configs/config_infer_primary_peoplenet.txt"
  tracker:  "configs/config_tracker_NvDCF.yml"

output:
  mode: stdout          # stdout | file
  # file: /tmp/edge_retail_events.jsonl
```

### `core_engine/configs/config_infer_primary_peoplenet.txt` (key fields)

| Field | Value | Notes |
|---|---|---|
| `onnx-file` | `../../models/peoplenet/resnet34_peoplenet_int8.onnx` | Relative to config file location |
| `model-engine-file` | `../../models/peoplenet/resnet34_peoplenet_int8.engine` | Auto-generated TRT cache |
| `network-mode` | `1` (INT8) | Switch to `2` (FP16) if INT8 calibration table is absent |
| `infer-dims` | `3;544;960` | PeopleNet input: C×H×W |
| `pre-cluster-threshold` | `0.3` (person), `0.2` (bag/face) | Confidence cutoff before NMS |

### `core_engine/configs/config_tracker_NvDCF.yml` (key fields)

| Field | Value | Notes |
|---|---|---|
| `maxTargetsPerStream` | `50` | Cap concurrent tracked identities |
| `probationAge` | `2` | Frames before a tentative track is confirmed |
| `maxShadowTrackingAge` | `51` | Frames to keep a lost target alive |
| `visualTrackerType` | `1` (NvDCF) | Correlation-filter visual tracker |

---

## Project Structure

```
edge-retail-intelligence/
├── core_engine/
│   ├── src/
│   │   ├── main.cpp        # Entry point, signal handling
│   │   ├── app.cpp         # YAML config loader, output routing
│   │   ├── pipeline.cpp    # DeepStream GStreamer pipeline
│   │   └── metadata.cpp    # FrameEvent / Detection JSON serialiser
│   ├── include/core_engine/
│   │   ├── app.hpp
│   │   ├── pipeline.hpp
│   │   └── metadata.hpp
│   ├── configs/
│   │   ├── default.yaml
│   │   ├── config_infer_primary_peoplenet.txt
│   │   ├── config_tracker_NvDCF.yml
│   │   └── peoplenet_labels.txt
│   └── CMakeLists.txt
├── models/peoplenet/       # Downloaded via scripts/download_peoplenet.sh
├── scripts/
│   └── download_peoplenet.sh
├── api/                    # Phase 5 — FastAPI (not yet implemented)
├── messaging/              # Phase 2 — ZeroMQ / Kafka (not yet implemented)
└── README.md
```

---

## Design Principles

- **Decoupled** — pipeline emits events; business logic lives upstream
- **Real-time first** — low-latency pad probe, fakesink (no display overhead)
- **Edge-native** — INT8 TensorRT, NvDCF tracker, Jetson-optimised
- **Extensible** — add models, trackers, or output adapters without touching the pipeline core
