# Edge Retail Intelligence Engine

Real-time edge AI for retail analytics: person detection, per-camera tracking, cross-camera re-identification (ReID), and loitering detection — running on **NVIDIA Jetson Orin NX**.

---

## Architecture

```
[ RTSP Cameras ]
      ↓
C++ Core Engine  (DeepStream 6.2 / GStreamer / TensorRT)
  • Person detection       — PeopleNet ResNet-34 INT8
  • Per-camera tracking    — NvDCF correlation-filter tracker
  • ReID embedding         — ResNet50 FP16 SGIE (256-dim per person crop)
  • Cross-camera global ID — cosine gallery in pad probe; color-coded OSD boxes
  • Metadata extraction    — GStreamer pad probe → JSONL events over ZeroMQ
      ↓
Message Bus  (ZeroMQ PUB/SUB)
      ↓
Application Layer  (Phase 5 — FastAPI)
  • Loitering detection (dwell-time + zone rules)
  • REST API
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Video pipeline | C++17, NVIDIA DeepStream 6.2, GStreamer 1.16 |
| Primary inference | PeopleNet v2.3.3 (ResNet-34), TensorRT INT8 |
| Secondary inference | ResNet50 (Market-1501 + AI City 156), TensorRT FP16, SGIE mode |
| Tracking | NvDCF (`libnvds_nvmultiobjecttracker`) |
| Messaging | ZeroMQ PUB/SUB |
| Config | YAML (yaml-cpp) + INI (nvinfer .txt format) |
| API layer (Phase 5+) | FastAPI, Python 3.8+ |

Target hardware: **NVIDIA Jetson Orin NX** — JetPack 5.x (L4T R35), CUDA 11.4, TensorRT 8.5.

---

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| 1 | C++ DeepStream pipeline — detection, tracking, JSONL output | **Done** |
| 2 | ZeroMQ / Kafka messaging | **Done** |
| 3 | ReID embedding extraction + cross-camera correlation | **Done** |
| 4 | Loitering detection engine (zones, dwell-time thresholds) | Planned |
| 5 | FastAPI control plane | Planned |
| 6 | Production hardening — Docker, metrics, storage, UI | Planned |

---

## Quick Start

### Prerequisites

```bash
# JetPack 5.x provides CUDA, TensorRT, cuDNN — skip if already on Jetson
# GStreamer dev headers
sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libyaml-cpp-dev

# Phase 2 (ZeroMQ)
./scripts/install_deps_phase2.sh

# Python tools for ReID model export
pip3 install torchreid gdown
```

### 1. Download models

```bash
# PeopleNet (Phase 1) — NGC account or API key required
./scripts/download_peoplenet.sh

# ReID — ResNet50 ONNX downloaded directly (no NGC account needed)
# Place resnet50_market1501_aicity156.onnx in models/reid/
# TRT engine is auto-built on first run (~3 min on Orin NX)
```

### 2. Build

```bash
cd core_engine
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### 3. Run

```bash
# Terminal 1 — inference pipeline
cd core_engine
./build/edge_retail_core_engine --config configs/default.yaml --verbose

# Terminal 2 — ZMQ consumer (raw frame events)
cd messaging/zmq
python3 consumer.py --endpoint tcp://localhost:5555 --filter person

# Terminal 2 (alt) — cross-camera ReID matcher
python3 reid_matcher.py --endpoint tcp://localhost:5555 --threshold 0.75
# Prints reid_identity (new person seen) and reid_link (same person confirmed across cameras)
# --status-interval 30  prints a periodic identity table to stderr

# Terminal 3 (optional) — embedding quality diagnostic
python3 reid_probe.py --endpoint tcp://localhost:5555 --frames 500
# Coverage should be 100%, inter-ID cosine sim mean < 0.55, gap >= 0.30
```

On first run, nvinfer auto-builds TRT engines from the ONNX files (~3–5 min each on Orin NX). Subsequent starts load the cached `.engine` files instantly.

**Annotated video + live display** — set `osd_file` and/or `osd_display` in `default.yaml`:

```yaml
output:
  osd_file: "/tmp/reid_annotated.mkv"   # record MKV with annotated boxes
  osd_display: true                     # open live preview window simultaneously
```

The OSD draws color-coded bounding boxes keyed by **global ID** (same color = same physical person across cameras) with `ID:N` and per-camera `tid:M` labels. The side-by-side tiled view shows all sources at 960×540 per tile.

```bash
# Playback recorded file (use ffplay — VLC 3.0.x crashes on Jetson ARM64):
ffplay /tmp/reid_annotated.mkv
```

---

## Output Format

### Frame events (C++ pipeline → ZMQ)

One JSON object per frame with ≥1 detection:

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
      "tid": 7,
      "emb": "<base64-encoded float32[256]>"
    }
  ]
}
```

| Field | Description |
|---|---|
| `ts_ms` | Wall-clock timestamp (Unix ms) |
| `source_id` | Camera index (0-based) |
| `tid` | Per-camera tracking ID assigned by NvDCF (resets per stream) |
| `emb` | Base64-encoded 256-d float32 ReID embedding; L2-normalised |

### ReID events (`reid_matcher.py` → stdout)

`reid_matcher.py` emits events **once per new identity or new cross-camera link** — not every frame:

```json
{"event":"reid_identity","ts_ms":1713000000000,"global_id":5,"source_id":0,"tid":3}
{"event":"reid_link","ts_ms":1713000000000,"global_id":5,
 "cam_a":{"source_id":0,"tid":3},"cam_b":{"source_id":1,"tid":7},"similarity":0.912}
```

`global_id` is the stable cross-camera identity number. The same person seen on cameras 0 and 1 will share a `global_id` and appear with the same color in the OSD annotated video.

---

## Configuration Reference

### `configs/default.yaml`

```yaml
sources:
  - uri: "rtsp://<host>:<port>/<path>"
    enabled: true
  - uri: "rtsp://<host>:<port>/<path2>"   # second camera for cross-camera ReID
    enabled: true

models:
  detector: "configs/config_infer_primary_peoplenet.txt"
  tracker:  "configs/config_tracker_NvDCF.yml"
  reid:     "configs/config_infer_secondary_reid.txt"

output:
  mode: zmq                    # zmq | stdout | file
  endpoint: "tcp://*:5555"
  osd_file: "/tmp/reid_annotated.mkv"  # remove this line to disable OSD recording
  osd_display: true            # open live preview window (nveglglessink); set false to disable
```

`osd_file` and `osd_display` each independently enable an OSD branch. When both are set, a `tee` element splits the stream:
- **`osd_file`** branch: `nvmultistreamtiler → nvdsosd → nvvideoconvert → nvv4l2h264enc → h264parse → matroskamux → filesink`
- **`osd_display`** branch: `nvmultistreamtiler → nvdsosd → nvvideoconvert → [caps:video/x-raw,format=RGBA] → nveglglessink`

The CPU caps (`video/x-raw, format=RGBA`) on the display branch are required — `nveglglessink` cannot handle NVMM surface arrays. MKV is used instead of MP4 because `matroskamux` writes its index progressively and survives Ctrl+C; `qtmux`/`mp4mux` write the moov atom only at EOS and produce unplayable files if interrupted.

### `configs/config_infer_secondary_reid.txt` (key fields)

| Field | Value | Notes |
|---|---|---|
| `process-mode` | `2` | Secondary GIE (SGIE) — runs on object crops, not full frames |
| ~~`operate-on-gie-id`~~ | *(removed)* | **Do not set.** NvDCF tracker changes `unique_component_id` on tracked objects, so filtering by PGIE ID causes ~95% of crops to be skipped silently |
| `operate-on-class-ids` | `0` | Person class only |
| `onnx-file` | `resnet50_market1501_aicity156.onnx` | Source model; TRT engine built from this on first run (~3 min) |
| `net-scale-factor` | `0.017352607709750568` | ImageNet std normalisation (≈ 1/57.63); **required** — model has no preprocessing baked in |
| `offsets` | `123.675;116.28;103.53` | ImageNet mean subtraction (R;G;B); **required** — without this all embeddings collapse to cosine sim ≈ 0.875 |
| `network-mode` | `2` | FP16 inference |
| `batch-size` | `8` | Max person crops per inference call |
| `network-type` | `100` | OTHER — raw tensor output (not classifier); do **not** use `1` (classifier) for embedding models |
| `output-tensor-meta` | `1` | Attaches raw float tensors to `NvDsObjectMeta` for pad-probe access |
| `gie-unique-id` | `2` | Must match `kReidGieId` constant in `pipeline.cpp` |
| `interval` | `0` | Infer every frame; `secondary-reinfer-interval` GObject property is additionally set to `1` in C++ |

---

## Project Structure

```
edge-retail-intelligence/
├── core_engine/
│   ├── src/
│   │   ├── main.cpp          # Entry point, signal handling
│   │   ├── app.cpp           # YAML config loader, output-mode routing
│   │   ├── pipeline.cpp      # DeepStream/GStreamer pipeline builder
│   │   └── metadata.cpp      # FrameEvent / Detection JSON serialiser
│   └── configs/
│       ├── default.yaml
│       ├── config_infer_primary_peoplenet.txt
│       ├── config_infer_secondary_reid.txt
│       ├── config_tracker_NvDCF.yml
│       └── peoplenet_labels.txt
├── models/
│   ├── peoplenet/            # Downloaded via download_peoplenet.sh
│   └── reid/                 # Generated via export_reid_onnx.py
├── scripts/
│   ├── download_peoplenet.sh
│   ├── download_reid.sh      # Downloads NVIDIA TAO ETLT (reference only — TAO key is not public)
│   └── export_reid_onnx.py   # Exports OSNet to ONNX (legacy — project now uses ResNet50)
└── messaging/zmq/
    ├── consumer.py           # Raw ZMQ subscriber — prints all frame events
    ├── reid_matcher.py       # Cross-camera ReID matcher — emits reid_identity / reid_link events
    └── reid_probe.py         # Embedding quality diagnostic — coverage, norm, cosine sim distribution
```

---

## Technical Reference — NVIDIA SDK

This section documents the non-obvious behaviour of DeepStream, TensorRT, TAO, and the Jetson environment. Each entry covers the *why* behind a design decision or a failure mode encountered during development.

---

### 1. DeepStream Pipeline Topology and the PGIE/SGIE Distinction

DeepStream's inference elements (`nvinfer`) run in two modes set by `process-mode`:

| Mode | Value | Input | Typical use |
|---|---|---|---|
| Primary GIE (PGIE) | `1` | Full decoded frame | Detection (PeopleNet) |
| Secondary GIE (SGIE) | `2` | Object-level crop from a PGIE | Classification, attribute extraction, ReID |

**How SGIE receives crops**: the PGIE attaches `NvDsObjectMeta` for each detection. The SGIE reads `operate-on-gie-id` and `operate-on-class-ids` to decide which detections to process. It automatically crops and resizes those bounding boxes to `infer-dims` before running inference — you do not crop manually.

**Pad probe placement matters**: the probe in this project sits on the **SGIE src pad** (falling back to the nvtracker src pad when SGIE is disabled). This means when the probe fires, both PGIE and SGIE have already run and the ReID embedding is populated in `NvDsObjectMeta` — you read it from the tensor metadata.

**Pipeline element order is critical**: SGIE must come **after** nvtracker in the pipeline graph. If SGIE is placed before nvtracker, NvDCF reconstructs `NvDsObjectMeta` entries as it runs — silently dropping any tensor metadata attached earlier. The symptom is 0% embedding coverage in the probe even though the SGIE appears to run without errors.

**`output-tensor-meta=1` is required for embeddings**: by default, SGIE only writes classifier output into `NvDsClassifierMeta`. Setting `output-tensor-meta=1` additionally attaches the raw float buffer to `NvDsObjectMeta::tensor_output_list`, which lets the pad probe read the full 256-d embedding vector.

**`network-type=100` (OTHER) for embedding models**: using `network-type=1` (classifier) causes nvinfer to interpret the model output as class logits and allocate `NvDsClassifierMeta` instead of raw tensor output. Embedding models must use `network-type=100`. With `output-tensor-meta=1` the raw tensor is available regardless, but `network-type=100` avoids nvinfer wasting memory on a spurious classifier path.

**`gie-unique-id` links pipeline elements**: every nvinfer has a unique integer ID. The C++ code uses `kReidGieId=2` to find the correct tensor output when iterating `NvDsObjectMeta`. If these IDs don't match, you get no embeddings silently.

**Do not set `operate-on-gie-id`**: after nvtracker runs, NvDCF changes `unique_component_id` on most tracked objects. If `operate-on-gie-id=1` is set in the SGIE config, only the small fraction of objects whose ID hasn't been overwritten will be processed — producing ~5% embedding coverage. Removing the field lets the SGIE process all class-0 objects unconditionally.

**`secondary-reinfer-interval` GObject property**: this property (not in the `.txt` config file) defaults to `0`, meaning nvinfer infers each tracking ID exactly **once** in its lifetime. The result is that only the first frame a person appears carries an embedding — all subsequent frames are empty. Set it to `1` in C++ code after the element is created:
```cpp
g_object_set(sgie_, "secondary-reinfer-interval", (guint)1, nullptr);
```

---

### 2. TensorRT Engine Build and Caching

TensorRT compiles a model (ONNX, UFF, or custom parser) into a platform-specific **engine** file optimised for the exact GPU it runs on.

**Auto-build flow in nvinfer**:
1. nvinfer checks for `model-engine-file` on disk.
2. If absent (or if the source model is newer), it builds the engine from `onnx-file`.
3. The built engine is serialised to `model-engine-file` and reused on every subsequent run.

**Engine naming convention**: nvinfer generates the engine filename automatically as:
```
<onnx-file>_b<batch>_gpu<id>_<precision>.engine
```
Example: `osnet_x1_0_market1501.onnx_b8_gpu0_fp16.engine`

**Why engines are non-portable**: TRT engines encode layer-fusion decisions and kernel tile sizes chosen for the specific GPU architecture (Ampere, Orin, Ada, ...). An engine built on an Orin NX will not load on a discrete RTX GPU and vice versa. This is why `models/*/` is gitignored — engines must always be rebuilt on the target device.

**Precision modes** (`network-mode` in nvinfer):

| Value | Mode | Notes |
|---|---|---|
| `0` | FP32 | Highest accuracy, slowest |
| `1` | INT8 | Fastest; requires a calibration cache file for post-training quantisation |
| `2` | FP16 | Good accuracy/speed trade-off; no calibration needed |

PeopleNet ships with an INT8 calibration table. OSNet/ReID uses FP16.

**First-run latency**: on Jetson Orin NX, building a TRT engine from ONNX takes 3–10 minutes depending on model size and precision. This is normal — it happens once per (model, GPU, precision) combination.

---

### 3. NVIDIA TAO Toolkit — ETLT Format (Dead End for This Project)

> **This project does not use TAO.** This section documents a failed evaluation path so others don't repeat it.

TAO (Train, Adapt, Optimize) is NVIDIA's MLOps toolkit for fine-tuning pretrained models and exporting them in **ETLT** (Encrypted TLT) format — an AES-256 encrypted ONNX wrapper.

**Why we tried it**: NVIDIA hosts a `reidentificationnet:deployable_v1.0` model on NGC that looks like a drop-in ReID solution for DeepStream.

**Why it didn't work**: ETLT decryption requires the key used at export time. NVIDIA's docs say public TAO models use key `nvidia_tlt`, but `reidentificationnet:deployable_v1.0` cannot be decrypted with `nvidia_tlt` or any other documented key. There is no public API to discover the correct key. `tao-converter v4.0.0` and `v3.22.05` both fail silently.

**What we use instead**: open-source OSNet ONNX exported via `scripts/export_reid_onnx.py`. Plain ONNX has no encryption, TRT builds it directly, and the architecture is well-documented.

**Reference — `tao-converter` usage** (if you have a model with a known key):
```bash
tao-converter \
  -k <key>             # decryption key
  -d 3,256,128         # input dims (C,H,W) — no batch dimension
  -o fc_pred           # output layer name
  -t fp16              # precision
  -m 8                 # max batch size
  -e output.engine     # TRT engine destination
  model.etlt
```
Platform-specific binary — for TRT 8.5.x on Jetson use `v4.0.0_trt8.5.2.2_aarch64` from NGC. Find the right version ID via the REST API (see §4 below).

---

### 4. NGC CLI — Model Discovery Tricks

The NGC CLI (`ngc`) that ships with JetPack uses a different binary path than `/usr/bin/ngc`. Always find it first:

```bash
find /usr /home /opt -name ngc -type f 2>/dev/null
```

**Listing available files in a model version**: the NGC CLI does not have a `file-list` subcommand (despite what docs suggest). To see what's in a version, download the whole version without `--file`:

```bash
ngc registry model download-version nvidia/tao/reidentificationnet:deployable_v1.0 \
    --dest ./models/reid
# Contents land in a subdirectory: reidentificationnet_vdeployable_v1.0/
```

**Discovering resource version IDs**: the NGC CLI has no `list-versions` for resources. Query the public REST API instead:

```bash
curl -s "https://api.ngc.nvidia.com/v2/resources/nvidia/tao/tao-converter/versions" \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
for v in data.get('recipeVersions', []):
    print(v['versionId'])
"
```

This returns strings like `v4.0.0_trt8.5.2.2_aarch64`, `v5.1.0_jp6.0_aarch64`, etc. Match your TRT version (`dpkg -l libnvinfer8`).

**`canGuestDownload: true` resources** (like `tao-converter`) can be queried without authentication via the public API endpoint:
```
https://api.ngc.nvidia.com/v2/resources/<org>/<team>/<name>/versions
```

---

### 5. ReID Model — ResNet50 and Phase 3 Implementation Findings

#### Active model: ResNet50 (Market-1501 + AI City Challenge 156)

This project uses a ResNet50 backbone trained on combined Market-1501 and AI City Challenge datasets. It replaced OSNet x1.0 during Phase 3 development.

| Property | ResNet50 (current) | OSNet x1.0 (replaced) |
|---|---|---|
| Input | `3×256×128`, RGB | `3×256×128`, RGB |
| Output dim | 256 | 512 |
| Engine size | ~47 MB FP16 | ~5 MB FP16 |
| Inference time (Orin NX) | ~11 ms/batch-8 | ~22 ms/batch-8 |
| Inter-ID mean cosine sim | ~0.42 | N/A (collapsed, see below) |

ResNet50 is faster on Jetson because TensorRT can fuse standard 3×3 convolutions into efficient kernels. OSNet's omni-scale blocks (parallel depthwise branches with learned gates) don't benefit from the same fusion, resulting in higher latency despite a smaller model file.

#### Normalization — the most common silent failure

The ResNet50 ONNX model has **no preprocessing baked in** (the first op is a Conv layer). Without explicit normalization in the nvinfer config, all embeddings collapse and inter-ID cosine similarity is ~0.875 — indistinguishable from random noise.

Required normalization (ImageNet statistics):
```ini
net-scale-factor=0.017352607709750568   # ≈ 1 / (255 * 0.226)
offsets=123.675;116.28;103.53           # ImageNet mean (R;G;B)
```

Impact on embedding discrimination:

| Config | Inter-ID mean cosine sim | Gap (intra−inter) | Verdict |
|---|---|---|---|
| `net-scale-factor=0.00392` (1/255 only) | 0.875 | 0.062 | ✗ Collapsed — unusable |
| + `offsets` + correct scale | 0.42 | 0.363 | ✓ Discriminative |

#### Diagnostic workflow with `reid_probe.py`

Run while the pipeline is streaming to check embedding health without modifying any pipeline state:

```bash
python3 messaging/zmq/reid_probe.py --frames 500
```

Key metrics to watch:
- **Coverage**: should be **100%** — every person detection carries an embedding. If low (<50%), check `secondary-reinfer-interval` and `operate-on-gie-id` (see §1 above)
- **Norm**: should be **1.0** — ResNet50's `fc_pred` output is BN-normalised
- **Intra-ID mean**: cosine similarity of same person across frames — should be **> 0.70**
- **Inter-ID mean**: cosine similarity across different identities — should be **< 0.55**
- **Gap (intra−inter)**: should be **≥ 0.30** for reliable matching

#### Cross-camera global ID and OSD color coding

The C++ pad probe (`on_buffer_probe` in `pipeline.cpp`) maintains a cross-camera gallery in memory:

1. Each `(source_id, tracking_id)` pair is looked up in `reid_gallery_`
2. On first appearance, the embedding is compared (cosine similarity) against all entries from **other** cameras
3. If similarity ≥ threshold: the new track inherits the matching global ID
4. If no match: a new global ID is assigned (`next_global_id_++`)
5. The assignment is stored in `track_global_` and frozen — the track keeps the same ID for its lifetime (no flickering even as embeddings update)

OSD rendering: `global_id % 8` selects from an 8-color palette. The label `ID:N tid:M` shows both the stable cross-camera global ID and the per-camera NvDCF tracking ID.

The screenshot below shows the system working — the same physical person in both camera tiles has the same `ID:N` label and bounding box color:

> Two-camera tiled OSD view: orange box "ID:35 tid:35" on cam0, matching yellow box on cam1 — same global ID, different per-camera TID. A third person (blue, "ID:28") appears on cam0 only.

#### OSNet evaluation (why it was replaced)

OSNet x1.0 from torchreid was evaluated first. The publicly available Market-1501 softmax checkpoint (`GDrive 1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA`) was found unsuitable for distance-based matching: the BatchNorm neck had `running_var ≈ 0.0001` (collapsed), causing mean pairwise cosine similarity of 0.956 on random identity pairs. Even the ImageNet backbone (no ReID training) produced a better-spread space (mean 0.741, std 0.084). The ResNet50 model with metric-loss training resolved this definitively.

---

### 6. ONNX Export Gotchas with DeepStream

**Dynamic axes**: always export with `dynamic_axes` on the batch dimension. nvinfer sets the actual runtime batch size based on how many object crops arrive per frame — it may vary from 1 to `batch-size`. A static-batch ONNX will fail with a shape mismatch at runtime.

**Output node name must match**: the `output_names=["fc_pred"]` in `torch.onnx.export` controls what nvinfer sees as the output tensor name. The name `fc_pred` matches the standard TAO ReIdentificationNet convention. If you change it, update the nvinfer config and the pad-probe code that reads `NvDsInferTensorMeta` by name.

**opset_version=11**: safe baseline for TensorRT 8.x. Higher opsets add operators that TRT may not support.

**OSNet uses a `return_featuremaps` branch**: torchreid's OSNet `forward()` has a conditional:
```python
if self.training:
    return v, y  # feature + classifier logit
else:
    return v     # feature vector only (512-d)
```
You must call `model.eval()` before ONNX export. If the model is in training mode, the export trace records the wrong graph branch and the output shape becomes wrong.

**TracerWarning about Python booleans**: the `if return_featuremaps:` check triggers a `TracerWarning: Converting a tensor to a Python boolean`. This is safe to ignore — `return_featuremaps` is a constructor-time constant (`False` by default), so the branch is always the same.

---

### 7. Working Around torchreid's Broken Import on Jetson

On Jetson JP5.x, the Jetson-specific `torch==2.4.1` wheel is not compatible with any pip-available `torchvision` wheel (all torchvision releases pin a specific torch version). Installing torchvision from the standard PyTorch wheel index (`https://download.pytorch.org/whl/cu118`) silently downgrades torch to 2.0.1.

`torchreid.__init__` transitively imports `torchvision.transforms`, which causes an `ImportError` before any model code runs.

**Fix**: bypass the package init entirely by loading only `torchreid/reid/models/osnet.py` as a standalone module:

```python
import importlib.util, os

def _find_osnet_py():
    import site
    for base in site.getsitepackages() + [os.path.expanduser("~/.local/lib/python3.8/site-packages")]:
        p = os.path.join(base, "torchreid", "reid", "models", "osnet.py")
        if os.path.exists(p):
            return p

spec = importlib.util.spec_from_file_location("osnet", _find_osnet_py())
osnet_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(osnet_mod)

model = osnet_mod.osnet_x1_0(num_classes=751, pretrained=True, loss="softmax")
```

The model file itself only imports `torch` and `torch.nn` — no torchvision.

---

### 8. Jetson-Specific Environment Quirks

**Finding the correct JetPack version**:
```bash
cat /etc/nv_tegra_release
# R35 (release), REVISION: 3.1  →  JetPack 5.1.1
```

**Checking TensorRT version** (critical for picking the right tao-converter build):
```bash
dpkg -l libnvinfer8
# 8.5.2-1+cuda11.4  →  TRT 8.5.2, CUDA 11.4
```

**NGC CLI is not in PATH on Jetson**:
```bash
# Common locations:
/home/$USER/work/ngc-cli/ngc
/home/$USER/Downloads/ngccli_arm64/ngc-cli/ngc
```

**GPU not detected by Python torch in shell sessions**: `torch.cuda.is_available()` may return `False` in terminal sessions without CUDA env vars. This is harmless for ONNX export (which runs on CPU). Verify CUDA is present separately:
```bash
nvidia-smi   # or:  /usr/local/cuda/bin/nvcc --version
```

**`libnvdsinfer_custom_impl_Tao.so` is not shipped with DeepStream 6.2**: the TAO custom inference library that nvinfer needs to parse ETLT files inline is only provided for x86 in some DeepStream packages. On Jetson 6.2, you must use `tao-converter` to pre-build the engine, or switch to plain ONNX.

---

### 9. NvDCF Tracker — Tuning and Known Issues

**Visual features and NVMM surface pinning**

NvDCF's visual tracking mode (`visualTrackerType=1`) pins NVMM surfaces for CPU access. This races with `nvdsosd` and any downstream element that also maps the buffer, causing pipeline crashes or hangs on Jetson.

Fix: set `visualTrackerType=0` (DUMMY mode) in `config_tracker_NvDCF.yml`. The tracker falls back to pure IoU + Kalman filter motion — no pixel access, no NVMM contention. Cross-camera ReID handles appearance matching separately.

**Class filtering — `filter-out-class-ids` in the PGIE config**

PeopleNet detects three classes: person (0), bag (1), face (2). NvDCF assigns tracking IDs from a single global counter across all classes. Without filtering, face and bag bboxes appear in the OSD with IDs in the same sequence as persons, making the video confusing.

Fix: add `filter-out-class-ids=1;2` to `config_infer_primary_peoplenet.txt`. This removes bag and face from `NvDsMeta` before it reaches the tracker — they are never tracked, never drawn by OSD, and never processed by the pad probe. Only person detections reach downstream elements.

**ID instability — motion model tuning**

With visual features disabled, the tracker relies entirely on a Kalman filter for motion prediction. The default `processNoiseVar4Vel=0.03` is tuned for slow, smooth motion. When a person changes direction quickly, the Kalman prediction diverges, the IoU overlap drops below `minMatchingScore4Iou`, and the track is dropped — creating a new ID on re-detection.

Key parameters to tune in `config_tracker_NvDCF.yml`:

| Parameter | Default | Tuned | Effect |
|---|---|---|---|
| `maxShadowTrackingAge` | 51 | 90 | Keeps lost tracks alive for ~3s before eviction |
| `processNoiseVar4Vel` | 0.03 | 0.15 | Allows Kalman to handle faster direction changes |

Without a visual ReID appearance model in the tracker itself (which requires re-enabling `visualTrackerType=1`), ID instability on occlusion and re-entry remains a fundamental limitation of pure IoU tracking. The production fix is to integrate OSNet embeddings directly into the NvDCF association cost matrix via the `matchingScoreWeight4VisualSimilarity` weight — but that requires enabling visual tracking features (and resolving the NVMM surface pinning issue separately, e.g. by using `nvbuffermap` with proper unpin).

---

### 10. ZeroMQ PUB/SUB Design Pattern

The pipeline uses ZMQ `ZMQ_DONTWAIT` on the publisher socket. If no consumer is connected or the high-water mark (HWM) is reached, frames are **dropped, not blocked**. This keeps the GStreamer pipeline real-time — the inference loop is never stalled by a slow downstream consumer.

The trade-off: if the consumer crashes and restarts, it misses events during downtime. For a retail analytics use case (aggregate statistics, not transaction-level reliability), this is acceptable. For event-level reliability, use a durable message broker (Kafka, NATS JetStream).

---

## Future Improvements

### ReID — Fine-tuning for Higher Accuracy

The current ResNet50 checkpoint (Market-1501 + AI City 156) produces an inter-ID vs intra-ID gap of ~0.36, which is functional for multi-camera matching but leaves room for improvement. Options:

**Option 1 — FastReID fine-tune on domain-specific data**

[FastReID](https://github.com/JDAI-CV/fast-reid) provides a training framework and ONNX export compatible with this project's `infer-dims=3;256;128` and `output-tensor-meta=1` config. Fine-tuning on footage from the actual deployment cameras would improve recall at the current threshold.

**Option 2 — Track-level gallery instead of per-frame updates**

Currently the gallery stores the **latest** embedding per track. Averaging embeddings over the first N confident frames (e.g., high detector confidence, large bbox area) would produce a more stable representative and reduce noise from partial occlusions.

**How to evaluate a new checkpoint before deploying**: run `reid_probe.py` and check:
- Coverage: **100%** (every detection has an embedding)
- Inter-ID mean cosine sim: **< 0.50**
- Gap (intra−inter): **≥ 0.30**

### Cross-Camera Matching — Algorithmic Improvements

| Improvement | Description |
|---|---|
| **Temporal smoothing** | Average embeddings over N frames per tracklet instead of using the latest frame; reduces noise from partial occlusions |
| **Re-ranking** | Apply k-reciprocal re-ranking (Zhong et al.) post-matching to boost precision without retraining |
| **Zone-aware filtering** | Suppress matches between cameras that have no plausible physical path (e.g., camera A is entrance, camera B is stockroom — the set of reachable pairs is constrained by store layout) |
| **Track-level gallery** | Replace per-frame embedding updates with a per-tracklet representative (mean of top-confidence frames); reduces gallery noise |

### Pipeline — OSD Improvements

The current OSD draws global-ID-colored bboxes with `ID:N tid:M` labels. Possible extensions:
- **Confidence display**: show the ReID gallery match similarity score alongside the global ID label
- **Zone overlay**: draw configured zone polygons on the OSD frame for loitering detection debugging (Phase 4)

### Tracker — Re-enable Visual ReID for Stable IDs

With `visualTrackerType=0`, ID stability degrades when persons are briefly occluded or change direction. The correct production upgrade is to re-enable NvDCF visual tracking with the ResNet50 embedding as the appearance descriptor, letting the tracker use appearance similarity during association:

1. Set `visualTrackerType=1` and `matchingScoreWeight4VisualSimilarity > 0` in `config_tracker_NvDCF.yml`
2. Resolve the NVMM surface pinning race between nvtracker and downstream elements (use `process-mode=0` CPU rendering for nvdsosd, which is already set)
3. Tune `minMatchingScore4VisualSimilarity` on real footage

---

## Concept Map

Quick reference linking each SDK concept to where it appears in this codebase.

| Concept | Where it appears | Key points |
|---|---|---|
| **DeepStream pipeline** | `pipeline.cpp` | GStreamer element graph; PGIE→tracker→SGIE order; pad probes vs appsink |
| **TensorRT engine build** | First pipeline run | ONNX→TRT compilation; precision (INT8/FP16); engine non-portability; layer fusion |
| **SGIE / object-level inference** | ReID nvinfer config | `process-mode=2`; automatic crop routing; do NOT set `operate-on-gie-id`; `secondary-reinfer-interval=1` GObject property |
| **TAO Toolkit** | `download_reid.sh` | ETLT format; tao-converter; key-based decryption; TAO vs open-source ONNX |
| **NGC CLI** | Model download scripts | Registry vs resource; version discovery via REST API; guest-downloadable resources |
| **NvDCF tracker** | Tracker config | Correlation filter; NVMM surface pinning; probation/shadow age |
| **ReID pipeline** | Phase 3 | ResNet50 SGIE; 256-dim embeddings; ImageNet normalization; `secondary-reinfer-interval`; cross-camera gallery; global ID color coding |
| **Jetson platform** | Hardware target | NVMM zero-copy; CUDA unified memory; DLA; power modes (`nvpmodel`, `jetson_clocks`) |
| **ONNX export** | `export_reid_onnx.py` | Dynamic axes; opset version; `eval()` vs `train()` branch tracing |
| **ZeroMQ** | `publisher.cpp` | PUB/SUB; HWM; DONTWAIT drop semantics; vs Kafka durability |
