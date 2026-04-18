# Edge Retail Intelligence Engine

Real-time edge AI for retail analytics: person detection, per-camera tracking, cross-camera re-identification (ReID), and loitering detection — running on **NVIDIA Jetson Orin NX**.

---

## Architecture

```
[ RTSP Cameras ]
      ↓
C++ Core Engine  (DeepStream 6.2 / GStreamer / TensorRT)
  • Person detection    — PeopleNet ResNet-34 INT8
  • Per-camera tracking — NvDCF correlation-filter tracker
  • ReID embedding      — OSNet-x1.0 FP16 SGIE (512-dim per person crop)
  • Metadata extraction — GStreamer pad probe → JSONL events over ZeroMQ
      ↓
Message Bus  (ZeroMQ PUB/SUB)
      ↓
Application Layer  (Phase 5 — FastAPI)
  • Cross-camera ReID correlation
  • Loitering detection (dwell-time + zone rules)
  • REST API
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Video pipeline | C++17, NVIDIA DeepStream 6.2, GStreamer 1.16 |
| Primary inference | PeopleNet v2.3.3 (ResNet-34), TensorRT INT8 |
| Secondary inference | OSNet-x1.0, TensorRT FP16, SGIE mode |
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
| 3 | ReID embedding extraction + cross-camera correlation | **In progress** |
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

### 1. Download & export models

```bash
# PeopleNet (Phase 1)
./scripts/download_peoplenet.sh

# ReID — exports OSNet-x1.0 to ONNX (no NGC account needed)
python3 scripts/export_reid_onnx.py
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

# Terminal 2 — ZMQ consumer
cd messaging/zmq
python3 consumer.py --endpoint tcp://localhost:5555 --filter person
```

On first run, nvinfer auto-builds TRT engines from the ONNX files (~5 min each on Orin NX). Subsequent starts load the cached `.engine` files instantly.

---

## Output Format

One JSON object per frame, written to the configured output (ZMQ / stdout / file):

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
      "reid_embedding": [0.023, -0.107, ...]
    }
  ]
}
```

| Field | Description |
|---|---|
| `ts_ms` | Wall-clock timestamp (Unix ms) |
| `source_id` | Camera index (0-based) |
| `tid` | Stable per-camera tracking ID (NvDCF) |
| `reid_embedding` | 512-d float vector; cosine-similarity across cameras identifies the same person |

---

## Configuration Reference

### `configs/default.yaml`

```yaml
sources:
  - uri: "rtsp://<host>:<port>/<path>"
    enabled: true

models:
  detector: "configs/config_infer_primary_peoplenet.txt"
  tracker:  "configs/config_tracker_NvDCF.yml"
  reid:     "configs/config_infer_secondary_reid.txt"

output:
  mode: zmq                    # zmq | stdout | file
  endpoint: "tcp://*:5555"
```

### `configs/config_infer_secondary_reid.txt` (key fields)

| Field | Value | Notes |
|---|---|---|
| `process-mode` | `2` | Secondary GIE (SGIE) — runs on object crops, not full frames |
| `operate-on-gie-id` | `1` | Receives crops from primary GIE (PeopleNet) |
| `operate-on-class-ids` | `0` | Person class only |
| `onnx-file` | `osnet_x1_0_market1501.onnx` | Source model; TRT engine built from this on first run |
| `network-mode` | `2` | FP16 inference |
| `batch-size` | `8` | Max person crops per inference call; tune to GPU memory |
| `network-type` | `1` | Classifier — allocates `NvDsClassifierMeta` per object |
| `output-tensor-meta` | `1` | Attaches raw float tensors to `NvDsObjectMeta` for pad-probe access |
| `gie-unique-id` | `2` | Must match `kReidGieId` constant in `pipeline.cpp` |

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
│   ├── download_reid.sh      # Downloads NVIDIA TAO ETLT (reference only)
│   └── export_reid_onnx.py   # Exports OSNet to ONNX — use this instead
└── messaging/zmq/
    └── consumer.py
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

**Pad probe placement matters**: the probe in this project sits on the **nvtracker src pad**, not the SGIE src pad. This means when the probe fires, the PGIE *and* SGIE have already run. The ReID embedding is already populated in `NvDsObjectMeta` — you just read it from the tensor metadata.

**`output-tensor-meta=1` is required for embeddings**: by default, SGIE with `network-type=1` (classifier) only writes the argmax class label into `NvDsClassifierMeta`. Setting `output-tensor-meta=1` additionally attaches the raw float buffer to `NvDsObjectMeta::tensor_output_list`, which lets the pad probe read the full 512-d embedding vector.

**`gie-unique-id` links pipeline elements**: every nvinfer has a unique integer ID. The SGIE declares `operate-on-gie-id=1` (the PGIE's ID) and the C++ code uses `kReidGieId=2` to find the correct tensor output when iterating `NvDsObjectMeta`. If these IDs don't match, you get no embeddings silently.

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

### 3. NVIDIA TAO Toolkit — ETLT Format and Its Pitfalls

TAO (Train, Adapt, Optimize) is NVIDIA's MLOps toolkit for fine-tuning and exporting models. It exports in **ETLT** (Encrypted TLT) format — an AES-256 encrypted ONNX file.

**ETLT decryption requires the correct key**: every ETLT is encrypted with a key set at export time. NVIDIA's public TAO models are documented as using key `nvidia_tlt`, but this is not universally true. The `reidentificationnet:deployable_v1.0` ETLT could not be decrypted with any commonly documented key (`nvidia_tlt`, `tlt_encode`).

**`tao-converter` is the official decrypt + TRT build tool**:
```bash
tao-converter \
  -k nvidia_tlt        # decryption key
  -d 3,256,128         # input dims (C,H,W) — no batch
  -o fc_pred           # output layer name
  -t fp16              # precision
  -m 8                 # max batch size
  -e output.engine     # engine destination
  model.etlt
```

Key facts about `tao-converter`:
- It is **platform-specific** — the aarch64 build for TRT 8.5.x is `v4.0.0_trt8.5.2.2_aarch64` on NGC. Download via `ngc registry resource download-version`.
- It links against `libcrypto.so.1.1` (OpenSSL 1.1). Verify with `ldd`.
- **`tao-converter` is deprecated** as of 2023 in favour of `nvidia-tao-deploy` pip package. However `nvidia-tao-deploy` is complex to install on Jetson due to CUDA dependency mismatches.
- Engines built by `tao-converter` can be consumed by nvinfer using only `model-engine-file` (no `onnx-file` or `tlt-encoded-model` needed).

**DeepStream can consume ETLT directly** via `tlt-encoded-model` + `tlt-model-key` in the nvinfer config (no tao-converter needed). However this calls the same decryption path internally, so a wrong key fails the same way.

**Practical lesson**: if an NVIDIA TAO ETLT from NGC fails to decrypt, the key is simply not `nvidia_tlt`. There is no public API to discover it. The pragmatic solution is to use an open-source ONNX model with equivalent architecture.

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

### 5. OSNet — Architecture and Why It's Preferred for ReID

OSNet (Omni-Scale Network, Zhou et al. ICCV 2019) is the dominant lightweight ReID backbone. Key architectural ideas:

- **Omni-scale feature learning**: each layer aggregates features at multiple spatial scales (1×1, 3×3, 5×5, 7×7) via depthwise separable convolution branches, then fuses them with a learned gate.
- **Unified aggregation gate (UAG)**: a channel-wise sigmoid gate controls how much each scale contributes, conditioned on the input feature map.
- **Output**: a global average pooling + BN layer produces a 512-d L2-normalised feature vector.

OSNet-x1.0 input/output for this project:
- Input: `(B, 3, 256, 128)` — standard Market-1501 crop size, RGB, normalised with `net-scale-factor=1/255`
- Output: `(B, 512)` — embedding vector (cosine similarity used for identity matching)

**Pretrained weights from torchreid**: the `torchreid` library (kaiyangzhou) provides pretrained OSNet checkpoints on Market-1501, DukeMTMC, and others. The `pretrained=True` flag in `osnet_x1_0()` downloads from Google Drive via `gdown`. Market-1501 fine-tuned weights produce ~94% Rank-1 accuracy on the Market-1501 benchmark.

**What this project currently uses**: `scripts/export_reid_onnx.py` loads the **ImageNet backbone** (`pretrained=True`, torchreid key `'osnet_x1_0'`). See the finding below for why.

**Finding: softmax-only Market-1501 training causes embedding collapse**

We evaluated the publicly available torchreid Market-1501 checkpoint (GDrive `1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA`) and found it unsuitable for distance-based matching:

| Metric | ImageNet backbone | Market-1501 softmax ckpt |
|---|---|---|
| Embedding norm (unit expected) | ~27 (raw) → 1.0 after L2-wrap | ~25 (raw) → 1.0 after L2-wrap |
| Mean pairwise cosine sim (random pairs) | **0.741** | **0.956** |
| Std of pairwise cosine sim | 0.084 | 0.008 |
| BN neck `fc.1.running_var` | normal | ~0.0001 (collapsed) |

Root cause: the checkpoint was trained with **softmax cross-entropy only** (no metric loss). Softmax optimises classification accuracy without constraining the geometry of the embedding space. The BatchNorm neck collapses (`running_var ≈ 0.0001`), causing all inputs — including random noise — to land in a tiny angular cluster. Pairwise cosine similarity of 0.956 on random inputs means the embeddings carry almost no identity information for retrieval.

The ImageNet backbone, despite having no ReID-specific training, produces a much better-spread embedding space (mean 0.741, std 0.084) and is the more useful baseline until a metric-loss trained model is available.

**Diagnostic tool**: `messaging/zmq/reid_probe.py` measures these statistics live from the ZMQ stream without modifying any pipeline state. Run it while the pipeline is streaming to check embedding health:
```bash
python3 messaging/zmq/reid_probe.py --frames 500
```
Key output to watch: norm statistics (should be 1.0 if L2-norm is baked in), mean pairwise cosine sim (should be <0.5 for a well-discriminating model), and fraction of pairs above the matcher threshold (should be low for different-identity pairs).

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

### 9. NvDCF Tracker — Visual Features and NVMM Surface Pinning

The NvDCF tracker in DeepStream supports a **visual tracking mode** that computes appearance features from frame crops. Enabling this (`visualTrackerType=1`) causes the tracker to pin NVMM surfaces for CPU access, which can race with the display (nvdsosd) or downstream elements if buffers are not properly unmapped.

**Symptom**: pipeline crashes or hangs with NVMM pinning errors in the log.

**Fix**: set `useColorNames=0` and `useHog=0` in `config_tracker_NvDCF.yml` to disable visual features and revert to pure correlation-filter tracking (IoU + motion). Cross-camera ReID handles appearance matching instead.

---

### 10. ZeroMQ PUB/SUB Design Pattern

The pipeline uses ZMQ `ZMQ_DONTWAIT` on the publisher socket. If no consumer is connected or the high-water mark (HWM) is reached, frames are **dropped, not blocked**. This keeps the GStreamer pipeline real-time — the inference loop is never stalled by a slow downstream consumer.

The trade-off: if the consumer crashes and restarts, it misses events during downtime. For a retail analytics use case (aggregate statistics, not transaction-level reliability), this is acceptable. For event-level reliability, use a durable message broker (Kafka, NATS JetStream).

---

## Future Improvements

### ReID — Upgrade to Metric-Loss Trained Weights

The biggest accuracy gain available with zero pipeline changes is replacing the ImageNet backbone with a model trained using **metric loss** (triplet, ArcFace, or contrastive). Softmax-only training collapses the embedding space; metric-loss training explicitly pushes different-identity embeddings apart and same-identity embeddings together.

**Option 1 — FastReID (JDAI-CV)**

[FastReID](https://github.com/JDAI-CV/fast-reid) is the current state-of-the-art open-source ReID library. It provides:
- OSNet-x1.0 trained with ArcFace loss on Market-1501 (expected: same-person cosine sim >0.85, different-person <0.40)
- Plug-and-play ONNX export compatible with this project's `infer-dims=3;256;128` and `output-tensor-meta=1` config
- Export command: `python tools/deploy/onnx_export.py --config-file configs/Market1501/osnet_ibn.yml MODEL.WEIGHTS path/to/model.pth`

**Option 2 — Strong-ReID Baseline (Luo et al.)**

The ["bag of tricks"](https://github.com/michuanhaohao/reid-strong-baseline) baseline trains ResNet-50 with triplet + softmax (ID loss). Pre-trained Market-1501 weights are available. OSNet-compatible alternative is available via [torchreid with triplet loss](https://kaiyangzhou.github.io/deep-person-reid/datasets.html).

**Option 3 — Torchreid with Metric Loss**

Torchreid supports `loss='triplet'` training mode. A Market-1501 OSNet-x1.0 checkpoint trained with triplet+softmax is available from the torchreid model zoo:
```python
# In torchreid (if torchvision conflict is resolved):
model = torchreid.models.build_model('osnet_x1_0', num_classes=751, loss='triplet')
torchreid.utils.load_pretrained_weights(model, 'osnet_x1_0_market_256x128_amsgrad_ep150_...')
```

**How to evaluate a new checkpoint before deploying**: run `reid_probe.py` and check:
- Mean pairwise cosine sim among random pairs should drop to **<0.45**
- Std should rise to **>0.15** (embedding space is spread out)
- BN neck `running_var` values should be in the normal range (0.01–1.0)

### Cross-Camera Matching — Algorithmic Improvements

| Improvement | Description |
|---|---|
| **Temporal smoothing** | Average embeddings over N frames per tracklet instead of using the latest frame; reduces noise from partial occlusions |
| **Re-ranking** | Apply k-reciprocal re-ranking (Zhong et al.) post-matching to boost precision without retraining |
| **Zone-aware filtering** | Suppress matches between cameras that have no plausible physical path (e.g., camera A is entrance, camera B is stockroom — the set of reachable pairs is constrained by store layout) |
| **Track-level gallery** | Replace per-frame embedding updates with a per-tracklet representative (mean of top-confidence frames); reduces gallery noise |

### Pipeline — OSD and Video Output

Adding `nvdsosd` enables visual debugging of detection boxes, tracking IDs, and (with custom overlays) ReID match events directly on the video stream. See the OSD integration notes in the pipeline source (`pipeline.cpp`) for the NVMM surface handling required to avoid pinning races.

---

## Concept Map

Quick reference linking each SDK concept to where it appears in this codebase.

| Concept | Where it appears | Key points |
|---|---|---|
| **DeepStream pipeline** | `pipeline.cpp` | GStreamer element graph; PGIE→tracker→SGIE order; pad probes vs appsink |
| **TensorRT engine build** | First pipeline run | ONNX→TRT compilation; precision (INT8/FP16); engine non-portability; layer fusion |
| **SGIE / object-level inference** | ReID nvinfer config | `process-mode=2`; automatic crop routing; `operate-on-gie-id` linkage |
| **TAO Toolkit** | `download_reid.sh` | ETLT format; tao-converter; key-based decryption; TAO vs open-source ONNX |
| **NGC CLI** | Model download scripts | Registry vs resource; version discovery via REST API; guest-downloadable resources |
| **NvDCF tracker** | Tracker config | Correlation filter; NVMM surface pinning; probation/shadow age |
| **ReID pipeline** | Phase 3 | Embedding extraction; cosine similarity; cross-camera gallery matching |
| **Jetson platform** | Hardware target | NVMM zero-copy; CUDA unified memory; DLA; power modes (`nvpmodel`, `jetson_clocks`) |
| **ONNX export** | `export_reid_onnx.py` | Dynamic axes; opset version; `eval()` vs `train()` branch tracing |
| **ZeroMQ** | `publisher.cpp` | PUB/SUB; HWM; DONTWAIT drop semantics; vs Kafka durability |
