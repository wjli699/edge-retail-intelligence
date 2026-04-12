# 🧠 Edge Retail Intelligence Engine (Loitering Detection + ReID)

## 📌 Overview

This project is a **production-oriented Edge AI system** for retail analytics, focused on:

* **Loitering detection**
* **Person tracking**
* **Cross-camera re-identification (ReID)**
* **Event-driven analytics API**

The system is designed with a **high-performance C++ core engine** for real-time video processing and a **FastAPI control plane** for orchestration, configuration, and integration.

---

## 🏗️ Architecture

```
[ Cameras / RTSP Streams ]
            ↓
   C++ Core Engine (DeepStream)
   - Detection (Person)
   - Tracking (Per-camera)
   - ReID Embedding Extraction
   - Metadata Serialization
            ↓
     Message Bus (ZeroMQ / Kafka)
            ↓
   Application Layer (FastAPI)
   - Multi-camera correlation
   - Loitering logic
   - Event management
   - REST APIs
            ↓
     Storage / Search / UI (Optional)
```

---

## 🎯 Key Features

### 🎥 C++ Core Engine

* Multi-stream video ingestion (RTSP, file)
* GPU-accelerated inference
* Person detection (YOLO or equivalent)
* Per-camera tracking
* ReID embedding extraction
* Configurable pipeline (YAML/INI)
* Metadata export (JSON / protobuf)

---

### 🧠 Intelligence Layer (FastAPI)

* Multi-camera identity correlation (ReID matching)
* Loitering detection:

  * dwell time threshold
  * zone-based monitoring
* Event aggregation and filtering
* REST API for:

  * querying events
  * configuring rules
  * system monitoring

---

### 🔄 Messaging Layer

* Decouples real-time inference from business logic
* Supports:

  * ZeroMQ (edge lightweight)
  * Kafka (scalable deployment)

---

### 📊 Optional Extensions

* Event storage (PostgreSQL / TimescaleDB)
* Search (vector DB for ReID embeddings)
* UI dashboard
* Alerting (webhook, email)

---

## 🧪 Example Use Case

**Retail Loitering Detection:**

1. Detect person entering a zone
2. Track movement over time
3. Extract ReID embedding
4. Correlate across cameras
5. If dwell time exceeds threshold → trigger event

---

## 🧰 Tech Stack

| Layer              | Technology                   |
| ------------------ | ---------------------------- |
| Video Pipeline     | C++ (DeepStream / GStreamer) |
| Inference          | TensorRT                     |
| Messaging          | ZeroMQ / Kafka               |
| API Layer          | FastAPI (Python)             |
| Storage (optional) | PostgreSQL / Redis           |
| Deployment         | Docker / Docker Compose      |

---

## 💻 Development Environment

### Hardware

* NVIDIA Jetson (Orin NX recommended) OR
* x86 + NVIDIA GPU (RTX series)

---

### Software Requirements

#### Core Engine (C++)

* Ubuntu 20.04/22.04
* CUDA Toolkit
* DeepStream SDK
* TensorRT
* GStreamer

#### API Layer (Python)

* Python 3.8+
* FastAPI
* Uvicorn
* Pydantic

---

### 📦 Setup (High-Level)

#### 1. Clone Repository

```
git clone <repo>
cd edge-retail-intelligence
```

#### 2. Build C++ Engine

```
cd core_engine
mkdir build && cd build
cmake ..
make -j
```

#### 3. Start API Service

```
cd api
pip install -r requirements.txt
uvicorn main:app --reload
```

#### 4. Run System

* Start message broker
* Launch C++ pipeline
* Start FastAPI

---

## 📁 Project Structure

```
edge-retail-intelligence/
│
├── core_engine/              # C++ DeepStream pipeline
│   ├── src/
│   ├── include/
│   ├── configs/
│   └── CMakeLists.txt
│
├── api/                     # FastAPI control plane
│   ├── main.py
│   ├── routes/
│   ├── services/
│   └── models/
│
├── messaging/               # Queue abstraction
│   ├── zmq/
│   └── kafka/
│
├── models/                  # AI models (ONNX / TRT engines)
│
├── scripts/                 # Setup / run scripts
│
└── README.md
```

---

## 🚀 Roadmap (Phased Development)

---

### 🟢 Phase 1 — C++ Core Pipeline (Foundation)

**Goal:** Real-time single-camera intelligence

* [ ] Build DeepStream pipeline
* [ ] Integrate person detection model
* [ ] Enable tracking (per stream)
* [ ] Implement `pad_probe` metadata extraction
* [ ] Output JSON events (stdout/file)

👉 Deliverable:

* Working pipeline with bounding boxes + tracking IDs

---

### 🟡 Phase 2 — Messaging & Decoupling

**Goal:** Separate pipeline from logic

* [ ] Integrate ZeroMQ or Kafka producer
* [ ] Serialize metadata (JSON/protobuf)
* [ ] Build basic consumer service
* [ ] Validate multi-stream ingestion

👉 Deliverable:

* Real-time metadata streaming system

---

### 🔵 Phase 3 — ReID + Multi-Camera Correlation

**Goal:** Identity across cameras

* [ ] Add ReID model (embedding extraction)
* [ ] Store embeddings temporarily
* [ ] Implement similarity matching (cosine distance)
* [ ] Build identity tracking across cameras

👉 Deliverable:

* Same person recognized across multiple streams

---

### 🟣 Phase 4 — Loitering Detection Engine

**Goal:** Business logic layer

* [ ] Define zones (ROI polygons)
* [ ] Track dwell time per identity
* [ ] Trigger loitering events
* [ ] Add configurable thresholds

👉 Deliverable:

* Accurate loitering alerts

---

### 🟠 Phase 5 — FastAPI Control Plane

**Goal:** System usability

* [ ] REST API for:

  * camera management
  * rule configuration
  * event query
* [ ] Health monitoring endpoints
* [ ] Config persistence

👉 Deliverable:

* Fully controllable system via API

---

### 🔴 Phase 6 — Production Readiness

**Goal:** Scale & polish

* [ ] Dockerize services
* [ ] Add logging + metrics
* [ ] Optimize batching / latency
* [ ] Add storage backend
* [ ] Optional UI dashboard

👉 Deliverable:

* Deployable edge AI product

---

## ⚖️ Design Principles

* **Decoupled architecture** (pipeline ≠ intelligence)
* **Real-time first** (low latency, high throughput)
* **Extensible** (add models, rules easily)
* **Edge-native** (Jetson optimized)
* **Hybrid stack** (C++ + Python)

---

## 🔥 Why This Project Matters

This is not just a learning exercise—it mirrors **real-world Edge AI systems** used in:

* Retail analytics
* Smart cities
* Security systems
* Autonomous monitoring

You will gain experience in:

* GPU pipelines
* distributed systems
* AI deployment
* system architecture

---

## 📌 Future Enhancements

* LLM-powered “search your cameras”
* Behavior prediction
* Cross-site analytics
* Federated edge deployment
* Integration with VMS systems

---

## 🤝 Contribution

Contributions are welcome:

* New models
* Performance optimizations
* API extensions
* Deployment improvements

---

## 📄 License

TBD
