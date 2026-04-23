"""
Cross-camera ReID matching service (Phase 3).

Subscribes to the ZMQ event stream, maintains a cross-camera identity table,
and emits ONE event when a new cross-camera link is first established — not
every frame.  Subsequent frames that confirm the same link are silent.

Output events (one JSON line each):

  New person first seen on any camera:
    {"event":"reid_identity","ts_ms":...,"global_id":0,
     "source_id":1,"tid":23}

  Same person confirmed across cameras (emitted ONCE per new link):
    {"event":"reid_link","ts_ms":...,"global_id":0,
     "cam_a":{"source_id":1,"tid":23},
     "cam_b":{"source_id":0,"tid":17},"similarity":0.832}

  Periodic status (stderr, every --status-interval frames):
    [reid] identities: {0: [(0,17),(1,23)], 1: [(0,24)], ...}

Usage:
    python3 reid_matcher.py [--endpoint tcp://localhost:5555]
                            [--threshold 0.70]
                            [--gallery-ttl 30]
                            [--status-interval 300]
"""

import argparse
import base64
import json
import logging
import signal
import sys
import time

import numpy as np
import zmq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [reid] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PERSON_CLASS = 0


# ---------------------------------------------------------------------------
# Identity registry
# ---------------------------------------------------------------------------

class IdentityRegistry:
    """
    Assigns a stable global_id to each physical person across cameras.

    Rules (same as the C++ OSD gallery):
    - A new track (source_id, tid) seen for the first time searches the
      other camera's gallery for the best embedding match.
    - If similarity >= threshold  →  inherits that global_id (cross-camera link).
    - Otherwise                   →  new global_id (new person).
    - Once assigned, the mapping never changes for the lifetime of the track.
    - The embedding gallery is updated every frame so matching improves over time.
    """

    def __init__(self, threshold: float, ttl: float):
        self._threshold = threshold
        self._ttl       = ttl
        # Stable assignment: (src, tid) → global_id
        self._track_global: dict[tuple, int] = {}
        # Rolling gallery:  (src, tid) → {"emb": np.ndarray, "ts": float}
        self._gallery: dict[tuple, dict] = {}
        # global_id → list of (src, tid) that share it
        self._identities: dict[int, list] = {}
        self._next_id = 0

    # ── public API ────────────────────────────────────────────────────────────

    def process(self, src: int, tid: int, emb: np.ndarray, ts_ms: int):
        """
        Process one detection.  Returns a dict describing what happened:
          {"action": "new_identity", "global_id": N}          – first encounter
          {"action": "link",  "global_id": N, "matched": (s,t), "sim": f}
                                                               – cross-camera link established
          {"action": "known", "global_id": N}                 – already tracked, no event needed
        """
        self._evict()
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
        key = (src, tid)

        # Already assigned — just refresh embedding and return silently.
        if key in self._track_global:
            self._gallery[key] = {"emb": emb_norm, "ts": time.monotonic()}
            return {"action": "known", "global_id": self._track_global[key]}

        # New track — find best match from OTHER cameras.
        best_sim, best_key = self._threshold, None
        for (s, t), entry in self._gallery.items():
            if s == src:
                continue
            sim = float(np.dot(emb_norm, entry["emb"]))
            if sim > best_sim:
                best_sim, best_key = sim, (s, t)

        if best_key is not None:
            global_id = self._track_global[best_key]
            self._track_global[key] = global_id
            self._identities[global_id].append(key)
            self._gallery[key] = {"emb": emb_norm, "ts": time.monotonic()}
            return {"action": "link", "global_id": global_id,
                    "matched": best_key, "sim": round(best_sim, 4)}
        else:
            global_id = self._next_id
            self._next_id += 1
            self._track_global[key] = global_id
            self._identities[global_id] = [key]
            self._gallery[key] = {"emb": emb_norm, "ts": time.monotonic()}
            return {"action": "new_identity", "global_id": global_id}

    def summary(self) -> str:
        """Human-readable identity table for status logging."""
        lines = []
        for gid, tracks in sorted(self._identities.items()):
            cams = ", ".join(f"cam{s}:tid{t}" for s, t in tracks)
            n_cams = len({s for s, _ in tracks})
            flag = " ✓" if n_cams > 1 else ""
            lines.append(f"  ID:{gid:3d}  [{cams}]{flag}")
        return "\n".join(lines) if lines else "  (empty)"

    # ── internals ─────────────────────────────────────────────────────────────

    def _evict(self):
        cutoff = time.monotonic() - self._ttl
        dead = [k for k, v in self._gallery.items() if v["ts"] < cutoff]
        for k in dead:
            del self._gallery[k]
            # Note: track_global and identities are kept even after eviction so
            # global IDs remain stable if the person re-appears before session end.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_embedding(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    return np.frombuffer(raw, dtype=np.float32).copy()


def process_event(event: dict, registry: IdentityRegistry) -> list:
    out = []
    src = event.get("source_id", -1)
    ts  = event.get("ts_ms", 0)

    for det in event.get("detections", []):
        if det.get("class") != PERSON_CLASS:
            continue
        if "emb" not in det:
            continue
        tid = det.get("tid")
        if tid is None:
            continue
        try:
            emb = decode_embedding(det["emb"])
        except Exception as exc:
            log.warning("Bad embedding src=%d tid=%s: %s", src, tid, exc)
            continue

        result = registry.process(src, tid, emb, ts)

        if result["action"] == "new_identity":
            out.append({"event": "reid_identity", "ts_ms": ts,
                        "global_id": result["global_id"],
                        "source_id": src, "tid": tid})

        elif result["action"] == "link":
            ms, mt = result["matched"]
            out.append({"event": "reid_link", "ts_ms": ts,
                        "global_id": result["global_id"],
                        "cam_a": {"source_id": src, "tid": tid},
                        "cam_b": {"source_id": ms,  "tid": mt},
                        "similarity": result["sim"]})

        # "known" → silent, no output

    return out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(endpoint: str, threshold: float, gallery_ttl: float,
        status_interval: int) -> None:
    ctx  = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.RCVTIMEO, 1000)
    sock.connect(endpoint)
    log.info("Connected to %s  threshold=%.2f  ttl=%.0fs", endpoint, threshold, gallery_ttl)

    registry = IdentityRegistry(threshold=threshold, ttl=gallery_ttl)

    running = True
    def _stop(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    frame_count = 0

    while running:
        try:
            raw = sock.recv_string()
        except zmq.Again:
            continue

        try:
            event = json.loads(raw)
        except json.JSONDecodeError as exc:
            log.warning("Malformed JSON: %s", exc)
            continue

        frame_count += 1
        for msg in process_event(event, registry):
            print(json.dumps(msg), flush=True)

        if status_interval > 0 and frame_count % status_interval == 0:
            log.info("frames=%d  identities:\n%s", frame_count, registry.summary())

    log.info("Stopped after %d frames\nFinal identities:\n%s",
             frame_count, registry.summary())
    sock.close()
    ctx.term()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-camera ReID matching service")
    p.add_argument("--endpoint",        default="tcp://localhost:5555")
    p.add_argument("--threshold",       type=float, default=0.70)
    p.add_argument("--gallery-ttl",     type=float, default=30.0)
    p.add_argument("--status-interval", type=int,   default=300,
                   help="Print identity table every N frames (0=off)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.endpoint, args.threshold, args.gallery_ttl, args.status_interval)
