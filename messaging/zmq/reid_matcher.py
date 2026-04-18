"""
Cross-camera ReID matching service (Phase 3).

Subscribes to the ZMQ event stream from the C++ pipeline, maintains a
per-camera rolling embedding gallery, and prints a reid_match JSON event
whenever the same person is likely seen on two different cameras.

Usage:
    python reid_matcher.py [--endpoint tcp://localhost:5555]
                           [--threshold 0.75]
                           [--gallery-ttl 30]

Output (one JSON line per cross-camera match):
    {"event":"reid_match","ts_ms":...,"cam_a":{"source_id":0,"tid":3},
     "cam_b":{"source_id":1,"tid":7},"similarity":0.912}

Similarity is cosine similarity in the L2-normalised embedding space [0, 1].
Threshold guidance:
    >= 0.80  high confidence same person
    0.70-0.79  likely same person, worth flagging
    <  0.70  different person (default threshold is 0.75)
"""

import argparse
import base64
import json
import logging
import signal
import sys
import time
from collections import defaultdict

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
# Gallery
# ---------------------------------------------------------------------------

class EmbeddingGallery:
    """
    Rolling gallery of (source_id, tracking_id) → embedding entries.

    Entries expire after `ttl` seconds of no update.  Only person-class
    detections with embeddings are stored.
    """

    def __init__(self, ttl: float = 30.0):
        self._ttl = ttl
        # key: (source_id, tid) → {"emb": np.ndarray, "ts": float}
        self._entries: dict = {}

    def update(self, source_id: int, tid: int, emb: np.ndarray) -> None:
        self._entries[(source_id, tid)] = {"emb": emb, "ts": time.monotonic()}
        self._evict()

    def _evict(self) -> None:
        cutoff = time.monotonic() - self._ttl
        dead = [k for k, v in self._entries.items() if v["ts"] < cutoff]
        for k in dead:
            del self._entries[k]

    def query(
        self, source_id: int, tid: int, emb: np.ndarray, threshold: float
    ) -> list:
        """
        Return matches on other cameras above `threshold`, sorted by similarity.
        Each match: {"source_id": int, "tid": int, "similarity": float}
        """
        q = emb / (np.linalg.norm(emb) + 1e-8)
        matches = []
        for (s, t), entry in self._entries.items():
            if s == source_id:
                continue  # skip same camera
            g = entry["emb"]
            g = g / (np.linalg.norm(g) + 1e-8)
            sim = float(np.dot(q, g))
            if sim >= threshold:
                matches.append({"source_id": s, "tid": t, "similarity": round(sim, 4)})
        return sorted(matches, key=lambda x: -x["similarity"])

    @property
    def size(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_embedding(b64: str) -> np.ndarray:
    """Decode a base64-encoded float32 embedding produced by the C++ pipeline."""
    raw = base64.b64decode(b64)
    return np.frombuffer(raw, dtype=np.float32).copy()


def process_event(event: dict, gallery: EmbeddingGallery, threshold: float) -> list:
    """
    Process one frame event.  Returns a (possibly empty) list of reid_match dicts.
    """
    matches_out = []
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
            log.warning("Failed to decode embedding for src=%d tid=%s: %s", src, tid, exc)
            continue

        # Query before updating so we don't match against ourselves
        for match in gallery.query(src, tid, emb, threshold):
            matches_out.append({
                "event":      "reid_match",
                "ts_ms":      ts,
                "cam_a":      {"source_id": src, "tid": tid},
                "cam_b":      match,
                "similarity": match["similarity"],
            })

        gallery.update(src, tid, emb)

    return matches_out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(endpoint: str, threshold: float, gallery_ttl: float) -> None:
    ctx  = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.RCVTIMEO, 1000)
    sock.connect(endpoint)
    log.info("Connected to %s  threshold=%.2f  gallery_ttl=%.0fs",
             endpoint, threshold, gallery_ttl)

    gallery = EmbeddingGallery(ttl=gallery_ttl)

    running = True
    def _stop(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    frame_count  = 0
    match_count  = 0
    log_interval = 200  # frames

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
        for match in process_event(event, gallery, threshold):
            match_count += 1
            print(json.dumps(match), flush=True)

        if frame_count % log_interval == 0:
            log.info("frames=%d  matches=%d  gallery_entries=%d",
                     frame_count, match_count, gallery.size)

    log.info("Stopped — frames=%d  matches=%d", frame_count, match_count)
    sock.close()
    ctx.term()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-camera ReID matching service")
    p.add_argument("--endpoint",    default="tcp://localhost:5555",
                   help="ZMQ endpoint to subscribe to")
    p.add_argument("--threshold",   type=float, default=0.75,
                   help="Cosine similarity threshold (default: 0.75)")
    p.add_argument("--gallery-ttl", type=float, default=30.0,
                   help="Seconds before an unseen person is evicted from gallery")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.endpoint, args.threshold, args.gallery_ttl)
