"""
Loitering Detection Service (Phase 4).

Subscribes to the ZMQ frame event stream, tests each tracked person's
centroid against configured zone polygons, and emits a loitering_alert
when a person has been inside a zone longer than the configured threshold.

No ReID required — works entirely on per-camera tracking IDs (tid).

Output events (one JSON line to stdout per alert):
  {"event":"loitering_alert","ts_ms":1713000000000,
   "source_id":0,"tid":7,"zone":"entrance","dwell_s":47.2,
   "bbox":{"l":312.0,"t":88.0,"w":96.0,"h":260.0}}

Periodic status (stderr, controlled by --status-interval):
  [loiter] cam0 entrance: tid=7 dwell=47.2s  ← active dwellers

Usage:
    python3 loitering_detector.py [--zones zones.yaml]
                                  [--endpoint tcp://localhost:5555]
                                  [--alert-cooldown 60]
                                  [--stale-ttl 5]
                                  [--status-interval 30]
"""

import argparse
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
import zmq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [loiter] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PERSON_CLASS = 0


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def point_in_polygon(px: float, py: float, polygon: List[Tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def bbox_centroid(bbox: dict) -> Tuple[float, float]:
    return bbox["l"] + bbox["w"] / 2.0, bbox["t"] + bbox["h"] / 2.0


# ---------------------------------------------------------------------------
# Zone data
# ---------------------------------------------------------------------------

@dataclass
class Zone:
    name: str
    polygon: List[Tuple[float, float]]
    dwell_threshold_s: float


# ---------------------------------------------------------------------------
# Per-track, per-zone dwell state
# ---------------------------------------------------------------------------

@dataclass
class DwellState:
    entry_time: float               # monotonic clock when person entered zone
    last_alert_time: float = 0.0   # last time an alert was emitted (for cooldown)
    last_seen: float = field(default_factory=time.monotonic)  # stale-track cleanup


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class LoiteringDetector:
    """
    Maintains dwell state for every (source_id, tid, zone) triple.

    Call process(event) with each parsed frame event dict.
    Returns a list of alert dicts to emit (may be empty).
    """

    def __init__(
        self,
        zones_path: str,
        alert_cooldown_s: float = 60.0,
        stale_ttl_s: float = 5.0,
    ):
        self.alert_cooldown_s = alert_cooldown_s
        self.stale_ttl_s = stale_ttl_s

        # source_id → list of Zone
        self.zones: Dict[int, List[Zone]] = {}
        self._load_zones(zones_path)

        # (source_id, tid, zone_name) → DwellState
        self._states: Dict[Tuple[int, int, str], DwellState] = {}
        self._last_cleanup = time.monotonic()

    def _load_zones(self, path: str) -> None:
        with open(path) as f:
            cfg = yaml.safe_load(f)

        for cam in cfg.get("cameras", []):
            src = int(cam["source_id"])
            zones = []
            for z in cam.get("zones", []):
                polygon = [tuple(pt) for pt in z["polygon"]]
                zones.append(Zone(
                    name=z["name"],
                    polygon=polygon,
                    dwell_threshold_s=float(z["dwell_threshold_s"]),
                ))
            self.zones[src] = zones
            log.info("Loaded %d zone(s) for source_id=%d", len(zones), src)

    def process(self, event: dict) -> List[dict]:
        if event.get("event") != "frame":
            return []

        source_id: int = event["source_id"]
        ts_ms: int = event["ts_ms"]
        now = time.monotonic()

        cam_zones = self.zones.get(source_id, [])
        alerts = []

        for det in event.get("detections", []):
            if det.get("class") != PERSON_CLASS:
                continue
            tid = det.get("tid", -1)
            if tid < 0:
                continue

            cx, cy = bbox_centroid(det["bbox"])

            for zone in cam_zones:
                key = (source_id, tid, zone.name)
                inside = point_in_polygon(cx, cy, zone.polygon)

                if inside:
                    if key not in self._states:
                        self._states[key] = DwellState(entry_time=now)
                        log.debug("tid=%d entered zone '%s' on cam%d", tid, zone.name, source_id)

                    state = self._states[key]
                    state.last_seen = now
                    dwell_s = now - state.entry_time

                    if (dwell_s >= zone.dwell_threshold_s and
                            now - state.last_alert_time >= self.alert_cooldown_s):
                        state.last_alert_time = now
                        alert = {
                            "event":     "loitering_alert",
                            "ts_ms":     ts_ms,
                            "source_id": source_id,
                            "tid":       tid,
                            "zone":      zone.name,
                            "dwell_s":   round(dwell_s, 1),
                            "bbox":      det["bbox"],
                        }
                        alerts.append(alert)
                        log.warning(
                            "LOITERING cam%d zone='%s' tid=%d dwell=%.1fs",
                            source_id, zone.name, tid, dwell_s,
                        )

                else:
                    # Person left the zone — reset their dwell timer
                    if key in self._states:
                        dwell_s = now - self._states[key].entry_time
                        log.debug(
                            "tid=%d left zone '%s' on cam%d after %.1fs",
                            tid, zone.name, source_id, dwell_s,
                        )
                        del self._states[key]

        # Periodic stale-track cleanup (tracks that disappeared without a zone-exit event)
        if now - self._last_cleanup > 2.0:
            self._cleanup_stale(now)
            self._last_cleanup = now

        return alerts

    def _cleanup_stale(self, now: float) -> None:
        stale = [k for k, s in self._states.items() if now - s.last_seen > self.stale_ttl_s]
        for k in stale:
            src, tid, zone = k
            log.debug("Evicting stale track tid=%d zone='%s' cam%d", tid, zone, src)
            del self._states[k]

    def active_dwellers(self) -> Dict[Tuple[int, int, str], float]:
        """Return {(source_id, tid, zone): dwell_s} for all currently tracked entries."""
        now = time.monotonic()
        return {k: round(now - s.entry_time, 1) for k, s in self._states.items()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Loitering detection service")
    p.add_argument("--zones",            default="zones.yaml",          help="Zone config YAML")
    p.add_argument("--endpoint",         default="tcp://localhost:5555", help="ZMQ PUB endpoint")
    p.add_argument("--alert-cooldown",   type=float, default=60.0,      help="Seconds between repeat alerts for same person+zone")
    p.add_argument("--stale-ttl",        type=float, default=5.0,       help="Seconds before a gone track's zone state is evicted")
    p.add_argument("--status-interval",  type=int,   default=30,        help="Print active dweller table every N seconds (0=off)")
    return p.parse_args()


def main():
    args = parse_args()

    zones_path = Path(args.zones)
    if not zones_path.exists():
        log.error("Zone config not found: %s", zones_path)
        sys.exit(1)

    detector = LoiteringDetector(
        zones_path=str(zones_path),
        alert_cooldown_s=args.alert_cooldown,
        stale_ttl_s=args.stale_ttl,
    )

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(args.endpoint)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    log.info("Connected to %s", args.endpoint)

    running = True
    def _stop(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    last_status = time.monotonic()
    frame_count = 0

    while running:
        try:
            raw = sub.recv_string(flags=zmq.NOBLOCK)
        except zmq.Again:
            time.sleep(0.001)
            continue
        except zmq.ZMQError:
            break

        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue

        frame_count += 1
        alerts = detector.process(event)

        for alert in alerts:
            print(json.dumps(alert), flush=True)

        # Periodic status to stderr
        if args.status_interval > 0:
            now = time.monotonic()
            if now - last_status >= args.status_interval:
                last_status = now
                dwellers = detector.active_dwellers()
                if dwellers:
                    log.info("Active dwellers (%d frames processed):", frame_count)
                    for (src, tid, zone), dwell_s in sorted(dwellers.items()):
                        log.info("  cam%d %-20s tid=%-5d dwell=%.1fs", src, zone, tid, dwell_s)
                else:
                    log.info("No active dwellers (%d frames processed)", frame_count)

    log.info("Shutting down")
    sub.close()
    ctx.term()


if __name__ == "__main__":
    main()
