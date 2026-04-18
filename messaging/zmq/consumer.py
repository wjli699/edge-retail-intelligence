"""
ZeroMQ SUB consumer for the edge retail intelligence event stream.

Connects to the C++ core engine's PUB socket and receives per-frame
detection events as newline-delimited JSON.

Usage:
    python consumer.py [--endpoint tcp://localhost:5555] [--filter person]

Phase 2: prints raw events.
Phase 5: this module will be imported by the FastAPI service for
         multi-camera correlation, loitering logic, and event routing.
"""

import argparse
import json
import logging
import signal
import sys

import zmq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [consumer] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Edge retail ZMQ event consumer")
    p.add_argument(
        "--endpoint",
        default="tcp://localhost:5555",
        help="ZMQ endpoint to connect to (default: tcp://localhost:5555)",
    )
    p.add_argument(
        "--filter",
        default="",
        help="Only print frames containing this label (e.g. 'person'). "
             "Empty string = all events.",
    )
    p.add_argument(
        "--stats-every",
        type=int,
        default=100,
        metavar="N",
        help="Log a throughput summary every N frames (0 = off)",
    )
    return p.parse_args()


def on_event(event: dict, label_filter: str) -> bool:
    """Return True if the event passes the filter and should be printed."""
    if not label_filter:
        return True
    return any(d.get("label") == label_filter for d in event.get("detections", []))


def run(endpoint: str, label_filter: str, stats_every: int) -> None:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")   # subscribe to all topics
    sock.setsockopt(zmq.RCVTIMEO, 1000)   # 1 s timeout so SIGINT is responsive
    sock.connect(endpoint)

    log.info("Connected to %s — waiting for events (filter=%r) ...", endpoint, label_filter or "all")

    # Graceful shutdown on Ctrl-C
    running = True
    def _stop(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    frame_count = 0
    person_count = 0

    while running:
        try:
            raw = sock.recv_string()
        except zmq.Again:
            continue  # timeout — check running flag

        try:
            event = json.loads(raw)
        except json.JSONDecodeError as exc:
            log.warning("Malformed JSON: %s", exc)
            continue

        frame_count += 1
        detections = event.get("detections", [])
        persons = [d for d in detections if d.get("label") == "person"]
        person_count += len(persons)

        if on_event(event, label_filter):
            src = event.get("source_id", "?")
            frame = event.get("frame", "?")
            ts = event.get("ts_ms", 0)
            det_summary = ", ".join(
                f"{d['label']}(tid={d.get('tid','?')} conf={d['conf']:.2f})"
                for d in detections
            )
            print(f"[src={src} frame={frame} ts={ts}] {det_summary}", flush=True)

        if stats_every and frame_count % stats_every == 0:
            log.info(
                "frames=%d  total_persons_detected=%d",
                frame_count, person_count,
            )

    log.info("Shutting down — frames received: %d", frame_count)
    sock.close()
    ctx.term()


if __name__ == "__main__":
    args = parse_args()
    run(args.endpoint, args.filter, args.stats_every)
