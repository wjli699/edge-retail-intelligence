"""
Zone Calibrator — interactive polygon picker for loitering zone setup (Phase 4).

Lets you define zone polygons by clicking on a camera frame, then prints
the resulting zones.yaml snippet ready to paste into your config.

Requires OpenCV (pre-installed on Jetson via JetPack, or: pip3 install opencv-python).

Input — one of:
  --image   path to a JPEG/PNG screenshot (e.g. captured from ffplay or OSD video)
  --rtsp    RTSP URL to grab one frame from the live camera

Controls:
  Left-click       Add a polygon vertex
  Right-click      Close the current polygon (connect last vertex to first)
  'n'              Name the current zone and start a new one
  'z'              Undo last vertex
  's'              Save — print zones.yaml snippet to stdout and exit
  'q' / Esc        Quit without saving

Usage:
    # From a saved screenshot:
    python3 zone_calibrator.py --image /tmp/frame.png --source-id 0

    # From a live RTSP stream (grabs one frame):
    python3 zone_calibrator.py --rtsp rtsp://192.168.9.146:18554/stream1 --source-id 0

    # Specify dwell threshold for all zones defined in this session:
    python3 zone_calibrator.py --image /tmp/frame.png --source-id 0 --dwell 45
"""

import argparse
import sys
from typing import List, Optional, Tuple

try:
    import cv2
    import numpy as np
except ImportError:
    print("OpenCV not found. Install with: pip3 install opencv-python", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

# Palette: one colour per zone (BGR for OpenCV)
_COLORS = [
    (0,  200, 255),   # orange
    (0,  255,   0),   # green
    (255,  80,  80),  # blue
    (0,  255, 255),   # yellow
    (200,   0, 255),  # purple
    (0,  200, 200),   # olive
    (255, 200,   0),  # cyan
    (180,   0, 180),  # magenta
]


def _zone_color(idx: int) -> Tuple[int, int, int]:
    return _COLORS[idx % len(_COLORS)]


def _draw_state(
    base: np.ndarray,
    finished_zones: List[dict],
    current_poly: List[Tuple[int, int]],
    zone_idx: int,
) -> np.ndarray:
    img = base.copy()
    h, w = img.shape[:2]

    # Draw finished zones
    for i, zone in enumerate(finished_zones):
        col = _zone_color(i)
        pts = np.array(zone["polygon"], dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], col)
        cv2.addWeighted(overlay, 0.20, img, 0.80, 0, img)
        cv2.polylines(img, [pts], isClosed=True, color=col, thickness=2)
        # Label at centroid
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        cv2.putText(img, zone["name"], (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Draw in-progress polygon
    if current_poly:
        col = _zone_color(zone_idx)
        for pt in current_poly:
            cv2.circle(img, pt, 5, col, -1)
        for a, b in zip(current_poly, current_poly[1:]):
            cv2.line(img, a, b, col, 2)
        if len(current_poly) >= 2:
            cv2.line(img, current_poly[-1], current_poly[0], col, 1)  # preview close

    # HUD
    lines = [
        "Left-click: add vertex | Right-click: close polygon",
        "n: name zone | z: undo vertex | s: save | q: quit",
        f"Zone {zone_idx + 1} in progress ({len(current_poly)} vertices)",
    ]
    for li, text in enumerate(lines):
        cv2.putText(img, text, (10, h - 10 - (len(lines) - 1 - li) * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1,
                    cv2.LINE_AA)

    return img


# ---------------------------------------------------------------------------
# Interaction
# ---------------------------------------------------------------------------

class Calibrator:
    def __init__(self, image: np.ndarray, source_id: int, default_dwell: float):
        self.base = image
        self.source_id = source_id
        self.default_dwell = default_dwell

        self.finished_zones: List[dict] = []
        self.current_poly: List[Tuple[int, int]] = []
        self.closed = False   # current polygon is closed, ready to name
        self._dirty = True

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_poly.append((x, y))
            self._dirty = True

        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.current_poly) >= 3:
                self.closed = True
                self._dirty = True

    def run(self) -> Optional[List[dict]]:
        win = "Zone Calibrator"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        # Verify the window was actually created (fails silently on headless displays)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 0:
            print("ERROR: OpenCV could not open a display window.", file=sys.stderr)
            print("  Check that DISPLAY is set and a GUI session is active.", file=sys.stderr)
            print("  Alternatively, save a frame with ffplay and use --image instead.", file=sys.stderr)
            return None
        cv2.setMouseCallback(win, self.mouse_callback)

        while True:
            if self._dirty:
                frame = _draw_state(
                    self.base, self.finished_zones,
                    self.current_poly, len(self.finished_zones),
                )
                cv2.imshow(win, frame)
                self._dirty = False

            key = cv2.waitKey(30) & 0xFF

            if key in (ord('q'), 27):  # q or Esc
                cv2.destroyAllWindows()
                return None

            elif key == ord('z'):  # undo last vertex
                if self.current_poly:
                    self.current_poly.pop()
                    self.closed = False
                    self._dirty = True

            elif key == ord('n') or (key == ord('n') and self.closed):
                self._finish_current_zone()

            elif key == ord('s'):
                if self.current_poly and len(self.current_poly) >= 3:
                    self._finish_current_zone()
                cv2.destroyAllWindows()
                return self.finished_zones if self.finished_zones else None

            # Auto-prompt to name after right-click close
            if self.closed:
                self._finish_current_zone()

        cv2.destroyAllWindows()
        return None

    def _finish_current_zone(self):
        if len(self.current_poly) < 3:
            print("[calibrator] Need at least 3 vertices to define a zone.", file=sys.stderr)
            self.closed = False
            return

        name = input(f"\nZone name (zone {len(self.finished_zones) + 1}): ").strip()
        if not name:
            name = f"zone-{len(self.finished_zones) + 1}"

        dwell_input = input(f"Dwell threshold in seconds [{self.default_dwell}]: ").strip()
        dwell = float(dwell_input) if dwell_input else self.default_dwell

        self.finished_zones.append({
            "name":              name,
            "dwell_threshold_s": dwell,
            "polygon":           list(self.current_poly),
        })
        print(f"[calibrator] Zone '{name}' saved ({len(self.current_poly)} vertices).",
              file=sys.stderr)

        self.current_poly = []
        self.closed = False
        self._dirty = True


# ---------------------------------------------------------------------------
# YAML output
# ---------------------------------------------------------------------------

def print_yaml(source_id: int, zones: List[dict]) -> None:
    print("\n# Paste this into zones.yaml under 'cameras:'")
    print(f"  - source_id: {source_id}")
    print(f"    zones:")
    for zone in zones:
        print(f"      - name: \"{zone['name']}\"")
        print(f"        dwell_threshold_s: {zone['dwell_threshold_s']}")
        print(f"        polygon:")
        for pt in zone["polygon"]:
            print(f"          - [{pt[0]}, {pt[1]}]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def grab_rtsp_frame(url: str, save_path: Optional[str] = None) -> np.ndarray:
    print(f"Connecting to {url} ...", file=sys.stderr)
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"Failed to open RTSP stream: {url}", file=sys.stderr)
        sys.exit(1)
    for _ in range(10):  # skip buffered frames
        cap.grab()
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("Failed to grab frame from RTSP stream.", file=sys.stderr)
        sys.exit(1)
    if save_path:
        cv2.imwrite(save_path, frame)
        print(f"Frame saved to {save_path}", file=sys.stderr)
    return frame


def main():
    p = argparse.ArgumentParser(description="Interactive zone polygon calibrator")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="Path to a camera screenshot (JPEG/PNG)")
    src.add_argument("--rtsp",  help="RTSP URL to grab one frame from")
    p.add_argument("--source-id",  type=int,   required=True,        help="Camera source_id (0-based)")
    p.add_argument("--dwell",      type=float, default=30.0,         help="Default dwell threshold (seconds)")
    p.add_argument("--save-frame", metavar="PATH", default=None,     help="Save grabbed RTSP frame to this path and exit (no GUI)")
    args = p.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Cannot read image: {args.image}", file=sys.stderr)
            sys.exit(1)
    else:
        frame = grab_rtsp_frame(args.rtsp, save_path=args.save_frame)
        if args.save_frame:
            print(f"Re-run with:  python3 zone_calibrator.py --image {args.save_frame} --source-id {args.source_id}")
            sys.exit(0)

    print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}", file=sys.stderr)
    print("Define zones by clicking vertices. Right-click to close a polygon.", file=sys.stderr)

    cal = Calibrator(frame, source_id=args.source_id, default_dwell=args.dwell)
    zones = cal.run()

    if zones:
        print_yaml(args.source_id, zones)
    else:
        print("No zones saved.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
