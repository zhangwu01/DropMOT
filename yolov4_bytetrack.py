"""
DropMOT — YOLOv4 + ByteTracker
================================
Replaces the original trackpy-based two-pass pipeline with an *online*
ByteTracker.  Every frame is detected and tracked in a single forward pass;
no buffering of all frames is required.

Key improvements over yolov4_trackpy.py
-----------------------------------------
1. **Online tracking** – detections are linked frame-by-frame using a
   Kalman filter, so the script can handle arbitrarily long videos without
   accumulating a large in-memory DataFrame.

2. **Two-stage association (ByteTrack)** – high-confidence detections are
   matched first with all active tracks via IoU; low-confidence detections
   then get a second chance with unmatched tracks.  This dramatically
   reduces ID switches when droplets are partially occluded or close
   together.

3. **Kalman-filter motion prediction** – predicts where each droplet will
   be in the next frame, enabling correct matching even across fast motion
   or short disappearances.

4. **IoU-based matching** – uses bounding-box overlap instead of centroid
   distance, so matching is invariant to droplet size variation.

5. **Track confirmation** – newly created tracks are marked "tentative"
   for the first N frames to suppress spurious one-frame detections.

Usage
-----
    python yolov4_bytetrack.py [--video PATH] [--conf FLOAT] [--low FLOAT]
                               [--max-lost INT] [--min-hits INT]
                               [--iou FLOAT] [--nms FLOAT] [--no-trails]

All arguments are optional; sensible defaults are provided.
"""

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from collections import deque

from byte_tracker import ByteTracker


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DropMOT — YOLOv4 + ByteTracker")

    p.add_argument("--video",    default="input/StorageChamber_short.mp4",
                   help="Path to input video")
    p.add_argument("--weights",  default="yolov4_1ob_best.weights")
    p.add_argument("--config",   default="yolov4_1ob.cfg")
    p.add_argument("--output",   default=None,
                   help="Output path (auto-generated if omitted)")

    # Detection thresholds
    p.add_argument("--conf",     type=float, default=0.5,
                   help="High-confidence detection threshold (ByteTrack stage 1)")
    p.add_argument("--low",      type=float, default=0.1,
                   help="Low-confidence threshold (ByteTrack stage 2)")
    p.add_argument("--nms",      type=float, default=0.4,
                   help="NMS IoU threshold for YOLOv4")

    # Tracker parameters
    p.add_argument("--max-lost", type=int,   default=5,
                   help="Frames to keep a lost track before deleting it")
    p.add_argument("--min-hits", type=int,   default=3,
                   help="Frames before a new track is reported (confirmation)")
    p.add_argument("--iou",      type=float, default=0.3,
                   help="IoU threshold for track–detection association")

    # Visualisation
    p.add_argument("--no-trails", action="store_true",
                   help="Disable trajectory trails")
    p.add_argument("--trail-len", type=int,   default=30,
                   help="Number of past positions shown as a trail")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Colour palette (HSV → BGR, one colour per track ID)
# ---------------------------------------------------------------------------

def _id_to_colour(track_id: int):
    hue = (track_id * 37) % 180           # spread IDs across the hue wheel
    hsv = np.uint8([[[hue, 220, 220]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# ---------------------------------------------------------------------------
# Draw helpers
# ---------------------------------------------------------------------------

def _draw_track(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    track_id: int,
    score: float,
    trail: deque,
) -> None:
    colour = _id_to_colour(track_id)

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

    # Label (ID + confidence)
    label = f"#{track_id}  {score:.2f}"
    font        = cv2.FONT_HERSHEY_SIMPLEX
    font_scale  = 0.35
    thickness   = 1
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
    lx = max(x1, 2)
    ly = max(y1 - 4, th + 2)
    cv2.rectangle(frame, (lx, ly - th - 2), (lx + tw + 2, ly + 2), colour, -1)
    cv2.putText(frame, label, (lx + 1, ly), font, font_scale, (0, 0, 0), thickness)

    # Trajectory trail
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    trail.append((cx, cy))

    for i in range(1, len(trail)):
        alpha = i / len(trail)                  # fade older points
        c = tuple(int(v * alpha) for v in colour)
        cv2.line(frame, trail[i - 1], trail[i], c, 1)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- Output path ----
    if args.output is None:
        base = os.path.splitext(os.path.basename(args.video))[0]
        args.output = os.path.join(
            "output",
            f"{base}_bytetrack_conf{args.conf}_lost{args.max_lost}.mp4",
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # ---- Load YOLOv4 ----
    print("Loading YOLOv4 …")
    net   = cv2.dnn.readNet(args.weights, args.config)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(640, 512), scale=1 / 255, swapRB=True)

    # ---- Video I/O ----
    cap    = cv2.VideoCapture(args.video)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # ---- Tracker ----
    tracker = ByteTracker(
        high_thresh = args.conf,
        low_thresh  = args.low,
        max_lost    = args.max_lost,
        min_hits    = args.min_hits,
        iou_thresh  = args.iou,
    )

    # Trajectory buffers: track_id → deque of (cx,cy)
    trails = {}  # track_id -> deque of (cx, cy)

    # ---- Stats ----
    total_dets = 0
    total_trks = 0
    id_set = set()

    print(f"\nTracking {args.video}  →  {args.output}")
    print(f"  high_thresh={args.conf}  low_thresh={args.low}  "
          f"max_lost={args.max_lost}  min_hits={args.min_hits}  iou={args.iou}\n")

    with tqdm(total=total, unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ---- Detect ----
            classes, scores, boxes = model.detect(frame, args.low, args.nms)

            # Build (N,5) detections array [x1,y1,x2,y2,score]
            dets = []
            for score, box in zip(scores, boxes):
                x, y, w, h = box
                dets.append([x, y, x + w, y + h, float(score)])
            dets_arr = np.array(dets, dtype=float) if dets else np.empty((0, 5))

            total_dets += len(dets_arr)

            # ---- Track ----
            # results: (M,6)  [x1,y1,x2,y2,track_id,score]
            results = tracker.update(dets_arr)
            total_trks += len(results)

            # ---- Draw ----
            for row in results:
                x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                track_id        = int(row[4])
                score           = float(row[5])

                # Clip to frame bounds
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(width - 1, x2); y2 = min(height - 1, y2)

                id_set.add(track_id)

                if not args.no_trails:
                    if track_id not in trails:
                        trails[track_id] = deque(maxlen=args.trail_len)
                    trail = trails[track_id]
                else:
                    trail = deque(maxlen=1)

                _draw_track(frame, x1, y1, x2, y2, track_id, score, trail)

            # Frame counter overlay
            cv2.putText(
                frame,
                f"Frame {tracker.frame_count}  |  Active tracks: {len(results)}",
                (8, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
            )

            out.write(frame)
            pbar.update(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # ---- Summary ----
    print(f"\n{'─'*55}")
    print(f"  Output           : {args.output}")
    print(f"  Total frames     : {tracker.frame_count}")
    print(f"  Total detections : {total_dets}")
    print(f"  Unique track IDs : {len(id_set)}")
    print(f"  Avg tracks/frame : {total_trks / max(tracker.frame_count, 1):.1f}")
    print(f"{'─'*55}\n")


if __name__ == "__main__":
    main()
