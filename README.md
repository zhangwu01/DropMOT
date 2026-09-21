# DropMOT
**Multi-object detection and tracking for microfluidic droplet videos using pre-trained YOLOv4 deep learning models and tracking algorithms**

**This repository will continue to be updated as new methods, improvements, and bug fixes for droplet detection and tracking are developed.**

---

## Tracking backends

| Script | Tracker | Architecture | Pros |
|---|---|---|---|
| `yolov4_trackpy.py` | TrackPy | Two-pass (offline) | Simple setup |
| `yolov4_bytetrack.py` ✨ | **ByteTracker** | Online (frame-by-frame) | Better ID stability, handles occlusion, Kalman prediction |

### Why ByteTracker?

The original `trackpy` pipeline buffers **all** detections before linking them, which is memory-intensive and prevents real-time use.  ByteTracker improves on this in three key ways:

1. **Online tracking** – processes one frame at a time with no buffering.
2. **Kalman filter** – predicts each droplet's position in the next frame, enabling robust matching even through fast motion or brief disappearance.
3. **Two-stage association** – high-confidence detections are matched first; low-confidence detections get a second chance with unmatched tracks, dramatically reducing ID switches during occlusions.

---

## Setup

### Step 1 – Download YOLOv4 weights
```
https://zenodo.org/records/10938306   →  yolov4_1ob_best.weights
```
Place the weights file in the repo root.

### Step 2 – Create environment
```bash
conda env create -f environment.yml
conda activate yolo
```

### Step 3 – Run tracking

**ByteTracker (recommended)**
```bash
python yolov4_bytetrack.py
```

With custom settings:
```bash
python yolov4_bytetrack.py \
  --video input/StorageChamber_short.mp4 \
  --conf 0.5 \       # high-confidence threshold (stage 1)
  --low  0.1 \       # low-confidence threshold  (stage 2)
  --max-lost 5 \     # frames before a lost track is deleted
  --min-hits 3 \     # frames before a new track is confirmed
  --iou  0.3 \       # IoU threshold for matching
  --trail-len 30     # length of trajectory trail drawn on video
```

**Original TrackPy (legacy)**
```bash
python yolov4_trackpy.py
```

### Step 4 – Deactivate environment
```bash
conda deactivate
```

---

## ByteTracker parameters

| Argument | Default | Description |
|---|---|---|
| `--conf` | 0.5 | High-confidence threshold; detections above this are used in stage-1 matching |
| `--low` | 0.1 | Low-confidence threshold; detections between `low` and `conf` get a second-chance match |
| `--max-lost` | 5 | Frames a track is kept alive without a detection before being deleted |
| `--min-hits` | 3 | Frames a new track must appear before it is drawn (suppresses false positives) |
| `--iou` | 0.3 | Minimum IoU for a track–detection match to be accepted |
| `--trail-len` | 30 | Number of past positions drawn as a trajectory trail |
| `--no-trails` | off | Pass this flag to disable trajectory trails |

---

## File overview
```
DropMOT/
├── yolov4_bytetrack.py   # ✨ main pipeline (YOLOv4 + ByteTracker)
├── byte_tracker.py       # ByteTracker + KalmanBoxTracker implementation
├── yolov4_trackpy.py     # original TrackPy pipeline (kept for reference)
├── yolov4_1ob.cfg        # YOLOv4 architecture config
├── obj.names / obj.data  # class labels
├── environment.yml       # conda environment spec
├── input/                # place input videos here
└── output/               # tracked output videos appear here
```
