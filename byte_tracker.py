"""
ByteTracker implementation for DropMOT.

Based on:  ByteTrack — Zhang et al., ECCV 2022
           "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"

Key improvements over the original trackpy approach:
  - Online tracking (no need to buffer all frames before linking)
  - Kalman filter for motion prediction between frames
  - Two-stage association: high-confidence detections first,
    then low-confidence detections matched to lost tracks
  - IoU-based matching instead of pure centroid distance
  - Robust ID preservation through occlusion / temporary disappearance
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


# ---------------------------------------------------------------------------
# Kalman filter state: [cx, cy, s, r, vx, vy, vs]
#   cx, cy  – bounding-box centre
#   s       – area (w*h)
#   r       – aspect ratio w/h  (treated as constant)
#   vx, vy  – velocity of centre
#   vs      – rate of change of area
# ---------------------------------------------------------------------------

class KalmanBoxTracker:
    """Tracks a single bounding box using a constant-velocity Kalman filter."""

    count = 0  # class-level counter for unique IDs

    def __init__(self, bbox: np.ndarray, score: float = 1.0):
        """
        bbox : [x1, y1, x2, y2]
        score: detection confidence
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition matrix (constant velocity)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=float)

        # Measurement matrix (observe cx, cy, s, r)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=float)

        # Measurement noise – tighter for small droplets
        self.kf.R[2:, 2:] *= 10.0

        # Covariance – high initial uncertainty on velocity
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0

        # Process noise
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = _bbox_to_z(bbox)

        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count

        self.score = score
        self.hits = 1
        self.hit_streak = 1
        self.age = 0                     # frames since last successful update
        self.time_since_update = 0
        self.state = "tentative"         # tentative → confirmed → lost

    # ------------------------------------------------------------------
    def predict(self):
        """Advance the state estimate one step."""
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return _z_to_bbox(self.kf.x)

    def update(self, bbox: np.ndarray, score: float = 1.0):
        """Correct the state with a new measurement."""
        self.score = score
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(_bbox_to_z(bbox))
        if self.hits >= 3:
            self.state = "confirmed"

    def get_state(self) -> np.ndarray:
        """Return the current bounding-box estimate as [x1,y1,x2,y2]."""
        return _z_to_bbox(self.kf.x)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """[x1,y1,x2,y2] → [cx, cy, s, r]"""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h) if h > 0 else 1.0
    return np.array([[cx], [cy], [s], [r]], dtype=float)


def _z_to_bbox(x: np.ndarray, score: float | None = None) -> np.ndarray:
    """[cx, cy, s, r, ...] → [x1,y1,x2,y2]"""
    w = np.sqrt(abs(x[2] * x[3]))
    h = x[2] / w if w > 0 else 0
    return np.array([
        x[0] - w / 2.0,
        x[1] - h / 2.0,
        x[0] + w / 2.0,
        x[1] + h / 2.0,
    ]).flatten()


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------

def _iou_batch(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of boxes.
    bboxes_a : (N, 4)  [x1,y1,x2,y2]
    bboxes_b : (M, 4)
    Returns  : (N, M) IoU matrix
    """
    if len(bboxes_a) == 0 or len(bboxes_b) == 0:
        return np.zeros((len(bboxes_a), len(bboxes_b)), dtype=float)

    a = bboxes_a[:, None, :]   # (N,1,4)
    b = bboxes_b[None, :, :]   # (1,M,4)

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (bboxes_a[:, 2] - bboxes_a[:, 0]) * (bboxes_a[:, 3] - bboxes_a[:, 1])
    area_b = (bboxes_b[:, 2] - bboxes_b[:, 0]) * (bboxes_b[:, 3] - bboxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - inter_area
    return inter_area / np.maximum(union, 1e-6)


# ---------------------------------------------------------------------------
# Hungarian matching
# ---------------------------------------------------------------------------

def _linear_assignment(cost_matrix: np.ndarray):
    """Wrapper around scipy's linear_sum_assignment."""
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return np.stack([row_ind, col_ind], axis=1)


def _associate(
    trackers: list,
    detections: np.ndarray,
    iou_threshold: float = 0.3,
) -> tuple[np.ndarray, list[int], list[int]]:
    """
    Associate detections to existing trackers using IoU + Hungarian matching.

    Returns
    -------
    matches        : (K,2) array of [tracker_idx, det_idx]
    unmatched_trks : list of tracker indices with no assignment
    unmatched_dets : list of detection indices with no assignment
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), [], list(range(len(detections)))
    if len(detections) == 0:
        return np.empty((0, 2), dtype=int), list(range(len(trackers))), []

    trk_boxes = np.array([t.get_state() for t in trackers])
    iou_matrix = _iou_batch(trk_boxes, detections[:, :4])  # (T, D)

    cost = 1.0 - iou_matrix
    assignments = _linear_assignment(cost)

    matched, unmatched_trks, unmatched_dets = [], [], []

    matched_trks = set()
    matched_dets = set()

    for t_idx, d_idx in assignments:
        if iou_matrix[t_idx, d_idx] < iou_threshold:
            unmatched_trks.append(t_idx)
            unmatched_dets.append(d_idx)
        else:
            matched.append([t_idx, d_idx])
            matched_trks.add(t_idx)
            matched_dets.add(d_idx)

    for i in range(len(trackers)):
        if i not in matched_trks:
            unmatched_trks.append(i)
    for i in range(len(detections)):
        if i not in matched_dets:
            unmatched_dets.append(i)

    return np.array(matched, dtype=int) if matched else np.empty((0, 2), dtype=int), \
           unmatched_trks, unmatched_dets


# ---------------------------------------------------------------------------
# ByteTracker
# ---------------------------------------------------------------------------

class ByteTracker:
    """
    Online multi-object tracker following the ByteTrack paper.

    Two-stage association:
      Stage 1 – high-confidence detections  ↔  all active tracks  (IoU)
      Stage 2 – low-confidence detections   ↔  unmatched tracks   (IoU)

    Parameters
    ----------
    high_thresh   : Score threshold separating high/low detections.
    low_thresh    : Minimum score for a detection to enter Stage 2.
    max_lost      : Frames to keep a lost track before deletion.
    min_hits      : Frames a track must be seen before being reported.
    iou_thresh    : IoU threshold used in both association stages.
    """

    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        max_lost: int = 5,
        min_hits: int = 3,
        iou_thresh: float = 0.3,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.max_lost = max_lost
        self.min_hits = min_hits
        self.iou_thresh = iou_thresh

        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0   # reset IDs at construction

    # ------------------------------------------------------------------
    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        Run one tracking step.

        Parameters
        ----------
        detections : (N, 5) array  [x1, y1, x2, y2, score]
                     or (0,) / empty array when no detections.

        Returns
        -------
        results : (M, 6) array  [x1, y1, x2, y2, track_id, score]
                  Only confirmed tracks (seen ≥ min_hits times) are returned.
        """
        self.frame_count += 1

        if len(detections) == 0:
            detections = np.empty((0, 5), dtype=float)

        # ---- Predict all existing trackers one step forward ----
        for t in self.trackers:
            t.predict()

        # ---- Split detections by confidence ----
        high_mask = detections[:, 4] >= self.high_thresh
        low_mask  = (detections[:, 4] >= self.low_thresh) & ~high_mask

        dets_high = detections[high_mask]
        dets_low  = detections[low_mask]

        # ---- Stage 1: high-confidence dets  ↔  all active trackers ----
        matched1, unmatched_trks1, unmatched_dets_high = _associate(
            self.trackers, dets_high, self.iou_thresh
        )

        for t_idx, d_idx in matched1:
            self.trackers[t_idx].update(dets_high[d_idx, :4], dets_high[d_idx, 4])

        # ---- Stage 2: low-confidence dets  ↔  unmatched trackers ----
        lost_trackers = [self.trackers[i] for i in unmatched_trks1]
        matched2, still_unmatched_trks, _ = _associate(
            lost_trackers, dets_low, self.iou_thresh
        )

        for t_idx, d_idx in matched2:
            original_idx = unmatched_trks1[t_idx]
            self.trackers[original_idx].update(dets_low[d_idx, :4], dets_low[d_idx, 4])

        # Mark truly unmatched trackers
        final_unmatched = [unmatched_trks1[i] for i in still_unmatched_trks]
        for i in final_unmatched:
            # mark as lost but don't delete yet
            pass  # time_since_update already incremented in predict()

        # ---- Create new trackers for unmatched high-conf detections ----
        for d_idx in unmatched_dets_high:
            self.trackers.append(
                KalmanBoxTracker(dets_high[d_idx, :4], dets_high[d_idx, 4])
            )

        # ---- Remove stale tracks ----
        self.trackers = [
            t for t in self.trackers
            if t.time_since_update <= self.max_lost
        ]

        # ---- Collect results (confirmed tracks only) ----
        results = []
        for t in self.trackers:
            if t.hits >= self.min_hits or self.frame_count <= self.min_hits:
                bbox = t.get_state()
                results.append([*bbox, t.id, t.score])

        return np.array(results, dtype=float) if results else np.empty((0, 6), dtype=float)
