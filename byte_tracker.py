"""
ByteTracker implementation for DropMOT.

Based on:  ByteTrack — Zhang et al., ECCV 2022
           "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"

Dependencies: numpy, scipy only (no filterpy).
The Kalman filter is implemented from scratch using pure numpy so there
are no scipy/filterpy version conflicts.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Minimal Kalman filter (pure numpy)
#
# State vector (7-dim): [cx, cy, s, r, vx, vy, vs]
#   cx, cy  – bounding-box centre
#   s       – area (w * h)
#   r       – aspect ratio w/h  (constant)
#   vx, vy  – velocity of centre
#   vs      – rate of change of area
#
# Measurement vector (4-dim): [cx, cy, s, r]
# ---------------------------------------------------------------------------

class _KalmanFilter:
    """Constant-velocity Kalman filter for a bounding box."""

    def __init__(self):
        # State transition (constant velocity)
        self.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=float)

        # Measurement matrix (observe cx, cy, s, r)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=float)

        # Measurement noise
        self.R = np.diag([1., 1., 10., 10.])

        # Process noise
        self.Q = np.eye(7) * 0.01
        self.Q[4, 4] = 0.01
        self.Q[5, 5] = 0.01
        self.Q[6, 6] = 0.0001

        # State covariance – high initial uncertainty on velocity
        self.P = np.eye(7)
        self.P[4:, 4:] *= 1000.0
        self.P *= 10.0

        # State estimate
        self.x = np.zeros((7, 1))

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray):
        """z : (4,1) measurement vector"""
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)       # Kalman gain
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(7) - K @ self.H) @ self.P


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """[x1,y1,x2,y2] → (4,1) [cx, cy, s, r]"""
    w  = bbox[2] - bbox[0]
    h  = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s  = w * h
    r  = w / float(h) if h > 0 else 1.0
    return np.array([[cx], [cy], [s], [r]], dtype=float)


def _z_to_bbox(x: np.ndarray) -> np.ndarray:
    """(7,1) or (7,) state → [x1,y1,x2,y2]"""
    x = x.flatten()
    w = float(np.sqrt(abs(float(x[2]) * float(x[3]))))
    h = float(x[2]) / w if w > 0 else 0.0
    cx, cy = float(x[0]), float(x[1])
    return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])


# ---------------------------------------------------------------------------
# Single-object tracker
# ---------------------------------------------------------------------------

class KalmanBoxTracker:
    """Tracks one bounding box with the constant-velocity Kalman filter."""

    count = 0

    def __init__(self, bbox: np.ndarray, score: float = 1.0):
        self.kf = _KalmanFilter()
        self.kf.x[:4] = _bbox_to_z(bbox)

        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count

        self.score           = score
        self.hits            = 1
        self.hit_streak      = 1
        self.age             = 0
        self.time_since_update = 0
        self.state           = "tentative"

    def predict(self) -> np.ndarray:
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return _z_to_bbox(self.kf.x)

    def update(self, bbox: np.ndarray, score: float = 1.0):
        self.score             = score
        self.time_since_update = 0
        self.hits             += 1
        self.hit_streak       += 1
        self.kf.update(_bbox_to_z(bbox))
        if self.hits >= 3:
            self.state = "confirmed"

    def get_state(self) -> np.ndarray:
        return _z_to_bbox(self.kf.x)


# ---------------------------------------------------------------------------
# IoU + Hungarian matching
# ---------------------------------------------------------------------------

def _iou_batch(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    if len(bboxes_a) == 0 or len(bboxes_b) == 0:
        return np.zeros((len(bboxes_a), len(bboxes_b)))

    a = bboxes_a[:, None, :]
    b = bboxes_b[None, :, :]

    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])

    inter = np.maximum(0., ix2 - ix1) * np.maximum(0., iy2 - iy1)
    area_a = (bboxes_a[:, 2]-bboxes_a[:, 0]) * (bboxes_a[:, 3]-bboxes_a[:, 1])
    area_b = (bboxes_b[:, 2]-bboxes_b[:, 0]) * (bboxes_b[:, 3]-bboxes_b[:, 1])
    union  = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def _associate(trackers, detections, iou_threshold):
    if len(trackers) == 0:
        return np.empty((0,2),int), [], list(range(len(detections)))
    if len(detections) == 0:
        return np.empty((0,2),int), list(range(len(trackers))), []

    trk_boxes  = np.array([t.get_state() for t in trackers])
    iou_matrix = _iou_batch(trk_boxes, detections[:, :4])
    row, col   = linear_sum_assignment(1.0 - iou_matrix)

    matched, unmatched_trks, unmatched_dets = [], [], []
    matched_t, matched_d = set(), set()

    for t_idx, d_idx in zip(row, col):
        if iou_matrix[t_idx, d_idx] < iou_threshold:
            unmatched_trks.append(t_idx)
            unmatched_dets.append(d_idx)
        else:
            matched.append([t_idx, d_idx])
            matched_t.add(t_idx); matched_d.add(d_idx)

    for i in range(len(trackers)):
        if i not in matched_t: unmatched_trks.append(i)
    for i in range(len(detections)):
        if i not in matched_d: unmatched_dets.append(i)

    m = np.array(matched, dtype=int) if matched else np.empty((0,2), dtype=int)
    return m, unmatched_trks, unmatched_dets


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
    high_thresh : score threshold separating high / low detections
    low_thresh  : minimum score for a detection to enter Stage 2
    max_lost    : frames to keep a lost track before deletion
    min_hits    : frames before a new track is reported
    iou_thresh  : IoU threshold for track–detection association
    """

    def __init__(self, high_thresh=0.5, low_thresh=0.1,
                 max_lost=5, min_hits=3, iou_thresh=0.3):
        self.high_thresh = high_thresh
        self.low_thresh  = low_thresh
        self.max_lost    = max_lost
        self.min_hits    = min_hits
        self.iou_thresh  = iou_thresh

        self.trackers = []  # list of KalmanBoxTracker
        self.frame_count = 0
        KalmanBoxTracker.count = 0

    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        detections : (N,5) [x1,y1,x2,y2,score] or empty

        Returns
        -------
        (M,6) [x1,y1,x2,y2,track_id,score]  — confirmed tracks only
        """
        self.frame_count += 1
        if len(detections) == 0:
            detections = np.empty((0,5), dtype=float)

        for t in self.trackers:
            t.predict()

        high_mask = detections[:, 4] >= self.high_thresh
        low_mask  = (detections[:, 4] >= self.low_thresh) & ~high_mask
        dets_high = detections[high_mask]
        dets_low  = detections[low_mask]

        # Stage 1
        m1, unmatched_trks1, unmatched_dets_high = _associate(
            self.trackers, dets_high, self.iou_thresh)
        for t_idx, d_idx in m1:
            self.trackers[t_idx].update(dets_high[d_idx,:4], dets_high[d_idx,4])

        # Stage 2
        lost = [self.trackers[i] for i in unmatched_trks1]
        m2, still_lost, _ = _associate(lost, dets_low, self.iou_thresh)
        for t_idx, d_idx in m2:
            self.trackers[unmatched_trks1[t_idx]].update(
                dets_low[d_idx,:4], dets_low[d_idx,4])

        # New tracks for unmatched high-conf detections
        for d_idx in unmatched_dets_high:
            self.trackers.append(
                KalmanBoxTracker(dets_high[d_idx,:4], dets_high[d_idx,4]))

        # Prune stale tracks
        self.trackers = [t for t in self.trackers
                         if t.time_since_update <= self.max_lost]

        # Return confirmed tracks
        results = []
        for t in self.trackers:
            if t.hits >= self.min_hits or self.frame_count <= self.min_hits:
                results.append([*t.get_state(), t.id, t.score])

        return np.array(results, dtype=float) if results else np.empty((0,6), dtype=float)
