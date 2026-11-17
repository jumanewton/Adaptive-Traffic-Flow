"""Tracker interface and ByteTrack implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import numpy as np
from filterpy.kalman import KalmanFilter

from traffic_config.settings import TrackingSettings, get_settings
from traffic_domain.schemas import Detection, Track, TrackStatus, BoundingBox, Point
from traffic_observability.logging import get_logger
from traffic_observability.metrics import get_metrics_registry


logger = get_logger(__name__)


@dataclass
class KalmanState:
    """Kalman filter state for a track."""
    kf: KalmanFilter
    track_id: Any
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    status: TrackStatus = TrackStatus.TENTATIVE
    bbox: BoundingBox | None = None
    class_id: int = 0
    class_name: str = ""

    def predict(self) -> BoundingBox:
        """Predict next state."""
        self.kf.predict()
        x = self.kf.x
        return BoundingBox(
            x1=float(x[0]),
            y1=float(x[1]),
            x2=float(x[2]),
            y2=float(x[3]),
        )

    def update(self, bbox: BoundingBox) -> None:
        """Update with measurement."""
        z = np.array([bbox.x1, bbox.y1, bbox.x2, bbox.y2], dtype=np.float32)
        self.kf.update(z)
        self.hits += 1
        self.time_since_update = 0
        self.bbox = bbox

    @property
    def world_position(self) -> Point | None:
        if self.bbox:
            return self.bbox.center
        return None


class Tracker(ABC):
    """Abstract tracker interface."""

    @abstractmethod
    def update(self, detections: list[Detection]) -> list[Track]:
        """Update tracker with new detections.

        Args:
            detections: List of detections for current frame

        Returns:
            List of active tracks
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset tracker state."""
        pass


def iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Calculate IoU between two bounding boxes."""
    x1 = max(box1.x1, box2.x1)
    y1 = max(box1.y1, box2.y1)
    x2 = min(box1.x2, box2.x2)
    y2 = min(box1.y2, box2.y2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = box1.area
    area2 = box2.area
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def iou_matrix(detections: list[Detection], tracks: list[KalmanState]) -> np.ndarray:
    """Compute IoU matrix between detections and tracks."""
    if not detections or not tracks:
        return np.empty((len(detections), len(tracks)))

    matrix = np.zeros((len(detections), len(tracks)), dtype=np.float32)
    for i, det in enumerate(detections):
        for j, track in enumerate(tracks):
            if track.bbox:
                matrix[i, j] = iou(det.bbox, track.bbox)
    return matrix


class ByteTracker(Tracker):
    """ByteTrack implementation for multi-object tracking.

    Based on: "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"
    (Zhang et al., 2021)
    """

    def __init__(self, settings: TrackingSettings | None = None):
        """Initialize ByteTracker."""
        self._settings = settings or get_settings().tracking
        self._tracks: dict[Any, KalmanState] = {}
        self._next_id = 1
        self._frame_count = 0
        self._metrics = get_metrics_registry()

    def _create_kalman_filter(self) -> KalmanFilter:
        """Create a Kalman filter for tracking (constant velocity model)."""
        # State: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
        # Measurement: [x1, y1, x2, y2]
        kf = KalmanFilter(dim_x=8, dim_z=4)

        # State transition matrix (constant velocity)
        dt = 1.0
        kf.F = np.array([
            [1, 0, 0, 0, dt, 0, 0, 0],
            [0, 1, 0, 0, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0, dt, 0],
            [0, 0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)

        # Measurement matrix
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=np.float32)

        # Measurement noise
        kf.R *= 10.0

        # Process noise
        kf.Q *= 0.01

        # Initial covariance
        kf.P *= 100.0

        return kf

    def _init_kalman(self, bbox: BoundingBox) -> KalmanFilter:
        """Initialize Kalman filter with bbox."""
        kf = self._create_kalman_filter()
        kf.x = np.array([
            bbox.x1, bbox.y1, bbox.x2, bbox.y2,
            0, 0, 0, 0  # Initial velocities
        ], dtype=np.float32).reshape(-1, 1)
        return kf

    def update(self, detections: list[Detection]) -> list[Track]:
        """Update tracker with new detections."""
        self._frame_count += 1

        # Separate high and low confidence detections
        high_conf = [d for d in detections if d.confidence >= self._settings.track_thresh]
        low_conf = [d for d in detections if d.confidence < self._settings.track_thresh]

        # Get predicted positions for existing tracks
        active_tracks = [t for t in self._tracks.values() if t.status != TrackStatus.REMOVED]

        # First association: high confidence detections with active tracks
        matched_dets, matched_trks, unmatched_dets, unmatched_trks = self._associate(
            high_conf, active_tracks, self._settings.match_thresh
        )

        # Update matched tracks
        for det_idx, trk_idx in zip(matched_dets, matched_trks):
            track = active_tracks[trk_idx]
            det = high_conf[det_idx]
            track.update(det.bbox)
            track.class_id = det.class_id
            track.class_name = det.class_name
            track.status = TrackStatus.CONFIRMED if track.hits >= self._settings.min_hits else TrackStatus.TENTATIVE

        # Second association: low confidence detections with unmatched tracks
        if low_conf and unmatched_trks:
            unmatched_track_objs = [active_tracks[i] for i in unmatched_trks]
            matched_dets2, matched_trks2, _, _ = self._associate(
                low_conf, unmatched_track_objs, self._settings.match_thresh
            )

            for det_idx, trk_idx in zip(matched_dets2, matched_trks2):
                track = unmatched_track_objs[trk_idx]
                det = low_conf[det_idx]
                track.update(det.bbox)
                track.class_id = det.class_id
                track.class_name = det.class_name
                track.status = TrackStatus.CONFIRMED if track.hits >= self._settings.min_hits else TrackStatus.TENTATIVE

        # Create new tracks for unmatched high-confidence detections
        for det_idx in unmatched_dets:
            det = high_conf[det_idx]
            if det.bbox.area >= self._settings.min_box_area:
                self._create_track(det)

        # Age unmatched tracks
        for trk_idx in unmatched_trks:
            track = active_tracks[trk_idx]
            track.time_since_update += 1
            track.age += 1
            track.predict()  # Predict next position

            # Mark as lost if too old
            if track.time_since_update > self._settings.max_age:
                track.status = TrackStatus.LOST

        # Remove dead tracks
        to_remove = []
        for track_id, track in self._tracks.items():
            if track.status == TrackStatus.LOST and track.time_since_update > self._settings.max_age:
                track.status = TrackStatus.REMOVED
                to_remove.append(track_id)

        for track_id in to_remove:
            del self._tracks[track_id]

        # Convert to Track objects
        result_tracks = []
        for track in self._tracks.values():
            if track.status != TrackStatus.REMOVED:
                result_tracks.append(self._to_track(track))

        # Update metrics
        self._metrics.set_gauge(
            "traffic_tracks_active",
            len([t for t in result_tracks if t.status == TrackStatus.CONFIRMED]),
            labels={"camera_id": "default"},
        )

        for track in result_tracks:
            self._metrics.observe_histogram(
                "traffic_track_age_frames",
                track.age,
                labels={"camera_id": "default"},
            )

        return result_tracks

    def _associate(
        self,
        detections: list[Detection],
        tracks: list[KalmanState],
        iou_threshold: float,
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        """Associate detections to tracks using Hungarian algorithm (greedy IoU)."""
        if not detections or not tracks:
            return [], [], list(range(len(detections))), list(range(len(tracks)))

        iou_mat = iou_matrix(detections, tracks)

        # Greedy matching by IoU
        matched_dets = []
        matched_trks = []
        used_dets = set()
        used_trks = set()

        # Sort by IoU descending
        pairs = []
        for i in range(iou_mat.shape[0]):
            for j in range(iou_mat.shape[1]):
                if iou_mat[i, j] > iou_threshold:
                    pairs.append((iou_mat[i, j], i, j))

        pairs.sort(reverse=True)

        for _, i, j in pairs:
            if i not in used_dets and j not in used_trks:
                matched_dets.append(i)
                matched_trks.append(j)
                used_dets.add(i)
                used_trks.add(j)

        unmatched_dets = [i for i in range(len(detections)) if i not in used_dets]
        unmatched_trks = [j for j in range(len(tracks)) if j not in used_trks]

        return matched_dets, matched_trks, unmatched_dets, unmatched_trks

    def _create_track(self, detection: Detection) -> KalmanState:
        """Create a new track from detection."""
        track_id = self._next_id
        self._next_id += 1

        kf = self._init_kalman(detection.bbox)
        track = KalmanState(
            kf=kf,
            track_id=track_id,
            class_id=detection.class_id,
            class_name=detection.class_name,
            bbox=detection.bbox,
            status=TrackStatus.TENTATIVE,
        )
        self._tracks[track_id] = track
        return track

    def _to_track(self, state: KalmanState) -> Track:
        """Convert KalmanState to domain Track."""
        detections = []
        if state.bbox:
            # Create a dummy detection for the current state
            det = Detection(
                detection_id=uuid4(),
                frame_id=uuid4(),
                timestamp=None,
                bbox=state.bbox,
                class_id=state.class_id,
                class_name=state.class_name,
                confidence=1.0,
            )
            detections = [det]

        return Track(
            track_id=state.track_id,
            class_id=state.class_id,
            class_name=state.class_name,
            status=state.status,
            detections=detections,
            world_position=state.world_position,
            world_velocity=None,  # Could extract from Kalman filter
            age=state.age,
            hits=state.hits,
            time_since_update=state.time_since_update,
        )

    def reset(self) -> None:
        """Reset tracker state."""
        self._tracks.clear()
        self._next_id = 1
        self._frame_count = 0
        logger.info("Tracker reset")


def create_tracker(tracker_type: str = "bytetrack", settings: TrackingSettings | None = None) -> Tracker:
    """Factory function to create tracker."""
    if tracker_type.lower() == "bytetrack":
        return ByteTracker(settings)
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")