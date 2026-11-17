"""Camera calibration and coordinate transforms."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from traffic_domain.schemas import CameraCalibration, Point, HomographyMatrix
from traffic_observability.logging import get_logger


logger = get_logger(__name__)


class CameraCalibrator:
    """Camera calibration utility for perspective transforms."""

    def __init__(self, calibration: CameraCalibration):
        """Initialize with calibration data."""
        self.calibration = calibration
        self._camera_matrix = None
        self._dist_coeffs = None
        self._homography = None
        self._inv_homography = None
        self._compute_matrices()

    def _compute_matrices(self) -> None:
        """Compute camera matrix and distortion coefficients."""
        # Camera matrix
        self._camera_matrix = np.array([
            [self.calibration.fx, 0, self.calibration.cx],
            [0, self.calibration.fy, self.calibration.cy],
            [0, 0, 1],
        ], dtype=np.float64)

        # Distortion coefficients
        self._dist_coeffs = np.array(self.calibration.distortion, dtype=np.float64)

        # Homography for ground plane
        if self.calibration.ground_homography:
            self._homography = np.array(self.calibration.ground_homography.matrix, dtype=np.float64)
            self._inv_homography = np.linalg.inv(self._homography)

    def undistort_points(self, points: list[Point]) -> list[Point]:
        """Undistort image points."""
        if not points:
            return []

        pts = np.array([[p.x, p.y] for p in points], dtype=np.float32).reshape(-1, 1, 2)
        undistorted = cv2.undistortPoints(pts, self._camera_matrix, self._dist_coeffs, P=self._camera_matrix)
        return [Point(x=float(p[0][0]), y=float(p[0][1])) for p in undistorted]

    def pixel_to_world(self, pixel: Point) -> Point | None:
        """Convert pixel coordinates to world coordinates (ground plane).

        Requires ground_homography to be set in calibration.
        """
        if self._homography is None:
            logger.warning("No ground homography available for pixel->world transform")
            return None

        pt = np.array([pixel.x, pixel.y, 1.0], dtype=np.float64)
        world_pt = self._homography @ pt
        world_pt = world_pt / world_pt[2]  # Normalize homogeneous coordinates
        return Point(x=float(world_pt[0]), y=float(world_pt[1]))

    def world_to_pixel(self, world: Point) -> Point | None:
        """Convert world coordinates to pixel coordinates."""
        if self._inv_homography is None:
            logger.warning("No inverse homography available for world->pixel transform")
            return None

        pt = np.array([world.x, world.y, 1.0], dtype=np.float64)
        pixel_pt = self._inv_homography @ pt
        pixel_pt = pixel_pt / pixel_pt[2]
        return Point(x=float(pixel_pt[0]), y=float(pixel_pt[1]))

    def bbox_center_to_world(self, bbox_center: Point) -> Point | None:
        """Convert bounding box center (pixel) to world position."""
        return self.pixel_to_world(bbox_center)


def load_calibration(file_path: str | Path) -> CameraCalibration:
    """Load camera calibration from JSON file."""
    path = Path(file_path)
    with open(path) as f:
        data = json.load(f)

    # Convert lists to proper types
    if "ground_homography" in data and data["ground_homography"]:
        data["ground_homography"] = HomographyMatrix(matrix=data["ground_homography"]["matrix"])

    return CameraCalibration(**data)


def save_calibration(calibration: CameraCalibration, file_path: str | Path) -> None:
    """Save camera calibration to JSON file."""
    path = Path(file_path)
    data = calibration.model_dump(mode="json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def create_calibration_from_points(
    camera_id: str,
    image_width: int,
    image_height: int,
    image_points: list[Point],  # Pixel coordinates
    world_points: list[Point],  # World coordinates (meters)
    fx: float | None = None,
    fy: float | None = None,
) -> CameraCalibration:
    """Create calibration from corresponding image/world points.

    Uses OpenCV's findHomography for ground plane.
    Requires at least 4 point correspondences.
    """
    if len(image_points) < 4 or len(world_points) < 4:
        raise ValueError("Need at least 4 point correspondences")

    if len(image_points) != len(world_points):
        raise ValueError("Number of image and world points must match")

    # Convert to numpy arrays
    img_pts = np.array([[p.x, p.y] for p in image_points], dtype=np.float32)
    world_pts = np.array([[p.x, p.y] for p in world_points], dtype=np.float32)

    # Compute homography (image -> world)
    homography_matrix, _ = cv2.findHomography(img_pts, world_pts, cv2.RANSAC, 5.0)

    if homography_matrix is None:
        raise ValueError("Failed to compute homography")

    # Estimate focal length if not provided
    if fx is None:
        fx = max(image_width, image_height) * 1.2
    if fy is None:
        fy = fx

    cx = image_width / 2
    cy = image_height / 2

    # Create calibration with identity rotation/translation (simplified)
    calibration = CameraCalibration(
        camera_id=camera_id,
        image_width=image_width,
        image_height=image_height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        distortion=[0.0, 0.0, 0.0, 0.0, 0.0],
        rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        translation=[0, 0, 5],  # 5m camera height assumption
        ground_homography=HomographyMatrix(matrix=homography_matrix.tolist()),
    )

    return calibration