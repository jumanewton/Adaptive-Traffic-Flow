#!/usr/bin/env python
"""Generate a synthetic test video with moving vehicles for testing."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def generate_test_video(
    output_path: str,
    duration_seconds: int = 10,
    fps: int = 30,
    resolution: tuple[int, int] = (1280, 720),
    num_vehicles: int = 5,
) -> None:
    """Generate a synthetic traffic video with moving vehicles."""
    width, height = resolution
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Vehicle properties
    vehicles = []
    for i in range(num_vehicles):
        # Random lane (y position)
        lane_y = np.random.randint(height // 4, 3 * height // 4)
        # Random start x position
        start_x = np.random.randint(-100, width // 2)
        # Random speed (pixels per frame)
        speed = np.random.uniform(2, 8)
        # Vehicle size
        vw, vh = np.random.randint(60, 100), np.random.randint(30, 50)
        # Color
        color = tuple(np.random.randint(50, 255, 3).tolist())

        vehicles.append({
            "x": start_x,
            "y": lane_y,
            "speed": speed,
            "w": vw,
            "h": vh,
            "color": color,
        })

    # Background (simple road)
    def draw_background(img):
        img[:] = (50, 50, 50)  # Dark gray road
        # Lane markings
        for y in [height // 4, height // 2, 3 * height // 4]:
            for x in range(0, width, 40):
                cv2.line(img, (x, y), (x + 20, y), (200, 200, 200), 2)

    total_frames = duration_seconds * fps

    for frame_idx in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        draw_background(frame)

        # Update and draw vehicles
        for v in vehicles:
            v["x"] += v["speed"]
            # Wrap around
            if v["x"] > width + v["w"]:
                v["x"] = -v["w"]
                v["y"] = np.random.randint(height // 4, 3 * height // 4)

            # Draw vehicle
            x, y = int(v["x"]), int(v["y"])
            cv2.rectangle(frame,
                         (x, y - v["h"] // 2),
                         (x + v["w"], y + v["h"] // 2),
                         v["color"], -1)
            # Windows
            cv2.rectangle(frame,
                         (x + 5, y - v["h"] // 2 + 5),
                         (x + v["w"] - 5, y - v["h"] // 2 + 15),
                         (100, 150, 255), -1)

        # Frame counter
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        writer.write(frame)

    writer.release()
    print(f"Generated {output_path}: {total_frames} frames, {duration_seconds}s at {fps}fps")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic traffic test video")
    parser.add_argument("-o", "--output", default="data/raw/test_traffic.mp4", help="Output video path")
    parser.add_argument("-d", "--duration", type=int, default=10, help="Duration in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("-W", "--width", type=int, default=1280, help="Width")
    parser.add_argument("-H", "--height", type=int, default=720, help="Height")
    parser.add_argument("-n", "--vehicles", type=int, default=5, help="Number of vehicles")

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_test_video(
        str(output_path),
        duration_seconds=args.duration,
        fps=args.fps,
        resolution=(args.width, args.height),
        num_vehicles=args.vehicles,
    )


if __name__ == "__main__":
    main()