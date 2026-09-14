from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class ScreenCalibration:
    source_points: np.ndarray
    output_size: tuple[int, int]
    text_roi: tuple[int, int, int, int] | None

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "ScreenCalibration | None":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        raw_points = data.get("source_points")
        if raw_points is None:
            return None

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "ScreenCalibration":
        raw_points = data.get("source_points")
        if raw_points is None:
            raise ValueError("source_points must not be null for a calibrated screen")

        points = np.asarray(raw_points, dtype=np.float32)

        raw_output_size = data.get("output_size")
        if raw_output_size is None:
            raise ValueError("output_size is required")

        output_size = tuple(int(value) for value in raw_output_size)

        raw_roi = data.get("text_roi")
        text_roi = (
            tuple(int(value) for value in raw_roi)
            if raw_roi is not None
            else None
        )

        return cls(
            source_points=points,
            output_size=output_size,
            text_roi=text_roi,
        )

    def to_dict(self) -> dict:
        return {
            "source_points": [
                [float(x), float(y)]
                for x, y in self.source_points.tolist()
            ],
            "output_size": [
                int(self.output_size[0]),
                int(self.output_size[1]),
            ],
            "text_roi": (
                [int(value) for value in self.text_roi]
                if self.text_roi is not None
                else None
            ),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def validate(self) -> None:
        points = np.asarray(self.source_points, dtype=np.float32)
        if points.shape != (4, 2):
            raise ValueError("source_points must have shape (4, 2)")

        if not np.isfinite(points).all():
            raise ValueError("source_points must contain only finite values")

        if len(self.output_size) != 2:
            raise ValueError("output_size must contain exactly two values")

        width, height = self.output_size
        if width <= 0 or height <= 0:
            raise ValueError("output_size values must be positive")

        contour = points.reshape((-1, 1, 2))
        if not cv2.isContourConvex(contour):
            raise ValueError(
                "source_points must form a convex quadrilateral in "
                "TL, TR, BR, BL order"
            )

        area = abs(float(cv2.contourArea(contour)))
        if area < 1.0:
            raise ValueError("source_points define a degenerate quadrilateral")

        if self.text_roi is None:
            return

        if len(self.text_roi) != 4:
            raise ValueError("text_roi must contain x, y, width, height")

        x, y, roi_width, roi_height = self.text_roi
        if x < 0 or y < 0:
            raise ValueError("text_roi x/y must be non-negative")
        if roi_width <= 0 or roi_height <= 0:
            raise ValueError("text_roi width/height must be positive")
        if x + roi_width > width or y + roi_height > height:
            raise ValueError("text_roi must stay inside the rectified screen")

    def rectify(
        self,
        frame_bgr: np.ndarray,
    ) -> np.ndarray:
        width, height = self.output_size

        destination = np.asarray(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ],
            dtype=np.float32,
        )
        homography = cv2.getPerspectiveTransform(
            self.source_points.astype(np.float32),
            destination,
        )

        return cv2.warpPerspective(
            frame_bgr,
            homography,
            (width, height),
        )

    def crop_text_roi(
        self,
        rectified_bgr: np.ndarray,
    ) -> np.ndarray | None:
        if self.text_roi is None:
            return None

        x, y, width, height = self.text_roi
        return rectified_bgr[
            y : y + height,
            x : x + width,
        ]
