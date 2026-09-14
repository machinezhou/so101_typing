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

        points = np.asarray(
            raw_points,
            dtype=np.float32,
        )

        if points.shape != (4, 2):
            raise ValueError(
                "source_points must have shape (4, 2)"
            )

        output_size = tuple(
            int(value)
            for value in data["output_size"]
        )

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
            self.source_points,
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
