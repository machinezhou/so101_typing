from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class TopMaskConfig:
    polygon: np.ndarray | None
    brightness_threshold: int = 245

    @classmethod
    def load(cls, path: str | Path) -> "TopMaskConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        polygon_raw = data.get("polygon")

        polygon = None

        if polygon_raw is not None:
            polygon = np.asarray(
                polygon_raw,
                dtype=np.int32,
            )

            if polygon.ndim != 2 or polygon.shape[1] != 2:
                raise ValueError(
                    "TOP mask polygon must have shape (N, 2)"
                )

        return cls(
            polygon=polygon,
            brightness_threshold=int(
                data.get("brightness_threshold", 245)
            ),
        )


def analyze_and_mask(
    frame_bgr: np.ndarray,
    config: TopMaskConfig,
) -> tuple[np.ndarray, dict]:
    if config.polygon is None:
        return frame_bgr.copy(), {
            "configured": False,
            "masked_fraction": 0.0,
            "bright_fraction_inside_screen": None,
        }

    height, width = frame_bgr.shape[:2]

    region_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    cv2.fillPoly(
        region_mask,
        [config.polygon],
        255,
    )

    masked = frame_bgr.copy()
    masked[region_mask > 0] = 0

    gray = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    inside = region_mask > 0
    pixel_count = int(inside.sum())

    bright_fraction = None

    if pixel_count:
        bright_fraction = float(
            np.mean(
                gray[inside]
                >= config.brightness_threshold
            )
        )

    return masked, {
        "configured": True,
        "masked_fraction": float(
            pixel_count / (height * width)
        ),
        "bright_fraction_inside_screen": bright_fraction,
    }
