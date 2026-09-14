from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np


def _normalize_polygon(raw_polygon: object) -> np.ndarray | None:
    if raw_polygon is None:
        return None

    polygon = np.asarray(raw_polygon, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("TOP mask polygon must have shape (N, 2)")
    if polygon.shape[0] < 3:
        raise ValueError("TOP mask polygon must contain at least three points")
    if not np.all(np.isfinite(polygon)):
        raise ValueError("TOP mask polygon contains non-finite coordinates")

    polygon_i32 = np.rint(polygon).astype(np.int32)
    area = abs(float(cv2.contourArea(polygon_i32.astype(np.float32))))
    if area < 1.0:
        raise ValueError("TOP mask polygon area must be non-zero")

    return polygon_i32


def _validate_brightness_threshold(value: int) -> int:
    threshold = int(value)
    if not 0 <= threshold <= 255:
        raise ValueError("brightness_threshold must be in [0, 255]")
    return threshold


@dataclass(frozen=True, slots=True)
class TopMaskConfig:
    polygon: np.ndarray | None
    brightness_threshold: int = 245

    def __post_init__(self) -> None:
        object.__setattr__(self, "polygon", _normalize_polygon(self.polygon))
        object.__setattr__(
            self,
            "brightness_threshold",
            _validate_brightness_threshold(self.brightness_threshold),
        )

    @classmethod
    def load(cls, path: str | Path) -> "TopMaskConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        return cls(
            polygon=data.get("polygon"),
            brightness_threshold=int(data.get("brightness_threshold", 245)),
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "polygon": (
                self.polygon.astype(int).tolist()
                if self.polygon is not None
                else None
            ),
            "brightness_threshold": self.brightness_threshold,
        }
        destination.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    def validate_for_frame(self, frame_bgr: np.ndarray) -> None:
        if self.polygon is None:
            raise ValueError("TOP mask polygon is not configured")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("TOP frame must be a BGR image with shape HxWx3")

        height, width = frame_bgr.shape[:2]
        xs = self.polygon[:, 0]
        ys = self.polygon[:, 1]
        if (
            np.any(xs < 0)
            or np.any(xs >= width)
            or np.any(ys < 0)
            or np.any(ys >= height)
        ):
            raise ValueError(
                "TOP mask polygon must stay inside the configured TOP frame"
            )


def analyze_and_mask(
    frame_bgr: np.ndarray,
    config: TopMaskConfig,
) -> tuple[np.ndarray, dict]:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("TOP frame must be a BGR image with shape HxWx3")

    if config.polygon is None:
        return frame_bgr.copy(), {
            "configured": False,
            "masked_fraction": 0.0,
            "screen_pixel_count": 0,
            "bright_fraction_inside_screen": None,
            "mean_inside_screen": None,
            "std_inside_screen": None,
        }

    config.validate_for_frame(frame_bgr)
    height, width = frame_bgr.shape[:2]

    region_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(region_mask, [config.polygon], 255)

    inside = region_mask > 0
    pixel_count = int(inside.sum())
    if pixel_count <= 0:
        raise ValueError("TOP mask polygon produced an empty screen region")

    masked = frame_bgr.copy()
    masked[inside] = 0

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    screen_gray = gray[inside]

    return masked, {
        "configured": True,
        "masked_fraction": float(pixel_count / (height * width)),
        "screen_pixel_count": pixel_count,
        "bright_fraction_inside_screen": float(
            np.mean(screen_gray >= config.brightness_threshold)
        ),
        "mean_inside_screen": float(np.mean(screen_gray)),
        "std_inside_screen": float(np.std(screen_gray)),
    }
