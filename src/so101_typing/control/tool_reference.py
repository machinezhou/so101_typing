from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _normalize_image_size(raw_size: Sequence[int]) -> tuple[int, int]:
    if len(raw_size) != 2:
        raise ValueError("image_size must contain exactly two values")

    width, height = raw_size
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, (int, np.integer))
        or not isinstance(height, (int, np.integer))
    ):
        raise TypeError("image_size values must be integers")

    normalized = (int(width), int(height))
    if normalized[0] <= 0 or normalized[1] <= 0:
        raise ValueError("image_size values must be positive")
    return normalized


@dataclass(frozen=True, slots=True)
class ToolReferenceCalibration:
    camera_name: str
    image_size: tuple[int, int]
    u: float
    v: float
    sample_count: int
    std_u_px: float
    std_v_px: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_size", _normalize_image_size(self.image_size))
        self.validate()

    @property
    def center(self) -> tuple[float, float]:
        return (float(self.u), float(self.v))

    def error_px(
        self,
        target_center: tuple[float, float],
    ) -> tuple[float, float]:
        target_u, target_v = target_center
        if not math.isfinite(float(target_u)) or not math.isfinite(float(target_v)):
            raise ValueError("target_center must contain only finite values")
        return (float(target_u) - self.u, float(target_v) - self.v)

    @classmethod
    def from_samples(
        cls,
        centers: Iterable[Sequence[float]],
        *,
        camera_name: str = "wrist",
        image_size: tuple[int, int] = (640, 480),
    ) -> ToolReferenceCalibration:
        size = _normalize_image_size(image_size)
        try:
            samples = np.asarray(list(centers), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("centers must contain numeric pixel coordinates") from exc

        if samples.ndim != 2 or samples.shape[1] != 2:
            raise ValueError("centers must have shape (N, 2)")
        if samples.shape[0] < 3:
            raise ValueError("at least three center samples are required")
        if not np.isfinite(samples).all():
            raise ValueError("centers must contain only finite values")

        width, height = size
        if (
            np.any(samples[:, 0] < 0)
            or np.any(samples[:, 0] >= width)
            or np.any(samples[:, 1] < 0)
            or np.any(samples[:, 1] >= height)
        ):
            raise ValueError("center samples must stay inside image bounds")

        median = np.median(samples, axis=0)
        std = np.std(samples, axis=0, ddof=0)
        return cls(
            camera_name=camera_name,
            image_size=size,
            u=float(median[0]),
            v=float(median[1]),
            sample_count=int(samples.shape[0]),
            std_u_px=float(std[0]),
            std_v_px=float(std[1]),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> ToolReferenceCalibration | None:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        u = data.get("u")
        v = data.get("v")
        if u is None and v is None:
            return None
        if u is None or v is None:
            raise ValueError("u and v must either both be null or both be calibrated")

        raw_image_size = data.get("image_size")
        if raw_image_size is None:
            raise ValueError("image_size is required")

        try:
            image_size = _normalize_image_size(raw_image_size)
            sample_count = data["sample_count"]
            std_u_px = data["std_u_px"]
            std_v_px = data["std_v_px"]
            camera_name = data["camera_name"]
        except KeyError as exc:
            raise ValueError(f"missing required calibration field: {exc.args[0]}") from exc

        return cls(
            camera_name=camera_name,
            image_size=image_size,
            u=float(u),
            v=float(v),
            sample_count=sample_count,
            std_u_px=float(std_u_px),
            std_v_px=float(std_v_px),
        )

    def to_dict(self) -> dict:
        self.validate()
        return {
            "camera_name": self.camera_name,
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "u": float(self.u),
            "v": float(self.v),
            "sample_count": int(self.sample_count),
            "std_u_px": float(self.std_u_px),
            "std_v_px": float(self.std_v_px),
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

    def validate(self) -> None:
        if not isinstance(self.camera_name, str) or not self.camera_name.strip():
            raise ValueError("camera_name must be a non-empty string")

        width, height = _normalize_image_size(self.image_size)
        if not math.isfinite(float(self.u)) or not math.isfinite(float(self.v)):
            raise ValueError("u and v must be finite")
        if not 0 <= self.u < width or not 0 <= self.v < height:
            raise ValueError("tool reference must stay inside image bounds")

        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, (int, np.integer))
            or int(self.sample_count) < 1
        ):
            raise ValueError("sample_count must be an integer >= 1")

        if not math.isfinite(float(self.std_u_px)) or not math.isfinite(float(self.std_v_px)):
            raise ValueError("standard deviations must be finite")
        if self.std_u_px < 0 or self.std_v_px < 0:
            raise ValueError("standard deviations must be non-negative")
