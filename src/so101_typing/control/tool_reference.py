from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


LEGACY_TARGET_DERIVED = "legacy_target_derived"
WRIST_TOOL_TIP = "wrist_tool_tip"
_VALID_REFERENCE_KINDS = frozenset({LEGACY_TARGET_DERIVED, WRIST_TOOL_TIP})


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
    reference_kind: str = WRIST_TOOL_TIP
    method: str = "direct_manual_click"
    rms_radial_deviation_px: float | None = None
    max_radial_deviation_px: float | None = None
    schema_version: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_size", _normalize_image_size(self.image_size))
        self.validate()

    @property
    def center(self) -> tuple[float, float]:
        return (float(self.u), float(self.v))

    @property
    def is_direct_tool_tip(self) -> bool:
        return self.reference_kind == WRIST_TOOL_TIP

    def require_direct_tool_tip(self) -> None:
        if not self.is_direct_tool_tip:
            raise RuntimeError(
                "TIP CALIBRATION REQUIRED: calibration/tool_reference.json is the legacy "
                "target-derived H3.1 reference, not a direct WRIST pencil-tip calibration. "
                "Run scripts/calibrate_tool_reference.py before Phase 3 autonomy."
            )

    def error_px(
        self,
        target_center: tuple[float, float],
    ) -> tuple[float, float]:
        """Return ``p_key - p_tip`` in WRIST pixels."""
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
        reference_kind: str = WRIST_TOOL_TIP,
        method: str = "direct_manual_click",
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
        radial = np.linalg.norm(samples - median, axis=1)
        return cls(
            camera_name=camera_name,
            image_size=size,
            u=float(median[0]),
            v=float(median[1]),
            sample_count=int(samples.shape[0]),
            std_u_px=float(std[0]),
            std_v_px=float(std[1]),
            reference_kind=reference_kind,
            method=method,
            rms_radial_deviation_px=float(np.sqrt(np.mean(radial**2))),
            max_radial_deviation_px=float(np.max(radial)),
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

        # Files produced before the direct-tip design are H3.1 target-derived
        # references.  Backward-compatible loading must never silently promote
        # them to a physical pencil-tip calibration.
        reference_kind = data.get("reference_kind", LEGACY_TARGET_DERIVED)
        method = data.get(
            "method",
            "legacy_h3_1_target_alignment"
            if reference_kind == LEGACY_TARGET_DERIVED
            else "direct_manual_click",
        )
        schema_version = int(data.get("schema_version", 1))
        rms_radial = data.get("rms_radial_deviation_px")
        max_radial = data.get("max_radial_deviation_px")

        return cls(
            camera_name=camera_name,
            image_size=image_size,
            u=float(u),
            v=float(v),
            sample_count=sample_count,
            std_u_px=float(std_u_px),
            std_v_px=float(std_v_px),
            reference_kind=str(reference_kind),
            method=str(method),
            rms_radial_deviation_px=None if rms_radial is None else float(rms_radial),
            max_radial_deviation_px=None if max_radial is None else float(max_radial),
            schema_version=schema_version,
        )

    def to_dict(self) -> dict:
        self.validate()
        return {
            "schema_version": int(self.schema_version),
            "reference_kind": self.reference_kind,
            "method": self.method,
            "camera_name": self.camera_name,
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "u": float(self.u),
            "v": float(self.v),
            "sample_count": int(self.sample_count),
            "std_u_px": float(self.std_u_px),
            "std_v_px": float(self.std_v_px),
            "rms_radial_deviation_px": (
                None
                if self.rms_radial_deviation_px is None
                else float(self.rms_radial_deviation_px)
            ),
            "max_radial_deviation_px": (
                None
                if self.max_radial_deviation_px is None
                else float(self.max_radial_deviation_px)
            ),
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

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

        for name, value in (
            ("rms_radial_deviation_px", self.rms_radial_deviation_px),
            ("max_radial_deviation_px", self.max_radial_deviation_px),
        ):
            if value is None:
                continue
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative when present")
        if (
            self.rms_radial_deviation_px is not None
            and self.max_radial_deviation_px is not None
            and self.rms_radial_deviation_px > self.max_radial_deviation_px + 1e-12
        ):
            raise ValueError("rms_radial_deviation_px must not exceed max_radial_deviation_px")

        if self.reference_kind not in _VALID_REFERENCE_KINDS:
            raise ValueError(
                f"reference_kind must be one of {sorted(_VALID_REFERENCE_KINDS)}, "
                f"got {self.reference_kind!r}"
            )
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError("method must be a non-empty string")
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise TypeError("schema_version must be an integer")
        if self.schema_version < 1:
            raise ValueError("schema_version must be >= 1")
