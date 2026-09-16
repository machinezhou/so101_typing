from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _as_finite_array(
    values: Iterable[Sequence[float]],
    *,
    name: str,
) -> np.ndarray:
    try:
        array = np.asarray(list(values), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc

    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _normalize_matrix(matrix: Sequence[Sequence[float]]) -> tuple[tuple[float, float], ...]:
    try:
        array = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("matrix must be numeric") from exc

    if array.shape != (2, 2):
        raise ValueError("matrix must have shape (2, 2)")
    if not np.isfinite(array).all():
        raise ValueError("matrix must contain only finite values")
    if np.linalg.matrix_rank(array) < 2:
        raise ValueError("matrix must have rank 2")

    return tuple(tuple(float(value) for value in row) for row in array)


def _normalize_singular_values(values: Sequence[float]) -> tuple[float, float]:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("singular_values must be numeric") from exc

    if array.shape != (2,):
        raise ValueError("singular_values must contain exactly two values")
    if not np.isfinite(array).all() or np.any(array <= 0.0):
        raise ValueError("singular_values must be finite and positive")
    if array[0] < array[1]:
        raise ValueError("singular_values must be stored in descending order")

    return (float(array[0]), float(array[1]))


@dataclass(frozen=True, slots=True)
class ImageJacobianCalibration:
    """Local mapping from small Cartesian XY motion to wrist-image motion.

    The matrix follows the column-vector convention::

        delta_p_px = J @ delta_x

    where ``delta_p_px = (delta_u_px, delta_v_px)`` and ``delta_x`` is a
    two-dimensional Cartesian perturbation expressed in ``motion_unit`` and
    ``motion_frame``.
    """

    motion_frame: str
    motion_unit: str
    matrix: tuple[tuple[float, float], ...]
    sample_count: int
    residual_rms_px: float
    singular_values: tuple[float, float]
    condition_number: float
    damping: float = 1e-6

    def __post_init__(self) -> None:
        if not isinstance(self.motion_frame, str):
            raise TypeError("motion_frame must be a string")
        if not self.motion_frame.strip():
            raise ValueError("motion_frame must be non-empty")
        if not isinstance(self.motion_unit, str):
            raise TypeError("motion_unit must be a string")
        if not self.motion_unit.strip():
            raise ValueError("motion_unit must be non-empty")

        normalized_matrix = _normalize_matrix(self.matrix)
        object.__setattr__(self, "matrix", normalized_matrix)

        if not isinstance(self.sample_count, (int, np.integer)):
            raise TypeError("sample_count must be an integer")
        if self.sample_count < 4:
            raise ValueError("sample_count must be at least 4")
        object.__setattr__(self, "sample_count", int(self.sample_count))

        residual = float(self.residual_rms_px)
        if not math.isfinite(residual) or residual < 0.0:
            raise ValueError("residual_rms_px must be finite and non-negative")
        object.__setattr__(self, "residual_rms_px", residual)

        singular_values = _normalize_singular_values(self.singular_values)
        object.__setattr__(self, "singular_values", singular_values)

        condition = float(self.condition_number)
        if not math.isfinite(condition) or condition < 1.0:
            raise ValueError("condition_number must be finite and at least 1")
        object.__setattr__(self, "condition_number", condition)

        damping = float(self.damping)
        if not math.isfinite(damping) or damping < 0.0:
            raise ValueError("damping must be finite and non-negative")
        object.__setattr__(self, "damping", damping)

    @property
    def array(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=np.float64)

    def predict_image_delta(self, motion_delta: Sequence[float]) -> tuple[float, float]:
        try:
            delta = np.asarray(motion_delta, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("motion_delta must be numeric") from exc

        if delta.shape != (2,):
            raise ValueError("motion_delta must contain exactly two values")
        if not np.isfinite(delta).all():
            raise ValueError("motion_delta must contain only finite values")

        predicted = self.array @ delta
        return (float(predicted[0]), float(predicted[1]))

    @classmethod
    def from_samples(
        cls,
        motion_deltas: Iterable[Sequence[float]],
        image_deltas_px: Iterable[Sequence[float]],
        *,
        motion_frame: str = "base_link_xy",
        motion_unit: str = "mm",
        damping: float = 1e-6,
        max_condition_number: float | None = None,
    ) -> ImageJacobianCalibration:
        motions = _as_finite_array(motion_deltas, name="motion_deltas")
        image_deltas = _as_finite_array(image_deltas_px, name="image_deltas_px")

        if motions.shape[0] != image_deltas.shape[0]:
            raise ValueError("motion_deltas and image_deltas_px must have equal length")
        if motions.shape[0] < 4:
            raise ValueError("at least 4 perturbation samples are required")
        if np.linalg.matrix_rank(motions) < 2:
            raise ValueError("motion_deltas must span two independent XY directions")

        # Rows are samples.  Solve P = X @ J.T, then transpose to the
        # documented column-vector convention delta_p = J @ delta_x.
        jacobian_t, _, _, _ = np.linalg.lstsq(motions, image_deltas, rcond=None)
        jacobian = jacobian_t.T

        if np.linalg.matrix_rank(jacobian) < 2:
            raise ValueError("estimated image Jacobian must have rank 2")

        predicted = motions @ jacobian.T
        residuals = image_deltas - predicted
        residual_rms_px = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))

        singular_values_array = np.linalg.svd(jacobian, compute_uv=False)
        condition_number = float(singular_values_array[0] / singular_values_array[-1])

        if max_condition_number is not None:
            threshold = float(max_condition_number)
            if not math.isfinite(threshold) or threshold < 1.0:
                raise ValueError("max_condition_number must be finite and at least 1")
            if condition_number > threshold:
                raise ValueError(
                    "estimated image Jacobian is ill-conditioned: "
                    f"condition_number={condition_number:.6g} > {threshold:.6g}"
                )

        return cls(
            motion_frame=motion_frame,
            motion_unit=motion_unit,
            matrix=tuple(tuple(float(value) for value in row) for row in jacobian),
            sample_count=int(motions.shape[0]),
            residual_rms_px=residual_rms_px,
            singular_values=(
                float(singular_values_array[0]),
                float(singular_values_array[1]),
            ),
            condition_number=condition_number,
            damping=damping,
        )

    @classmethod
    def load(cls, path: str | Path) -> ImageJacobianCalibration | None:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        matrix = data.get("matrix")
        if matrix is None:
            return None

        required = (
            "motion_frame",
            "motion_unit",
            "sample_count",
            "residual_rms_px",
            "singular_values",
            "condition_number",
            "damping",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"calibrated image Jacobian is missing fields: {missing}")

        return cls(
            motion_frame=data["motion_frame"],
            motion_unit=data["motion_unit"],
            matrix=matrix,
            sample_count=data["sample_count"],
            residual_rms_px=data["residual_rms_px"],
            singular_values=data["singular_values"],
            condition_number=data["condition_number"],
            damping=data["damping"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "motion_frame": self.motion_frame,
            "motion_unit": self.motion_unit,
            "matrix": [list(row) for row in self.matrix],
            "sample_count": self.sample_count,
            "residual_rms_px": self.residual_rms_px,
            "singular_values": list(self.singular_values),
            "condition_number": self.condition_number,
            "damping": self.damping,
        }

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(f".{output_path.name}.tmp")
        temporary_path.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(output_path)
