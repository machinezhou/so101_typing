from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from so101_typing.control.image_jacobian import ImageJacobianCalibration


def _as_vector2(values: Sequence[float], *, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc

    if array.shape != (2,):
        raise ValueError(f"{name} must contain exactly two values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def damped_pseudoinverse(
    matrix: Sequence[Sequence[float]],
    *,
    damping: float,
) -> np.ndarray:
    """Return the SVD damped pseudoinverse of a 2x2 image Jacobian."""
    try:
        array = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("matrix must be numeric") from exc

    if array.shape != (2, 2):
        raise ValueError("matrix must have shape (2, 2)")
    if not np.isfinite(array).all():
        raise ValueError("matrix must contain only finite values")

    damping_value = float(damping)
    if not math.isfinite(damping_value) or damping_value < 0.0:
        raise ValueError("damping must be finite and non-negative")

    u, singular_values, vt = np.linalg.svd(array, full_matrices=False)
    if damping_value == 0.0 and np.any(singular_values == 0.0):
        raise ValueError("undamped pseudoinverse requires a full-rank matrix")

    denominator = singular_values**2 + damping_value**2
    factors = np.divide(
        singular_values,
        denominator,
        out=np.zeros_like(singular_values),
        where=denominator > 0.0,
    )
    return vt.T @ np.diag(factors) @ u.T


@dataclass(frozen=True, slots=True)
class VisualServoConfig:
    """Pure-math bounds for one XY visual-servo correction step."""

    gain: float = 1.0
    convergence_threshold_px: float = 3.0
    max_step_norm: float = 2.0
    max_total_correction_norm: float = 10.0

    def __post_init__(self) -> None:
        gain = float(self.gain)
        if not math.isfinite(gain) or not 0.0 < gain <= 1.0:
            raise ValueError("gain must be finite and in (0, 1]")
        object.__setattr__(self, "gain", gain)

        threshold = float(self.convergence_threshold_px)
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("convergence_threshold_px must be finite and positive")
        object.__setattr__(self, "convergence_threshold_px", threshold)

        max_step = float(self.max_step_norm)
        if not math.isfinite(max_step) or max_step <= 0.0:
            raise ValueError("max_step_norm must be finite and positive")
        object.__setattr__(self, "max_step_norm", max_step)

        max_total = float(self.max_total_correction_norm)
        if not math.isfinite(max_total) or max_total <= 0.0:
            raise ValueError("max_total_correction_norm must be finite and positive")
        object.__setattr__(self, "max_total_correction_norm", max_total)


class VisualServoStepStatus(StrEnum):
    CORRECTION = "correction"
    WITHIN_TOLERANCE = "within_tolerance"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True, slots=True)
class VisualServoStep:
    status: VisualServoStepStatus
    error_px: tuple[float, float]
    error_norm_px: float
    raw_correction: tuple[float, float]
    correction: tuple[float, float]
    correction_norm: float
    cumulative_before: tuple[float, float]
    cumulative_after: tuple[float, float]
    step_limited: bool
    total_limited: bool
    motion_frame: str
    motion_unit: str


def _limit_norm(vector: np.ndarray, limit: float) -> tuple[np.ndarray, bool]:
    norm = float(np.linalg.norm(vector))
    if norm <= limit:
        return vector, False
    return vector * (limit / norm), True


def _limit_total_correction(
    cumulative: np.ndarray,
    step: np.ndarray,
    limit: float,
) -> tuple[np.ndarray, bool]:
    if float(np.linalg.norm(cumulative + step)) <= limit:
        return step, False

    step_squared = float(step @ step)
    if step_squared == 0.0:
        return step, False

    b = 2.0 * float(cumulative @ step)
    c = float(cumulative @ cumulative) - limit**2
    discriminant = max(0.0, b**2 - 4.0 * step_squared * c)
    alpha = (-b + math.sqrt(discriminant)) / (2.0 * step_squared)
    alpha = min(1.0, max(0.0, alpha))
    return step * alpha, True


def compute_visual_servo_step(
    error_px: Sequence[float],
    calibration: ImageJacobianCalibration,
    *,
    config: VisualServoConfig | None = None,
    cumulative_correction: Sequence[float] = (0.0, 0.0),
) -> VisualServoStep:
    """Compute one bounded XY correction without commanding any robot motion.

    Sign convention is frozen as ``error = p - p_star`` and
    ``delta_x = -gain * J_damped_pinv @ error``.

    ``WITHIN_TOLERANCE`` means only that this single observation is inside the
    pixel threshold.  Stable multi-frame convergence belongs to the runtime
    state machine and is intentionally not declared here.
    """
    if config is None:
        config = VisualServoConfig()

    error = _as_vector2(error_px, name="error_px")
    cumulative = _as_vector2(cumulative_correction, name="cumulative_correction")
    cumulative_norm = float(np.linalg.norm(cumulative))
    if cumulative_norm > config.max_total_correction_norm + 1e-12:
        raise ValueError("cumulative_correction already exceeds the configured total bound")

    error_norm = float(np.linalg.norm(error))
    zero = (0.0, 0.0)
    cumulative_tuple = (float(cumulative[0]), float(cumulative[1]))

    if error_norm < config.convergence_threshold_px:
        return VisualServoStep(
            status=VisualServoStepStatus.WITHIN_TOLERANCE,
            error_px=(float(error[0]), float(error[1])),
            error_norm_px=error_norm,
            raw_correction=zero,
            correction=zero,
            correction_norm=0.0,
            cumulative_before=cumulative_tuple,
            cumulative_after=cumulative_tuple,
            step_limited=False,
            total_limited=False,
            motion_frame=calibration.motion_frame,
            motion_unit=calibration.motion_unit,
        )

    inverse = damped_pseudoinverse(calibration.matrix, damping=calibration.damping)
    raw = -config.gain * (inverse @ error)
    step, step_limited = _limit_norm(raw, config.max_step_norm)
    step, total_limited = _limit_total_correction(
        cumulative,
        step,
        config.max_total_correction_norm,
    )
    after = cumulative + step
    correction_norm = float(np.linalg.norm(step))

    status = VisualServoStepStatus.CORRECTION
    if correction_norm <= 1e-12 and total_limited:
        status = VisualServoStepStatus.BUDGET_EXHAUSTED

    return VisualServoStep(
        status=status,
        error_px=(float(error[0]), float(error[1])),
        error_norm_px=error_norm,
        raw_correction=(float(raw[0]), float(raw[1])),
        correction=(float(step[0]), float(step[1])),
        correction_norm=correction_norm,
        cumulative_before=cumulative_tuple,
        cumulative_after=(float(after[0]), float(after[1])),
        step_limited=step_limited,
        total_limited=total_limited,
        motion_frame=calibration.motion_frame,
        motion_unit=calibration.motion_unit,
    )
