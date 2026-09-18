from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np


TAlignment = TypeVar("TAlignment")
TZStep = TypeVar("TZStep")


# Frozen pixel/command context of the accepted H3.2 calibration checkpoint.
# The canonical JSON intentionally remains untouched; Phase 3 validates the
# runtime against the context in which that canonical mapping was accepted.
H32_CAMERA_NAME = "wrist"
H32_IMAGE_SIZE = (640, 480)
H32_MOTION_FRAME = "base_link_xy"
H32_MOTION_UNIT = "mm"



def validate_h32_runtime_context(
    *,
    camera_name: str,
    configured_image_size: Sequence[int],
    jacobian_motion_frame: str,
    jacobian_motion_unit: str,
) -> None:
    """Fail closed if Phase-3 runtime leaves the accepted H3.2 coordinate space."""
    try:
        width, height = configured_image_size
    except (TypeError, ValueError) as exc:
        raise ValueError("configured_image_size must contain width and height") from exc
    size = (int(width), int(height))
    if camera_name != H32_CAMERA_NAME:
        raise RuntimeError(
            f"H3.2 context mismatch: camera={camera_name!r}, expected {H32_CAMERA_NAME!r}"
        )
    if size != H32_IMAGE_SIZE:
        raise RuntimeError(
            "H3.2 context mismatch: configured WRIST image size "
            f"{size} != accepted {H32_IMAGE_SIZE}"
        )
    if jacobian_motion_frame != H32_MOTION_FRAME:
        raise RuntimeError(
            "H3.2 context mismatch: J_cmd motion_frame "
            f"{jacobian_motion_frame!r} != {H32_MOTION_FRAME!r}"
        )
    if jacobian_motion_unit != H32_MOTION_UNIT:
        raise RuntimeError(
            "H3.2 context mismatch: J_cmd motion_unit "
            f"{jacobian_motion_unit!r} != {H32_MOTION_UNIT!r}"
        )


def validate_frame_image_size(
    image_shape: Sequence[int],
    *,
    expected_image_size: Sequence[int] = H32_IMAGE_SIZE,
) -> None:
    """Reject camera frames whose actual pixel geometry differs from calibration."""
    if len(image_shape) < 2:
        raise ValueError("image_shape must contain at least height and width")
    actual = (int(image_shape[1]), int(image_shape[0]))
    expected = (int(expected_image_size[0]), int(expected_image_size[1]))
    if actual != expected:
        raise RuntimeError(
            "WRIST frame geometry mismatch: actual image size "
            f"{actual} != expected calibration size {expected}"
        )

def _vec2(values: Sequence[float], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain two finite values")
    return array


@dataclass(frozen=True, slots=True)
class ResponseGateConfig:
    """Small first-command image-response gate for each new fixed XY anchor."""

    min_cosine: float = 0.0
    min_magnitude_ratio: float = 0.15
    max_magnitude_ratio: float = 4.0
    max_error_increase_px: float = 3.0
    min_actual_response_px: float = 0.5

    def __post_init__(self) -> None:
        values = (
            self.min_cosine,
            self.min_magnitude_ratio,
            self.max_magnitude_ratio,
            self.max_error_increase_px,
            self.min_actual_response_px,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("response-gate values must be finite")
        if not -1.0 <= self.min_cosine <= 1.0:
            raise ValueError("min_cosine must be in [-1, 1]")
        if self.min_magnitude_ratio < 0.0:
            raise ValueError("min_magnitude_ratio must be >= 0")
        if self.max_magnitude_ratio <= self.min_magnitude_ratio:
            raise ValueError("max_magnitude_ratio must exceed min_magnitude_ratio")
        if self.max_error_increase_px < 0.0:
            raise ValueError("max_error_increase_px must be >= 0")
        if self.min_actual_response_px < 0.0:
            raise ValueError("min_actual_response_px must be >= 0")


@dataclass(frozen=True, slots=True)
class ResponseAssessment:
    accepted: bool
    cosine: float | None
    magnitude_ratio: float | None
    predicted_norm_px: float
    actual_norm_px: float
    error_before_px: float
    error_after_px: float
    reason: str


def assess_image_response(
    predicted_duv: Sequence[float],
    actual_duv: Sequence[float],
    *,
    error_before_px: float,
    error_after_px: float,
    config: ResponseGateConfig | None = None,
) -> ResponseAssessment:
    """Check whether a small first command supports trusting canonical J at this anchor."""
    cfg = config or ResponseGateConfig()
    predicted = _vec2(predicted_duv, name="predicted_duv")
    actual = _vec2(actual_duv, name="actual_duv")
    before = float(error_before_px)
    after = float(error_after_px)
    if not math.isfinite(before) or not math.isfinite(after) or before < 0.0 or after < 0.0:
        raise ValueError("error norms must be finite and non-negative")

    predicted_norm = float(np.linalg.norm(predicted))
    actual_norm = float(np.linalg.norm(actual))
    if predicted_norm <= 1e-9:
        return ResponseAssessment(
            False,
            None,
            None,
            predicted_norm,
            actual_norm,
            before,
            after,
            "predicted response is zero",
        )
    if actual_norm < cfg.min_actual_response_px:
        return ResponseAssessment(
            False,
            None,
            actual_norm / predicted_norm,
            predicted_norm,
            actual_norm,
            before,
            after,
            "actual image response is too small",
        )

    cosine = float(np.dot(predicted, actual) / (predicted_norm * actual_norm))
    ratio = actual_norm / predicted_norm
    if cosine < cfg.min_cosine:
        reason = "actual image response points away from the canonical-J prediction"
        accepted = False
    elif not cfg.min_magnitude_ratio <= ratio <= cfg.max_magnitude_ratio:
        reason = "actual/predicted image-response magnitude ratio is outside the readiness range"
        accepted = False
    elif after > before + cfg.max_error_increase_px:
        reason = "visual error increased too much during readiness correction"
        accepted = False
    else:
        reason = "ready"
        accepted = True

    return ResponseAssessment(
        accepted,
        cosine,
        ratio,
        predicted_norm,
        actual_norm,
        before,
        after,
        reason,
    )


@dataclass(slots=True)
class SegmentReadinessGate:
    """Finite probe/return conditioning state for one fixed XY anchor."""

    max_attempts: int = 3
    attempts_started: int = 0
    accepted: bool = False
    return_required: bool = False
    exhausted_after_return: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int):
            raise TypeError("max_attempts must be an integer")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

    def begin_probe(self) -> int:
        if self.accepted:
            raise RuntimeError("segment readiness is already accepted")
        if self.return_required:
            raise RuntimeError("must return to the fixed anchor before another readiness probe")
        if self.exhausted_after_return or self.attempts_started >= self.max_attempts:
            raise RuntimeError("segment readiness attempts are exhausted")
        self.attempts_started += 1
        return self.attempts_started

    def record_assessment(self, assessment: ResponseAssessment) -> None:
        if self.attempts_started < 1:
            raise RuntimeError("no readiness probe has been started")
        if self.accepted or self.return_required:
            raise RuntimeError("readiness assessment is not currently expected")
        if assessment.accepted:
            self.accepted = True
            return
        self.return_required = True

    def record_return_to_anchor(self) -> None:
        if not self.return_required:
            raise RuntimeError("no readiness return is required")
        self.return_required = False
        if self.attempts_started >= self.max_attempts:
            self.exhausted_after_return = True



def validate_joint_target_step(
    observation: Mapping[str, object],
    planned_action: Mapping[str, object],
    motor_names: Sequence[str],
    *,
    max_relative_target_deg: float,
) -> dict[str, float]:
    """Fail before send_action if a planned arm target would trigger relative clipping.

    LeRobot remains the owner of servo calibration and hardware joint limits.  This
    project-side check only prevents the known ``max_relative_target`` clipping path
    from being discovered after a Cartesian command has already been sent.
    """

    limit = float(max_relative_target_deg)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("max_relative_target_deg must be finite and positive")

    deltas: dict[str, float] = {}
    for name in motor_names:
        if name == "gripper":
            continue
        key = f"{name}.pos"
        if key not in observation or key not in planned_action:
            raise ValueError(f"joint target precheck is missing {key}")
        current = float(observation[key])
        target = float(planned_action[key])
        if not math.isfinite(current) or not math.isfinite(target):
            raise ValueError(f"joint target precheck found non-finite {key}")
        delta = abs(target - current)
        deltas[name] = delta
        if delta > limit + 1e-9:
            raise RuntimeError(
                "Refusing send_action: planned relative joint step "
                f"{name}={delta:.3f} deg exceeds follower max_relative_target "
                f"{limit:.3f} deg"
            )
    return deltas


@dataclass(slots=True)
class StagedDescentGate:
    """Pure state for the Phase-3 small-budget Z primitive."""

    required_stable_bursts: int = 3
    max_z_steps: int = 2
    z_step_mm: float = -0.5
    stable_bursts: int = 0
    z_steps_done: int = 0
    alignment_valid: bool = False

    def __post_init__(self) -> None:
        if self.required_stable_bursts < 1:
            raise ValueError("required_stable_bursts must be >= 1")
        if self.max_z_steps < 1:
            raise ValueError("max_z_steps must be >= 1")
        if not math.isfinite(float(self.z_step_mm)) or self.z_step_mm >= 0.0:
            raise ValueError("z_step_mm must be finite and negative")

    @property
    def complete(self) -> bool:
        return self.z_steps_done >= self.max_z_steps

    @property
    def cumulative_z_mm(self) -> float:
        return self.z_steps_done * float(self.z_step_mm)

    def observe_alignment(self, within_tolerance: bool) -> bool:
        if within_tolerance:
            self.stable_bursts += 1
        else:
            self.stable_bursts = 0
            self.alignment_valid = False
        if self.stable_bursts >= self.required_stable_bursts:
            self.alignment_valid = True
        return self.alignment_valid

    def accept_verified_alignment(self) -> None:
        """Record an alignment routine that already enforced the configured burst count."""
        self.stable_bursts = self.required_stable_bursts
        self.alignment_valid = True

    def authorize_z_step(self) -> None:
        if self.complete:
            raise RuntimeError("Phase-3 Z budget is already complete")
        if not self.alignment_valid:
            raise RuntimeError("Z step requires stable current-level alignment")
        # Every Z step invalidates the alignment that authorized it.
        self.z_steps_done += 1
        self.alignment_valid = False
        self.stable_bursts = 0


@dataclass(frozen=True, slots=True)
class StagedSequenceResult:
    aligned_levels: int
    z_steps_done: int
    cumulative_z_mm: float


def run_staged_sequence(
    gate: StagedDescentGate,
    *,
    align_level: Callable[[int], TAlignment],
    command_z_step: Callable[[int], TZStep],
    on_aligned: Callable[[int, TAlignment], None] | None = None,
    on_z_complete: Callable[[int, TZStep], None] | None = None,
) -> StagedSequenceResult:
    """Execute the Phase-3 align -> Z -> reobserve/realign sequencing contract.

    The callbacks own hardware/perception details.  This function guarantees that
    every Z step is preceded by a completed alignment callback and that the final
    Z step is followed by one more completed alignment callback before success is
    reported.  Exceptions fail closed and stop the sequence immediately.
    """

    level_index = 0
    while True:
        level_index += 1
        alignment = align_level(level_index)
        gate.accept_verified_alignment()
        if on_aligned is not None:
            on_aligned(level_index, alignment)

        if gate.complete:
            return StagedSequenceResult(
                aligned_levels=level_index,
                z_steps_done=gate.z_steps_done,
                cumulative_z_mm=gate.cumulative_z_mm,
            )

        next_step = gate.z_steps_done + 1
        gate.authorize_z_step()
        z_result = command_z_step(next_step)
        if on_z_complete is not None:
            on_z_complete(next_step, z_result)
