from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from so101_typing.control.image_jacobian import ImageJacobianCalibration
from so101_typing.control.tool_reference import ToolReferenceCalibration
from so101_typing.control.visual_servo import VisualServoConfig
from so101_typing.control.visual_servo_runtime import (
    VisualServoRuntimeConfig,
    VisualServoRuntimeDecision,
    VisualServoRuntimeStatus,
)

_SCHEMA_VERSION = 1
_TERMINAL_STATUSES = frozenset(
    {
        VisualServoRuntimeStatus.ALIGNED,
        VisualServoRuntimeStatus.TARGET_LOST,
        VisualServoRuntimeStatus.MAX_ITERATIONS,
        VisualServoRuntimeStatus.TIMEOUT,
        VisualServoRuntimeStatus.BUDGET_EXHAUSTED,
    }
)


def _normalize_target_label(value: str) -> str:
    target = str(value).strip().upper()
    if len(target) != 1 or not target.isalpha() or not target.isascii():
        raise ValueError("target_label must be one A-Z letter")
    return target


def _finite_non_negative(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return normalized


def _finite_positive(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return normalized


def _normalize_vector2(
    value: Sequence[float] | None,
    *,
    name: str,
) -> tuple[float, float] | None:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    normalized = (float(value[0]), float(value[1]))
    if not all(math.isfinite(component) for component in normalized):
        raise ValueError(f"{name} must contain only finite values")
    return normalized


def _optional_non_negative(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    return _finite_non_negative(value, name=name)


def _non_negative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _positive_int(value: int, *, name: str) -> int:
    normalized = _non_negative_int(value, name=name)
    if normalized < 1:
        raise ValueError(f"{name} must be at least 1")
    return normalized


@dataclass(frozen=True, slots=True)
class VisualServoAttemptMetrics:
    """Serializable measurements for one complete visual-servo attempt.

    ``initial_error_*`` is the first valid target error observed.  The
    ``final_observed_error_*`` fields contain the latest valid target error,
    which can precede a terminal timeout or target-loss decision.  For an
    ``ALIGNED`` attempt it is the terminal aligned error.
    """

    attempt_id: str
    target_label: str
    terminal_status: VisualServoRuntimeStatus
    decision_count: int
    correction_count: int
    stale_frame_count: int
    iteration_count: int
    elapsed_s: float
    initial_error_px: tuple[float, float] | None
    initial_error_norm_px: float | None
    final_observed_error_px: tuple[float, float] | None
    final_observed_error_norm_px: float | None
    max_frame_age_ms: float
    cumulative_correction: tuple[float, float]
    cumulative_correction_norm: float
    motion_frame: str
    motion_unit: str

    servo_gain: float
    convergence_threshold_px: float
    max_step_norm: float
    max_total_correction_norm: float
    max_frame_age_limit_ms: float
    required_stable_frames: int
    max_iterations: int
    timeout_s: float

    tool_reference_sample_count: int
    tool_reference_std_u_px: float
    tool_reference_std_v_px: float
    jacobian_sample_count: int
    jacobian_residual_rms_px: float
    jacobian_condition_number: float
    jacobian_damping: float

    def __post_init__(self) -> None:
        if not isinstance(self.attempt_id, str) or not self.attempt_id.strip():
            raise ValueError("attempt_id must be a non-empty string")
        object.__setattr__(self, "target_label", _normalize_target_label(self.target_label))

        try:
            status = VisualServoRuntimeStatus(self.terminal_status)
        except ValueError as exc:
            raise ValueError("terminal_status is not a valid visual-servo status") from exc
        if status not in _TERMINAL_STATUSES:
            raise ValueError("terminal_status must be terminal")
        object.__setattr__(self, "terminal_status", status)

        for field_name in (
            "decision_count",
            "correction_count",
            "stale_frame_count",
            "iteration_count",
        ):
            object.__setattr__(
                self,
                field_name,
                _non_negative_int(getattr(self, field_name), name=field_name),
            )
        if self.decision_count < 1:
            raise ValueError("decision_count must be at least 1")
        if self.correction_count != self.iteration_count:
            raise ValueError("correction_count must equal iteration_count")
        if self.stale_frame_count > self.decision_count:
            raise ValueError("stale_frame_count cannot exceed decision_count")

        object.__setattr__(
            self,
            "elapsed_s",
            _finite_non_negative(self.elapsed_s, name="elapsed_s"),
        )
        object.__setattr__(
            self,
            "max_frame_age_ms",
            _finite_non_negative(self.max_frame_age_ms, name="max_frame_age_ms"),
        )

        initial = _normalize_vector2(self.initial_error_px, name="initial_error_px")
        final = _normalize_vector2(
            self.final_observed_error_px,
            name="final_observed_error_px",
        )
        initial_norm = _optional_non_negative(
            self.initial_error_norm_px,
            name="initial_error_norm_px",
        )
        final_norm = _optional_non_negative(
            self.final_observed_error_norm_px,
            name="final_observed_error_norm_px",
        )
        if (initial is None) != (initial_norm is None):
            raise ValueError(
                "initial error vector and norm must either both be present or both be null"
            )
        if (final is None) != (final_norm is None):
            raise ValueError(
                "final error vector and norm must either both be present or both be null"
            )
        if initial is not None and not math.isclose(
            math.hypot(*initial), initial_norm, abs_tol=1e-9
        ):
            raise ValueError("initial_error_norm_px does not match initial_error_px")
        if final is not None and not math.isclose(math.hypot(*final), final_norm, abs_tol=1e-9):
            raise ValueError("final_observed_error_norm_px does not match final_observed_error_px")
        object.__setattr__(self, "initial_error_px", initial)
        object.__setattr__(self, "initial_error_norm_px", initial_norm)
        object.__setattr__(self, "final_observed_error_px", final)
        object.__setattr__(self, "final_observed_error_norm_px", final_norm)

        cumulative = _normalize_vector2(
            self.cumulative_correction,
            name="cumulative_correction",
        )
        if cumulative is None:
            raise ValueError("cumulative_correction must contain exactly two values")
        cumulative_norm = _finite_non_negative(
            self.cumulative_correction_norm,
            name="cumulative_correction_norm",
        )
        if not math.isclose(math.hypot(*cumulative), cumulative_norm, abs_tol=1e-9):
            raise ValueError("cumulative_correction_norm does not match cumulative_correction")
        object.__setattr__(self, "cumulative_correction", cumulative)
        object.__setattr__(self, "cumulative_correction_norm", cumulative_norm)

        if not isinstance(self.motion_frame, str) or not self.motion_frame.strip():
            raise ValueError("motion_frame must be a non-empty string")
        if not isinstance(self.motion_unit, str) or not self.motion_unit.strip():
            raise ValueError("motion_unit must be a non-empty string")

        gain = float(self.servo_gain)
        if not math.isfinite(gain) or not 0.0 < gain <= 1.0:
            raise ValueError("servo_gain must be finite and in (0, 1]")
        object.__setattr__(self, "servo_gain", gain)
        for field_name in (
            "convergence_threshold_px",
            "max_step_norm",
            "max_total_correction_norm",
            "max_frame_age_limit_ms",
            "timeout_s",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_positive(getattr(self, field_name), name=field_name),
            )
        object.__setattr__(
            self,
            "required_stable_frames",
            _positive_int(self.required_stable_frames, name="required_stable_frames"),
        )
        object.__setattr__(
            self,
            "max_iterations",
            _positive_int(self.max_iterations, name="max_iterations"),
        )
        if self.iteration_count > self.max_iterations:
            raise ValueError("iteration_count cannot exceed max_iterations")
        if cumulative_norm > self.max_total_correction_norm + 1e-9:
            raise ValueError("cumulative correction exceeds max_total_correction_norm")
        if status is VisualServoRuntimeStatus.ALIGNED:
            if final_norm is None:
                raise ValueError("aligned attempts must contain a final observed error")
            if final_norm >= self.convergence_threshold_px:
                raise ValueError("aligned final error must be below convergence_threshold_px")

        object.__setattr__(
            self,
            "tool_reference_sample_count",
            _positive_int(
                self.tool_reference_sample_count,
                name="tool_reference_sample_count",
            ),
        )
        object.__setattr__(
            self,
            "jacobian_sample_count",
            _positive_int(self.jacobian_sample_count, name="jacobian_sample_count"),
        )
        if self.jacobian_sample_count < 4:
            raise ValueError("jacobian_sample_count must be at least 4")
        for field_name in (
            "tool_reference_std_u_px",
            "tool_reference_std_v_px",
            "jacobian_residual_rms_px",
            "jacobian_damping",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_non_negative(getattr(self, field_name), name=field_name),
            )
        condition = float(self.jacobian_condition_number)
        if not math.isfinite(condition) or condition < 1.0:
            raise ValueError("jacobian_condition_number must be finite and at least 1")
        object.__setattr__(self, "jacobian_condition_number", condition)

    @property
    def success(self) -> bool:
        return self.terminal_status is VisualServoRuntimeStatus.ALIGNED

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "attempt_id": self.attempt_id,
            "target_label": self.target_label,
            "terminal_status": self.terminal_status.value,
            "success": self.success,
            "decision_count": self.decision_count,
            "correction_count": self.correction_count,
            "stale_frame_count": self.stale_frame_count,
            "iteration_count": self.iteration_count,
            "elapsed_s": self.elapsed_s,
            "initial_error_px": list(self.initial_error_px) if self.initial_error_px else None,
            "initial_error_norm_px": self.initial_error_norm_px,
            "final_observed_error_px": (
                list(self.final_observed_error_px) if self.final_observed_error_px else None
            ),
            "final_observed_error_norm_px": self.final_observed_error_norm_px,
            "max_frame_age_ms": self.max_frame_age_ms,
            "cumulative_correction": list(self.cumulative_correction),
            "cumulative_correction_norm": self.cumulative_correction_norm,
            "motion_frame": self.motion_frame,
            "motion_unit": self.motion_unit,
            "servo_gain": self.servo_gain,
            "convergence_threshold_px": self.convergence_threshold_px,
            "max_step_norm": self.max_step_norm,
            "max_total_correction_norm": self.max_total_correction_norm,
            "max_frame_age_limit_ms": self.max_frame_age_limit_ms,
            "required_stable_frames": self.required_stable_frames,
            "max_iterations": self.max_iterations,
            "timeout_s": self.timeout_s,
            "tool_reference_sample_count": self.tool_reference_sample_count,
            "tool_reference_std_u_px": self.tool_reference_std_u_px,
            "tool_reference_std_v_px": self.tool_reference_std_v_px,
            "jacobian_sample_count": self.jacobian_sample_count,
            "jacobian_residual_rms_px": self.jacobian_residual_rms_px,
            "jacobian_condition_number": self.jacobian_condition_number,
            "jacobian_damping": self.jacobian_damping,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> VisualServoAttemptMetrics:
        if data.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError(f"unsupported metrics schema_version: {data.get('schema_version')!r}")

        payload = dict(data)
        payload.pop("schema_version", None)
        serialized_success = payload.pop("success", None)
        record = cls(**payload)  # type: ignore[arg-type]
        if serialized_success is not None and serialized_success is not record.success:
            raise ValueError("serialized success flag does not match terminal_status")
        return record

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> VisualServoAttemptMetrics:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise TypeError("metrics file must contain a JSON object")
        return cls.from_dict(data)


class VisualServoAttemptRecorder:
    """Collect runtime decisions and freeze one terminal attempt into metrics."""

    def __init__(
        self,
        *,
        attempt_id: str,
        target_label: str,
        tool_reference: ToolReferenceCalibration,
        image_jacobian: ImageJacobianCalibration,
        servo_config: VisualServoConfig,
        runtime_config: VisualServoRuntimeConfig,
    ) -> None:
        if not isinstance(attempt_id, str) or not attempt_id.strip():
            raise ValueError("attempt_id must be a non-empty string")
        self.attempt_id = attempt_id
        self.target_label = _normalize_target_label(target_label)
        self.tool_reference = tool_reference
        self.image_jacobian = image_jacobian
        self.servo_config = servo_config
        self.runtime_config = runtime_config

        self._decision_count = 0
        self._correction_count = 0
        self._stale_frame_count = 0
        self._max_frame_age_ms = 0.0
        self._initial_error_px: tuple[float, float] | None = None
        self._initial_error_norm_px: float | None = None
        self._final_error_px: tuple[float, float] | None = None
        self._final_error_norm_px: float | None = None
        self._terminal_decision: VisualServoRuntimeDecision | None = None

    @property
    def terminal(self) -> bool:
        return self._terminal_decision is not None

    def record(self, decision: VisualServoRuntimeDecision) -> None:
        if self._terminal_decision is not None:
            raise RuntimeError("cannot record after a terminal decision")
        if decision.motion_frame != self.image_jacobian.motion_frame:
            raise ValueError("decision motion_frame does not match image Jacobian calibration")
        if decision.motion_unit != self.image_jacobian.motion_unit:
            raise ValueError("decision motion_unit does not match image Jacobian calibration")
        if decision.required_stable_frames != self.runtime_config.required_stable_frames:
            raise ValueError("decision required_stable_frames does not match runtime config")

        frame_age_ms = _finite_non_negative(
            decision.frame_age_ms,
            name="decision.frame_age_ms",
        )
        _finite_non_negative(decision.elapsed_s, name="decision.elapsed_s")

        correction_increment = int(decision.status is VisualServoRuntimeStatus.CORRECTION)
        stale_increment = int(decision.status is VisualServoRuntimeStatus.STALE_FRAME)
        next_correction_count = self._correction_count + correction_increment
        if decision.iteration_count != next_correction_count:
            raise ValueError(
                "decision iteration_count does not match corrections recorded from attempt start"
            )

        error: tuple[float, float] | None = None
        error_norm: float | None = None
        if decision.error_px is not None:
            error = _normalize_vector2(decision.error_px, name="decision.error_px")
            if decision.error_norm_px is None:
                raise ValueError(
                    "decision error vector and norm must either both be present or both be null"
                )
            error_norm = _finite_non_negative(
                decision.error_norm_px,
                name="decision.error_norm_px",
            )
            if error is None:
                raise RuntimeError("normalized decision error unexpectedly missing")
            if not math.isclose(math.hypot(*error), error_norm, abs_tol=1e-9):
                raise ValueError("decision error norm does not match error vector")
        elif decision.error_norm_px is not None:
            raise ValueError(
                "decision error vector and norm must either both be present or both be null"
            )

        self._decision_count += 1
        self._correction_count = next_correction_count
        self._stale_frame_count += stale_increment
        self._max_frame_age_ms = max(self._max_frame_age_ms, frame_age_ms)

        if error is not None:
            if self._initial_error_px is None:
                self._initial_error_px = error
                self._initial_error_norm_px = error_norm
            self._final_error_px = error
            self._final_error_norm_px = error_norm

        if decision.terminal:
            self._terminal_decision = decision

    def finalize(self) -> VisualServoAttemptMetrics:
        terminal = self._terminal_decision
        if terminal is None:
            raise RuntimeError("cannot finalize before a terminal decision")

        cumulative_norm = math.hypot(*terminal.cumulative_correction)
        return VisualServoAttemptMetrics(
            attempt_id=self.attempt_id,
            target_label=self.target_label,
            terminal_status=terminal.status,
            decision_count=self._decision_count,
            correction_count=self._correction_count,
            stale_frame_count=self._stale_frame_count,
            iteration_count=terminal.iteration_count,
            elapsed_s=terminal.elapsed_s,
            initial_error_px=self._initial_error_px,
            initial_error_norm_px=self._initial_error_norm_px,
            final_observed_error_px=self._final_error_px,
            final_observed_error_norm_px=self._final_error_norm_px,
            max_frame_age_ms=self._max_frame_age_ms,
            cumulative_correction=terminal.cumulative_correction,
            cumulative_correction_norm=cumulative_norm,
            motion_frame=terminal.motion_frame,
            motion_unit=terminal.motion_unit,
            servo_gain=self.servo_config.gain,
            convergence_threshold_px=self.servo_config.convergence_threshold_px,
            max_step_norm=self.servo_config.max_step_norm,
            max_total_correction_norm=self.servo_config.max_total_correction_norm,
            max_frame_age_limit_ms=self.runtime_config.max_frame_age_ms,
            required_stable_frames=self.runtime_config.required_stable_frames,
            max_iterations=self.runtime_config.max_iterations,
            timeout_s=self.runtime_config.timeout_s,
            tool_reference_sample_count=self.tool_reference.sample_count,
            tool_reference_std_u_px=self.tool_reference.std_u_px,
            tool_reference_std_v_px=self.tool_reference.std_v_px,
            jacobian_sample_count=self.image_jacobian.sample_count,
            jacobian_residual_rms_px=self.image_jacobian.residual_rms_px,
            jacobian_condition_number=self.image_jacobian.condition_number,
            jacobian_damping=self.image_jacobian.damping,
        )


@dataclass(frozen=True, slots=True)
class VisualServoEvaluationSummary:
    attempt_count: int
    aligned_count: int
    convergence_rate: float
    failure_rate: float
    timeout_rate: float
    target_loss_rate: float
    mean_initial_error_px: float | None
    mean_final_error_px_aligned: float | None
    max_final_error_px_aligned: float | None
    mean_iterations_aligned: float | None
    mean_convergence_time_s_aligned: float | None


def summarize_visual_servo_attempts(
    attempts: Iterable[VisualServoAttemptMetrics],
) -> VisualServoEvaluationSummary:
    records = list(attempts)
    if not records:
        raise ValueError("at least one visual-servo attempt is required")

    aligned = [record for record in records if record.success]
    attempts_with_initial_error = [
        record for record in records if record.initial_error_norm_px is not None
    ]
    attempt_count = len(records)
    aligned_count = len(aligned)
    timeout_count = sum(
        record.terminal_status is VisualServoRuntimeStatus.TIMEOUT for record in records
    )
    target_loss_count = sum(
        record.terminal_status is VisualServoRuntimeStatus.TARGET_LOST for record in records
    )

    if attempts_with_initial_error:
        mean_initial_error = sum(
            record.initial_error_norm_px
            for record in attempts_with_initial_error
            if record.initial_error_norm_px is not None
        ) / len(attempts_with_initial_error)
    else:
        mean_initial_error = None

    if aligned:
        final_errors = [
            record.final_observed_error_norm_px
            for record in aligned
            if record.final_observed_error_norm_px is not None
        ]
        if len(final_errors) != len(aligned):
            raise ValueError("aligned attempts must contain final observed error metrics")
        mean_final_error = sum(final_errors) / len(final_errors)
        max_final_error = max(final_errors)
        mean_iterations = sum(record.iteration_count for record in aligned) / aligned_count
        mean_time = sum(record.elapsed_s for record in aligned) / aligned_count
    else:
        mean_final_error = None
        max_final_error = None
        mean_iterations = None
        mean_time = None

    return VisualServoEvaluationSummary(
        attempt_count=attempt_count,
        aligned_count=aligned_count,
        convergence_rate=aligned_count / attempt_count,
        failure_rate=(attempt_count - aligned_count) / attempt_count,
        timeout_rate=timeout_count / attempt_count,
        target_loss_rate=target_loss_count / attempt_count,
        mean_initial_error_px=mean_initial_error,
        mean_final_error_px_aligned=mean_final_error,
        max_final_error_px_aligned=max_final_error,
        mean_iterations_aligned=mean_iterations,
        mean_convergence_time_s_aligned=mean_time,
    )
