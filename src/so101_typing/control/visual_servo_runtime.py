from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from so101_typing.contracts import CameraFrame
from so101_typing.control.image_jacobian import ImageJacobianCalibration
from so101_typing.control.tool_reference import ToolReferenceCalibration
from so101_typing.control.visual_servo import (
    VisualServoConfig,
    VisualServoStepStatus,
    compute_visual_servo_step,
)


class TargetObservationLike(Protocol):
    """Minimal perception contract required by the servo runtime."""

    target_label: str
    found: bool
    center_px: tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class VisualServoRuntimeConfig:
    """Freshness, stability, and attempt bounds for one servo alignment run."""

    max_frame_age_ms: float = 100.0
    required_stable_frames: int = 3
    max_iterations: int = 20
    timeout_s: float = 5.0

    def __post_init__(self) -> None:
        max_age = float(self.max_frame_age_ms)
        if not math.isfinite(max_age) or max_age <= 0.0:
            raise ValueError("max_frame_age_ms must be finite and positive")
        object.__setattr__(self, "max_frame_age_ms", max_age)

        if isinstance(self.required_stable_frames, bool) or not isinstance(
            self.required_stable_frames, int
        ):
            raise TypeError("required_stable_frames must be an integer")
        if self.required_stable_frames < 1:
            raise ValueError("required_stable_frames must be at least 1")

        if isinstance(self.max_iterations, bool) or not isinstance(self.max_iterations, int):
            raise TypeError("max_iterations must be an integer")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

        timeout = float(self.timeout_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        object.__setattr__(self, "timeout_s", timeout)


class VisualServoRuntimeStatus(StrEnum):
    CORRECTION = "correction"
    STABILIZING = "stabilizing"
    ALIGNED = "aligned"
    STALE_FRAME = "stale_frame"
    TARGET_LOST = "target_lost"
    MAX_ITERATIONS = "max_iterations"
    TIMEOUT = "timeout"
    BUDGET_EXHAUSTED = "budget_exhausted"


_TERMINAL_STATUSES = frozenset(
    {
        VisualServoRuntimeStatus.ALIGNED,
        VisualServoRuntimeStatus.TARGET_LOST,
        VisualServoRuntimeStatus.MAX_ITERATIONS,
        VisualServoRuntimeStatus.TIMEOUT,
        VisualServoRuntimeStatus.BUDGET_EXHAUSTED,
    }
)


@dataclass(frozen=True, slots=True)
class VisualServoRuntimeDecision:
    status: VisualServoRuntimeStatus
    frame_id: int
    frame_age_ms: float
    elapsed_s: float
    error_px: tuple[float, float] | None
    error_norm_px: float | None
    correction: tuple[float, float]
    correction_norm: float
    iteration_count: int
    stable_frame_count: int
    required_stable_frames: int
    cumulative_correction: tuple[float, float]
    motion_frame: str
    motion_unit: str

    @property
    def terminal(self) -> bool:
        return self.status in _TERMINAL_STATUSES


class VisualServoRuntime:
    """State machine around the pure visual-servo step computation.

    This class never commands robot motion.  It only decides whether a fresh
    wrist observation may produce an XY correction and when alignment has
    accumulated enough consecutive fresh in-tolerance observations.
    """

    def __init__(
        self,
        *,
        target_label: str,
        tool_reference: ToolReferenceCalibration,
        image_jacobian: ImageJacobianCalibration,
        servo_config: VisualServoConfig | None = None,
        runtime_config: VisualServoRuntimeConfig | None = None,
        start_timestamp: float | None = None,
    ) -> None:
        target = str(target_label).strip().upper()
        if len(target) != 1 or not target.isalpha() or not target.isascii():
            raise ValueError("target_label must be one A-Z letter")
        if tool_reference.camera_name != "wrist":
            raise ValueError("tool reference must be calibrated for the wrist camera")

        self.target_label = target
        self.tool_reference = tool_reference
        self.image_jacobian = image_jacobian
        self.servo_config = servo_config or VisualServoConfig()
        self.runtime_config = runtime_config or VisualServoRuntimeConfig()

        start = time.monotonic() if start_timestamp is None else float(start_timestamp)
        if not math.isfinite(start):
            raise ValueError("start_timestamp must be finite")
        self._start_timestamp = start
        self._last_frame_id: int | None = None
        self._last_capture_timestamp: float | None = None
        self._iteration_count = 0
        self._stable_frame_count = 0
        self._cumulative_correction = (0.0, 0.0)
        self._terminal_status: VisualServoRuntimeStatus | None = None

    @property
    def iteration_count(self) -> int:
        return self._iteration_count

    @property
    def stable_frame_count(self) -> int:
        return self._stable_frame_count

    @property
    def cumulative_correction(self) -> tuple[float, float]:
        return self._cumulative_correction

    @property
    def terminal_status(self) -> VisualServoRuntimeStatus | None:
        return self._terminal_status

    def _decision(
        self,
        *,
        status: VisualServoRuntimeStatus,
        frame: CameraFrame,
        frame_age_ms: float,
        elapsed_s: float,
        error_px: tuple[float, float] | None = None,
        error_norm_px: float | None = None,
        correction: tuple[float, float] = (0.0, 0.0),
        correction_norm: float = 0.0,
    ) -> VisualServoRuntimeDecision:
        return VisualServoRuntimeDecision(
            status=status,
            frame_id=int(frame.frame_id),
            frame_age_ms=frame_age_ms,
            elapsed_s=elapsed_s,
            error_px=error_px,
            error_norm_px=error_norm_px,
            correction=correction,
            correction_norm=correction_norm,
            iteration_count=self._iteration_count,
            stable_frame_count=self._stable_frame_count,
            required_stable_frames=self.runtime_config.required_stable_frames,
            cumulative_correction=self._cumulative_correction,
            motion_frame=self.image_jacobian.motion_frame,
            motion_unit=self.image_jacobian.motion_unit,
        )

    def _latch_terminal(self, status: VisualServoRuntimeStatus) -> None:
        if status not in _TERMINAL_STATUSES:
            raise ValueError("only terminal statuses may be latched")
        self._terminal_status = status

    def process(
        self,
        frame: CameraFrame,
        observation: TargetObservationLike,
        *,
        now_timestamp: float | None = None,
    ) -> VisualServoRuntimeDecision:
        """Evaluate one wrist frame without sending any robot command."""
        if self._terminal_status is not None:
            raise RuntimeError(f"visual servo runtime is terminal: {self._terminal_status}")

        now = time.monotonic() if now_timestamp is None else float(now_timestamp)
        if not math.isfinite(now):
            raise ValueError("now_timestamp must be finite")
        if now < self._start_timestamp:
            raise ValueError("now_timestamp must not precede start_timestamp")

        capture_timestamp = float(frame.capture_timestamp)
        processing_timestamp = float(frame.processing_timestamp)
        if not math.isfinite(capture_timestamp) or not math.isfinite(processing_timestamp):
            raise ValueError("frame timestamps must be finite")
        if capture_timestamp > now:
            raise ValueError("frame capture_timestamp must not be in the future")
        if processing_timestamp < capture_timestamp:
            raise ValueError("frame processing_timestamp must not precede capture_timestamp")
        if frame.camera_name != self.tool_reference.camera_name:
            raise ValueError(
                f"expected camera {self.tool_reference.camera_name!r}, got {frame.camera_name!r}"
            )

        observed_target = str(observation.target_label).strip().upper()
        if observed_target != self.target_label:
            raise ValueError(
                f"expected target {self.target_label!r}, got observation for {observed_target!r}"
            )

        elapsed_s = now - self._start_timestamp
        frame_age_ms = max(
            frame.frame_age_ms,
            max(0.0, (now - capture_timestamp) * 1000.0),
        )

        if elapsed_s >= self.runtime_config.timeout_s:
            self._stable_frame_count = 0
            self._latch_terminal(VisualServoRuntimeStatus.TIMEOUT)
            return self._decision(
                status=VisualServoRuntimeStatus.TIMEOUT,
                frame=frame,
                frame_age_ms=frame_age_ms,
                elapsed_s=elapsed_s,
            )

        fresh_identity = (
            (self._last_frame_id is None or frame.frame_id > self._last_frame_id)
            and (
                self._last_capture_timestamp is None
                or capture_timestamp > self._last_capture_timestamp
            )
        )
        if not fresh_identity or frame_age_ms > self.runtime_config.max_frame_age_ms:
            self._stable_frame_count = 0
            return self._decision(
                status=VisualServoRuntimeStatus.STALE_FRAME,
                frame=frame,
                frame_age_ms=frame_age_ms,
                elapsed_s=elapsed_s,
            )

        self._last_frame_id = int(frame.frame_id)
        self._last_capture_timestamp = capture_timestamp

        if not observation.found:
            self._stable_frame_count = 0
            self._latch_terminal(VisualServoRuntimeStatus.TARGET_LOST)
            return self._decision(
                status=VisualServoRuntimeStatus.TARGET_LOST,
                frame=frame,
                frame_age_ms=frame_age_ms,
                elapsed_s=elapsed_s,
            )

        if observation.center_px is None:
            raise ValueError("found target observation must provide center_px")

        error_px = self.tool_reference.error_px(observation.center_px)
        error_norm_px = math.hypot(*error_px)

        if error_norm_px < self.servo_config.convergence_threshold_px:
            self._stable_frame_count += 1
            if self._stable_frame_count >= self.runtime_config.required_stable_frames:
                self._latch_terminal(VisualServoRuntimeStatus.ALIGNED)
                status = VisualServoRuntimeStatus.ALIGNED
            else:
                status = VisualServoRuntimeStatus.STABILIZING
            return self._decision(
                status=status,
                frame=frame,
                frame_age_ms=frame_age_ms,
                elapsed_s=elapsed_s,
                error_px=error_px,
                error_norm_px=error_norm_px,
            )

        self._stable_frame_count = 0
        if self._iteration_count >= self.runtime_config.max_iterations:
            self._latch_terminal(VisualServoRuntimeStatus.MAX_ITERATIONS)
            return self._decision(
                status=VisualServoRuntimeStatus.MAX_ITERATIONS,
                frame=frame,
                frame_age_ms=frame_age_ms,
                elapsed_s=elapsed_s,
                error_px=error_px,
                error_norm_px=error_norm_px,
            )

        step = compute_visual_servo_step(
            error_px,
            self.image_jacobian,
            config=self.servo_config,
            cumulative_correction=self._cumulative_correction,
        )
        if step.status is VisualServoStepStatus.BUDGET_EXHAUSTED:
            self._latch_terminal(VisualServoRuntimeStatus.BUDGET_EXHAUSTED)
            return self._decision(
                status=VisualServoRuntimeStatus.BUDGET_EXHAUSTED,
                frame=frame,
                frame_age_ms=frame_age_ms,
                elapsed_s=elapsed_s,
                error_px=error_px,
                error_norm_px=error_norm_px,
            )
        if step.status is not VisualServoStepStatus.CORRECTION:
            raise RuntimeError(f"unexpected visual-servo step status: {step.status}")

        self._iteration_count += 1
        self._cumulative_correction = step.cumulative_after
        return self._decision(
            status=VisualServoRuntimeStatus.CORRECTION,
            frame=frame,
            frame_age_ms=frame_age_ms,
            elapsed_s=elapsed_s,
            error_px=step.error_px,
            error_norm_px=step.error_norm_px,
            correction=step.correction,
            correction_norm=step.correction_norm,
        )
