from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np

from so101_typing.contracts import CameraFrame
from so101_typing.control.target_measurement import (
    TargetBurstConfig,
    TargetBurstResult,
    robust_target_center,
)


class TargetObservationLike(Protocol):
    target_label: str
    found: bool
    center_px: tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class TargetMeasurementRuntimeConfig:
    """Fresh-frame constraints for one fixed-pose target burst."""

    max_frame_age_ms: float = 100.0
    burst: TargetBurstConfig = TargetBurstConfig()

    def __post_init__(self) -> None:
        max_age = float(self.max_frame_age_ms)

        if not math.isfinite(max_age) or max_age <= 0.0:
            raise ValueError(
                "max_frame_age_ms must be finite and positive"
            )

        object.__setattr__(
            self,
            "max_frame_age_ms",
            max_age,
        )


class TargetMeasurementStatus(StrEnum):
    STALE_FRAME = "stale_frame"
    TARGET_MISSING = "target_missing"
    COLLECTING = "collecting"
    READY = "ready"
    CLUSTER_REJECTED = "cluster_rejected"


_TERMINAL_STATUSES = frozenset(
    {
        TargetMeasurementStatus.READY,
        TargetMeasurementStatus.CLUSTER_REJECTED,
    }
)


@dataclass(frozen=True, slots=True)
class TargetMeasurementUpdate:
    status: TargetMeasurementStatus
    frame_id: int
    frame_age_ms: float
    accepted_observation_count: int
    required_observation_count: int
    result: TargetBurstResult | None = None

    @property
    def terminal(self) -> bool:
        return self.status in _TERMINAL_STATUSES


class TargetBurstAccumulator:
    """Collect one robust target measurement while the robot is fixed.

    This class consumes already-produced CameraFrame and target
    observations.  It never opens a camera and never commands hardware.
    """

    def __init__(
        self,
        *,
        target_label: str,
        camera_name: str = "wrist",
        config: TargetMeasurementRuntimeConfig | None = None,
    ) -> None:
        target = str(target_label).strip().upper()

        if (
            len(target) != 1
            or not target.isalpha()
            or not target.isascii()
        ):
            raise ValueError(
                "target_label must be one ASCII A-Z letter"
            )

        camera = str(camera_name).strip()

        if not camera:
            raise ValueError(
                "camera_name must be nonempty"
            )

        self.target_label = target
        self.camera_name = camera
        self.config = (
            config
            or TargetMeasurementRuntimeConfig()
        )

        self._last_frame_id: int | None = None
        self._last_capture_timestamp: float | None = None
        self._centers: list[tuple[float, float]] = []
        self._terminal_status: TargetMeasurementStatus | None = None

    @property
    def accepted_observation_count(self) -> int:
        return len(self._centers)

    @property
    def terminal_status(
        self,
    ) -> TargetMeasurementStatus | None:
        return self._terminal_status

    def _update(
        self,
        *,
        status: TargetMeasurementStatus,
        frame: CameraFrame,
        frame_age_ms: float,
        result: TargetBurstResult | None = None,
    ) -> TargetMeasurementUpdate:
        return TargetMeasurementUpdate(
            status=status,
            frame_id=int(frame.frame_id),
            frame_age_ms=float(frame_age_ms),
            accepted_observation_count=len(self._centers),
            required_observation_count=(
                self.config.burst.frame_count
            ),
            result=result,
        )

    def process(
        self,
        frame: CameraFrame,
        observation: TargetObservationLike,
        *,
        now_timestamp: float,
    ) -> TargetMeasurementUpdate:
        """Consume one candidate wrist observation."""

        if self._terminal_status is not None:
            raise RuntimeError(
                "target burst accumulator is terminal: "
                f"{self._terminal_status}"
            )

        now = float(now_timestamp)

        if not math.isfinite(now):
            raise ValueError(
                "now_timestamp must be finite"
            )

        capture_timestamp = float(
            frame.capture_timestamp
        )

        processing_timestamp = float(
            frame.processing_timestamp
        )

        if (
            not math.isfinite(capture_timestamp)
            or not math.isfinite(processing_timestamp)
        ):
            raise ValueError(
                "frame timestamps must be finite"
            )

        if capture_timestamp > now:
            raise ValueError(
                "frame capture_timestamp must not be in the future"
            )

        if processing_timestamp < capture_timestamp:
            raise ValueError(
                "frame processing_timestamp must not precede "
                "capture_timestamp"
            )

        if frame.camera_name != self.camera_name:
            raise ValueError(
                f"expected camera {self.camera_name!r}, "
                f"got {frame.camera_name!r}"
            )

        observed_target = str(
            observation.target_label
        ).strip().upper()

        if observed_target != self.target_label:
            raise ValueError(
                f"expected target {self.target_label!r}, "
                f"got observation for {observed_target!r}"
            )

        frame_age_ms = max(
            float(frame.frame_age_ms),
            max(
                0.0,
                (now - capture_timestamp)
                * 1000.0,
            ),
        )

        fresh_identity = (
            (
                self._last_frame_id is None
                or frame.frame_id > self._last_frame_id
            )
            and (
                self._last_capture_timestamp is None
                or capture_timestamp
                > self._last_capture_timestamp
            )
        )

        if (
            not fresh_identity
            or frame_age_ms
            > self.config.max_frame_age_ms
        ):
            return self._update(
                status=TargetMeasurementStatus.STALE_FRAME,
                frame=frame,
                frame_age_ms=frame_age_ms,
            )

        # This fresh frame is now consumed even if the target
        # is missing.  The same physical frame must never be
        # counted later with a different observation.
        self._last_frame_id = int(
            frame.frame_id
        )

        self._last_capture_timestamp = (
            capture_timestamp
        )

        if not observation.found:
            return self._update(
                status=TargetMeasurementStatus.TARGET_MISSING,
                frame=frame,
                frame_age_ms=frame_age_ms,
            )

        if observation.center_px is None:
            raise ValueError(
                "found target observation must provide center_px"
            )

        center = np.asarray(
            observation.center_px,
            dtype=np.float64,
        )

        if center.shape != (2,):
            raise ValueError(
                "observation center_px must contain two values"
            )

        if not np.isfinite(center).all():
            raise ValueError(
                "observation center_px must be finite"
            )

        self._centers.append(
            (
                float(center[0]),
                float(center[1]),
            )
        )

        if (
            len(self._centers)
            < self.config.burst.frame_count
        ):
            return self._update(
                status=TargetMeasurementStatus.COLLECTING,
                frame=frame,
                frame_age_ms=frame_age_ms,
            )

        result = robust_target_center(
            self._centers,
            config=self.config.burst,
        )

        if result.accepted:
            status = TargetMeasurementStatus.READY
        else:
            status = (
                TargetMeasurementStatus.CLUSTER_REJECTED
            )

        self._terminal_status = status

        return self._update(
            status=status,
            frame=frame,
            frame_age_ms=frame_age_ms,
            result=result,
        )
