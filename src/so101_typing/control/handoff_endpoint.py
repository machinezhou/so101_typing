from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import cv2
import numpy as np


SCHEMA = "phase8.handoff_endpoint_contract.v1"


@dataclass(frozen=True, slots=True)
class HandoffEndpointEvaluation:
    error_px: tuple[float, float]
    signed_distance_to_hull_px: float
    acceptance_clearance_px: float
    accepted: bool


@dataclass(frozen=True, slots=True)
class HandoffEndpointContract:
    hull_vertices_error_px: tuple[tuple[float, float], ...]
    acceptance_margin_px: float
    required_consecutive_frames: int
    verified_episode_count: int
    reliable_endpoint_count: int
    measurement_repeatability_p95_px: float

    @classmethod
    def load(cls, path: str | Path) -> "HandoffEndpointContract":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        if payload.get("schema") != SCHEMA:
            raise ValueError(
                f"Unexpected handoff endpoint schema: "
                f"{payload.get('schema')!r}"
            )

        raw_vertices = payload.get("hull_vertices_error_px")
        if not isinstance(raw_vertices, list) or len(raw_vertices) < 3:
            raise ValueError(
                "handoff endpoint hull must contain at least 3 vertices"
            )

        vertices: list[tuple[float, float]] = []

        for raw in raw_vertices:
            if not isinstance(raw, list) or len(raw) != 2:
                raise ValueError(
                    "every handoff hull vertex must contain exactly x,y"
                )

            x = float(raw[0])
            y = float(raw[1])

            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(
                    "handoff hull vertices must be finite"
                )

            vertices.append((x, y))

        contour = np.asarray(
            vertices,
            dtype=np.float32,
        ).reshape(-1, 1, 2)

        if abs(float(cv2.contourArea(contour))) <= 1e-6:
            raise ValueError("handoff endpoint hull has zero area")

        if not bool(cv2.isContourConvex(contour)):
            raise ValueError(
                "handoff endpoint vertices must form a convex hull"
            )

        margin = float(payload["acceptance_margin_px"])
        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError(
                "acceptance_margin_px must be finite and >= 0"
            )

        required = int(payload["required_consecutive_frames"])
        if required < 1:
            raise ValueError(
                "required_consecutive_frames must be >= 1"
            )

        repeatability = float(
            payload["measurement_repeatability_p95_px"]
        )
        if not math.isfinite(repeatability) or repeatability < 0.0:
            raise ValueError(
                "measurement_repeatability_p95_px must be finite and >= 0"
            )

        verified_count = int(payload["verified_episode_count"])
        reliable_count = int(payload["reliable_endpoint_count"])

        if verified_count <= 0:
            raise ValueError("verified_episode_count must be positive")

        if reliable_count <= 0 or reliable_count > verified_count:
            raise ValueError(
                "reliable_endpoint_count must be in "
                "[1, verified_episode_count]"
            )

        return cls(
            hull_vertices_error_px=tuple(vertices),
            acceptance_margin_px=margin,
            required_consecutive_frames=required,
            verified_episode_count=verified_count,
            reliable_endpoint_count=reliable_count,
            measurement_repeatability_p95_px=repeatability,
        )

    def _contour(self) -> np.ndarray:
        return np.asarray(
            self.hull_vertices_error_px,
            dtype=np.float32,
        ).reshape(-1, 1, 2)

    def evaluate(
        self,
        error_px: tuple[float, float] | np.ndarray,
    ) -> HandoffEndpointEvaluation:
        error = np.asarray(
            error_px,
            dtype=np.float64,
        )

        if error.shape != (2,) or not np.isfinite(error).all():
            raise ValueError(
                "error_px must contain exactly two finite values"
            )

        x = float(error[0])
        y = float(error[1])

        # OpenCV convention:
        #   positive = inside raw verified hull
        #   zero     = on hull
        #   negative = outside hull
        signed_distance = float(
            cv2.pointPolygonTest(
                self._contour(),
                (x, y),
                True,
            )
        )

        # The accepted region is the empirical hull expanded outward
        # by the measured P95 repeatability margin.
        clearance = (
            signed_distance
            + float(self.acceptance_margin_px)
        )

        return HandoffEndpointEvaluation(
            error_px=(x, y),
            signed_distance_to_hull_px=signed_distance,
            acceptance_clearance_px=float(clearance),
            accepted=bool(clearance >= -1e-6),
        )



@dataclass(frozen=True, slots=True)
class HandoffFrameConfirmation:
    """Track consecutive eligible observations on distinct WRIST frames.

    Re-reading the same camera frame is a no-op. A fresh ineligible frame
    resets the streak. This state machine contains no perception or robot I/O
    so the ownership boundary can be regression-tested directly.
    """

    last_frame_id: int | None = None
    streak: int = 0

    def observe(
        self,
        frame_id: int,
        *,
        eligible: bool,
    ) -> tuple["HandoffFrameConfirmation", bool]:
        frame_id = int(frame_id)

        if frame_id < 0:
            raise ValueError("frame_id must be >= 0")

        if (
            self.last_frame_id is not None
            and frame_id < self.last_frame_id
        ):
            raise ValueError(
                "WRIST frame_id moved backwards: "
                f"{frame_id} < {self.last_frame_id}"
            )

        if self.last_frame_id == frame_id:
            return self, False

        next_streak = (
            self.streak + 1
            if bool(eligible)
            else 0
        )

        return (
            HandoffFrameConfirmation(
                last_frame_id=frame_id,
                streak=next_streak,
            ),
            True,
        )

    def confirmed(self, required_frames: int) -> bool:
        required_frames = int(required_frames)

        if required_frames < 1:
            raise ValueError(
                "required_frames must be >= 1"
            )

        return self.streak >= required_frames
