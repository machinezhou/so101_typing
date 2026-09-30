from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import validate_phase5_single_key as phase5


@dataclass
class _State:
    x_mm: float = 0.0
    y_mm: float = 0.0
    z_mm: float = -4.0
    max_xy_norm_mm: float = 80.0
    max_xyz_norm_mm: float = 80.0

    @property
    def xy_mm(self):
        return (self.x_mm, self.y_mm)


@dataclass
class _Observation:
    center_px: tuple[float, float]
    source: str = "geometry"


@dataclass
class _Frame:
    frame_id: int


class _ToolReference:
    def error_px(self, center_px):
        return center_px


class _Jacobian:
    motion_frame = "tool_xy"
    motion_unit = "mm"


class Phase5WristOutlierGuardTests(unittest.TestCase):
    def _run(self, centers):
        calls = {"count": 0}

        def fake_capture_locked_target(*args, **kwargs):
            del args, kwargs
            index = calls["count"]
            calls["count"] += 1
            center = centers[index]
            return (
                _Observation(center_px=center),
                _Frame(frame_id=index + 1),
            )

        with patch.object(
            phase5,
            "capture_locked_target",
            side_effect=fake_capture_locked_target,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "WRIST_TRACK_LOST",
            ):
                phase5.align_wrist(
                    robot=object(),
                    wrist_camera=object(),
                    recognizer=object(),
                    semantic_lock=object(),
                    tool_reference=_ToolReference(),
                    jacobian=_Jacobian(),
                    planner=object(),
                    state=_State(),
                    controller=object(),
                    motor_names=[],
                    last_frame_id=0,
                    min_timestamp=0.0,
                    threshold_px=6.0,
                    max_commands=None,
                    sent_state_tracker=object(),
                )

        return calls["count"]

    def test_five_consecutive_geometry_outliers_fail_closed(self):
        count = self._run(
            [(100.0, 0.0)] * phase5.MAX_CONSECUTIVE_WRIST_OUTLIERS
        )
        self.assertEqual(
            count,
            phase5.MAX_CONSECUTIVE_WRIST_OUTLIERS,
        )

    def test_non_outlier_resets_consecutive_streak(self):
        centers = (
            [(100.0, 0.0)] * 4
            + [(0.0, 0.0)]
            + [(100.0, 0.0)] * phase5.MAX_CONSECUTIVE_WRIST_OUTLIERS
        )
        count = self._run(centers)
        self.assertEqual(
            count,
            4 + 1 + phase5.MAX_CONSECUTIVE_WRIST_OUTLIERS,
        )


if __name__ == "__main__":
    unittest.main()
