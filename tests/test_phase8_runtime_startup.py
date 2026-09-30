from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import phase8_single_key_integration as phase8


EXPECTED_MOTORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _home_baseline() -> dict[str, float]:
    return {f"{name}.pos": 0.0 for name in EXPECTED_MOTORS}


class Phase8RuntimeStartupTests(unittest.TestCase):
    def _devices(self, motor_names=EXPECTED_MOTORS):
        top = Mock()
        wrist = Mock()
        side = Mock()
        side.latest.return_value = object()

        robot = Mock()
        robot.bus.motors = {name: object() for name in motor_names}
        return top, wrist, side, robot

    def test_start_runtime_devices_preserves_phase8_startup_contract(self) -> None:
        top, wrist, side, robot = self._devices()
        baseline = {"baseline": True}

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            with (
                patch.object(phase8, "_wait_initial_frames") as wait_frames,
                patch.object(
                    phase8,
                    "_capture_dynamic_side_snapshot",
                    return_value=(baseline, {"ignored": True}),
                ) as capture_baseline,
                patch.object(phase8.time, "sleep") as sleep,
            ):
                motor_names, keys, screen_baseline = phase8._start_runtime_devices(
                    top=top,
                    wrist=wrist,
                    side=side,
                    robot=robot,
                    home_baseline=_home_baseline(),
                    screen_calibration=object(),
                    run_dir=run_dir,
                )

        top.start.assert_called_once_with()
        wrist.start.assert_called_once_with()
        side.start.assert_called_once_with()
        wait_frames.assert_called_once_with(top, wrist)
        sleep.assert_called_once_with(0.5)
        robot.connect.assert_called_once_with()
        capture_baseline.assert_called_once()
        self.assertEqual(motor_names, EXPECTED_MOTORS)
        self.assertEqual(keys, [f"{name}.pos" for name in EXPECTED_MOTORS])
        self.assertIs(screen_baseline, baseline)

    def test_missing_home_key_is_rejected(self) -> None:
        top, wrist, side, robot = self._devices()
        home = _home_baseline()
        del home["gripper.pos"]

        with tempfile.TemporaryDirectory() as td:
            with (
                patch.object(phase8, "_wait_initial_frames"),
                patch.object(
                    phase8,
                    "_capture_dynamic_side_snapshot",
                    return_value=({"baseline": True}, None),
                ),
                patch.object(phase8.time, "sleep"),
            ):
                with self.assertRaisesRegex(RuntimeError, "Recovery HOME missing keys"):
                    phase8._start_runtime_devices(
                        top=top,
                        wrist=wrist,
                        side=side,
                        robot=robot,
                        home_baseline=home,
                        screen_calibration=object(),
                        run_dir=Path(td),
                    )

    def test_motor_order_mismatch_is_rejected(self) -> None:
        bad_order = EXPECTED_MOTORS.copy()
        bad_order[0], bad_order[1] = bad_order[1], bad_order[0]
        top, wrist, side, robot = self._devices(bad_order)

        with tempfile.TemporaryDirectory() as td:
            with (
                patch.object(phase8, "_wait_initial_frames"),
                patch.object(
                    phase8,
                    "_capture_dynamic_side_snapshot",
                    return_value=({"baseline": True}, None),
                ),
                patch.object(phase8.time, "sleep"),
            ):
                with self.assertRaisesRegex(RuntimeError, "motor order"):
                    phase8._start_runtime_devices(
                        top=top,
                        wrist=wrist,
                        side=side,
                        robot=robot,
                        home_baseline=_home_baseline(),
                        screen_calibration=object(),
                        run_dir=Path(td),
                    )


if __name__ == "__main__":
    unittest.main()
