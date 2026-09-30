from __future__ import annotations

from pathlib import Path
import sys
import unittest
from threading import Event
from unittest.mock import patch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import phase8_single_key_integration as phase8


class Phase8HomeRecoveryTests(unittest.TestCase):
    def test_restore_canonical_home_preserves_accepted_sequence(self) -> None:
        robot = object()
        motor_names = ["m0", "m1", "m2", "m3", "m4", "m5"]
        baseline = {"m0.pos": 1.0}
        home_stop = Event()

        with (
            patch.object(
                phase8,
                "_synchronize_goal_to_present",
                return_value={"hold": True},
            ) as synchronize,
            patch.object(
                phase8,
                "_auto_restore_recorded_start",
                return_value={"restore": True},
            ) as restore,
            patch.object(
                phase8,
                "_verify_home",
                return_value=0.044,
            ) as verify,
        ):
            result = phase8._restore_canonical_home(
                robot=robot,
                motor_names=motor_names,
                home_baseline=baseline,
                home_stop=home_stop,
            )

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["present_hold"], {"hold": True})
        self.assertEqual(result["restore"], {"restore": True})
        self.assertAlmostEqual(result["final_goal_diff_deg"], 0.044)

        synchronize.assert_called_once_with(
            robot=robot,
            motor_names=motor_names,
            max_sent_diff_deg=phase8.STARTUP_SENT_DIFF_DEG,
        )
        restore.assert_called_once_with(
            robot=robot,
            baseline=baseline,
            motor_names=motor_names,
            step_deg=phase8.HOME_STEP_DEG,
            max_sent_diff_deg=phase8.STARTUP_SENT_DIFF_DEG,
            label="RECOVERY HOME",
        )
        verify.assert_called_once_with(
            robot=robot,
            motor_names=motor_names,
            home_baseline=baseline,
            max_diff_deg=phase8.STARTUP_SENT_DIFF_DEG,
        )

    def test_operator_stop_after_goal_sync_blocks_home_motion(self) -> None:
        robot = object()
        motor_names = ["m0", "m1", "m2", "m3", "m4", "m5"]
        baseline = {"m0.pos": 1.0}
        home_stop = Event()

        def synchronize_and_stop(**kwargs):
            del kwargs
            home_stop.set()
            return {"hold": True}

        with (
            patch.object(
                phase8,
                "_synchronize_goal_to_present",
                side_effect=synchronize_and_stop,
            ),
            patch.object(
                phase8,
                "_auto_restore_recorded_start",
            ) as restore,
            patch.object(
                phase8,
                "_verify_home",
            ) as verify,
        ):
            with self.assertRaises(phase8.HomeStop):
                phase8._restore_canonical_home(
                    robot=robot,
                    motor_names=motor_names,
                    home_baseline=baseline,
                    home_stop=home_stop,
                )

        restore.assert_not_called()
        verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
