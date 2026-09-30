from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import phase9_multi_key_typing as phase9


class _DummyPolicy:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return object()


def _fake_phase8():
    phase5 = SimpleNamespace(
        ACTIVE_SCREEN_WATCHER=None,
        raise_if_screen_event=object(),
        wait_motion_stable=object(),
    )
    return SimpleNamespace(
        ThreadedOpenCVCamera=Mock(side_effect=lambda spec: object()),
        SO101Follower=Mock(side_effect=lambda config: object()),
        ACTPolicy=_DummyPolicy,
        make_pre_post_processors=Mock(return_value=(object(), object())),
        _start_runtime_devices=Mock(),
        _stop_runtime_devices=Mock(),
        _restore_canonical_home=Mock(
            return_value={
                "status": "PASS",
                "final_goal_diff_deg": 0.044,
            }
        ),
        _run_full_deterministic_press=Mock(),
        phase5=phase5,
        _action_key=lambda name: f"{name}.pos",
        _wait_initial_frames=Mock(),
        _capture_dynamic_side_snapshot=Mock(
            return_value=({"status": "DYNAMIC_SNAPSHOT"}, 123)
        ),
    )


class Phase9RuntimeReuseTests(unittest.TestCase):
    def test_outcome_mapping_preserves_wrong_and_uncertain(self) -> None:
        self.assertEqual(
            phase9._map_phase8_outcome(
                {"success": False, "outcome": "WRONG_KEY"},
                {"task_status": "FAIL_PRESS_OUTCOME"},
            ),
            phase9.SingleKeyAttemptOutcome.WRONG,
        )
        self.assertEqual(
            phase9._map_phase8_outcome(
                {"success": False, "outcome": "UNCERTAIN_EXHAUSTED"},
                {"task_status": "FAIL_PRESS_OUTCOME"},
            ),
            phase9.SingleKeyAttemptOutcome.UNCERTAIN,
        )

    def test_intermediate_success_defers_home(self) -> None:
        fake = _fake_phase8()
        adapter = phase9._ReusablePhase8Session(fake, total_attempts=3)
        adapter.begin_attempt(0, "C")
        adapter.current_success = True

        result = adapter._restore_canonical_home()

        self.assertEqual(result["status"], "DEFERRED_INTER_KEY_SUCCESS")
        self.assertEqual(adapter.deferred_home_calls, 1)
        self.assertEqual(adapter.real_home_calls, 0)
        fake._restore_canonical_home.assert_not_called()

    def test_final_success_uses_real_home(self) -> None:
        fake = _fake_phase8()
        adapter = phase9._ReusablePhase8Session(fake, total_attempts=3)
        adapter.begin_attempt(2, "T")
        adapter.current_success = True

        result = adapter._restore_canonical_home(
            robot=object(),
            motor_names=[],
            home_baseline={},
            home_stop=object(),
        )

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(adapter.real_home_calls, 1)
        fake._restore_canonical_home.assert_called_once()

    def test_reused_key_gets_fresh_side_baseline_without_reconnect(self) -> None:
        fake = _fake_phase8()
        adapter = phase9._ReusablePhase8Session(fake, total_attempts=3)

        top, wrist, side = Mock(), Mock(), Mock()
        side.latest.return_value = object()
        robot = Mock()
        robot.is_connected = True
        robot.bus.motors = {
            "shoulder_pan": object(),
            "shoulder_lift": object(),
            "elbow_flex": object(),
            "wrist_flex": object(),
            "wrist_roll": object(),
            "gripper": object(),
        }

        adapter.session_started = True
        adapter.camera_objects = [top, wrist, side]
        adapter.robot_object = robot
        adapter.begin_attempt(1, "A")

        home = {key: 0.0 for key in adapter.EXPECTED_KEYS}
        with patch.object(phase9.time, "sleep"):
            motor_names, keys, baseline = adapter._start_runtime_devices(
                top=top,
                wrist=wrist,
                side=side,
                robot=robot,
                home_baseline=home,
                screen_calibration=object(),
                run_dir=Path("artifacts/test_phase9"),
            )

        self.assertEqual(len(motor_names), 6)
        self.assertEqual(keys, adapter.EXPECTED_KEYS)
        self.assertEqual(baseline["status"], "DYNAMIC_SNAPSHOT")
        self.assertEqual(adapter.session_reuse_calls, 1)
        fake._capture_dynamic_side_snapshot.assert_called_once()
        fake._start_runtime_devices.assert_not_called()

    def test_intermediate_teardown_keeps_hardware_session_alive(self) -> None:
        fake = _fake_phase8()
        adapter = phase9._ReusablePhase8Session(fake, total_attempts=3)
        adapter.begin_attempt(0, "C")
        adapter.current_success = True

        top, wrist, side, robot = Mock(), Mock(), Mock(), Mock()
        original_raise = object()
        original_wait = object()

        adapter._stop_runtime_devices(
            top=top,
            wrist=wrist,
            side=side,
            robot=robot,
            original_raise_if_screen_event=original_raise,
            original_wait_motion_stable=original_wait,
            signal_installed=False,
            previous_sigint=object(),
        )

        top.stop.assert_not_called()
        wrist.stop.assert_not_called()
        side.stop.assert_not_called()
        robot.disconnect.assert_not_called()
        fake._stop_runtime_devices.assert_not_called()
        self.assertIs(fake.phase5.raise_if_screen_event, original_raise)
        self.assertIs(fake.phase5.wait_motion_stable, original_wait)
        self.assertEqual(adapter.deferred_teardown_calls, 1)


if __name__ == "__main__":
    unittest.main()
