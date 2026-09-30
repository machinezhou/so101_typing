from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import phase8_single_key_integration as phase8
from so101_typing.control.fixed_anchor_planner import (
    RelativeJointSafetyError,
)


@dataclass(frozen=True, slots=True)
class FakeState:
    z_mm: float
    x_mm: float = 0.0
    y_mm: float = 0.0

    @property
    def xyz_mm(self):
        return (
            float(self.x_mm),
            float(self.y_mm),
            float(self.z_mm),
        )

    def with_z_level(self, z_mm: float):
        return replace(
            self,
            z_mm=float(z_mm),
        )


@dataclass(frozen=True, slots=True)
class FakePlan:
    predicted_delta_mm: tuple[float, float, float]


class Phase8AdaptiveRetractTests(unittest.TestCase):
    def test_joint_slew_failure_subdivides_only_failed_segment(self):
        successful_targets: list[float] = []
        rejected_targets: list[float] = []
        last_sent_z = {"value": -20.0}
        events: list[dict] = []

        def fake_send_command_state(
            robot,
            planner,
            state,
            motor_names,
            *,
            label,
            sent_state_tracker,
            event_capture_timestamp=None,
        ):
            del (
                robot,
                planner,
                motor_names,
                label,
                sent_state_tracker,
                event_capture_timestamp,
            )

            step = float(state.z_mm) - float(last_sent_z["value"])

            if step > 5.0 + 1e-12:
                rejected_targets.append(float(state.z_mm))
                raise RelativeJointSafetyError(
                    violations=(("elbow_flex", 10.5),),
                    limit_deg=10.0,
                )

            successful_targets.append(float(state.z_mm))
            last_sent_z["value"] = float(state.z_mm)

            return (
                0.0,
                FakePlan(
                    predicted_delta_mm=(
                        0.0,
                        0.0,
                        float(state.z_mm),
                    )
                ),
            )

        with patch.object(
            phase8.phase5,
            "send_command_state",
            side_effect=fake_send_command_state,
        ):
            final_state = phase8._retract_to_z0(
                robot=object(),
                planner=object(),
                state=FakeState(-20.0),
                motor_names=["elbow_flex"],
                sent_state_tracker=object(),
                events=events,
            )

        self.assertEqual(
            successful_targets,
            [-15.0, -10.0, -5.0, 0.0],
        )
        self.assertEqual(
            rejected_targets,
            [-10.0, 0.0],
        )
        self.assertEqual(
            final_state.z_mm,
            0.0,
        )

        subdivides = [
            event
            for event in events
            if event["event"] == "retract_subdivide"
        ]
        sends = [
            event
            for event in events
            if event["event"] == "retract"
        ]

        self.assertEqual(len(subdivides), 2)
        self.assertEqual(len(sends), 4)
        self.assertTrue(
            all(
                event["joint_slew_limit_deg"] == 10.0
                for event in subdivides
            )
        )

    def test_release_mode_reuses_adaptive_segmentation_to_z0(self):
        successful_targets: list[float] = []
        rejected_targets: list[float] = []
        capture_timestamps: list[float | None] = []
        labels: list[str] = []
        last_sent_z = {"value": -30.0}
        events: list[dict] = []

        def fake_send_command_state(
            robot,
            planner,
            state,
            motor_names,
            *,
            label,
            sent_state_tracker,
            event_capture_timestamp=None,
        ):
            del robot, planner, motor_names, sent_state_tracker

            labels.append(str(label))
            capture_timestamps.append(event_capture_timestamp)

            step = float(state.z_mm) - float(last_sent_z["value"])
            if step > 10.0 + 1e-12:
                rejected_targets.append(float(state.z_mm))
                raise RelativeJointSafetyError(
                    violations=(("elbow_flex", 10.5),),
                    limit_deg=10.0,
                )

            successful_targets.append(float(state.z_mm))
            last_sent_z["value"] = float(state.z_mm)
            return (
                0.0,
                FakePlan(
                    predicted_delta_mm=(
                        0.0,
                        0.0,
                        float(state.z_mm),
                    )
                ),
            )

        with patch.object(
            phase8.phase5,
            "send_command_state",
            side_effect=fake_send_command_state,
        ):
            final_state = phase8._retract_to_z0(
                robot=object(),
                planner=object(),
                state=FakeState(-30.0),
                motor_names=["elbow_flex"],
                sent_state_tracker=object(),
                events=events,
                label_prefix="RELEASE",
                nominal_step_mm=20.0,
                event_capture_timestamp=123.456,
            )

        self.assertEqual(rejected_targets, [-10.0])
        self.assertEqual(successful_targets, [-20.0, -10.0, 0.0])
        self.assertEqual(final_state.z_mm, 0.0)
        self.assertTrue(all(label.startswith("RELEASE ") for label in labels))
        self.assertEqual(capture_timestamps[:2], [123.456, 123.456])
        self.assertTrue(
            all(value is None for value in capture_timestamps[2:])
        )
        self.assertEqual(
            [event["event"] for event in events],
            ["release_subdivide", "release", "release", "release"],
        )

    def test_unrelated_runtime_error_is_not_subdivided(self):
        calls = {"count": 0}

        def fake_send_command_state(*args, **kwargs):
            del args, kwargs
            calls["count"] += 1
            raise RuntimeError("different safety failure")

        with patch.object(
            phase8.phase5,
            "send_command_state",
            side_effect=fake_send_command_state,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "different safety failure",
            ):
                phase8._retract_to_z0(
                    robot=object(),
                    planner=object(),
                    state=FakeState(-10.0),
                    motor_names=["elbow_flex"],
                    sent_state_tracker=object(),
                    events=[],
                )

        self.assertEqual(
            calls["count"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
