import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from so101_typing.control.cartesian_kinematics import (
    DEFAULT_MAX_DELTA_NORM_MM,
    MM_TO_M,
    MOTION_FRAME,
    MOTION_UNIT,
    SO101_MOTOR_NAMES,
    build_official_cartesian_pipeline,
    make_cartesian_delta_action,
)


class TestCartesianDeltaAction(unittest.TestCase):
    def test_contract_is_mm_in_base_frame(self):
        action = make_cartesian_delta_action(
            10.0,
            -7.5,
            delta_z_mm=2.0,
        )

        self.assertEqual(MOTION_FRAME, "base_link_xy")
        self.assertEqual(MOTION_UNIT, "mm")
        self.assertEqual(MM_TO_M, 0.001)

        self.assertTrue(action["enabled"])

        self.assertEqual(action["target_x"], 10.0)
        self.assertEqual(action["target_y"], -7.5)
        self.assertEqual(action["target_z"], 2.0)

        self.assertEqual(action["target_wx"], 0.0)
        self.assertEqual(action["target_wy"], 0.0)
        self.assertEqual(action["target_wz"], 0.0)

        self.assertEqual(action["gripper_vel"], 0.0)

    def test_centimetre_scale_request_is_allowed_by_adapter(self):
        action = make_cartesian_delta_action(
            10.0,
            0.0,
        )

        self.assertEqual(action["target_x"], 10.0)

    def test_two_centimetre_request_is_allowed(self):
        action = make_cartesian_delta_action(
            20.0,
            0.0,
        )

        self.assertEqual(
            DEFAULT_MAX_DELTA_NORM_MM,
            20.0,
        )
        self.assertEqual(action["target_x"], 20.0)

    def test_first_command_over_two_centimetres_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "max_delta_norm_mm",
        ):
            make_cartesian_delta_action(
                20.01,
                0.0,
            )

    def test_custom_delta_limit_is_supported(self):
        with self.assertRaisesRegex(
            ValueError,
            "max_delta_norm_mm",
        ):
            make_cartesian_delta_action(
                10.01,
                0.0,
                max_delta_norm_mm=10.0,
            )

    def test_nonfinite_delta_is_rejected(self):
        for bad in (
            math.nan,
            math.inf,
            -math.inf,
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    make_cartesian_delta_action(
                        bad,
                        0.0,
                    )


class TestOfficialPipeline(unittest.TestCase):
    def test_pipeline_uses_lerobot_processors(self):
        with TemporaryDirectory() as tmp:
            urdf = Path(tmp) / "robot.urdf"
            urdf.write_text(
                "<robot name='dummy'/>",
                encoding="utf-8",
            )

            fake_kinematics = object()

            with patch(
                "so101_typing.control.cartesian_kinematics."
                "RobotKinematics",
                return_value=fake_kinematics,
            ):
                pipeline = build_official_cartesian_pipeline(
                    urdf,
                    motor_names=SO101_MOTOR_NAMES,
                    end_effector_bounds={
                        "min": [-1.0, -1.0, -1.0],
                        "max": [1.0, 1.0, 1.0],
                    },
                    max_ee_step_m=0.02,
                )

        self.assertIsNotNone(pipeline)

        step_names = [
            type(step).__name__
            for step in pipeline.steps
        ]

        self.assertEqual(
            step_names,
            [
                "EEReferenceAndDelta",
                "EEBoundsAndSafety",
                "GripperVelocityToJoint",
                "InverseKinematicsEEToJoints",
            ],
        )


if __name__ == "__main__":
    unittest.main()
