import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from so101_typing.control.cartesian_kinematics import (
    DEFAULT_MAX_DELTA_NORM_MM,
    MM_TO_M,
    MOTION_FRAME,
    MOTION_UNIT,
    SO101_MOTOR_NAMES,
    build_official_cartesian_pipeline,
    end_effector_xyz_mm,
    joint_vector_deg,
    make_cartesian_delta_action,
    ordered_joint_observation,
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


class TestJointOrderingAndFK(unittest.TestCase):
    def test_observation_is_canonicalized_to_motor_order(self):
        observation = {
            "wrist_roll.pos": 5.0,
            "shoulder_pan.pos": 1.0,
            "gripper.pos": 6.0,
            "elbow_flex.pos": 3.0,
            "shoulder_lift.pos": 2.0,
            "wrist_flex.pos": 4.0,
            "extra": 99,
        }
        ordered = ordered_joint_observation(observation, SO101_MOTOR_NAMES)
        self.assertEqual(
            [key for key in ordered if key.endswith(".pos")],
            [f"{name}.pos" for name in SO101_MOTOR_NAMES],
        )
        np.testing.assert_allclose(
            joint_vector_deg(ordered, SO101_MOTOR_NAMES),
            [1, 2, 3, 4, 5, 6],
        )

    def test_fk_xyz_mm_uses_explicit_motor_order(self):
        class FakeKinematics:
            def forward_kinematics(self, q):
                np.testing.assert_allclose(q, [1, 2, 3, 4, 5, 6])
                pose = np.eye(4)
                pose[:3, 3] = [0.1, -0.2, 0.3]
                return pose

        joints = {f"{name}.pos": i + 1 for i, name in enumerate(SO101_MOTOR_NAMES)}
        np.testing.assert_allclose(
            end_effector_xyz_mm(FakeKinematics(), joints, SO101_MOTOR_NAMES),
            [100.0, -200.0, 300.0],
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
        self.assertFalse(pipeline.steps[0].use_latched_reference)

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

    def test_pipeline_can_use_fixed_latched_reference(self):
        with TemporaryDirectory() as tmp:
            urdf = Path(tmp) / "robot.urdf"
            urdf.write_text("<robot name='dummy'/>", encoding="utf-8")
            with patch(
                "so101_typing.control.cartesian_kinematics.RobotKinematics",
                return_value=object(),
            ):
                pipeline = build_official_cartesian_pipeline(
                    urdf,
                    motor_names=SO101_MOTOR_NAMES,
                    end_effector_bounds={
                        "min": [-1.0, -1.0, -1.0],
                        "max": [1.0, 1.0, 1.0],
                    },
                    use_latched_reference=True,
                )
        self.assertTrue(pipeline.steps[0].use_latched_reference)


if __name__ == "__main__":
    unittest.main()
