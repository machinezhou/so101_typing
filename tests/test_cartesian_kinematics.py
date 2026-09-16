import unittest

import numpy as np

from so101_typing.control.cartesian_kinematics import (
    CartesianIKSafetyConfig,
    MOTION_FRAME,
    SafeCartesianIK,
)


JOINTS = (
    "j1",
    "j2",
    "j3",
    "j4",
    "j5",
)

LIMITS = {
    name: (-100.0, 100.0)
    for name in JOINTS
}


class FakeKinematics:
    def __init__(
        self,
        *,
        achieved_xyz_mm=(1.0, 0.0, 0.0),
        dq_deg=(0.5, 0.0, 0.0, 0.0, 0.0),
        orientation_change_deg=0.0,
    ):
        self.achieved_xyz_mm = np.asarray(
            achieved_xyz_mm,
            dtype=np.float64,
        )

        self.dq_deg = np.asarray(
            dq_deg,
            dtype=np.float64,
        )

        self.orientation_change_deg = float(
            orientation_change_deg
        )

        self.q0 = None
        self.q1 = None

        self.warmed = False
        self.inverse_saw_warm_state = False

    def forward_kinematics(
        self,
        joint_pos_deg,
    ):
        q = np.asarray(
            joint_pos_deg,
            dtype=np.float64,
        )

        if self.q0 is None:
            self.q0 = q.copy()
            self.warmed = True

            return np.eye(
                4,
                dtype=np.float64,
            )

        if (
            self.q1 is not None
            and np.allclose(
                q,
                self.q1,
            )
        ):
            T = np.eye(
                4,
                dtype=np.float64,
            )

            T[:3, 3] = (
                self.achieved_xyz_mm
                / 1000.0
            )

            angle = np.deg2rad(
                self.orientation_change_deg
            )

            c = np.cos(angle)
            s = np.sin(angle)

            T[:3, :3] = np.array(
                [
                    [c, -s, 0.0],
                    [s, c, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )

            return T

        return np.eye(
            4,
            dtype=np.float64,
        )

    def inverse_kinematics(
        self,
        current_joint_pos,
        desired_ee_pose,
        position_weight=1.0,
        orientation_weight=0.01,
    ):
        self.inverse_saw_warm_state = (
            self.warmed
        )

        self.q1 = (
            np.asarray(
                current_joint_pos,
                dtype=np.float64,
            )
            + self.dq_deg
        )

        return self.q1.copy()


def planner(
    fake,
    *,
    config=None,
    limits=None,
):
    return SafeCartesianIK(
        fake,
        joint_names=JOINTS,
        joint_limits_deg=(
            limits
            or LIMITS
        ),
        config=config,
    )


class TestSafeCartesianIK(unittest.TestCase):
    def test_accepts_small_backprojected_candidate(self):
        fake = FakeKinematics(
            achieved_xyz_mm=(
                0.99,
                0.0,
                0.004,
            )
        )

        result = planner(
            fake
        ).candidate_xy(
            np.zeros(5),
            (1.0, 0.0),
        )

        self.assertTrue(
            result.accepted
        )

        self.assertEqual(
            result.rejection_reasons,
            (),
        )

        self.assertEqual(
            result.motion_frame,
            MOTION_FRAME,
        )

        self.assertEqual(
            result.motion_frame,
            "base_link_xy",
        )

        self.assertEqual(
            result.motion_unit,
            "mm",
        )

        self.assertTrue(
            fake.inverse_saw_warm_state
        )

        self.assertLess(
            result.position_error_norm_mm,
            0.05,
        )

    def test_rejects_request_over_one_mm(self):
        fake = FakeKinematics()

        with self.assertRaisesRegex(
            ValueError,
            "max_request_norm_mm",
        ):
            planner(
                fake
            ).candidate_xy(
                np.zeros(5),
                (1.01, 0.0),
            )

    def test_rejects_large_joint_step(self):
        fake = FakeKinematics(
            dq_deg=(
                2.1,
                0.0,
                0.0,
                0.0,
                0.0,
            )
        )

        result = planner(
            fake
        ).candidate_xy(
            np.zeros(5),
            (1.0, 0.0),
        )

        self.assertFalse(
            result.accepted
        )

        self.assertIn(
            "joint_step",
            result.rejection_reasons,
        )

    def test_rejects_unintended_z(self):
        fake = FakeKinematics(
            achieved_xyz_mm=(
                1.0,
                0.0,
                0.06,
            )
        )

        result = planner(
            fake
        ).candidate_xy(
            np.zeros(5),
            (1.0, 0.0),
        )

        self.assertFalse(
            result.accepted
        )

        self.assertIn(
            "unintended_z",
            result.rejection_reasons,
        )

    def test_rejects_backprojection_error(self):
        fake = FakeKinematics(
            achieved_xyz_mm=(
                0.90,
                0.0,
                0.0,
            )
        )

        result = planner(
            fake
        ).candidate_xy(
            np.zeros(5),
            (1.0, 0.0),
        )

        self.assertFalse(
            result.accepted
        )

        self.assertIn(
            "position_error",
            result.rejection_reasons,
        )

    def test_rejects_orientation_change(self):
        fake = FakeKinematics(
            orientation_change_deg=0.11
        )

        result = planner(
            fake
        ).candidate_xy(
            np.zeros(5),
            (1.0, 0.0),
        )

        self.assertFalse(
            result.accepted
        )

        self.assertIn(
            "orientation_change",
            result.rejection_reasons,
        )

    def test_fails_before_kinematics_on_current_joint_margin(self):
        fake = FakeKinematics()

        q0 = np.array(
            [
                91.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        with self.assertRaisesRegex(
            ValueError,
            "current joint margin",
        ):
            planner(
                fake
            ).candidate_xy(
                q0,
                (1.0, 0.0),
            )

        self.assertFalse(
            fake.warmed
        )

        self.assertFalse(
            fake.inverse_saw_warm_state
        )

    def test_fails_before_kinematics_on_current_joint_outside_urdf(self):
        fake = FakeKinematics()

        q0 = np.array(
            [
                101.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        with self.assertRaisesRegex(
            ValueError,
            "outside URDF",
        ):
            planner(
                fake
            ).candidate_xy(
                q0,
                (1.0, 0.0),
            )

        self.assertFalse(
            fake.warmed
        )

        self.assertFalse(
            fake.inverse_saw_warm_state
        )

    def test_rejects_candidate_joint_margin(self):
        fake = FakeKinematics(
            dq_deg=(
                2.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
        )

        q0 = np.array(
            [
                89.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        result = planner(
            fake
        ).candidate_xy(
            q0,
            (1.0, 0.0),
        )

        self.assertFalse(
            result.accepted
        )

        self.assertIn(
            "candidate_joint_margin",
            result.rejection_reasons,
        )

    def test_config_rejects_invalid_threshold(self):
        with self.assertRaisesRegex(
            ValueError,
            "max_joint_step_deg",
        ):
            CartesianIKSafetyConfig(
                max_joint_step_deg=0.0
            )


if __name__ == "__main__":
    unittest.main()
