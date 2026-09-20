import unittest

import numpy as np

from so101_typing.control.fixed_anchor_planner import (
    FixedAnchorCartesianPlanner,
    FixedAnchorPlannerConfig,
)
from so101_typing.control.fixed_anchor_xyz import (
    FixedAnchorXYZCommandState,
    FixedGoalAnchor,
)


MOTORS = (
    "j0",
    "j1",
    "j2",
    "j3",
    "j4",
    "gripper",
)


class FakeKinematics:
    def forward_kinematics(
        self,
        joints,
    ):
        q = np.asarray(
            joints,
            dtype=np.float64,
        )

        transform = np.eye(
            4,
            dtype=np.float64,
        )

        # Treat first three fake joint values as Cartesian millimetres.
        transform[:3, 3] = (
            q[:3] * 0.001
        )

        return transform


class FakeLatchedPipeline:
    def __init__(
        self,
        *,
        nonzero_offset=(0.0, 0.0, 0.0),
        flip_z=False,
        zero_joint_shift=0.0,
    ):
        self.calls = 0

        self.nonzero_offset = np.asarray(
            nonzero_offset,
            dtype=np.float64,
        )

        self.flip_z = bool(
            flip_z
        )

        self.zero_joint_shift = float(
            zero_joint_shift
        )

    def __call__(
        self,
        payload,
    ):
        action, observation = payload

        self.calls += 1

        x = float(
            action["target_x"]
        )

        y = float(
            action["target_y"]
        )

        z = float(
            action["target_z"]
        )

        is_zero = (
            abs(x) < 1e-12
            and abs(y) < 1e-12
            and abs(z) < 1e-12
        )

        if is_zero:
            return {
                "j0.pos": (
                    float(
                        observation["j0.pos"]
                    )
                    + self.zero_joint_shift
                ),
                "j1.pos": float(
                    observation["j1.pos"]
                ),
                "j2.pos": float(
                    observation["j2.pos"]
                ),
                "j3.pos": float(
                    observation["j3.pos"]
                ),
                "j4.pos": float(
                    observation["j4.pos"]
                ),
                "gripper.pos": float(
                    observation[
                        "gripper.pos"
                    ]
                ),
            }

        xyz = np.asarray(
            [x, y, z],
            dtype=np.float64,
        )

        xyz = (
            xyz
            + self.nonzero_offset
        )

        if self.flip_z:
            xyz[2] = abs(
                xyz[2]
            )

        # Deliberately ignore Present_Position here.  This fake models the
        # accepted latched-reference semantics.
        return {
            "j0.pos": float(
                xyz[0]
            ),
            "j1.pos": float(
                xyz[1]
            ),
            "j2.pos": float(
                xyz[2]
            ),
            "j3.pos": 0.0,
            "j4.pos": 0.0,
            "gripper.pos": float(
                observation[
                    "gripper.pos"
                ]
            ),
        }


class TestFixedAnchorPlanner(
    unittest.TestCase
):
    def make_anchor(self):
        return (
            FixedGoalAnchor
            .from_goal_positions(
                {
                    name: 0.0
                    for name in MOTORS
                },
                MOTORS,
            )
        )

    def make_planner(
        self,
        pipeline=None,
        *,
        config=None,
    ):
        anchor = (
            self.make_anchor()
        )

        planner = (
            FixedAnchorCartesianPlanner(
                pipeline=(
                    pipeline
                    or FakeLatchedPipeline()
                ),
                validation_kinematics=(
                    FakeKinematics()
                ),
                motor_names=MOTORS,
                anchor=anchor,
                anchor_xyz_mm=(
                    0.0,
                    0.0,
                    0.0,
                ),
                config=config,
            )
        )

        return (
            planner,
            anchor,
        )

    def present(
        self,
        *,
        j0=0.0,
        j1=0.0,
        j2=0.0,
    ):
        return {
            "j0.pos": j0,
            "j1.pos": j1,
            "j2.pos": j2,
            "j3.pos": 0.0,
            "j4.pos": 0.0,
            "gripper.pos": 15.0,
        }

    def test_zero_delta_latches_once(self):
        planner, _ = (
            self.make_planner()
        )

        shift = (
            planner.latch_zero_delta()
        )

        self.assertEqual(
            shift,
            0.0,
        )

        self.assertTrue(
            planner.latched
        )

        with self.assertRaises(
            RuntimeError
        ):
            planner.latch_zero_delta()

    def test_zero_delta_shift_is_rejected(self):
        planner, _ = (
            self.make_planner(
                FakeLatchedPipeline(
                    zero_joint_shift=0.8
                )
            )
        )

        with self.assertRaises(
            RuntimeError
        ):
            planner.latch_zero_delta()

    def test_plan_requires_latched_anchor(self):
        planner, anchor = (
            self.make_planner()
        )

        state = (
            FixedAnchorXYZCommandState
            .at_anchor(anchor)
            .with_xy_target(
                3.0,
                2.0,
            )
        )

        with self.assertRaises(
            RuntimeError
        ):
            planner.plan(
                state,
                self.present(),
            )

    def test_present_position_does_not_rebase_command(self):
        planner, anchor = (
            self.make_planner()
        )

        planner.latch_zero_delta()

        state = (
            FixedAnchorXYZCommandState
            .at_anchor(anchor)
            .with_xy_target(
                5.0,
                -2.0,
            )
            .with_z_level(
                -3.0
            )
        )

        plan = planner.plan(
            state,
            self.present(
                j0=7.0,
                j1=-4.0,
                j2=-3.0,
            ),
        )

        self.assertEqual(
            plan.requested_xyz_mm,
            (
                5.0,
                -2.0,
                -3.0,
            ),
        )

        self.assertEqual(
            plan.predicted_delta_mm,
            (
                5.0,
                -2.0,
                -3.0,
            ),
        )

    def test_same_z_realign_preserves_z(self):
        planner, anchor = (
            self.make_planner()
        )

        planner.latch_zero_delta()

        state = (
            FixedAnchorXYZCommandState
            .at_anchor(anchor)
            .with_xy_target(
                4.0,
                2.0,
            )
            .with_z_level(
                -6.0
            )
            .with_xy_increment(
                -1.0,
                0.5,
            )
        )

        plan = planner.plan(
            state,
            self.present(
                j0=3.0,
                j1=2.5,
                j2=-6.0,
            ),
        )

        self.assertEqual(
            plan.requested_xyz_mm,
            (
                3.0,
                2.5,
                -6.0,
            ),
        )

    def test_model_error_is_rejected(self):
        planner, anchor = (
            self.make_planner(
                FakeLatchedPipeline(
                    nonzero_offset=(
                        2.0,
                        0.0,
                        0.0,
                    )
                )
            )
        )

        planner.latch_zero_delta()

        state = (
            FixedAnchorXYZCommandState
            .at_anchor(anchor)
            .with_xy_target(
                3.0,
                0.0,
            )
        )

        with self.assertRaises(
            RuntimeError
        ):
            planner.plan(
                state,
                self.present(),
            )

    def test_wrong_downward_direction_is_rejected(self):
        planner, anchor = (
            self.make_planner(
                FakeLatchedPipeline(
                    flip_z=True
                )
            )
        )

        planner.latch_zero_delta()

        state = (
            FixedAnchorXYZCommandState
            .at_anchor(anchor)
            .with_z_level(
                -3.0
            )
        )

        with self.assertRaises(
            RuntimeError
        ):
            planner.plan(
                state,
                self.present(),
            )

    def test_sent_action_clipping_is_rejected(self):
        planner, anchor = (
            self.make_planner()
        )

        planner.latch_zero_delta()

        state = (
            FixedAnchorXYZCommandState
            .at_anchor(anchor)
            .with_xy_target(
                2.0,
                1.0,
            )
        )

        plan = planner.plan(
            state,
            self.present(),
        )

        sent = dict(
            plan.joint_action
        )

        sent["j0.pos"] = (
            float(sent["j0.pos"])
            - 0.2
        )

        with self.assertRaises(
            RuntimeError
        ):
            planner.validate_sent_action(
                plan,
                sent,
            )


    def test_explicit_xy_error_override_is_local_to_call(self):
        planner, anchor = self.make_planner(
            FakeLatchedPipeline(
                nonzero_offset=(1.5, 0.0, 0.0),
            )
        )
        planner.latch_zero_delta()

        state = (
            FixedAnchorXYZCommandState
            .at_anchor(anchor)
            .with_xy_target(5.0, 0.0)
        )

        with self.assertRaises(RuntimeError):
            planner.plan(
                state,
                self.present(),
            )

        plan = planner.plan(
            state,
            self.present(),
            max_model_xy_error_mm=2.0,
        )

        self.assertAlmostEqual(
            plan.model_xy_error_mm,
            1.5,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
