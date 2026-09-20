import unittest

from so101_typing.control.fixed_anchor_xyz import (
    FixedAnchorXYZCommandState,
    FixedGoalAnchor,
    stepped_z_targets_to_zero,
)


MOTORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


class TestFixedAnchorXYZ(unittest.TestCase):
    def make_anchor(self):
        return FixedGoalAnchor.from_goal_positions(
            {
                name: 10.0 + index
                for index, name in enumerate(MOTORS)
            },
            MOTORS,
        )

    def test_goal_anchor_preserves_exact_values(self):
        anchor = self.make_anchor()

        self.assertEqual(
            anchor.motor_names,
            MOTORS,
        )

        self.assertEqual(
            anchor.as_observation()["shoulder_pan.pos"],
            10.0,
        )

        self.assertEqual(
            anchor.as_observation()["gripper.pos"],
            15.0,
        )

    def test_initial_state_is_zero_delta(self):
        state = FixedAnchorXYZCommandState.at_anchor(
            self.make_anchor()
        )

        self.assertEqual(
            state.xyz_mm,
            (0.0, 0.0, 0.0),
        )

    def test_xy_commands_accumulate(self):
        state = FixedAnchorXYZCommandState.at_anchor(
            self.make_anchor()
        )

        state = state.with_xy_increment(
            3.0,
            -2.0,
        )

        state = state.with_xy_increment(
            2.0,
            1.0,
        )

        self.assertEqual(
            state.xyz_mm,
            (5.0, -1.0, 0.0),
        )

    def test_z_level_holds_xy(self):
        state = FixedAnchorXYZCommandState.at_anchor(
            self.make_anchor()
        )

        state = state.with_xy_target(
            6.0,
            4.0,
        )

        lowered = state.with_z_level(
            -3.0
        )

        self.assertEqual(
            lowered.xyz_mm,
            (6.0, 4.0, -3.0),
        )

    def test_same_z_realign_preserves_z(self):
        state = FixedAnchorXYZCommandState.at_anchor(
            self.make_anchor()
        )

        state = (
            state
            .with_xy_target(5.0, 2.0)
            .with_z_level(-6.0)
        )

        realigned = state.with_xy_increment(
            -1.5,
            0.5,
        )

        self.assertEqual(
            realigned.xyz_mm,
            (3.5, 2.5, -6.0),
        )

    def test_anchor_identity_never_changes(self):
        anchor = self.make_anchor()

        state = FixedAnchorXYZCommandState.at_anchor(
            anchor
        )

        moved = (
            state
            .with_xy_increment(3.0, 2.0)
            .with_z_level(-3.0)
            .with_xy_increment(1.0, -1.0)
        )

        self.assertIs(
            moved.anchor,
            anchor,
        )

    def test_retract_preserves_xy_and_returns_z_zero(self):
        state = FixedAnchorXYZCommandState.at_anchor(
            self.make_anchor()
        )

        state = (
            state
            .with_xy_target(4.0, 3.0)
            .with_z_level(-6.0)
        )

        retracted = state.retracted()

        self.assertEqual(
            retracted.xyz_mm,
            (4.0, 3.0, 0.0),
        )

    def test_budget_is_enforced(self):
        state = FixedAnchorXYZCommandState.at_anchor(
            self.make_anchor(),
            max_xy_norm_mm=5.0,
            max_xyz_norm_mm=6.0,
        )

        with self.assertRaises(ValueError):
            state.with_xy_target(
                5.0,
                1.0,
            )


    def test_large_negative_retract_is_split_into_bounded_steps(self):
        targets = (
            stepped_z_targets_to_zero(
                -64.0,
                max_step_mm=10.0,
            )
        )

        self.assertEqual(
            targets,
            (
                -54.0,
                -44.0,
                -34.0,
                -24.0,
                -14.0,
                -4.0,
                0.0,
            ),
        )

        previous = -64.0

        for target in targets:
            self.assertLessEqual(
                abs(target - previous),
                10.0,
            )

            previous = target


    def test_retract_target_generator_handles_zero_and_positive_z(self):
        self.assertEqual(
            stepped_z_targets_to_zero(
                0.0,
                max_step_mm=10.0,
            ),
            (),
        )

        self.assertEqual(
            stepped_z_targets_to_zero(
                23.0,
                max_step_mm=10.0,
            ),
            (
                13.0,
                3.0,
                0.0,
            ),
        )


if __name__ == "__main__":
    unittest.main()
