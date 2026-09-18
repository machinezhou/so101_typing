import unittest

from so101_typing.control.phase3_staged import (
    ResponseGateConfig,
    SegmentReadinessGate,
    StagedDescentGate,
    assess_image_response,
    run_staged_sequence,
    validate_frame_image_size,
    validate_h32_runtime_context,
    validate_joint_target_step,
)


class TestH32RuntimeContext(unittest.TestCase):
    def test_accepts_frozen_h32_coordinate_space(self):
        validate_h32_runtime_context(
            camera_name="wrist",
            configured_image_size=(640, 480),
            jacobian_motion_frame="base_link_xy",
            jacobian_motion_unit="mm",
        )
        validate_frame_image_size((480, 640, 3), expected_image_size=(640, 480))

    def test_rejects_wrong_jacobian_frame_or_unit(self):
        for frame, unit in (("tool_xy", "mm"), ("base_link_xy", "m")):
            with self.subTest(frame=frame, unit=unit):
                with self.assertRaisesRegex(RuntimeError, "H3.2 context mismatch"):
                    validate_h32_runtime_context(
                        camera_name="wrist",
                        configured_image_size=(640, 480),
                        jacobian_motion_frame=frame,
                        jacobian_motion_unit=unit,
                    )

    def test_rejects_configured_or_actual_resolution_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "configured WRIST image size"):
            validate_h32_runtime_context(
                camera_name="wrist",
                configured_image_size=(1280, 720),
                jacobian_motion_frame="base_link_xy",
                jacobian_motion_unit="mm",
            )
        with self.assertRaisesRegex(RuntimeError, "frame geometry mismatch"):
            validate_frame_image_size((720, 1280, 3), expected_image_size=(640, 480))


class TestResponseGate(unittest.TestCase):
    def test_accepts_same_general_direction_and_bounded_ratio(self):
        result = assess_image_response(
            (2.0, 0.0),
            (3.0, 1.0),
            error_before_px=20.0,
            error_after_px=17.0,
        )
        self.assertTrue(result.accepted)
        self.assertGreater(result.cosine, 0.0)

    def test_rejects_opposite_response(self):
        result = assess_image_response(
            (2.0, 0.0),
            (-2.0, 0.0),
            error_before_px=20.0,
            error_after_px=22.0,
        )
        self.assertFalse(result.accepted)
        self.assertLess(result.cosine, 0.0)

    def test_rejects_large_error_increase(self):
        result = assess_image_response(
            (2.0, 0.0),
            (2.0, 0.0),
            error_before_px=20.0,
            error_after_px=24.0,
            config=ResponseGateConfig(max_error_increase_px=3.0),
        )
        self.assertFalse(result.accepted)


class TestSegmentReadinessGate(unittest.TestCase):
    def test_failed_first_probe_requires_return_before_retry(self):
        gate = SegmentReadinessGate(max_attempts=3)
        self.assertEqual(gate.begin_probe(), 1)
        failed = assess_image_response(
            (2.0, 0.0),
            (-2.0, 0.0),
            error_before_px=20.0,
            error_after_px=22.0,
        )
        gate.record_assessment(failed)
        self.assertTrue(gate.return_required)
        with self.assertRaisesRegex(RuntimeError, "return"):
            gate.begin_probe()

        gate.record_return_to_anchor()
        self.assertEqual(gate.begin_probe(), 2)
        passed = assess_image_response(
            (2.0, 0.0),
            (2.2, 0.1),
            error_before_px=20.0,
            error_after_px=18.0,
        )
        gate.record_assessment(passed)
        self.assertTrue(gate.accepted)

    def test_exhaustion_occurs_only_after_final_return(self):
        gate = SegmentReadinessGate(max_attempts=2)
        failed = assess_image_response(
            (2.0, 0.0),
            (-2.0, 0.0),
            error_before_px=20.0,
            error_after_px=22.0,
        )
        for _ in range(2):
            gate.begin_probe()
            gate.record_assessment(failed)
            gate.record_return_to_anchor()
        self.assertTrue(gate.exhausted_after_return)
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            gate.begin_probe()


class TestJointTargetPrecheck(unittest.TestCase):
    def test_accepts_bounded_relative_joint_targets(self):
        observation = {
            "shoulder_pan.pos": 1.0,
            "shoulder_lift.pos": 2.0,
            "gripper.pos": 3.0,
        }
        planned = {
            "shoulder_pan.pos": 4.0,
            "shoulder_lift.pos": -1.0,
            "gripper.pos": 90.0,
        }
        deltas = validate_joint_target_step(
            observation,
            planned,
            ("shoulder_pan", "shoulder_lift", "gripper"),
            max_relative_target_deg=10.0,
        )
        self.assertEqual(deltas, {"shoulder_pan": 3.0, "shoulder_lift": 3.0})

    def test_rejects_before_send_when_relative_target_would_clip(self):
        with self.assertRaisesRegex(RuntimeError, "Refusing send_action"):
            validate_joint_target_step(
                {"shoulder_pan.pos": 0.0},
                {"shoulder_pan.pos": 10.1},
                ("shoulder_pan",),
                max_relative_target_deg=10.0,
            )


class TestStagedDescentGate(unittest.TestCase):
    def test_requires_stable_alignment_and_invalidates_after_z(self):
        gate = StagedDescentGate(required_stable_bursts=3, max_z_steps=2, z_step_mm=-0.5)
        self.assertFalse(gate.observe_alignment(True))
        self.assertFalse(gate.observe_alignment(True))
        self.assertTrue(gate.observe_alignment(True))
        gate.authorize_z_step()
        self.assertFalse(gate.alignment_valid)
        self.assertEqual(gate.stable_bursts, 0)
        self.assertEqual(gate.z_steps_done, 1)
        self.assertAlmostEqual(gate.cumulative_z_mm, -0.5)

    def test_cannot_descend_without_alignment(self):
        gate = StagedDescentGate()
        with self.assertRaisesRegex(RuntimeError, "alignment"):
            gate.authorize_z_step()

    def test_budget_stops_additional_descent(self):
        gate = StagedDescentGate(required_stable_bursts=1, max_z_steps=1, z_step_mm=-0.5)
        gate.observe_alignment(True)
        gate.authorize_z_step()
        self.assertTrue(gate.complete)
        gate.observe_alignment(True)
        with self.assertRaisesRegex(RuntimeError, "already complete"):
            gate.authorize_z_step()


class TestFakeHardwareStagedSequence(unittest.TestCase):
    def test_full_two_step_sequence_reobserves_after_every_z_and_final_z(self):
        events: list[str] = []
        gate = StagedDescentGate(required_stable_bursts=3, max_z_steps=2, z_step_mm=-0.5)

        def align(level: int) -> str:
            events.append(f"align:{level}")
            return f"alignment-{level}"

        def z_step(step: int) -> str:
            events.append(f"z:{step}")
            return f"z-{step}"

        result = run_staged_sequence(gate, align_level=align, command_z_step=z_step)

        self.assertEqual(events, ["align:1", "z:1", "align:2", "z:2", "align:3"])
        self.assertEqual(result.aligned_levels, 3)
        self.assertEqual(result.z_steps_done, 2)
        self.assertAlmostEqual(result.cumulative_z_mm, -1.0)

    def test_target_loss_or_readiness_failure_stops_before_later_commands(self):
        for failure_label in ("target_lost", "readiness_failed"):
            with self.subTest(failure_label=failure_label):
                events: list[str] = []
                gate = StagedDescentGate(max_z_steps=2)

                def align(level: int) -> None:
                    events.append(f"align:{level}")
                    if level == 2:
                        raise RuntimeError(failure_label)

                def z_step(step: int) -> None:
                    events.append(f"z:{step}")

                with self.assertRaisesRegex(RuntimeError, failure_label):
                    run_staged_sequence(gate, align_level=align, command_z_step=z_step)
                self.assertEqual(events, ["align:1", "z:1", "align:2"])

    def test_z_plan_failure_stops_before_reobserve_or_second_z(self):
        events: list[str] = []
        gate = StagedDescentGate(max_z_steps=2)

        def align(level: int) -> None:
            events.append(f"align:{level}")

        def z_step(step: int) -> None:
            events.append(f"z:{step}")
            raise RuntimeError("z_plan_failed")

        with self.assertRaisesRegex(RuntimeError, "z_plan_failed"):
            run_staged_sequence(gate, align_level=align, command_z_step=z_step)
        self.assertEqual(events, ["align:1", "z:1"])


if __name__ == "__main__":
    unittest.main()
