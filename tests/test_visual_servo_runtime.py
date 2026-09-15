import unittest
from dataclasses import dataclass

import numpy as np

from so101_typing.contracts import CameraFrame
from so101_typing.control.image_jacobian import ImageJacobianCalibration
from so101_typing.control.tool_reference import ToolReferenceCalibration
from so101_typing.control.visual_servo import VisualServoConfig
from so101_typing.control.visual_servo_runtime import (
    VisualServoRuntime,
    VisualServoRuntimeConfig,
    VisualServoRuntimeStatus,
)


@dataclass(frozen=True, slots=True)
class _Observation:
    target_label: str = "G"
    found: bool = True
    center_px: tuple[float, float] | None = (330.0, 240.0)


def _frame(
    frame_id: int,
    capture_timestamp: float,
    *,
    processing_timestamp: float | None = None,
    camera_name: str = "wrist",
) -> CameraFrame:
    if processing_timestamp is None:
        processing_timestamp = capture_timestamp
    return CameraFrame(
        camera_name=camera_name,
        frame_id=frame_id,
        capture_timestamp=capture_timestamp,
        processing_timestamp=processing_timestamp,
        image=np.zeros((2, 2, 3), dtype=np.uint8),
    )


def _tool_reference() -> ToolReferenceCalibration:
    return ToolReferenceCalibration(
        camera_name="wrist",
        image_size=(640, 480),
        u=320.0,
        v=240.0,
        sample_count=5,
        std_u_px=0.5,
        std_v_px=0.5,
    )


def _jacobian() -> ImageJacobianCalibration:
    return ImageJacobianCalibration(
        motion_frame="keyboard_plane_xy",
        motion_unit="mm",
        matrix=((2.0, 0.0), (0.0, 3.0)),
        sample_count=4,
        residual_rms_px=0.0,
        singular_values=(3.0, 2.0),
        condition_number=1.5,
        damping=0.0,
    )


def _runtime(
    *,
    stable_frames: int = 3,
    max_iterations: int = 20,
    timeout_s: float = 5.0,
    max_frame_age_ms: float = 100.0,
    max_step_norm: float = 10.0,
    max_total_correction_norm: float = 20.0,
) -> VisualServoRuntime:
    return VisualServoRuntime(
        target_label="G",
        tool_reference=_tool_reference(),
        image_jacobian=_jacobian(),
        servo_config=VisualServoConfig(
            gain=1.0,
            convergence_threshold_px=3.0,
            max_step_norm=max_step_norm,
            max_total_correction_norm=max_total_correction_norm,
        ),
        runtime_config=VisualServoRuntimeConfig(
            max_frame_age_ms=max_frame_age_ms,
            required_stable_frames=stable_frames,
            max_iterations=max_iterations,
            timeout_s=timeout_s,
        ),
        start_timestamp=100.0,
    )


class TestVisualServoRuntimeConfig(unittest.TestCase):
    def test_rejects_invalid_runtime_bounds(self):
        with self.assertRaisesRegex(ValueError, "max_frame_age_ms"):
            VisualServoRuntimeConfig(max_frame_age_ms=0.0)
        with self.assertRaisesRegex(TypeError, "required_stable_frames"):
            VisualServoRuntimeConfig(required_stable_frames=3.0)
        with self.assertRaisesRegex(ValueError, "required_stable_frames"):
            VisualServoRuntimeConfig(required_stable_frames=0)
        with self.assertRaisesRegex(TypeError, "max_iterations"):
            VisualServoRuntimeConfig(max_iterations=True)
        with self.assertRaisesRegex(ValueError, "max_iterations"):
            VisualServoRuntimeConfig(max_iterations=0)
        with self.assertRaisesRegex(ValueError, "timeout_s"):
            VisualServoRuntimeConfig(timeout_s=float("inf"))


class TestVisualServoRuntime(unittest.TestCase):
    def test_fresh_target_produces_bounded_correction(self):
        runtime = _runtime(max_step_norm=2.0)
        decision = runtime.process(
            _frame(1, 100.01),
            _Observation(center_px=(330.0, 240.0)),
            now_timestamp=100.02,
        )
        self.assertEqual(decision.status, VisualServoRuntimeStatus.CORRECTION)
        np.testing.assert_allclose(decision.error_px, (10.0, 0.0), atol=1e-12)
        np.testing.assert_allclose(decision.correction, (-2.0, 0.0), atol=1e-12)
        self.assertEqual(decision.iteration_count, 1)
        self.assertEqual(runtime.cumulative_correction, (-2.0, 0.0))
        self.assertFalse(decision.terminal)

    def test_old_frame_is_rejected_without_correction(self):
        runtime = _runtime(max_frame_age_ms=50.0)
        decision = runtime.process(
            _frame(1, 100.01),
            _Observation(),
            now_timestamp=100.20,
        )
        self.assertEqual(decision.status, VisualServoRuntimeStatus.STALE_FRAME)
        self.assertEqual(decision.correction, (0.0, 0.0))
        self.assertEqual(decision.iteration_count, 0)
        self.assertGreater(decision.frame_age_ms, 50.0)

    def test_duplicate_frame_id_is_rejected_and_breaks_stability_streak(self):
        runtime = _runtime(stable_frames=2)
        first = runtime.process(
            _frame(1, 100.01),
            _Observation(center_px=(321.0, 240.0)),
            now_timestamp=100.02,
        )
        self.assertEqual(first.status, VisualServoRuntimeStatus.STABILIZING)
        self.assertEqual(first.stable_frame_count, 1)

        duplicate = runtime.process(
            _frame(1, 100.01),
            _Observation(center_px=(321.0, 240.0)),
            now_timestamp=100.03,
        )
        self.assertEqual(duplicate.status, VisualServoRuntimeStatus.STALE_FRAME)
        self.assertEqual(duplicate.stable_frame_count, 0)

    def test_non_increasing_capture_timestamp_is_rejected(self):
        runtime = _runtime()
        runtime.process(
            _frame(1, 100.01),
            _Observation(),
            now_timestamp=100.02,
        )
        decision = runtime.process(
            _frame(2, 100.01),
            _Observation(),
            now_timestamp=100.03,
        )
        self.assertEqual(decision.status, VisualServoRuntimeStatus.STALE_FRAME)
        self.assertEqual(decision.iteration_count, 1)

    def test_target_loss_is_terminal_and_never_produces_correction(self):
        runtime = _runtime()
        decision = runtime.process(
            _frame(1, 100.01),
            _Observation(found=False, center_px=None),
            now_timestamp=100.02,
        )
        self.assertEqual(decision.status, VisualServoRuntimeStatus.TARGET_LOST)
        self.assertEqual(decision.correction, (0.0, 0.0))
        self.assertTrue(decision.terminal)
        self.assertEqual(runtime.terminal_status, VisualServoRuntimeStatus.TARGET_LOST)

    def test_alignment_requires_consecutive_fresh_in_tolerance_frames(self):
        runtime = _runtime(stable_frames=3)
        statuses = []
        for frame_id in range(1, 4):
            decision = runtime.process(
                _frame(frame_id, 100.00 + 0.01 * frame_id),
                _Observation(center_px=(321.0, 241.0)),
                now_timestamp=100.005 + 0.01 * frame_id,
            )
            statuses.append(decision.status)

        self.assertEqual(
            statuses,
            [
                VisualServoRuntimeStatus.STABILIZING,
                VisualServoRuntimeStatus.STABILIZING,
                VisualServoRuntimeStatus.ALIGNED,
            ],
        )
        self.assertEqual(runtime.stable_frame_count, 3)
        self.assertEqual(runtime.iteration_count, 0)
        self.assertEqual(runtime.terminal_status, VisualServoRuntimeStatus.ALIGNED)

    def test_outside_tolerance_resets_stability_streak(self):
        runtime = _runtime(stable_frames=2)
        first = runtime.process(
            _frame(1, 100.01),
            _Observation(center_px=(321.0, 240.0)),
            now_timestamp=100.02,
        )
        self.assertEqual(first.stable_frame_count, 1)

        correction = runtime.process(
            _frame(2, 100.03),
            _Observation(center_px=(330.0, 240.0)),
            now_timestamp=100.04,
        )
        self.assertEqual(correction.status, VisualServoRuntimeStatus.CORRECTION)
        self.assertEqual(correction.stable_frame_count, 0)

        third = runtime.process(
            _frame(3, 100.05),
            _Observation(center_px=(321.0, 240.0)),
            now_timestamp=100.06,
        )
        self.assertEqual(third.status, VisualServoRuntimeStatus.STABILIZING)
        self.assertEqual(third.stable_frame_count, 1)

    def test_max_iterations_blocks_an_additional_correction(self):
        runtime = _runtime(max_iterations=1)
        first = runtime.process(
            _frame(1, 100.01),
            _Observation(),
            now_timestamp=100.02,
        )
        self.assertEqual(first.status, VisualServoRuntimeStatus.CORRECTION)
        self.assertEqual(first.iteration_count, 1)

        second = runtime.process(
            _frame(2, 100.03),
            _Observation(),
            now_timestamp=100.04,
        )
        self.assertEqual(second.status, VisualServoRuntimeStatus.MAX_ITERATIONS)
        self.assertEqual(second.correction, (0.0, 0.0))
        self.assertEqual(second.iteration_count, 1)
        self.assertTrue(second.terminal)

    def test_max_iterations_does_not_block_final_stability_observation(self):
        runtime = _runtime(stable_frames=1, max_iterations=1)
        runtime.process(
            _frame(1, 100.01),
            _Observation(),
            now_timestamp=100.02,
        )
        aligned = runtime.process(
            _frame(2, 100.03),
            _Observation(center_px=(321.0, 240.0)),
            now_timestamp=100.04,
        )
        self.assertEqual(aligned.status, VisualServoRuntimeStatus.ALIGNED)
        self.assertEqual(aligned.iteration_count, 1)

    def test_timeout_is_terminal_without_correction(self):
        runtime = _runtime(timeout_s=0.1)
        decision = runtime.process(
            _frame(1, 100.05),
            _Observation(),
            now_timestamp=100.11,
        )
        self.assertEqual(decision.status, VisualServoRuntimeStatus.TIMEOUT)
        self.assertEqual(decision.correction, (0.0, 0.0))
        self.assertTrue(decision.terminal)

    def test_total_correction_budget_exhaustion_is_terminal(self):
        runtime = _runtime(
            max_iterations=5,
            max_step_norm=1.0,
            max_total_correction_norm=1.0,
        )
        first = runtime.process(
            _frame(1, 100.01),
            _Observation(),
            now_timestamp=100.02,
        )
        self.assertEqual(first.status, VisualServoRuntimeStatus.CORRECTION)
        np.testing.assert_allclose(first.cumulative_correction, (-1.0, 0.0), atol=1e-12)

        second = runtime.process(
            _frame(2, 100.03),
            _Observation(),
            now_timestamp=100.04,
        )
        self.assertEqual(second.status, VisualServoRuntimeStatus.BUDGET_EXHAUSTED)
        self.assertEqual(second.correction, (0.0, 0.0))
        self.assertEqual(second.iteration_count, 1)
        self.assertTrue(second.terminal)

    def test_camera_mismatch_is_rejected(self):
        runtime = _runtime()
        with self.assertRaisesRegex(ValueError, "expected camera"):
            runtime.process(
                _frame(1, 100.01, camera_name="top"),
                _Observation(),
                now_timestamp=100.02,
            )

    def test_target_label_mismatch_is_rejected(self):
        runtime = _runtime()
        with self.assertRaisesRegex(ValueError, "expected target"):
            runtime.process(
                _frame(1, 100.01),
                _Observation(target_label="H"),
                now_timestamp=100.02,
            )

    def test_found_observation_requires_center(self):
        runtime = _runtime()
        with self.assertRaisesRegex(ValueError, "center_px"):
            runtime.process(
                _frame(1, 100.01),
                _Observation(found=True, center_px=None),
                now_timestamp=100.02,
            )

    def test_terminal_runtime_rejects_further_processing(self):
        runtime = _runtime(stable_frames=1)
        runtime.process(
            _frame(1, 100.01),
            _Observation(center_px=(320.0, 240.0)),
            now_timestamp=100.02,
        )
        with self.assertRaisesRegex(RuntimeError, "terminal"):
            runtime.process(
                _frame(2, 100.03),
                _Observation(center_px=(320.0, 240.0)),
                now_timestamp=100.04,
            )

    def test_rejects_clock_inconsistencies(self):
        runtime = _runtime()
        with self.assertRaisesRegex(ValueError, "start_timestamp"):
            runtime.process(
                _frame(1, 100.01),
                _Observation(),
                now_timestamp=99.0,
            )

        with self.assertRaisesRegex(ValueError, "future"):
            runtime.process(
                _frame(1, 101.0),
                _Observation(),
                now_timestamp=100.5,
            )


if __name__ == "__main__":
    unittest.main()
