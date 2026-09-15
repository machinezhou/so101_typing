from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from so101_typing.contracts import CameraFrame
from so101_typing.control.image_jacobian import ImageJacobianCalibration
from so101_typing.control.tool_reference import ToolReferenceCalibration
from so101_typing.control.visual_servo import VisualServoConfig
from so101_typing.control.visual_servo_metrics import (
    VisualServoAttemptMetrics,
    VisualServoAttemptRecorder,
    summarize_visual_servo_attempts,
)
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


def _frame(frame_id: int, capture_timestamp: float) -> CameraFrame:
    return CameraFrame(
        camera_name="wrist",
        frame_id=frame_id,
        capture_timestamp=capture_timestamp,
        processing_timestamp=capture_timestamp,
        image=np.zeros((2, 2, 3), dtype=np.uint8),
    )


def _tool_reference() -> ToolReferenceCalibration:
    return ToolReferenceCalibration(
        camera_name="wrist",
        image_size=(640, 480),
        u=320.0,
        v=240.0,
        sample_count=5,
        std_u_px=0.4,
        std_v_px=0.6,
    )


def _jacobian() -> ImageJacobianCalibration:
    return ImageJacobianCalibration(
        motion_frame="keyboard_plane_xy",
        motion_unit="mm",
        matrix=((2.0, 0.0), (0.0, 3.0)),
        sample_count=8,
        residual_rms_px=0.25,
        singular_values=(3.0, 2.0),
        condition_number=1.5,
        damping=0.0,
    )


def _runtime(*, stable_frames: int = 2, timeout_s: float = 2.0) -> VisualServoRuntime:
    return VisualServoRuntime(
        target_label="G",
        tool_reference=_tool_reference(),
        image_jacobian=_jacobian(),
        servo_config=VisualServoConfig(
            gain=1.0,
            convergence_threshold_px=3.0,
            max_step_norm=2.0,
            max_total_correction_norm=10.0,
        ),
        runtime_config=VisualServoRuntimeConfig(
            max_frame_age_ms=100.0,
            required_stable_frames=stable_frames,
            max_iterations=5,
            timeout_s=timeout_s,
        ),
        start_timestamp=100.0,
    )


def _recorder(
    attempt_id: str = "attempt-001",
    *,
    stable_frames: int = 2,
    timeout_s: float = 2.0,
) -> VisualServoAttemptRecorder:
    return VisualServoAttemptRecorder(
        attempt_id=attempt_id,
        target_label="G",
        tool_reference=_tool_reference(),
        image_jacobian=_jacobian(),
        servo_config=VisualServoConfig(
            gain=1.0,
            convergence_threshold_px=3.0,
            max_step_norm=2.0,
            max_total_correction_norm=10.0,
        ),
        runtime_config=VisualServoRuntimeConfig(
            max_frame_age_ms=100.0,
            required_stable_frames=stable_frames,
            max_iterations=5,
            timeout_s=timeout_s,
        ),
    )


def _aligned_metrics(
    *,
    attempt_id: str = "aligned",
    final_error: float = 1.0,
    iterations: int = 2,
    elapsed_s: float = 0.5,
) -> VisualServoAttemptMetrics:
    return VisualServoAttemptMetrics(
        attempt_id=attempt_id,
        target_label="G",
        terminal_status=VisualServoRuntimeStatus.ALIGNED,
        decision_count=iterations + 2,
        correction_count=iterations,
        stale_frame_count=0,
        iteration_count=iterations,
        elapsed_s=elapsed_s,
        initial_error_px=(10.0, 0.0),
        initial_error_norm_px=10.0,
        final_observed_error_px=(final_error, 0.0),
        final_observed_error_norm_px=final_error,
        max_frame_age_ms=20.0,
        cumulative_correction=(-4.0, 0.0),
        cumulative_correction_norm=4.0,
        motion_frame="keyboard_plane_xy",
        motion_unit="mm",
        servo_gain=1.0,
        convergence_threshold_px=3.0,
        max_step_norm=2.0,
        max_total_correction_norm=10.0,
        max_frame_age_limit_ms=100.0,
        required_stable_frames=2,
        max_iterations=5,
        timeout_s=2.0,
        tool_reference_sample_count=5,
        tool_reference_std_u_px=0.4,
        tool_reference_std_v_px=0.6,
        jacobian_sample_count=8,
        jacobian_residual_rms_px=0.25,
        jacobian_condition_number=1.5,
        jacobian_damping=0.0,
    )


class TestVisualServoAttemptRecorder(unittest.TestCase):
    def test_aligned_attempt_captures_acceptance_metrics(self):
        runtime = _runtime(stable_frames=2)
        recorder = _recorder()

        decisions = [
            runtime.process(_frame(1, 100.01), _Observation(), now_timestamp=100.02),
            runtime.process(_frame(1, 100.01), _Observation(), now_timestamp=100.03),
            runtime.process(
                _frame(2, 100.04),
                _Observation(center_px=(321.0, 240.0)),
                now_timestamp=100.05,
            ),
            runtime.process(
                _frame(3, 100.06),
                _Observation(center_px=(320.5, 240.0)),
                now_timestamp=100.07,
            ),
        ]
        for decision in decisions:
            recorder.record(decision)

        metrics = recorder.finalize()
        self.assertTrue(metrics.success)
        self.assertEqual(metrics.terminal_status, VisualServoRuntimeStatus.ALIGNED)
        self.assertEqual(metrics.decision_count, 4)
        self.assertEqual(metrics.correction_count, 1)
        self.assertEqual(metrics.stale_frame_count, 1)
        self.assertEqual(metrics.iteration_count, 1)
        self.assertEqual(metrics.initial_error_px, (10.0, 0.0))
        self.assertAlmostEqual(metrics.initial_error_norm_px, 10.0)
        self.assertEqual(metrics.final_observed_error_px, (0.5, 0.0))
        self.assertAlmostEqual(metrics.final_observed_error_norm_px, 0.5)
        self.assertAlmostEqual(metrics.elapsed_s, 0.07)
        self.assertGreater(metrics.max_frame_age_ms, 0.0)
        self.assertEqual(metrics.cumulative_correction, (-2.0, 0.0))
        self.assertEqual(metrics.tool_reference_sample_count, 5)
        self.assertEqual(metrics.jacobian_sample_count, 8)
        self.assertAlmostEqual(metrics.jacobian_condition_number, 1.5)
        self.assertAlmostEqual(metrics.convergence_threshold_px, 3.0)
        self.assertEqual(metrics.required_stable_frames, 2)
        self.assertEqual(metrics.max_iterations, 5)
        self.assertAlmostEqual(metrics.timeout_s, 2.0)
        self.assertAlmostEqual(metrics.jacobian_damping, 0.0)

    def test_target_loss_before_any_valid_error_is_recordable(self):
        runtime = _runtime()
        recorder = _recorder()
        recorder.record(
            runtime.process(
                _frame(1, 100.01),
                _Observation(found=False, center_px=None),
                now_timestamp=100.02,
            )
        )
        metrics = recorder.finalize()
        self.assertFalse(metrics.success)
        self.assertEqual(metrics.terminal_status, VisualServoRuntimeStatus.TARGET_LOST)
        self.assertIsNone(metrics.initial_error_px)
        self.assertIsNone(metrics.final_observed_error_px)

    def test_timeout_retains_last_valid_observed_error(self):
        runtime = _runtime(timeout_s=0.1)
        recorder = _recorder(timeout_s=0.1)
        recorder.record(
            runtime.process(_frame(1, 100.01), _Observation(), now_timestamp=100.02)
        )
        recorder.record(
            runtime.process(_frame(2, 100.10), _Observation(), now_timestamp=100.11)
        )
        metrics = recorder.finalize()
        self.assertEqual(metrics.terminal_status, VisualServoRuntimeStatus.TIMEOUT)
        self.assertEqual(metrics.initial_error_px, (10.0, 0.0))
        self.assertEqual(metrics.final_observed_error_px, (10.0, 0.0))
        self.assertEqual(metrics.iteration_count, 1)

    def test_correction_count_tracks_runtime_iterations(self):
        runtime = _runtime(stable_frames=1)
        recorder = _recorder(stable_frames=1)
        first = runtime.process(_frame(1, 100.01), _Observation(), now_timestamp=100.02)
        second = runtime.process(
            _frame(2, 100.03),
            _Observation(center_px=(320.0, 240.0)),
            now_timestamp=100.04,
        )
        recorder.record(first)
        recorder.record(second)
        metrics = recorder.finalize()
        self.assertEqual(metrics.correction_count, 1)
        self.assertEqual(metrics.iteration_count, 1)

    def test_record_after_terminal_is_rejected(self):
        runtime = _runtime(stable_frames=1)
        recorder = _recorder(stable_frames=1)
        terminal = runtime.process(
            _frame(1, 100.01),
            _Observation(center_px=(320.0, 240.0)),
            now_timestamp=100.02,
        )
        recorder.record(terminal)
        with self.assertRaisesRegex(RuntimeError, "terminal"):
            recorder.record(terminal)

    def test_finalize_before_terminal_is_rejected(self):
        runtime = _runtime()
        recorder = _recorder()
        recorder.record(
            runtime.process(_frame(1, 100.01), _Observation(), now_timestamp=100.02)
        )
        with self.assertRaisesRegex(RuntimeError, "terminal"):
            recorder.finalize()

    def test_mismatched_motion_frame_is_rejected(self):
        runtime = _runtime()
        recorder = _recorder()
        decision = runtime.process(_frame(1, 100.01), _Observation(), now_timestamp=100.02)
        with self.assertRaisesRegex(ValueError, "motion_frame"):
            recorder.record(replace(decision, motion_frame="other_frame"))

    def test_mismatched_motion_unit_is_rejected(self):
        runtime = _runtime()
        recorder = _recorder()
        decision = runtime.process(_frame(1, 100.01), _Observation(), now_timestamp=100.02)
        with self.assertRaisesRegex(ValueError, "motion_unit"):
            recorder.record(replace(decision, motion_unit="m"))

    def test_recorder_requires_decisions_from_attempt_start(self):
        runtime = _runtime()
        recorder = _recorder()
        decision = runtime.process(_frame(1, 100.01), _Observation(), now_timestamp=100.02)
        with self.assertRaisesRegex(ValueError, "iteration_count"):
            recorder.record(replace(decision, status=VisualServoRuntimeStatus.STABILIZING))

    def test_mismatched_required_stable_frames_is_rejected(self):
        runtime = _runtime(stable_frames=3)
        recorder = _recorder(stable_frames=2)
        decision = runtime.process(_frame(1, 100.01), _Observation(), now_timestamp=100.02)
        with self.assertRaisesRegex(ValueError, "required_stable_frames"):
            recorder.record(decision)

    def test_inconsistent_decision_error_norm_is_rejected(self):
        runtime = _runtime()
        recorder = _recorder()
        decision = runtime.process(_frame(1, 100.01), _Observation(), now_timestamp=100.02)
        with self.assertRaisesRegex(ValueError, "error norm"):
            recorder.record(replace(decision, error_norm_px=999.0))


class TestVisualServoAttemptMetrics(unittest.TestCase):
    def test_save_load_round_trip(self):
        metrics = _aligned_metrics()
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "attempt.json"
            metrics.save(path)
            loaded = VisualServoAttemptMetrics.load(path)
            self.assertEqual(loaded, metrics)
            self.assertTrue(path.read_text(encoding="utf-8").endswith("\n"))

    def test_unknown_schema_version_is_rejected(self):
        data = _aligned_metrics().to_dict()
        data["schema_version"] = 999
        with self.assertRaisesRegex(ValueError, "schema_version"):
            VisualServoAttemptMetrics.from_dict(data)

    def test_inconsistent_serialized_success_flag_is_rejected(self):
        data = _aligned_metrics().to_dict()
        data["success"] = False
        with self.assertRaisesRegex(ValueError, "success"):
            VisualServoAttemptMetrics.from_dict(data)

    def test_aligned_attempt_requires_final_error(self):
        metrics = _aligned_metrics()
        with self.assertRaisesRegex(ValueError, "aligned"):
            replace(
                metrics,
                final_observed_error_px=None,
                final_observed_error_norm_px=None,
            )

    def test_aligned_attempt_requires_error_below_recorded_threshold(self):
        with self.assertRaisesRegex(ValueError, "convergence_threshold_px"):
            replace(
                _aligned_metrics(),
                final_observed_error_px=(3.0, 0.0),
                final_observed_error_norm_px=3.0,
            )

    def test_cumulative_norm_must_match_vector(self):
        with self.assertRaisesRegex(ValueError, "cumulative_correction_norm"):
            replace(_aligned_metrics(), cumulative_correction_norm=99.0)

    def test_json_file_must_be_object(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "attempt.json"
            path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            with self.assertRaisesRegex(TypeError, "JSON object"):
                VisualServoAttemptMetrics.load(path)


class TestVisualServoEvaluationSummary(unittest.TestCase):
    def test_summary_reports_phase3_acceptance_metrics(self):
        aligned_a = _aligned_metrics(
            attempt_id="a",
            final_error=1.0,
            iterations=2,
            elapsed_s=0.4,
        )
        aligned_b = _aligned_metrics(
            attempt_id="b",
            final_error=2.0,
            iterations=4,
            elapsed_s=0.8,
        )
        timeout = replace(
            _aligned_metrics(attempt_id="timeout"),
            terminal_status=VisualServoRuntimeStatus.TIMEOUT,
        )
        target_loss = replace(
            _aligned_metrics(attempt_id="lost"),
            terminal_status=VisualServoRuntimeStatus.TARGET_LOST,
        )

        summary = summarize_visual_servo_attempts(
            [aligned_a, aligned_b, timeout, target_loss]
        )
        self.assertEqual(summary.attempt_count, 4)
        self.assertEqual(summary.aligned_count, 2)
        self.assertAlmostEqual(summary.convergence_rate, 0.5)
        self.assertAlmostEqual(summary.failure_rate, 0.5)
        self.assertAlmostEqual(summary.timeout_rate, 0.25)
        self.assertAlmostEqual(summary.target_loss_rate, 0.25)
        self.assertAlmostEqual(summary.mean_initial_error_px, 10.0)
        self.assertAlmostEqual(summary.mean_final_error_px_aligned, 1.5)
        self.assertAlmostEqual(summary.max_final_error_px_aligned, 2.0)
        self.assertAlmostEqual(summary.mean_iterations_aligned, 3.0)
        self.assertAlmostEqual(summary.mean_convergence_time_s_aligned, 0.6)

    def test_summary_without_aligned_attempts_has_null_success_aggregates(self):
        timeout = replace(
            _aligned_metrics(),
            terminal_status=VisualServoRuntimeStatus.TIMEOUT,
        )
        summary = summarize_visual_servo_attempts([timeout])
        self.assertEqual(summary.convergence_rate, 0.0)
        self.assertEqual(summary.failure_rate, 1.0)
        self.assertIsNone(summary.mean_final_error_px_aligned)
        self.assertIsNone(summary.mean_iterations_aligned)
        self.assertIsNone(summary.mean_convergence_time_s_aligned)

    def test_summary_rejects_empty_input(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            summarize_visual_servo_attempts([])


if __name__ == "__main__":
    unittest.main()
