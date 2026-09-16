import unittest

import numpy as np

from so101_typing.control.image_jacobian import ImageJacobianCalibration
from so101_typing.control.visual_servo import (
    VisualServoConfig,
    VisualServoStepStatus,
    compute_visual_servo_step,
    damped_pseudoinverse,
)


def _calibration(
    matrix=((2.0, 0.0), (0.0, 3.0)),
    *,
    damping=0.0,
):
    singular_values = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    return ImageJacobianCalibration(
        motion_frame="base_link_xy",
        motion_unit="mm",
        matrix=matrix,
        sample_count=4,
        residual_rms_px=0.0,
        singular_values=(float(singular_values[0]), float(singular_values[1])),
        condition_number=float(singular_values[0] / singular_values[1]),
        damping=damping,
    )


class TestDampedPseudoinverse(unittest.TestCase):
    def test_zero_damping_matches_inverse_for_full_rank_matrix(self):
        matrix = np.array([[2.0, -0.5], [0.25, 1.5]], dtype=np.float64)
        actual = damped_pseudoinverse(matrix, damping=0.0)
        np.testing.assert_allclose(actual, np.linalg.inv(matrix), atol=1e-12)

    def test_positive_damping_regularizes_small_singular_value(self):
        matrix = np.array([[2.0, 0.0], [0.0, 0.01]], dtype=np.float64)
        undamped = damped_pseudoinverse(matrix, damping=0.0)
        damped = damped_pseudoinverse(matrix, damping=0.1)
        self.assertLess(abs(damped[1, 1]), abs(undamped[1, 1]))

    def test_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            damped_pseudoinverse([[1.0, 0.0]], damping=0.0)
        with self.assertRaisesRegex(ValueError, "damping"):
            damped_pseudoinverse([[1.0, 0.0], [0.0, 1.0]], damping=-1.0)
        with self.assertRaisesRegex(ValueError, "full-rank"):
            damped_pseudoinverse([[1.0, 0.0], [0.0, 0.0]], damping=0.0)


class TestVisualServoStep(unittest.TestCase):
    def test_sign_convention_moves_against_positive_pixel_error(self):
        result = compute_visual_servo_step(
            (10.0, 0.0),
            _calibration(),
            config=VisualServoConfig(
                convergence_threshold_px=1.0,
                max_step_norm=10.0,
                max_total_correction_norm=20.0,
            ),
        )
        self.assertEqual(result.status, VisualServoStepStatus.CORRECTION)
        np.testing.assert_allclose(result.raw_correction, (-5.0, 0.0), atol=1e-12)
        np.testing.assert_allclose(result.correction, (-5.0, 0.0), atol=1e-12)

    def test_gain_scales_unbounded_correction(self):
        result = compute_visual_servo_step(
            (10.0, 0.0),
            _calibration(),
            config=VisualServoConfig(
                gain=0.5,
                convergence_threshold_px=1.0,
                max_step_norm=10.0,
                max_total_correction_norm=20.0,
            ),
        )
        np.testing.assert_allclose(result.correction, (-2.5, 0.0), atol=1e-12)

    def test_step_norm_is_bounded_without_changing_direction(self):
        result = compute_visual_servo_step(
            (12.0, 0.0),
            _calibration(),
            config=VisualServoConfig(
                convergence_threshold_px=1.0,
                max_step_norm=2.0,
                max_total_correction_norm=10.0,
            ),
        )
        self.assertTrue(result.step_limited)
        self.assertFalse(result.total_limited)
        self.assertAlmostEqual(result.correction_norm, 2.0)
        np.testing.assert_allclose(result.correction, (-2.0, 0.0), atol=1e-12)

    def test_total_bound_clips_step_at_servo_radius(self):
        result = compute_visual_servo_step(
            (-10.0, 0.0),
            _calibration(),
            config=VisualServoConfig(
                convergence_threshold_px=1.0,
                max_step_norm=10.0,
                max_total_correction_norm=5.0,
            ),
            cumulative_correction=(4.0, 0.0),
        )
        self.assertTrue(result.total_limited)
        np.testing.assert_allclose(result.correction, (1.0, 0.0), atol=1e-12)
        np.testing.assert_allclose(result.cumulative_after, (5.0, 0.0), atol=1e-12)

    def test_total_bound_allows_motion_back_toward_origin(self):
        result = compute_visual_servo_step(
            (4.0, 0.0),
            _calibration(),
            config=VisualServoConfig(
                convergence_threshold_px=1.0,
                max_step_norm=3.0,
                max_total_correction_norm=5.0,
            ),
            cumulative_correction=(5.0, 0.0),
        )
        self.assertFalse(result.total_limited)
        np.testing.assert_allclose(result.correction, (-2.0, 0.0), atol=1e-12)
        np.testing.assert_allclose(result.cumulative_after, (3.0, 0.0), atol=1e-12)

    def test_outward_motion_at_exhausted_budget_returns_zero(self):
        result = compute_visual_servo_step(
            (-4.0, 0.0),
            _calibration(),
            config=VisualServoConfig(
                convergence_threshold_px=1.0,
                max_step_norm=3.0,
                max_total_correction_norm=5.0,
            ),
            cumulative_correction=(5.0, 0.0),
        )
        self.assertEqual(result.status, VisualServoStepStatus.BUDGET_EXHAUSTED)
        self.assertTrue(result.total_limited)
        np.testing.assert_allclose(result.correction, (0.0, 0.0), atol=1e-12)

    def test_single_frame_inside_threshold_is_not_declared_converged(self):
        result = compute_visual_servo_step(
            (2.0, 0.0),
            _calibration(),
            config=VisualServoConfig(convergence_threshold_px=3.0),
        )
        self.assertEqual(result.status, VisualServoStepStatus.WITHIN_TOLERANCE)
        self.assertEqual(result.correction, (0.0, 0.0))

    def test_threshold_comparison_is_strict(self):
        result = compute_visual_servo_step(
            (3.0, 0.0),
            _calibration(),
            config=VisualServoConfig(
                convergence_threshold_px=3.0,
                max_step_norm=2.0,
                max_total_correction_norm=10.0,
            ),
        )
        self.assertEqual(result.status, VisualServoStepStatus.CORRECTION)

    def test_coupled_jacobian_matches_linear_solve_when_undamped(self):
        matrix = ((2.0, -0.5), (0.25, 1.5))
        error = np.array([4.0, -3.0], dtype=np.float64)
        result = compute_visual_servo_step(
            error,
            _calibration(matrix),
            config=VisualServoConfig(
                convergence_threshold_px=1.0,
                max_step_norm=100.0,
                max_total_correction_norm=100.0,
            ),
        )
        expected = -np.linalg.solve(np.asarray(matrix), error)
        np.testing.assert_allclose(result.correction, expected, atol=1e-12)

    def test_rejects_cumulative_correction_outside_total_bound(self):
        with self.assertRaisesRegex(ValueError, "already exceeds"):
            compute_visual_servo_step(
                (10.0, 0.0),
                _calibration(),
                config=VisualServoConfig(max_total_correction_norm=5.0),
                cumulative_correction=(5.1, 0.0),
            )

    def test_config_rejects_invalid_bounds(self):
        with self.assertRaisesRegex(ValueError, "gain"):
            VisualServoConfig(gain=0.0)
        with self.assertRaisesRegex(ValueError, "convergence_threshold_px"):
            VisualServoConfig(convergence_threshold_px=0.0)
        with self.assertRaisesRegex(ValueError, "max_step_norm"):
            VisualServoConfig(max_step_norm=0.0)
        with self.assertRaisesRegex(ValueError, "max_total_correction_norm"):
            VisualServoConfig(max_total_correction_norm=float("inf"))


if __name__ == "__main__":
    unittest.main()
