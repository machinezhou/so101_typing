import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from so101_typing.control.image_jacobian import ImageJacobianCalibration


class TestImageJacobianCalibration(unittest.TestCase):
    def test_uncalibrated_placeholder_returns_none(self):
        payload = {
            "motion_frame": "keyboard_plane_xy",
            "motion_unit": "mm",
            "matrix": None,
            "sample_count": 0,
            "residual_rms_px": None,
            "singular_values": None,
            "condition_number": None,
            "damping": 1e-6,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image_jacobian.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(ImageJacobianCalibration.load(path))

    def test_from_samples_recovers_known_jacobian(self):
        expected = np.array([[2.0, -0.5], [0.25, 1.5]], dtype=np.float64)
        motions = np.array(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
                [0.5, 0.5],
                [-0.5, 0.5],
            ],
            dtype=np.float64,
        )
        image_deltas = motions @ expected.T

        calibration = ImageJacobianCalibration.from_samples(motions, image_deltas)

        np.testing.assert_allclose(calibration.array, expected, atol=1e-12)
        self.assertEqual(calibration.sample_count, 6)
        self.assertAlmostEqual(calibration.residual_rms_px, 0.0, places=12)
        self.assertGreaterEqual(calibration.condition_number, 1.0)

    def test_least_squares_handles_small_measurement_noise(self):
        expected = np.array([[3.0, 0.4], [-0.2, 2.0]], dtype=np.float64)
        motions = np.array(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
                [0.5, -0.5],
                [-0.5, -0.5],
            ],
            dtype=np.float64,
        )
        noise = np.array(
            [
                [0.02, -0.01],
                [-0.01, 0.02],
                [0.01, 0.01],
                [-0.02, -0.01],
                [0.01, -0.02],
                [0.00, 0.01],
                [-0.01, 0.00],
                [0.02, 0.01],
            ],
            dtype=np.float64,
        )
        image_deltas = motions @ expected.T + noise

        calibration = ImageJacobianCalibration.from_samples(motions, image_deltas)

        np.testing.assert_allclose(calibration.array, expected, atol=0.03)
        self.assertGreater(calibration.residual_rms_px, 0.0)
        self.assertLess(calibration.residual_rms_px, 0.05)

    def test_predict_image_delta_uses_column_vector_convention(self):
        calibration = ImageJacobianCalibration(
            motion_frame="keyboard_plane_xy",
            motion_unit="mm",
            matrix=((2.0, -1.0), (0.5, 3.0)),
            sample_count=4,
            residual_rms_px=0.0,
            singular_values=(3.2, 2.0),
            condition_number=1.6,
        )
        self.assertEqual(calibration.predict_image_delta((4.0, 2.0)), (6.0, 8.0))

    def test_save_load_round_trip(self):
        calibration = ImageJacobianCalibration(
            motion_frame="keyboard_plane_xy",
            motion_unit="mm",
            matrix=((2.0, -0.5), (0.25, 1.5)),
            sample_count=6,
            residual_rms_px=0.12,
            singular_values=(2.1, 1.4),
            condition_number=1.5,
            damping=1e-5,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "image_jacobian.json"
            calibration.save(path)
            loaded = ImageJacobianCalibration.load(path)
            self.assertEqual(loaded, calibration)
            self.assertTrue(path.read_text(encoding="utf-8").endswith("\n"))

    def test_rejects_too_few_samples(self):
        with self.assertRaisesRegex(ValueError, "at least 4"):
            ImageJacobianCalibration.from_samples(
                [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]],
                [[2.0, 0.0], [-2.0, 0.0], [0.0, 2.0]],
            )

    def test_rejects_mismatched_sample_counts(self):
        with self.assertRaisesRegex(ValueError, "equal length"):
            ImageJacobianCalibration.from_samples(
                [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
                [[2.0, 0.0], [-2.0, 0.0], [0.0, 2.0]],
            )

    def test_rejects_motion_samples_that_do_not_span_xy(self):
        with self.assertRaisesRegex(ValueError, "two independent XY directions"):
            ImageJacobianCalibration.from_samples(
                [[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0], [-2.0, 0.0]],
                [[2.0, 0.0], [-2.0, 0.0], [4.0, 0.0], [-4.0, 0.0]],
            )

    def test_rejects_rank_deficient_estimated_jacobian(self):
        motions = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
        image_deltas = [[2.0, 0.0], [-2.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]
        with self.assertRaisesRegex(ValueError, "rank 2"):
            ImageJacobianCalibration.from_samples(motions, image_deltas)

    def test_optional_condition_number_limit_rejects_bad_fit(self):
        jacobian = np.array([[1.0, 0.0], [0.0, 0.01]], dtype=np.float64)
        motions = np.array(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
            dtype=np.float64,
        )
        image_deltas = motions @ jacobian.T

        with self.assertRaisesRegex(ValueError, "ill-conditioned"):
            ImageJacobianCalibration.from_samples(
                motions,
                image_deltas,
                max_condition_number=50.0,
            )

    def test_rejects_non_finite_samples(self):
        motions = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, float("nan")]]
        image_deltas = [[2.0, 0.0], [-2.0, 0.0], [0.0, 2.0], [0.0, -2.0]]
        with self.assertRaisesRegex(ValueError, "finite"):
            ImageJacobianCalibration.from_samples(motions, image_deltas)

    def test_rejects_malformed_sample_shape(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            ImageJacobianCalibration.from_samples(
                [[1.0], [-1.0], [0.0], [0.0]],
                [[2.0, 0.0], [-2.0, 0.0], [0.0, 2.0], [0.0, -2.0]],
            )

    def test_rejects_invalid_max_condition_number(self):
        motions = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
        image_deltas = [[2.0, 0.0], [-2.0, 0.0], [0.0, 2.0], [0.0, -2.0]]
        with self.assertRaisesRegex(ValueError, "at least 1"):
            ImageJacobianCalibration.from_samples(
                motions,
                image_deltas,
                max_condition_number=0.5,
            )

    def test_calibrated_json_requires_diagnostics(self):
        payload = {
            "motion_frame": "keyboard_plane_xy",
            "motion_unit": "mm",
            "matrix": [[2.0, 0.0], [0.0, 2.0]],
            "damping": 1e-6,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image_jacobian.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing fields"):
                ImageJacobianCalibration.load(path)

    def test_rejects_invalid_damping(self):
        with self.assertRaisesRegex(ValueError, "damping"):
            ImageJacobianCalibration(
                motion_frame="keyboard_plane_xy",
                motion_unit="mm",
                matrix=((2.0, 0.0), (0.0, 2.0)),
                sample_count=4,
                residual_rms_px=0.0,
                singular_values=(2.0, 2.0),
                condition_number=1.0,
                damping=-1.0,
            )


if __name__ == "__main__":
    unittest.main()
