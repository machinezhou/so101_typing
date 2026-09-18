import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from so101_typing.control.tool_reference import (
    LEGACY_TARGET_DERIVED,
    WRIST_TOOL_TIP,
    ToolReferenceCalibration,
)


class TestToolReferenceCalibration(unittest.TestCase):

    def test_old_json_is_loaded_as_legacy_not_direct_tip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tool_reference.json"
            path.write_text(
                json.dumps(
                    {
                        "camera_name": "wrist",
                        "image_size": [640, 480],
                        "u": 303.0,
                        "v": 227.5,
                        "sample_count": 7,
                        "std_u_px": 1.5,
                        "std_v_px": 1.3,
                    }
                ),
                encoding="utf-8",
            )
            calibration = ToolReferenceCalibration.load(path)

        assert calibration is not None
        self.assertEqual(calibration.reference_kind, LEGACY_TARGET_DERIVED)
        self.assertFalse(calibration.is_direct_tool_tip)
        with self.assertRaisesRegex(RuntimeError, "TIP CALIBRATION REQUIRED"):
            calibration.require_direct_tool_tip()

    def test_direct_tip_round_trip_preserves_reference_kind(self):
        calibration = ToolReferenceCalibration(
            camera_name="wrist",
            image_size=(640, 480),
            u=310.0,
            v=225.0,
            sample_count=5,
            std_u_px=0.5,
            std_v_px=0.6,
            reference_kind=WRIST_TOOL_TIP,
            method="direct_manual_click_browser",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tool_reference.json"
            calibration.save(path)
            loaded = ToolReferenceCalibration.load(path)
        assert loaded is not None
        self.assertTrue(loaded.is_direct_tool_tip)
        loaded.require_direct_tool_tip()
        self.assertEqual(loaded.reference_kind, WRIST_TOOL_TIP)

    def test_uncalibrated_placeholder_returns_none(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tool_reference.json"
            path.write_text(
                json.dumps(
                    {
                        "camera_name": "wrist",
                        "image_size": [640, 480],
                        "u": None,
                        "v": None,
                        "sample_count": 0,
                        "std_u_px": None,
                        "std_v_px": None,
                    }
                ),
                encoding="utf-8",
            )
            self.assertIsNone(ToolReferenceCalibration.load(path))

    def test_valid_json_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tool_reference.json"
            path.write_text(
                json.dumps(
                    {
                        "camera_name": "wrist",
                        "image_size": [640, 480],
                        "u": 321.5,
                        "v": 247.0,
                        "sample_count": 30,
                        "std_u_px": 0.8,
                        "std_v_px": 1.1,
                    }
                ),
                encoding="utf-8",
            )
            calibration = ToolReferenceCalibration.load(path)

        self.assertIsNotNone(calibration)
        assert calibration is not None
        self.assertEqual(calibration.center, (321.5, 247.0))
        self.assertEqual(calibration.sample_count, 30)

    def test_save_load_round_trip(self):
        calibration = ToolReferenceCalibration(
            camera_name="wrist",
            image_size=(640, 480),
            u=321.5,
            v=247.0,
            sample_count=30,
            std_u_px=0.8,
            std_v_px=1.1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "tool_reference.json"
            calibration.save(path)
            loaded = ToolReferenceCalibration.load(path)
            raw = path.read_text(encoding="utf-8")

        self.assertEqual(loaded, calibration)
        self.assertTrue(raw.endswith("\n"))

    def test_from_samples_uses_axiswise_median_and_records_statistics(self):
        samples = [
            [320.0, 240.0],
            [321.0, 241.0],
            [319.0, 239.0],
            [500.0, 100.0],
        ]
        calibration = ToolReferenceCalibration.from_samples(samples)

        self.assertEqual(calibration.center, (320.5, 239.5))
        self.assertEqual(calibration.sample_count, 4)
        self.assertGreaterEqual(calibration.std_u_px, 0.0)
        self.assertGreaterEqual(calibration.std_v_px, 0.0)
        self.assertAlmostEqual(
            calibration.std_u_px,
            float(np.std(np.asarray(samples)[:, 0], ddof=0)),
        )
        self.assertAlmostEqual(
            calibration.std_v_px,
            float(np.std(np.asarray(samples)[:, 1], ddof=0)),
        )
        array = np.asarray(samples, dtype=np.float64)
        median = np.median(array, axis=0)
        radial = np.linalg.norm(array - median, axis=1)
        self.assertAlmostEqual(
            calibration.rms_radial_deviation_px,
            float(np.sqrt(np.mean(radial**2))),
        )
        self.assertAlmostEqual(
            calibration.max_radial_deviation_px,
            float(np.max(radial)),
        )

    def test_legacy_file_without_radial_fields_keeps_them_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tool_reference.json"
            path.write_text(
                json.dumps(
                    {
                        "camera_name": "wrist",
                        "image_size": [640, 480],
                        "u": 303.0,
                        "v": 227.5,
                        "sample_count": 7,
                        "std_u_px": 1.5,
                        "std_v_px": 1.3,
                    }
                ),
                encoding="utf-8",
            )
            calibration = ToolReferenceCalibration.load(path)
        assert calibration is not None
        self.assertIsNone(calibration.rms_radial_deviation_px)
        self.assertIsNone(calibration.max_radial_deviation_px)

    def test_error_px_uses_readme_sign_convention(self):
        calibration = ToolReferenceCalibration(
            camera_name="wrist",
            image_size=(640, 480),
            u=320.0,
            v=240.0,
            sample_count=3,
            std_u_px=0.0,
            std_v_px=0.0,
        )
        self.assertEqual(calibration.error_px((330.0, 235.0)), (10.0, -5.0))

    def test_rejects_too_few_samples(self):
        with self.assertRaisesRegex(ValueError, "at least three"):
            ToolReferenceCalibration.from_samples([[1.0, 2.0], [3.0, 4.0]])

    def test_rejects_non_finite_samples(self):
        for bad in (float("nan"), float("inf")):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "finite"):
                ToolReferenceCalibration.from_samples(
                    [[10.0, 10.0], [20.0, 20.0], [bad, 30.0]]
                )

    def test_rejects_out_of_frame_samples(self):
        with self.assertRaisesRegex(ValueError, "image bounds"):
            ToolReferenceCalibration.from_samples(
                [[10.0, 10.0], [20.0, 20.0], [640.0, 30.0]]
            )

    def test_rejects_malformed_sample_shape(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            ToolReferenceCalibration.from_samples(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
            )

    def test_rejects_partially_null_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tool_reference.json"
            path.write_text(
                json.dumps(
                    {
                        "camera_name": "wrist",
                        "image_size": [640, 480],
                        "u": 320.0,
                        "v": None,
                        "sample_count": 3,
                        "std_u_px": 0.0,
                        "std_v_px": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "both"):
                ToolReferenceCalibration.load(path)

    def test_rejects_invalid_image_size(self):
        cases = (
            ((0, 480), ValueError),
            ((640, -1), ValueError),
            ((640.0, 480), TypeError),
        )
        for image_size, expected_error in cases:
            with self.subTest(image_size=image_size), self.assertRaises(expected_error):
                ToolReferenceCalibration(
                    camera_name="wrist",
                    image_size=image_size,
                    u=320.0,
                    v=240.0,
                    sample_count=3,
                    std_u_px=0.0,
                    std_v_px=0.0,
                )

    def test_rejects_negative_standard_deviation(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            ToolReferenceCalibration(
                camera_name="wrist",
                image_size=(640, 480),
                u=320.0,
                v=240.0,
                sample_count=3,
                std_u_px=-0.1,
                std_v_px=0.0,
            )

    def test_rejects_invalid_sample_count(self):
        with self.assertRaisesRegex(ValueError, "integer >= 1"):
            ToolReferenceCalibration(
                camera_name="wrist",
                image_size=(640, 480),
                u=320.0,
                v=240.0,
                sample_count=0,
                std_u_px=0.0,
                std_v_px=0.0,
            )


if __name__ == "__main__":
    unittest.main()
