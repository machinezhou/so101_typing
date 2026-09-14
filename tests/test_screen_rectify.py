import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from so101_typing.perception.screen_rectify import ScreenCalibration


class TestScreenRectification(unittest.TestCase):
    def test_uncalibrated_screen_returns_none(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(
                json.dumps(
                    {
                        "source_points": None,
                        "output_size": [1280, 800],
                        "text_roi": None,
                    }
                ),
                encoding="utf-8",
            )

            calibration = ScreenCalibration.load(path)

            self.assertIsNone(calibration)

    def test_valid_calibration_rectifies_and_crops(self):
        calibration = ScreenCalibration(
            source_points=np.asarray(
                [
                    [10, 10],
                    [110, 10],
                    [110, 70],
                    [10, 70],
                ],
                dtype=np.float32,
            ),
            output_size=(200, 100),
            text_roi=(20, 15, 80, 30),
        )
        frame = np.zeros((80, 120, 3), dtype=np.uint8)

        rectified = calibration.rectify(frame)
        roi = calibration.crop_text_roi(rectified)

        self.assertEqual(rectified.shape, (100, 200, 3))
        self.assertIsNotNone(roi)
        assert roi is not None
        self.assertEqual(roi.shape, (30, 80, 3))

    def test_rejects_non_convex_source_points(self):
        with self.assertRaisesRegex(ValueError, "convex quadrilateral"):
            ScreenCalibration(
                source_points=np.asarray(
                    [
                        [0, 0],
                        [100, 0],
                        [0, 100],
                        [100, 100],
                    ],
                    dtype=np.float32,
                ),
                output_size=(1280, 800),
                text_roi=None,
            )

    def test_rejects_roi_outside_rectified_screen(self):
        with self.assertRaisesRegex(ValueError, "inside the rectified screen"):
            ScreenCalibration(
                source_points=np.asarray(
                    [
                        [0, 0],
                        [100, 0],
                        [100, 100],
                        [0, 100],
                    ],
                    dtype=np.float32,
                ),
                output_size=(1280, 800),
                text_roi=(1200, 700, 200, 200),
            )

    def test_save_and_load_round_trip(self):
        calibration = ScreenCalibration(
            source_points=np.asarray(
                [
                    [12, 20],
                    [600, 24],
                    [590, 460],
                    [18, 455],
                ],
                dtype=np.float32,
            ),
            output_size=(1280, 800),
            text_roi=(100, 120, 900, 420),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            calibration.save(path)
            loaded = ScreenCalibration.load(path)

        self.assertIsNotNone(loaded)
        assert loaded is not None
        np.testing.assert_allclose(loaded.source_points, calibration.source_points)
        self.assertEqual(loaded.output_size, calibration.output_size)
        self.assertEqual(loaded.text_roi, calibration.text_roi)


if __name__ == "__main__":
    unittest.main()
