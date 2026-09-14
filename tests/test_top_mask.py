import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from so101_typing.perception.top_mask import (
    TopMaskConfig,
    analyze_and_mask,
)


class TestTopMask(unittest.TestCase):
    def test_top_mask_and_statistics(self):
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        config = TopMaskConfig(
            polygon=np.asarray(
                [[10, 10], [50, 10], [50, 50], [10, 50]],
                dtype=np.int32,
            )
        )

        masked, stats = analyze_and_mask(frame, config)

        self.assertTrue(stats["configured"])
        self.assertEqual(stats["bright_fraction_inside_screen"], 1.0)
        self.assertEqual(stats["mean_inside_screen"], 255.0)
        self.assertEqual(stats["std_inside_screen"], 0.0)
        self.assertGreater(stats["screen_pixel_count"], 0)
        self.assertTrue(np.all(masked[20, 20] == 0))
        self.assertTrue(np.all(masked[5, 5] == 255))

    def test_unconfigured_mask_is_passthrough(self):
        frame = np.full((20, 30, 3), 123, dtype=np.uint8)
        masked, stats = analyze_and_mask(frame, TopMaskConfig(polygon=None))

        self.assertFalse(stats["configured"])
        self.assertEqual(stats["masked_fraction"], 0.0)
        self.assertTrue(np.array_equal(masked, frame))
        self.assertIsNot(masked, frame)

    def test_invalid_polygon_rejected(self):
        with self.assertRaises(ValueError):
            TopMaskConfig(polygon=[[1, 1], [2, 2]])

        with self.assertRaises(ValueError):
            TopMaskConfig(
                polygon=[[1, 1], [2, 2], [3, 3]],
            )

    def test_out_of_frame_polygon_rejected_at_use(self):
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        config = TopMaskConfig(
            polygon=[[1, 1], [31, 1], [31, 10], [1, 10]],
        )
        with self.assertRaises(ValueError):
            analyze_and_mask(frame, config)

    def test_save_load_round_trip(self):
        config = TopMaskConfig(
            polygon=[[2, 3], [20, 3], [20, 15], [2, 15]],
            brightness_threshold=240,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "top_screen_mask.json"
            config.save(path)
            loaded = TopMaskConfig.load(path)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded.brightness_threshold, 240)
        self.assertTrue(np.array_equal(loaded.polygon, config.polygon))
        self.assertEqual(payload["brightness_threshold"], 240)
        self.assertEqual(payload["polygon"][0], [2, 3])

    def test_brightness_threshold_validation(self):
        with self.assertRaises(ValueError):
            TopMaskConfig(polygon=None, brightness_threshold=256)


if __name__ == "__main__":
    unittest.main()
