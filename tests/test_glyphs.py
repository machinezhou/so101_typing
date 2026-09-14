import unittest

import cv2
import numpy as np

from so101_typing.perception.glyphs import (
    GlyphPreprocessConfig,
    orient_glyph_crop,
    preprocess_glyph,
)


class TestGlyphPreprocessing(unittest.TestCase):
    def test_fixed_ccw_rotation(self):
        image = np.zeros((40, 60, 3), dtype=np.uint8)
        image[4:12, 6:14] = (255, 255, 255)

        rotated = orient_glyph_crop(image, 1)
        self.assertEqual(rotated.shape, (60, 40, 3))

        # Original top-left marker moves to bottom-left after 90° CCW.
        ys, xs = np.where(rotated[:, :, 0] > 0)
        self.assertGreater(float(ys.mean()), rotated.shape[0] / 2)
        self.assertLess(float(xs.mean()), rotated.shape[1] / 2)

    def test_preprocess_shape_and_dtype(self):
        image = np.full((64, 64, 3), 20, dtype=np.uint8)
        cv2.putText(
            image,
            "E",
            (20, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )
        result = preprocess_glyph(image)
        self.assertEqual(result.shape, (32, 32))
        self.assertEqual(result.dtype, np.uint8)
        self.assertGreater(int(result.max()), int(result.min()))

    def test_rejects_invalid_margin(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        config = GlyphPreprocessConfig(crop_margin=8)
        with self.assertRaises(ValueError):
            preprocess_glyph(image, config)


if __name__ == "__main__":
    unittest.main()
