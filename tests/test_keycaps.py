import unittest

import cv2
import numpy as np

from so101_typing.perception.keycaps import detect_keycaps


class TestKeycapDetection(unittest.TestCase):
    def test_detect_dark_keycap_candidate(self):
        image = np.full(
            (240, 320, 3),
            255,
            dtype=np.uint8,
        )

        cv2.rectangle(
            image,
            (100, 80),
            (155, 130),
            (20, 20, 20),
            -1,
        )

        candidates = detect_keycaps(image)

        self.assertEqual(len(candidates), 1)

        center = candidates[0].center

        self.assertGreater(center[0], 120)
        self.assertLess(center[0], 135)

        self.assertGreater(center[1], 100)
        self.assertLess(center[1], 115)


if __name__ == "__main__":
    unittest.main()
