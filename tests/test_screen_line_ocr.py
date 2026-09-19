import unittest

import cv2
import numpy as np

from so101_typing.perception.screen_ocr import (
    detect_screen_text_lines,
)


class TestScreenLineOCR(unittest.TestCase):
    def test_detects_two_text_lines(self):
        image = np.full(
            (220, 800, 3),
            255,
            dtype=np.uint8,
        )

        cv2.putText(
            image,
            "KEYPRESS",
            (40, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (50, 50, 50),
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            "SECONDLINE",
            (40, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (50, 50, 50),
            3,
            cv2.LINE_AA,
        )

        boxes = detect_screen_text_lines(
            image
        )

        self.assertEqual(
            len(boxes),
            2,
        )

        self.assertLess(
            boxes[0].y,
            boxes[1].y,
        )

    def test_blank_roi_has_no_lines(self):
        image = np.full(
            (220, 800, 3),
            255,
            dtype=np.uint8,
        )

        self.assertEqual(
            detect_screen_text_lines(image),
            (),
        )


if __name__ == "__main__":
    unittest.main()
