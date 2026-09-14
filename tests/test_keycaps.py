import unittest

import cv2
import numpy as np

from so101_typing.perception.keycaps import (
    KeycapCandidate,
    detect_keycaps,
    order_quad_points,
    rectify_keycap,
)


class TestKeycapDetection(unittest.TestCase):
    def test_detect_dark_keycap_candidate(self):
        image = np.full((240, 320, 3), 230, dtype=np.uint8)
        cv2.rectangle(image, (100, 80), (155, 130), (20, 20, 20), -1)
        cv2.rectangle(image, (100, 80), (155, 130), (180, 180, 180), 2)

        candidates = detect_keycaps(image)
        self.assertEqual(len(candidates), 1)

        center = candidates[0].center
        self.assertGreater(center[0], 120)
        self.assertLess(center[0], 135)
        self.assertGreater(center[1], 100)
        self.assertLess(center[1], 115)

    def test_rejects_bright_rectangle(self):
        image = np.full((240, 320, 3), 25, dtype=np.uint8)
        cv2.rectangle(image, (100, 80), (155, 130), (235, 235, 235), -1)
        cv2.rectangle(image, (100, 80), (155, 130), (80, 80, 80), 2)

        candidates = detect_keycaps(image)
        self.assertEqual(candidates, [])

    def test_detects_rotated_dark_keycap(self):
        image = np.full((260, 340, 3), 220, dtype=np.uint8)
        rect = ((170.0, 130.0), (58.0, 52.0), 27.0)
        quad = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillConvexPoly(image, quad, (20, 20, 20))
        cv2.polylines(image, [quad], True, (175, 175, 175), 2)

        candidates = detect_keycaps(image)
        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0].center[0], 170.0, delta=4.0)
        self.assertAlmostEqual(candidates[0].center[1], 130.0, delta=4.0)

    def test_order_quad_points(self):
        quad = np.array(
            [[90, 30], [30, 90], [100, 100], [20, 20]],
            dtype=np.float32,
        )
        ordered = order_quad_points(quad)
        np.testing.assert_allclose(ordered[0], [20, 20])
        np.testing.assert_allclose(ordered[1], [90, 30])
        np.testing.assert_allclose(ordered[2], [100, 100])
        np.testing.assert_allclose(ordered[3], [30, 90])

    def test_rectify_keycap_shape(self):
        image = np.zeros((160, 200, 3), dtype=np.uint8)
        rect = ((100.0, 80.0), (70.0, 50.0), -20.0)
        quad = cv2.boxPoints(rect).astype(np.float32)
        cv2.fillConvexPoly(image, quad.astype(np.int32), (50, 50, 50))
        cv2.putText(
            image,
            "A",
            (88, 91),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )

        candidate = KeycapCandidate(
            bbox=(60, 45, 80, 70),
            center=(100.0, 80.0),
            area=3500.0,
            aspect_ratio=1.1,
            rectangularity=0.9,
            quad=quad,
        )
        crop = rectify_keycap(image, candidate, (64, 64))
        self.assertEqual(crop.shape, (64, 64, 3))
        self.assertGreater(float(crop.mean()), 5.0)


if __name__ == "__main__":
    unittest.main()
