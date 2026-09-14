import unittest
from unittest.mock import patch

import numpy as np

from so101_typing.adapters.cameras import (
    CameraSpec,
    ThreadedOpenCVCamera,
)
from so101_typing.contracts import CameraFrame


class TestCameraFrame(unittest.TestCase):
    def test_frame_age_ms(self):
        frame = CameraFrame(
            camera_name="test",
            frame_id=1,
            capture_timestamp=10.000,
            processing_timestamp=10.025,
            image=np.zeros(
                (10, 10, 3),
                dtype=np.uint8,
            ),
        )

        self.assertAlmostEqual(
            frame.frame_age_ms,
            25.0,
            places=6,
        )

    def test_frame_age_never_negative(self):
        frame = CameraFrame(
            camera_name="test",
            frame_id=1,
            capture_timestamp=10.100,
            processing_timestamp=10.000,
            image=np.zeros(
                (10, 10, 3),
                dtype=np.uint8,
            ),
        )

        self.assertEqual(
            frame.frame_age_ms,
            0.0,
        )


class TestThreadedOpenCVCamera(unittest.TestCase):
    def test_latest_sets_processing_timestamp(self):
        spec = CameraSpec(
            name="test",
            index_or_path=0,
            width=640,
            height=480,
            fps=30,
            fourcc="YUYV",
        )

        camera = ThreadedOpenCVCamera(spec)

        source_image = np.zeros(
            (10, 10, 3),
            dtype=np.uint8,
        )

        stored_frame = CameraFrame(
            camera_name="test",
            frame_id=42,
            capture_timestamp=5.000,
            processing_timestamp=5.000,
            image=source_image,
        )

        with camera._lock:
            camera._latest = stored_frame

        with patch(
            "so101_typing.adapters.cameras.time.monotonic",
            return_value=5.025,
        ):
            result = camera.latest(
                copy_image=True
            )

        self.assertIsNotNone(result)

        assert result is not None

        self.assertEqual(
            result.camera_name,
            "test",
        )

        self.assertEqual(
            result.frame_id,
            42,
        )

        self.assertEqual(
            result.capture_timestamp,
            5.000,
        )

        self.assertEqual(
            result.processing_timestamp,
            5.025,
        )

        self.assertAlmostEqual(
            result.frame_age_ms,
            25.0,
            places=6,
        )

        self.assertIsNot(
            result.image,
            source_image,
        )


if __name__ == "__main__":
    unittest.main()
