import json
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
