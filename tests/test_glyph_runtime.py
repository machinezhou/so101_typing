import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from so101_typing.perception.glyph_runtime import RuntimeHOGGlyphRecognizer


class TestGlyphRuntime(unittest.TestCase):
    @staticmethod
    def glyph(letter: str, dx: int = 0) -> np.ndarray:
        image = np.full((32, 32), 20, np.uint8)
        cv2.putText(
            image,
            letter,
            (5 + dx, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            235,
            2,
            cv2.LINE_AA,
        )
        return image

    def model(self) -> RuntimeHOGGlyphRecognizer:
        labels = ["A", "A", "B", "B"]
        images = [
            self.glyph("A", 0),
            self.glyph("A", 1),
            self.glyph("B", 0),
            self.glyph("B", 1),
        ]
        return RuntimeHOGGlyphRecognizer.fit(labels, images, augment=True)

    def test_prediction_contains_acceptance_signals(self):
        model = self.model()
        prediction = model.predict(self.glyph("A"))
        self.assertEqual(prediction.label, "A")
        self.assertGreaterEqual(prediction.vote_fraction, 0.0)
        self.assertLessEqual(prediction.vote_fraction, 1.0)
        self.assertGreaterEqual(prediction.quality_score, 0.0)
        self.assertLessEqual(prediction.quality_score, 1.0)

    def test_save_load_round_trip(self):
        model = self.model()
        before = model.predict(self.glyph("B"))
        with tempfile.TemporaryDirectory() as directory:
            model.save(directory)
            loaded = RuntimeHOGGlyphRecognizer.load(directory)
            after = loaded.predict(self.glyph("B"))
            self.assertEqual(before.label, after.label)
            self.assertEqual(before.accepted, after.accepted)
            self.assertAlmostEqual(before.similarity, after.similarity, places=5)
            self.assertTrue((Path(directory) / "svm.xml").exists())
            self.assertTrue((Path(directory) / "model.json").exists())
            self.assertTrue((Path(directory) / "centroids.npz").exists())


if __name__ == "__main__":
    unittest.main()
