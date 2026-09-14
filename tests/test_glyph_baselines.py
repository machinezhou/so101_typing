import unittest

import cv2
import numpy as np

from so101_typing.perception.glyph_baselines import (
    HOGLinearSVMGlyphClassifier,
    TemplateGlyphClassifier,
    augment_glyph,
    shifted_normalized_correlation,
)


class TestGlyphBaselines(unittest.TestCase):
    @staticmethod
    def glyph(letter: str) -> np.ndarray:
        image = np.full((32, 32), 20, np.uint8)
        cv2.putText(
            image,
            letter,
            (5, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            235,
            2,
            cv2.LINE_AA,
        )
        return image

    def test_template_self_match(self):
        a = self.glyph("A")
        b = self.glyph("B")
        model = TemplateGlyphClassifier()
        model.fit(["A", "B"], [a, b])
        self.assertEqual(model.predict(a), "A")

    def test_correlation_prefers_same_glyph(self):
        a = self.glyph("A")
        b = self.glyph("B")
        self.assertGreater(
            shifted_normalized_correlation(a, a),
            shifted_normalized_correlation(a, b),
        )

    def test_hog_descriptor_size(self):
        model = HOGLinearSVMGlyphClassifier()
        feature = model.feature(self.glyph("A"))
        self.assertEqual(feature.shape, (324,))

    def test_augmentation_does_not_use_quarter_turns(self):
        image = self.glyph("R")
        variants = augment_glyph(image)
        self.assertEqual(len(variants), 9)
        self.assertTrue(np.array_equal(variants[0], image))

    def test_hog_svm_can_fit_simple_synthetic_classes(self):
        a = self.glyph("A")
        b = self.glyph("B")
        model = HOGLinearSVMGlyphClassifier()
        model.fit(["A", "A", "B", "B"], [a, a, b, b], augment=False)
        self.assertEqual(model.predict(a), "A")
        self.assertEqual(model.predict(b), "B")


if __name__ == "__main__":
    unittest.main()
