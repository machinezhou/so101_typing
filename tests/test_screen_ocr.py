import unittest

import numpy as np

from so101_typing.perception.screen_ocr import (
    ScreenOCRObservation,
    consensus_screen_text,
    normalize_ocr_text,
    parse_tesseract_tsv,
    preprocess_screen_roi,
)


class TestScreenOCR(unittest.TestCase):
    def observation(
        self,
        text: str,
        confidence: float,
    ) -> ScreenOCRObservation:
        return ScreenOCRObservation(
            raw_text=text,
            normalized_text=text,
            mean_confidence=confidence,
            word_count=1,
            processed_image=np.zeros(
                (2, 2),
                dtype=np.uint8,
            ),
        )

    def test_normalize_text(self):
        self.assertEqual(
            normalize_ocr_text(
                "  SO101   Phase4  \n\n  ABC  "
            ),
            "SO101 Phase4\nABC",
        )

    def test_preprocess_doubles_image(self):
        image = np.zeros(
            (20, 30, 3),
            dtype=np.uint8,
        )

        processed = preprocess_screen_roi(image)

        self.assertEqual(
            processed.shape,
            (40, 60),
        )

    def test_parse_tsv(self):
        tsv = (
            "level\tpage_num\tblock_num\tpar_num\tline_num\t"
            "word_num\tleft\ttop\twidth\theight\tconf\ttext\n"
            "5\t1\t1\t1\t1\t1\t0\t0\t10\t10\t92\tSO101\n"
            "5\t1\t1\t1\t1\t2\t20\t0\t10\t10\t88\tPhase4\n"
            "5\t1\t1\t1\t2\t1\t0\t20\t10\t10\t95\tABC\n"
        )

        text, confidence, count = parse_tesseract_tsv(
            tsv
        )

        self.assertEqual(
            text,
            "SO101 Phase4\nABC",
        )
        self.assertEqual(count, 3)
        self.assertAlmostEqual(
            confidence,
            (92 + 88 + 95) / 3,
        )

    def test_consensus_accepts_stable_majority(self):
        observations = [
            self.observation("ABC", 90),
            self.observation("ABC", 88),
            self.observation("ABC", 91),
            self.observation("ABC", 93),
            self.observation("A8C", 70),
        ]

        result = consensus_screen_text(
            observations,
            min_vote_fraction=0.60,
            min_mean_confidence=60.0,
        )

        self.assertTrue(result.stable)
        self.assertEqual(result.text, "ABC")
        self.assertEqual(result.votes, 4)

    def test_consensus_rejects_disagreement(self):
        observations = [
            self.observation("ABC", 90),
            self.observation("A8C", 90),
            self.observation("ABG", 90),
        ]

        result = consensus_screen_text(
            observations,
            min_vote_fraction=0.60,
            min_mean_confidence=60.0,
        )

        self.assertFalse(result.stable)

    def test_consensus_rejects_low_confidence(self):
        observations = [
            self.observation("ABC", 30),
            self.observation("ABC", 35),
            self.observation("ABC", 40),
        ]

        result = consensus_screen_text(
            observations,
            min_vote_fraction=0.60,
            min_mean_confidence=60.0,
        )

        self.assertFalse(result.stable)


if __name__ == "__main__":
    unittest.main()
