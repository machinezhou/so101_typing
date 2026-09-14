import unittest

import numpy as np

from so101_typing.perception.glyph_runtime import GlyphPrediction
from so101_typing.perception.wrist_target import (
    KeycapGlyphObservation,
    select_target_observation,
)


def observation(label: str, accepted: bool, quality: float, index: int):
    prediction = GlyphPrediction(
        label=label,
        accepted=accepted,
        vote_fraction=1.0,
        similarity=0.95,
        similarity_threshold=0.80,
        quality_score=quality,
        votes={label: 9},
    )
    return KeycapGlyphObservation(
        candidate_index=index,
        center_px=(100.0 + index, 200.0),
        center_norm=(0.5, 0.5),
        error_norm=(0.0, 0.0),
        bbox=(90, 190, 20, 20),
        quad=np.zeros((4, 2), dtype=np.float32),
        prediction=prediction,
        rectangularity=0.9,
    )


class TestWristTarget(unittest.TestCase):
    def test_missing_when_only_rejected_target_exists(self):
        result = select_target_observation(
            "A",
            [observation("A", False, 0.9, 0)],
        )
        self.assertFalse(result.found)
        self.assertEqual(result.target_label, "A")

    def test_selects_highest_quality_accepted_target(self):
        result = select_target_observation(
            "a",
            [
                observation("A", True, 0.7, 1),
                observation("B", True, 0.95, 2),
                observation("A", True, 0.9, 3),
            ],
        )
        self.assertTrue(result.found)
        self.assertEqual(result.target_label, "A")
        self.assertEqual(result.candidate_index, 3)

    def test_rejects_invalid_target(self):
        with self.assertRaises(ValueError):
            select_target_observation("ENTER", [])


if __name__ == "__main__":
    unittest.main()
