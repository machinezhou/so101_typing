import unittest
from dataclasses import dataclass

from scripts.build_glyph_round2_dataset import match_anchor


@dataclass
class Candidate:
    center: tuple[float, float]


class TestRound2AnchorMatching(unittest.TestCase):
    def test_matches_nearest_normalized_center(self):
        candidates = [
            Candidate((64.0, 48.0)),
            Candidate((320.0, 240.0)),
            Candidate((576.0, 432.0)),
        ]
        index, distance = match_anchor(
            candidates,
            (0.5, 0.5),
            width=640,
            height=480,
            max_distance=0.02,
        )
        self.assertEqual(index, 1)
        self.assertAlmostEqual(distance, 0.0, places=6)

    def test_excluded_candidate_is_not_reused(self):
        candidates = [
            Candidate((320.0, 240.0)),
            Candidate((326.0, 240.0)),
        ]
        index, _ = match_anchor(
            candidates,
            (0.5, 0.5),
            width=640,
            height=480,
            max_distance=0.02,
            excluded={0},
        )
        self.assertEqual(index, 1)

    def test_rejects_far_anchor(self):
        candidates = [Candidate((64.0, 48.0))]
        with self.assertRaises(ValueError):
            match_anchor(
                candidates,
                (0.9, 0.9),
                width=640,
                height=480,
                max_distance=0.02,
            )


if __name__ == "__main__":
    unittest.main()
