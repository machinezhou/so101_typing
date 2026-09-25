import unittest

import numpy as np

from scripts.phase7_act_temporal_contract import (
    CHUNK_SIZE,
    FPS,
    _episode_lengths,
    _valid_steps_for_episode,
)


class TestPhase7ActTemporalContract(unittest.TestCase):
    def test_frozen_constants(self):
        self.assertEqual(FPS, 15)
        self.assertEqual(CHUNK_SIZE, 20)

    def test_episode_lengths(self):
        episode_indices = np.asarray(
            [
                0, 0, 0,
                1, 1,
                3, 3, 3, 3,
            ],
            dtype=np.int64,
        )

        lengths = _episode_lengths(
            episode_indices,
            [0, 1, 3],
        )

        self.assertEqual(
            lengths,
            {
                0: 3,
                1: 2,
                3: 4,
            },
        )

    def test_valid_steps_without_padding_at_start(self):
        values = _valid_steps_for_episode(
            episode_length=30,
            chunk_size=20,
        )

        self.assertEqual(len(values), 30)

        self.assertEqual(
            values[:5],
            [20, 20, 20, 20, 20],
        )

    def test_valid_steps_decay_at_episode_end(self):
        values = _valid_steps_for_episode(
            episode_length=30,
            chunk_size=20,
        )

        self.assertEqual(
            values[-5:],
            [5, 4, 3, 2, 1],
        )

    def test_full_window_count_matches_formula(self):
        episode_length = 30
        chunk_size = 20

        values = _valid_steps_for_episode(
            episode_length,
            chunk_size,
        )

        full_windows = sum(
            value == chunk_size
            for value in values
        )

        self.assertEqual(
            full_windows,
            episode_length - chunk_size + 1,
        )

        self.assertEqual(
            full_windows,
            11,
        )

    def test_chunk_larger_than_episode_has_no_full_window(self):
        values = _valid_steps_for_episode(
            episode_length=30,
            chunk_size=100,
        )

        full_windows = sum(
            value == 100
            for value in values
        )

        self.assertEqual(
            full_windows,
            0,
        )

        self.assertEqual(
            values[0],
            30,
        )

        self.assertEqual(
            values[-1],
            1,
        )

    def test_invalid_episode_length_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "episode_length must be positive",
        ):
            _valid_steps_for_episode(
                episode_length=0,
                chunk_size=20,
            )

    def test_invalid_chunk_size_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "chunk_size must be positive",
        ):
            _valid_steps_for_episode(
                episode_length=30,
                chunk_size=0,
            )


if __name__ == "__main__":
    unittest.main()
