import unittest

import numpy as np

from so101_typing.perception.top_mask import (
    TopMaskConfig,
    analyze_and_mask,
)


class TestTopMask(unittest.TestCase):
    def test_top_mask(self):
        frame = np.full(
            (100, 100, 3),
            255,
            dtype=np.uint8,
        )

        config = TopMaskConfig(
            polygon=np.asarray(
                [
                    [10, 10],
                    [50, 10],
                    [50, 50],
                    [10, 50],
                ],
                dtype=np.int32,
            )
        )

        masked, stats = analyze_and_mask(
            frame,
            config,
        )

        self.assertTrue(stats["configured"])

        self.assertEqual(
            stats["bright_fraction_inside_screen"],
            1.0,
        )

        self.assertTrue(
            np.all(masked[20, 20] == 0)
        )


if __name__ == "__main__":
    unittest.main()
