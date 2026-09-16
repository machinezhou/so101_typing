import unittest

import numpy as np

from so101_typing.control.target_measurement import (
    TargetBurstConfig,
    robust_target_center,
)


class TestTargetBurstMeasurement(unittest.TestCase):
    def test_h31_sample_5_rejects_single_near_contact_occlusion_outlier(self):
        centers = np.array(
            [
                [440.362, 286.155],
                [302.000, 227.500],
                [302.000, 227.500],
                [302.000, 227.500],
                [302.000, 227.000],
            ]
        )

        result = robust_target_center(
            centers
        )

        self.assertTrue(result.accepted)
        self.assertEqual(
            result.inlier_count,
            4,
        )
        self.assertEqual(
            result.outlier_indices,
            (0,),
        )
        np.testing.assert_allclose(
            result.center_px,
            (302.0, 227.5),
            atol=1e-12,
        )

    def test_h31_sample_7_rejects_single_near_contact_occlusion_outlier(self):
        centers = np.array(
            [
                [301.000, 224.500],
                [302.000, 226.000],
                [302.500, 226.000],
                [440.000, 284.500],
                [302.000, 226.500],
            ]
        )

        result = robust_target_center(
            centers
        )

        self.assertTrue(result.accepted)
        self.assertEqual(
            result.inlier_count,
            4,
        )
        self.assertEqual(
            result.outlier_indices,
            (3,),
        )
        np.testing.assert_allclose(
            result.center_px,
            (302.0, 226.0),
            atol=1e-12,
        )

    def test_normal_h31_burst_keeps_all_frames(self):
        centers = np.array(
            [
                [303.424, 226.469],
                [304.467, 226.502],
                [304.501, 226.534],
                [302.938, 225.697],
                [302.380, 225.137],
            ]
        )

        result = robust_target_center(
            centers
        )

        self.assertTrue(result.accepted)
        self.assertEqual(
            result.inlier_count,
            5,
        )
        self.assertEqual(
            result.outlier_indices,
            (),
        )

    def test_rejects_when_no_four_frame_consensus_exists(self):
        centers = np.array(
            [
                [100.0, 100.0],
                [120.0, 100.0],
                [140.0, 100.0],
                [160.0, 100.0],
                [180.0, 100.0],
            ]
        )

        result = robust_target_center(
            centers
        )

        self.assertFalse(
            result.accepted
        )

        self.assertIsNone(
            result.center_px
        )

    def test_rejects_three_inliers_two_outliers(self):
        centers = np.array(
            [
                [300.0, 220.0],
                [301.0, 220.0],
                [300.5, 221.0],
                [400.0, 300.0],
                [500.0, 350.0],
            ]
        )

        result = robust_target_center(
            centers
        )

        self.assertFalse(
            result.accepted
        )

    def test_custom_config(self):
        config = TargetBurstConfig(
            frame_count=3,
            min_inliers=2,
            cluster_radius_px=2.0,
        )

        result = robust_target_center(
            [
                (10.0, 10.0),
                (11.0, 10.0),
                (100.0, 100.0),
            ],
            config=config,
        )

        self.assertTrue(
            result.accepted
        )

        self.assertEqual(
            result.inlier_count,
            2,
        )

    def test_requires_exact_burst_size(self):
        with self.assertRaisesRegex(
            ValueError,
            "shape",
        ):
            robust_target_center(
                [
                    (1.0, 2.0),
                    (1.0, 2.0),
                ]
            )

    def test_rejects_nonfinite_center(self):
        centers = np.zeros(
            (5, 2),
            dtype=np.float64,
        )

        centers[2, 0] = np.nan

        with self.assertRaisesRegex(
            ValueError,
            "finite",
        ):
            robust_target_center(
                centers
            )

    def test_config_rejects_invalid_inlier_count(self):
        with self.assertRaisesRegex(
            ValueError,
            "min_inliers",
        ):
            TargetBurstConfig(
                frame_count=5,
                min_inliers=6,
            )


if __name__ == "__main__":
    unittest.main()
