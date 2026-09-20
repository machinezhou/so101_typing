import unittest

import numpy as np

from so101_typing.perception.semantic_target_lock import (
    SemanticTargetLock,
    estimate_keyboard_similarity,
    estimate_keyboard_translation,
)


def keyboard_grid():
    return np.asarray(
        [
            (x * 40.0, y * 40.0)
            for y in range(3)
            for x in range(5)
        ],
        dtype=np.float64,
    )


def apply_similarity(
    points,
    *,
    scale,
    rotation_deg,
    translation,
):
    theta = np.deg2rad(
        rotation_deg
    )

    c = float(
        np.cos(theta)
    )

    s = float(
        np.sin(theta)
    )

    row_rotation = np.asarray(
        [
            [c, s],
            [-s, c],
        ],
        dtype=np.float64,
    )

    return (
        float(scale)
        * (
            np.asarray(
                points,
                dtype=np.float64,
            )
            @ row_rotation
        )
        + np.asarray(
            translation,
            dtype=np.float64,
        )
    )


class TestSemanticTargetLock(unittest.TestCase):
    def test_translation_estimate(self):
        before = keyboard_grid()

        shift = np.asarray(
            [7.0, -4.0]
        )

        after = before + shift

        estimate = (
            estimate_keyboard_translation(
                before,
                after,
            )
        )

        self.assertIsNotNone(
            estimate
        )

        self.assertAlmostEqual(
            estimate.translation_px[0],
            7.0,
        )

        self.assertAlmostEqual(
            estimate.translation_px[1],
            -4.0,
        )

        self.assertGreaterEqual(
            estimate.inliers,
            8,
        )

    def test_geometry_requires_semantic_seed(self):
        lock = SemanticTargetLock(
            "G"
        )

        result = (
            lock.propagate_geometry(
                keyboard_grid()
            )
        )

        self.assertFalse(
            result.found
        )

        self.assertEqual(
            result.source,
            "unlocked",
        )

    def test_semantic_seed_establishes_identity(self):
        lock = SemanticTargetLock(
            "G"
        )

        result = lock.observe_semantic(
            target_center_px=(123.0, 88.0),
            keycap_centers=keyboard_grid(),
        )

        self.assertTrue(
            result.found
        )

        self.assertEqual(
            result.target_label,
            "G",
        )

        self.assertEqual(
            result.source,
            "semantic",
        )

    def test_geometry_propagates_same_target(self):
        lock = SemanticTargetLock(
            "G"
        )

        before = keyboard_grid()

        lock.observe_semantic(
            target_center_px=(123.0, 88.0),
            keycap_centers=before,
        )

        shift = np.asarray(
            [5.0, -3.0]
        )

        result = (
            lock.propagate_geometry(
                before + shift
            )
        )

        self.assertTrue(
            result.found
        )

        self.assertEqual(
            result.target_label,
            "G",
        )

        self.assertEqual(
            result.source,
            "geometry",
        )

        self.assertAlmostEqual(
            result.center_px[0],
            128.0,
        )

        self.assertAlmostEqual(
            result.center_px[1],
            85.0,
        )

    def test_rejected_geometry_does_not_move_lock(self):
        lock = SemanticTargetLock(
            "G"
        )

        before = keyboard_grid()

        lock.observe_semantic(
            target_center_px=(123.0, 88.0),
            keycap_centers=before,
        )

        original = (
            lock.target_center_px
        )

        bad = before.copy()

        bad[:, 0] += np.linspace(
            -40.0,
            40.0,
            len(bad),
        )

        result = (
            lock.propagate_geometry(
                bad
            )
        )

        self.assertFalse(
            result.found
        )

        self.assertEqual(
            lock.target_center_px,
            original,
        )

    def test_sequential_geometry_is_cumulative(self):
        lock = SemanticTargetLock(
            "G"
        )

        centers = keyboard_grid()

        lock.observe_semantic(
            target_center_px=(100.0, 100.0),
            keycap_centers=centers,
        )

        first = centers + np.asarray(
            [4.0, 2.0]
        )

        result1 = (
            lock.propagate_geometry(
                first
            )
        )

        second = first + np.asarray(
            [-1.0, 3.0]
        )

        result2 = (
            lock.propagate_geometry(
                second
            )
        )

        self.assertTrue(
            result1.found
        )

        self.assertTrue(
            result2.found
        )

        self.assertAlmostEqual(
            result2.center_px[0],
            103.0,
        )

        self.assertAlmostEqual(
            result2.center_px[1],
            105.0,
        )


    def test_similarity_bridges_motion_larger_than_raw_pair_gate(self):
        before = keyboard_grid()

        after = (
            before
            + np.asarray(
                [22.0, -3.0]
            )
        )

        estimate = (
            estimate_keyboard_similarity(
                before,
                after,
            )
        )

        self.assertIsNotNone(
            estimate
        )

        self.assertAlmostEqual(
            estimate.translation_px[0],
            22.0,
            places=6,
        )

        self.assertAlmostEqual(
            estimate.translation_px[1],
            -3.0,
            places=6,
        )

        self.assertGreaterEqual(
            estimate.inliers,
            8,
        )


    def test_similarity_handles_scale_rotation_and_partial_visibility(self):
        before = keyboard_grid()

        transformed = apply_similarity(
            before,
            scale=1.024,
            rotation_deg=0.47,
            translation=(22.0, -3.0),
        )

        current = np.delete(
            transformed,
            [0, 4, 7, 14],
            axis=0,
        )

        current = np.vstack(
            [
                current,
                np.asarray(
                    [
                        [300.0, 300.0],
                        [10.0, 150.0],
                        [180.0, -20.0],
                    ],
                    dtype=np.float64,
                ),
            ]
        )

        estimate = (
            estimate_keyboard_similarity(
                before,
                current,
            )
        )

        self.assertIsNotNone(
            estimate
        )

        self.assertAlmostEqual(
            estimate.scale,
            1.024,
            places=3,
        )

        self.assertAlmostEqual(
            estimate.rotation_deg,
            0.47,
            places=2,
        )

        self.assertGreaterEqual(
            estimate.inliers,
            8,
        )

        self.assertLess(
            estimate.median_residual_px,
            1.0,
        )


    def test_similarity_propagates_locked_target(self):
        lock = SemanticTargetLock(
            "G"
        )

        before = keyboard_grid()

        target = np.asarray(
            [123.0, 88.0],
            dtype=np.float64,
        )

        lock.observe_semantic(
            target_center_px=target,
            keycap_centers=before,
        )

        after = apply_similarity(
            before,
            scale=1.024,
            rotation_deg=0.47,
            translation=(22.0, -3.0),
        )

        expected = apply_similarity(
            target,
            scale=1.024,
            rotation_deg=0.47,
            translation=(22.0, -3.0),
        )

        result = (
            lock.propagate_geometry(
                after
            )
        )

        self.assertTrue(
            result.found
        )

        self.assertEqual(
            result.source,
            "geometry",
        )

        self.assertAlmostEqual(
            result.center_px[0],
            expected[0],
            places=3,
        )

        self.assertAlmostEqual(
            result.center_px[1],
            expected[1],
            places=3,
        )

        self.assertAlmostEqual(
            result.scale,
            1.024,
            places=3,
        )


    def test_excessive_similarity_change_is_rejected_without_moving_lock(self):
        lock = SemanticTargetLock(
            "G"
        )

        before = keyboard_grid()

        lock.observe_semantic(
            target_center_px=(100.0, 60.0),
            keycap_centers=before,
        )

        original = (
            lock.target_center_px
        )

        after = apply_similarity(
            before,
            scale=1.15,
            rotation_deg=0.0,
            translation=(2.0, 0.0),
        )

        result = (
            lock.propagate_geometry(
                after
            )
        )

        self.assertFalse(
            result.found
        )

        self.assertEqual(
            result.source,
            "geometry_rejected",
        )

        self.assertEqual(
            lock.target_center_px,
            original,
        )


if __name__ == "__main__":
    unittest.main()
