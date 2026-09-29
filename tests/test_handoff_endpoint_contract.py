from __future__ import annotations

import unittest
from pathlib import Path

from so101_typing.control.handoff_endpoint import (
    HandoffEndpointContract,
    HandoffFrameConfirmation,
)


CONTRACT_PATH = Path(
    "calibration/handoff_endpoint_contract.json"
)


class HandoffEndpointContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = HandoffEndpointContract.load(
            CONTRACT_PATH
        )

    def test_frozen_provenance_and_margin(self) -> None:
        self.assertEqual(
            self.contract.verified_episode_count,
            158,
        )
        self.assertEqual(
            self.contract.reliable_endpoint_count,
            133,
        )
        self.assertAlmostEqual(
            self.contract.measurement_repeatability_p95_px,
            10.44,
            places=2,
        )
        self.assertAlmostEqual(
            self.contract.acceptance_margin_px,
            11.0,
            places=6,
        )
        self.assertEqual(
            self.contract.required_consecutive_frames,
            1,
        )

    def test_raw_hull_vertices_are_accepted(self) -> None:
        for point in self.contract.hull_vertices_error_px:
            result = self.contract.evaluate(point)
            self.assertTrue(
                result.accepted,
                msg=f"hull vertex rejected: {point}",
            )

    def test_margin_accepts_small_outside_repeatability_error(self) -> None:
        # Raw hull right edge is around x=45 px here.
        # This point is about 9.84 px outside, so the frozen
        # 11 px repeatability margin must still accept it.
        result = self.contract.evaluate(
            (55.0, -17.97)
        )

        self.assertLess(
            result.signed_distance_to_hull_px,
            0.0,
        )
        self.assertGreater(
            result.acceptance_clearance_px,
            0.0,
        )
        self.assertTrue(result.accepted)

    def test_margin_rejects_point_beyond_repeatability_allowance(self) -> None:
        # About 11.84 px outside the raw hull: just beyond
        # the frozen 11 px allowance.
        result = self.contract.evaluate(
            (57.0, -17.97)
        )

        self.assertLess(
            result.acceptance_clearance_px,
            0.0,
        )
        self.assertFalse(result.accepted)

    def test_failed_phase8_f_endpoint_is_clearly_outside(self) -> None:
        result = self.contract.evaluate(
            (68.50, -38.25)
        )

        self.assertFalse(result.accepted)

        self.assertAlmostEqual(
            result.signed_distance_to_hull_px,
            -25.16,
            delta=0.10,
        )

        self.assertAlmostEqual(
            result.acceptance_clearance_px,
            -14.16,
            delta=0.10,
        )

    def test_typical_verified_endpoint_region_is_accepted(self) -> None:
        result = self.contract.evaluate(
            (10.0, -37.0)
        )
        self.assertTrue(result.accepted)

    def test_confirmation_requires_two_distinct_fresh_frames(self) -> None:
        state = HandoffFrameConfirmation()

        state, fresh = state.observe(
            100,
            eligible=True,
        )
        self.assertTrue(fresh)
        self.assertEqual(state.streak, 1)
        self.assertFalse(state.confirmed(2))

        state, fresh = state.observe(
            101,
            eligible=True,
        )
        self.assertTrue(fresh)
        self.assertEqual(state.streak, 2)
        self.assertTrue(state.confirmed(2))

    def test_duplicate_frame_does_not_advance_streak(self) -> None:
        state = HandoffFrameConfirmation()

        state, fresh = state.observe(
            200,
            eligible=True,
        )
        self.assertTrue(fresh)
        self.assertEqual(state.streak, 1)

        duplicate, fresh = state.observe(
            200,
            eligible=True,
        )
        self.assertFalse(fresh)
        self.assertEqual(duplicate.streak, 1)
        self.assertEqual(
            duplicate.last_frame_id,
            200,
        )

    def test_fresh_ineligible_frame_resets_streak(self) -> None:
        state = HandoffFrameConfirmation()

        state, _ = state.observe(
            300,
            eligible=True,
        )
        self.assertEqual(state.streak, 1)

        state, fresh = state.observe(
            301,
            eligible=False,
        )
        self.assertTrue(fresh)
        self.assertEqual(state.streak, 0)
        self.assertFalse(state.confirmed(2))

    def test_duplicate_ineligible_read_does_not_reset_existing_streak(self) -> None:
        state = HandoffFrameConfirmation()

        state, _ = state.observe(
            400,
            eligible=True,
        )
        self.assertEqual(state.streak, 1)

        # Same physical camera frame being interpreted differently
        # must not mutate the confirmation state.
        duplicate, fresh = state.observe(
            400,
            eligible=False,
        )
        self.assertFalse(fresh)
        self.assertEqual(duplicate.streak, 1)

    def test_frame_id_cannot_move_backwards(self) -> None:
        state = HandoffFrameConfirmation()

        state, _ = state.observe(
            500,
            eligible=True,
        )

        with self.assertRaises(ValueError):
            state.observe(
                499,
                eligible=True,
            )


if __name__ == "__main__":
    unittest.main()
