import unittest
from dataclasses import dataclass

import numpy as np

from so101_typing.contracts import CameraFrame
from so101_typing.control.target_measurement_runtime import (
    TargetBurstAccumulator,
    TargetMeasurementRuntimeConfig,
    TargetMeasurementStatus,
)


@dataclass(frozen=True, slots=True)
class _Observation:
    target_label: str = "G"
    found: bool = True
    center_px: tuple[float, float] | None = (
        302.0,
        227.5,
    )


def _frame(
    frame_id: int,
    capture_timestamp: float,
    *,
    processing_timestamp: float | None = None,
    camera_name: str = "wrist",
) -> CameraFrame:
    if processing_timestamp is None:
        processing_timestamp = capture_timestamp

    return CameraFrame(
        camera_name=camera_name,
        frame_id=frame_id,
        capture_timestamp=capture_timestamp,
        processing_timestamp=processing_timestamp,
        image=np.zeros(
            (2, 2, 3),
            dtype=np.uint8,
        ),
    )


class TestTargetBurstAccumulator(unittest.TestCase):
    def test_h31_near_contact_occlusion_outlier_becomes_robust_ready_measurement(self):
        accumulator = TargetBurstAccumulator(
            target_label="G"
        )

        centers = [
            (440.362, 286.155),
            (302.0, 227.5),
            (302.0, 227.5),
            (302.0, 227.5),
            (302.0, 227.0),
        ]

        update = None

        for index, center in enumerate(
            centers,
            start=1,
        ):
            capture = 100.0 + index * 0.01

            update = accumulator.process(
                _frame(
                    index,
                    capture,
                ),
                _Observation(
                    center_px=center
                ),
                now_timestamp=(
                    capture + 0.005
                ),
            )

        assert update is not None

        self.assertEqual(
            update.status,
            TargetMeasurementStatus.READY,
        )

        self.assertTrue(
            update.terminal
        )

        self.assertIsNotNone(
            update.result
        )

        assert update.result is not None

        self.assertEqual(
            update.result.outlier_indices,
            (0,),
        )

        np.testing.assert_allclose(
            update.result.center_px,
            (302.0, 227.5),
            atol=1e-12,
        )

    def test_stale_frame_is_not_counted(self):
        accumulator = TargetBurstAccumulator(
            target_label="G",
            config=TargetMeasurementRuntimeConfig(
                max_frame_age_ms=50.0
            ),
        )

        update = accumulator.process(
            _frame(
                1,
                100.0,
            ),
            _Observation(),
            now_timestamp=100.2,
        )

        self.assertEqual(
            update.status,
            TargetMeasurementStatus.STALE_FRAME,
        )

        self.assertEqual(
            accumulator.accepted_observation_count,
            0,
        )

    def test_duplicate_frame_is_not_counted_twice(self):
        accumulator = TargetBurstAccumulator(
            target_label="G"
        )

        first = accumulator.process(
            _frame(
                1,
                100.01,
            ),
            _Observation(),
            now_timestamp=100.02,
        )

        self.assertEqual(
            first.status,
            TargetMeasurementStatus.COLLECTING,
        )

        duplicate = accumulator.process(
            _frame(
                1,
                100.01,
            ),
            _Observation(),
            now_timestamp=100.03,
        )

        self.assertEqual(
            duplicate.status,
            TargetMeasurementStatus.STALE_FRAME,
        )

        self.assertEqual(
            accumulator.accepted_observation_count,
            1,
        )

    def test_missing_target_does_not_count_but_consumes_frame(self):
        accumulator = TargetBurstAccumulator(
            target_label="G"
        )

        missing = accumulator.process(
            _frame(
                1,
                100.01,
            ),
            _Observation(
                found=False,
                center_px=None,
            ),
            now_timestamp=100.02,
        )

        self.assertEqual(
            missing.status,
            TargetMeasurementStatus.TARGET_MISSING,
        )

        repeated_same_frame = accumulator.process(
            _frame(
                1,
                100.01,
            ),
            _Observation(),
            now_timestamp=100.03,
        )

        self.assertEqual(
            repeated_same_frame.status,
            TargetMeasurementStatus.STALE_FRAME,
        )

        self.assertEqual(
            accumulator.accepted_observation_count,
            0,
        )

    def test_rejects_wrong_camera(self):
        accumulator = TargetBurstAccumulator(
            target_label="G"
        )

        with self.assertRaisesRegex(
            ValueError,
            "expected camera",
        ):
            accumulator.process(
                _frame(
                    1,
                    100.01,
                    camera_name="top",
                ),
                _Observation(),
                now_timestamp=100.02,
            )

    def test_rejects_wrong_target_label(self):
        accumulator = TargetBurstAccumulator(
            target_label="G"
        )

        with self.assertRaisesRegex(
            ValueError,
            "expected target",
        ):
            accumulator.process(
                _frame(
                    1,
                    100.01,
                ),
                _Observation(
                    target_label="H"
                ),
                now_timestamp=100.02,
            )

    def test_found_target_requires_center(self):
        accumulator = TargetBurstAccumulator(
            target_label="G"
        )

        with self.assertRaisesRegex(
            ValueError,
            "center_px",
        ):
            accumulator.process(
                _frame(
                    1,
                    100.01,
                ),
                _Observation(
                    found=True,
                    center_px=None,
                ),
                now_timestamp=100.02,
            )

    def test_non_consensus_burst_is_terminal_rejection(self):
        accumulator = TargetBurstAccumulator(
            target_label="G"
        )

        centers = [
            (100.0, 100.0),
            (120.0, 100.0),
            (140.0, 100.0),
            (160.0, 100.0),
            (180.0, 100.0),
        ]

        update = None

        for index, center in enumerate(
            centers,
            start=1,
        ):
            capture = 100.0 + index * 0.01

            update = accumulator.process(
                _frame(
                    index,
                    capture,
                ),
                _Observation(
                    center_px=center
                ),
                now_timestamp=(
                    capture + 0.005
                ),
            )

        assert update is not None

        self.assertEqual(
            update.status,
            TargetMeasurementStatus.CLUSTER_REJECTED,
        )

        self.assertTrue(
            update.terminal
        )

        self.assertIsNotNone(
            update.result
        )

        assert update.result is not None

        self.assertFalse(
            update.result.accepted
        )

    def test_terminal_accumulator_cannot_be_reused(self):
        accumulator = TargetBurstAccumulator(
            target_label="G"
        )

        for index in range(1, 6):
            capture = 100.0 + index * 0.01

            accumulator.process(
                _frame(
                    index,
                    capture,
                ),
                _Observation(),
                now_timestamp=(
                    capture + 0.005
                ),
            )

        self.assertEqual(
            accumulator.terminal_status,
            TargetMeasurementStatus.READY,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "terminal",
        ):
            accumulator.process(
                _frame(
                    6,
                    100.06,
                ),
                _Observation(),
                now_timestamp=100.065,
            )


if __name__ == "__main__":
    unittest.main()
