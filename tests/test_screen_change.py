import unittest

import cv2
import numpy as np

from so101_typing.perception.screen_change import (
    FastScreenChangeConfig,
    FastScreenChangeModel,
)


def make_roi(*, cursor=False, character=False):
    image = np.full((140, 520, 3), 255, dtype=np.uint8)
    cv2.putText(
        image,
        "KEYPRESS",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )

    if cursor:
        cv2.line(image, (282, 42), (282, 96), (0, 0, 0), 2)

    if character:
        cv2.putText(
            image,
            "g",
            (288, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )

    return image


class TestFastScreenChange(unittest.TestCase):
    def make_model(self):
        baseline = [
            make_roi(cursor=False),
            make_roi(cursor=True),
            make_roi(cursor=False),
            make_roi(cursor=True),
        ]
        # These tests exercise the ordinary 2-frame path only.
        # Disable the strong-event fast path here so the two paths are
        # tested independently.
        return FastScreenChangeModel.from_baseline_rois(
            baseline,
            config=FastScreenChangeConfig(
                strong_min_novel_pixels=10000,
                strong_min_component_area_px=10000,
            ),
        )

    def test_cursor_blink_is_absorbed_by_baseline(self):
        model = self.make_model()
        for cursor in (False, True, False, True):
            observation = model.observe(make_roi(cursor=cursor))
            self.assertFalse(observation.triggered)

    def test_one_changed_frame_does_not_trigger(self):
        model = self.make_model()
        first = model.observe(make_roi(character=True))
        self.assertFalse(first.triggered)
        self.assertEqual(first.consecutive_frames, 1)

    def test_persistent_new_character_triggers_on_second_frame(self):
        model = self.make_model()
        model.observe(make_roi(character=True))
        second = model.observe(make_roi(character=True))
        self.assertTrue(second.triggered)
        self.assertGreater(second.novel_pixels, 0)
        self.assertIsNotNone(second.bbox_xywh)


    def test_strong_character_triggers_on_first_frame(self):
        baseline = [
            make_roi(cursor=False),
            make_roi(cursor=True),
            make_roi(cursor=False),
            make_roi(cursor=True),
        ]

        model = FastScreenChangeModel.from_baseline_rois(
            baseline,
            config=FastScreenChangeConfig(
                strong_min_novel_pixels=300,
                strong_min_component_area_px=250,
                required_consecutive_frames=2,
            ),
        )

        first = model.observe(
            make_roi(character=True)
        )

        self.assertTrue(first.changed)
        self.assertTrue(first.strong_changed)
        self.assertTrue(first.triggered)
        self.assertEqual(
            first.consecutive_frames,
            1,
        )


if __name__ == "__main__":
    unittest.main()
