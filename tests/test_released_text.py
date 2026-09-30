import unittest

import cv2
import numpy as np

from so101_typing.perception.released_text import build_temporal_clean_text


class _FakeChangeModel:
    def __init__(self, masks, continuation_x0=0):
        self._masks = list(masks)
        self._index = 0
        self.continuation_x0 = int(continuation_x0)

    def _novel_mask(self, _roi):
        mask = self._masks[self._index]
        self._index += 1
        return mask.copy()


class TemporalReleasedTextTests(unittest.TestCase):
    def test_transient_cursor_removed_but_two_persistent_components_survive(self):
        height, width = 80, 140
        rois = []
        masks = []

        for index in range(10):
            roi = np.full((height, width, 3), 240, dtype=np.uint8)
            mask = np.zeros((height, width), dtype=np.uint8)

            cv2.rectangle(roi, (30, 22), (45, 58), (30, 30, 30), -1)
            cv2.rectangle(roi, (58, 22), (73, 58), (30, 30, 30), -1)
            cv2.rectangle(mask, (30, 22), (45, 58), 255, -1)
            cv2.rectangle(mask, (58, 22), (73, 58), 255, -1)

            if index < 4:
                cv2.rectangle(roi, (82, 18), (84, 62), (0, 0, 0), -1)
                cv2.rectangle(mask, (82, 18), (84, 62), 255, -1)

            rois.append(roi)
            masks.append(mask)

        result = build_temporal_clean_text(
            _FakeChangeModel(masks, continuation_x0=20),
            rois,
            persistence_fraction=0.70,
            crop_margin_px=6,
        )

        self.assertEqual(len(result.components), 2)
        self.assertGreater(
            int(np.count_nonzero(result.selected_mask[:, 30:74])),
            0,
        )
        self.assertEqual(
            int(np.count_nonzero(result.selected_mask[:, 82:85])),
            0,
        )
        self.assertGreater(result.clean_crop.size, 0)

    def test_recognition_pixels_come_from_original_roi_not_binary_mask(self):
        height, width = 60, 100
        rois = []
        masks = []

        for _ in range(8):
            roi = np.full((height, width, 3), 220, dtype=np.uint8)
            roi[20:45, 35:55] = (71, 83, 97)
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[20:45, 35:55] = 255
            rois.append(roi)
            masks.append(mask)

        result = build_temporal_clean_text(
            _FakeChangeModel(masks, continuation_x0=10),
            rois,
            crop_margin_px=2,
        )

        pixels = result.clean_crop.reshape(-1, 3)
        self.assertTrue(
            np.any(np.all(pixels == np.array([71, 83, 97]), axis=1))
        )


    def test_event_anchor_rejects_persistent_off_line_occluder(self):
        height, width = 160, 180
        rois = []
        masks = []

        for _ in range(10):
            roi = np.full((height, width, 3), 245, dtype=np.uint8)
            mask = np.zeros((height, width), dtype=np.uint8)

            # Real released glyph at the triggering event location.
            cv2.rectangle(roi, (60, 35), (91, 74), (40, 40, 40), -1)
            cv2.rectangle(mask, (60, 35), (91, 74), 255, -1)

            # Persistent arm/shadow fragment below the typing line.
            cv2.rectangle(roi, (54, 96), (64, 117), (70, 70, 70), -1)
            cv2.rectangle(mask, (54, 96), (64, 117), 255, -1)

            rois.append(roi)
            masks.append(mask)

        result = build_temporal_clean_text(
            _FakeChangeModel(masks, continuation_x0=45),
            rois,
            persistence_fraction=0.70,
            crop_margin_px=10,
            event_bbox_xywh=(59, 34, 34, 42),
        )

        self.assertEqual(len(result.components), 1)

        # The 2x2 morphology close can shift the connected-component boundary
        # by one pixel, so validate semantic geometry instead of exact x.
        self.assertGreater(
            int(np.count_nonzero(result.selected_mask[30:80, 55:100])),
            0,
        )
        self.assertEqual(
            int(np.count_nonzero(result.selected_mask[90:130, 45:80])),
            0,
        )

        self.assertLess(result.stable_bbox_xywh[3], 50)
        self.assertGreater(result.recognition_padding_px, 0)

    def test_event_anchor_preserves_adjacent_same_line_repeat(self):
        height, width = 100, 180
        rois = []
        masks = []

        for _ in range(10):
            roi = np.full((height, width, 3), 245, dtype=np.uint8)
            mask = np.zeros((height, width), dtype=np.uint8)

            cv2.rectangle(roi, (40, 25), (58, 65), (30, 30, 30), -1)
            cv2.rectangle(roi, (68, 25), (86, 65), (30, 30, 30), -1)
            cv2.rectangle(mask, (40, 25), (58, 65), 255, -1)
            cv2.rectangle(mask, (68, 25), (86, 65), 255, -1)

            rois.append(roi)
            masks.append(mask)

        result = build_temporal_clean_text(
            _FakeChangeModel(masks, continuation_x0=30),
            rois,
            persistence_fraction=0.70,
            crop_margin_px=10,
            event_bbox_xywh=(39, 24, 21, 43),
        )

        self.assertEqual(len(result.components), 2)
        self.assertGreater(
            int(np.count_nonzero(result.selected_mask[:, 68:87])),
            0,
        )


if __name__ == "__main__":
    unittest.main()
