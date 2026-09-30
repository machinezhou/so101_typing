import unittest

import cv2
import numpy as np

from so101_typing.perception.screen_ocr import (
    TesseractScreenLineOCR,
    TesseractSingleCharacterOCR,
    detect_screen_text_lines,
)


class TestScreenLineOCR(unittest.TestCase):
    def test_detects_two_text_lines(self):
        image = np.full(
            (220, 800, 3),
            255,
            dtype=np.uint8,
        )

        cv2.putText(
            image,
            "KEYPRESS",
            (40, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (50, 50, 50),
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            "SECONDLINE",
            (40, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (50, 50, 50),
            3,
            cv2.LINE_AA,
        )

        boxes = detect_screen_text_lines(
            image
        )

        self.assertEqual(
            len(boxes),
            2,
        )

        self.assertLess(
            boxes[0].y,
            boxes[1].y,
        )

    def test_blank_roi_has_no_lines(self):
        image = np.full(
            (220, 800, 3),
            255,
            dtype=np.uint8,
        )

        self.assertEqual(
            detect_screen_text_lines(image),
            (),
        )


    def test_whole_line_whitelist_stays_uppercase_only(self):
        whitelist = TesseractScreenLineOCR.DEFAULT_WHITELIST
        for character in "abcdefghijklmnopqrstuvwxyz":
            self.assertNotIn(character, whitelist)

    def test_single_character_whitelist_accepts_lowercase(self):
        whitelist = TesseractSingleCharacterOCR.DEFAULT_WHITELIST
        for character in "abcdefghijklmnopqrstuvwxyz":
            self.assertIn(character, whitelist)

    def test_single_character_psm_is_configurable(self):
        ocr = TesseractSingleCharacterOCR(
            psm=13,
            whitelist="abcdefghijklmnopqrstuvwxyz",
        )
        self.assertEqual(ocr.psm, 13)
        self.assertEqual(
            ocr.whitelist,
            "abcdefghijklmnopqrstuvwxyz",
        )

    def test_single_character_rejects_invalid_psm(self):
        with self.assertRaises(ValueError):
            TesseractSingleCharacterOCR(psm=-1)

    def test_multi_character_observation_is_preserved_without_changing_single_character_api(self):
        ocr = TesseractSingleCharacterOCR(
            psm=13,
            whitelist="abcdefghijklmnopqrstuvwxyz",
        )
        image = np.zeros((20, 40, 3), dtype=np.uint8)

        original = ocr._recognize_line
        try:
            ocr._recognize_line = lambda _image: ("qq", 91.5, 1)

            observed, confidence = ocr.recognize_characters(image)
            self.assertEqual(observed, "QQ")
            self.assertAlmostEqual(confidence, 91.5)

            character, confidence = ocr.recognize_character(image)
            self.assertIsNone(character)
            self.assertAlmostEqual(confidence, 91.5)
        finally:
            ocr._recognize_line = original


if __name__ == "__main__":
    unittest.main()
