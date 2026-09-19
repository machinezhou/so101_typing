import unittest

from so101_typing.supervisor.verification import (
    FrameEvidenceKind,
    ScreenVerificationStatus,
    classify_frame_text,
    verify_screen_texts,
)


class TestScreenVerification(unittest.TestCase):
    def test_line_number_does_not_break_prefix_match(self):
        evidence = classify_frame_text(
            "1 KEYPRESS",
            confirmed_prefix="KEYPRESS",
            expected_char="G",
        )

        self.assertEqual(
            evidence.kind,
            FrameEvidenceKind.NO_CHANGE,
        )

    def test_success_after_prefix(self):
        evidence = classify_frame_text(
            "1 KEYPRESSG",
            confirmed_prefix="KEYPRESS",
            expected_char="G",
        )

        self.assertEqual(
            evidence.kind,
            FrameEvidenceKind.SUCCESS,
        )
        self.assertEqual(
            evidence.continuation,
            "G",
        )

    def test_wrong_character_after_prefix(self):
        evidence = classify_frame_text(
            "1 KEYPRESSH",
            confirmed_prefix="KEYPRESS",
            expected_char="G",
        )

        self.assertEqual(
            evidence.kind,
            FrameEvidenceKind.WRONG,
        )
        self.assertEqual(
            evidence.continuation,
            "H",
        )

    def test_one_prefix_ocr_error_is_tolerated(self):
        evidence = classify_frame_text(
            "1 KEYPRE5SG",
            confirmed_prefix="KEYPRESS",
            expected_char="G",
        )

        self.assertEqual(
            evidence.kind,
            FrameEvidenceKind.SUCCESS,
        )
        self.assertEqual(
            evidence.prefix_distance,
            1,
        )

    def test_bad_prefix_is_uncertain(self):
        evidence = classify_frame_text(
            "UNRELATED TEXT",
            confirmed_prefix="KEYPRESS",
            expected_char="G",
        )

        self.assertEqual(
            evidence.kind,
            FrameEvidenceKind.UNCERTAIN,
        )

    def test_multi_frame_success(self):
        texts = [
            "1 KEYPRESSG",
            "1 KEYPRESSG",
            "1 KEYPRES5G",
            "1 KEYPRESSG",
            "1 KEYPRESSG",
            "1 KEYPRESSG",
            "garbage",
            "1 KEYPRESS",
            "1 KEYPRESSH",
        ]

        result = verify_screen_texts(
            texts,
            confirmed_prefix="KEYPRESS",
            expected_char="G",
        )

        self.assertEqual(
            result.status,
            ScreenVerificationStatus.CONFIRMED_SUCCESS,
        )

    def test_multi_frame_no_change(self):
        texts = [
            "1 KEYPRESS",
            "1 KEYPRESS",
            "1 KEYPRE5S",
            "1 KEYPRESS",
            "1 KEYPRESS",
            "1 KEYPRESS",
            "garbage",
            "1 KEYPRESSG",
            "1 KEYPRESSH",
        ]

        result = verify_screen_texts(
            texts,
            confirmed_prefix="KEYPRESS",
            expected_char="G",
        )

        self.assertEqual(
            result.status,
            ScreenVerificationStatus.CONFIRMED_NO_CHANGE,
        )

    def test_wrong_requires_same_wrong_character(self):
        texts = [
            "1 KEYPRESSH",
            "1 KEYPRESSH",
            "1 KEYPRE5SH",
            "1 KEYPRESSH",
            "1 KEYPRESSH",
            "1 KEYPRESSH",
            "1 KEYPRESSJ",
            "1 KEYPRESSK",
            "garbage",
        ]

        result = verify_screen_texts(
            texts,
            confirmed_prefix="KEYPRESS",
            expected_char="G",
        )

        self.assertEqual(
            result.status,
            ScreenVerificationStatus.CONFIRMED_WRONG,
        )

        self.assertEqual(
            result.wrong_character,
            "H",
        )

    def test_varying_wrong_characters_are_uncertain(self):
        texts = [
            "1 KEYPRESSH",
            "1 KEYPRESSJ",
            "1 KEYPRESSK",
            "1 KEYPRESSL",
            "1 KEYPRESSM",
            "1 KEYPRESSN",
            "garbage",
            "1 KEYPRESS",
            "1 KEYPRESSG",
        ]

        result = verify_screen_texts(
            texts,
            confirmed_prefix="KEYPRESS",
            expected_char="G",
        )

        self.assertEqual(
            result.status,
            ScreenVerificationStatus.UNCERTAIN,
        )


if __name__ == "__main__":
    unittest.main()
