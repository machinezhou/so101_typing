import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import phase8_single_key_integration as phase8
import validate_phase5_single_key as phase5


class Phase8ReleasedCharacterTests(unittest.TestCase):
    def test_single_expected_character_is_success(self):
        result = phase8._vote_released_characters(
            ["Q", "Q", "Q", "Q", "Q"],
            target="Q",
        )
        self.assertEqual(
            result.status,
            phase5.ScreenVerificationStatus.CONFIRMED_SUCCESS,
        )
        self.assertEqual(result.success_votes, 5)

    def test_stable_repeated_character_is_confirmed_wrong_not_uncertain(self):
        result = phase8._vote_released_characters(
            ["QQ", "QQ", "QQ", "QQ", "QQ"],
            target="Q",
        )
        self.assertEqual(
            result.status,
            phase5.ScreenVerificationStatus.CONFIRMED_WRONG,
        )
        self.assertEqual(result.wrong_character, "QQ")
        self.assertEqual(result.wrong_votes, 5)
        self.assertEqual(result.uncertain_votes, 0)

    def test_missing_observation_remains_uncertain(self):
        result = phase8._vote_released_characters(
            [None, None, None, None, None],
            target="Q",
        )
        self.assertEqual(
            result.status,
            phase5.ScreenVerificationStatus.UNCERTAIN,
        )
        self.assertEqual(result.uncertain_votes, 5)


    def test_zero_confidence_wrong_character_is_uncertain_evidence(self):
        observed = phase8._released_character_for_verdict(
            "Y",
            0.0,
            target="O",
        )
        self.assertIsNone(observed)

        result = phase8._vote_released_characters(
            [observed],
            target="O",
        )
        self.assertEqual(
            result.status,
            phase5.ScreenVerificationStatus.UNCERTAIN,
        )

    def test_zero_confidence_expected_character_is_not_rejected(self):
        observed = phase8._released_character_for_verdict(
            "O",
            0.0,
            target="O",
        )
        self.assertEqual(observed, "O")

    def test_zero_confidence_repeat_of_expected_remains_wrong_evidence(self):
        observed = phase8._released_character_for_verdict(
            "QQ",
            0.0,
            target="Q",
        )
        self.assertEqual(observed, "QQ")

        result = phase8._vote_released_characters(
            [observed],
            target="Q",
        )
        self.assertEqual(
            result.status,
            phase5.ScreenVerificationStatus.CONFIRMED_WRONG,
        )

    def test_positive_confidence_wrong_character_is_preserved(self):
        observed = phase8._released_character_for_verdict(
            "Y",
            1.0,
            target="O",
        )
        self.assertEqual(observed, "Y")


if __name__ == "__main__":
    unittest.main()
