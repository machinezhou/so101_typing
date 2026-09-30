from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import phase8_single_key_integration as phase8


class Phase8AttemptResetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.saved_target = phase8.phase5.TARGET
        self.saved_watcher = phase8.phase5.ACTIVE_SCREEN_WATCHER
        self.saved_reentry = phase8.phase5.SEMANTIC_REENTRY_GUARDED

    def tearDown(self) -> None:
        phase8.phase5.TARGET = self.saved_target
        phase8.phase5.ACTIVE_SCREEN_WATCHER = self.saved_watcher
        phase8.phase5.SEMANTIC_REENTRY_GUARDED = self.saved_reentry

    def test_reset_clears_all_attempt_local_state(self) -> None:
        sentinel_watcher = object()
        phase8.phase5.TARGET = "Q"
        phase8.phase5.ACTIVE_SCREEN_WATCHER = sentinel_watcher
        phase8.phase5.SEMANTIC_REENTRY_GUARDED = True

        policy = Mock()
        preprocessor = Mock()
        postprocessor = Mock()

        normalized = phase8._reset_single_key_attempt_state(
            target=" a ",
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
        )

        self.assertEqual(normalized, "A")
        self.assertEqual(phase8.phase5.TARGET, "A")
        self.assertIsNone(phase8.phase5.ACTIVE_SCREEN_WATCHER)
        self.assertFalse(phase8.phase5.SEMANTIC_REENTRY_GUARDED)
        policy.reset.assert_called_once_with()
        preprocessor.reset.assert_called_once_with()
        postprocessor.reset.assert_called_once_with()

    def test_invalid_target_fails_before_mutating_or_resetting(self) -> None:
        sentinel_watcher = object()
        phase8.phase5.TARGET = "Q"
        phase8.phase5.ACTIVE_SCREEN_WATCHER = sentinel_watcher
        phase8.phase5.SEMANTIC_REENTRY_GUARDED = True

        policy = Mock()
        preprocessor = Mock()
        postprocessor = Mock()

        with self.assertRaises(ValueError):
            phase8._reset_single_key_attempt_state(
                target="A1",
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
            )

        self.assertEqual(phase8.phase5.TARGET, "Q")
        self.assertIs(phase8.phase5.ACTIVE_SCREEN_WATCHER, sentinel_watcher)
        self.assertTrue(phase8.phase5.SEMANTIC_REENTRY_GUARDED)
        policy.reset.assert_not_called()
        preprocessor.reset.assert_not_called()
        postprocessor.reset.assert_not_called()


if __name__ == "__main__":
    unittest.main()
