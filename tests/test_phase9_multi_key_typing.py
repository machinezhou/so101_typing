from __future__ import annotations

from pathlib import Path
import sys
import unittest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import phase9_multi_key_typing as phase9


class Phase9MultiKeyTypingTests(unittest.TestCase):
    def test_target_is_normalized_to_uppercase(self) -> None:
        supervisor = phase9.MultiKeyTypingSupervisor(" cat ")
        self.assertEqual(supervisor.target_text, "CAT")
        self.assertEqual(supervisor.current_target, "C")

    def test_success_advances_to_next_character(self) -> None:
        supervisor = phase9.MultiKeyTypingSupervisor("CAT")
        snapshot = supervisor.record_attempt(phase9.SingleKeyAttemptOutcome.SUCCESS)
        self.assertEqual(snapshot.completed_text, "C")
        self.assertEqual(snapshot.current_index, 1)
        self.assertEqual(snapshot.current_target, "A")
        self.assertEqual(snapshot.state, phase9.MultiKeyTaskState.RUNNING)

    def test_all_successes_complete_whole_string(self) -> None:
        supervisor = phase9.MultiKeyTypingSupervisor("CAT")
        for _ in "CAT":
            snapshot = supervisor.record_attempt(
                phase9.SingleKeyAttemptOutcome.SUCCESS
            )
        self.assertEqual(snapshot.state, phase9.MultiKeyTaskState.SUCCEEDED)
        self.assertEqual(snapshot.completed_text, "CAT")
        self.assertIsNone(snapshot.current_target)

    def test_any_non_success_stops_without_advancing(self) -> None:
        for outcome in (
            phase9.SingleKeyAttemptOutcome.WRONG,
            phase9.SingleKeyAttemptOutcome.UNCERTAIN,
            phase9.SingleKeyAttemptOutcome.HANDOFF_FAILURE,
            phase9.SingleKeyAttemptOutcome.TIMEOUT,
            phase9.SingleKeyAttemptOutcome.ABORTED,
        ):
            with self.subTest(outcome=outcome):
                supervisor = phase9.MultiKeyTypingSupervisor("CAT")
                supervisor.record_attempt(phase9.SingleKeyAttemptOutcome.SUCCESS)
                snapshot = supervisor.record_attempt(outcome)
                self.assertEqual(snapshot.state, phase9.MultiKeyTaskState.FAILED)
                self.assertEqual(snapshot.completed_text, "C")
                self.assertEqual(snapshot.failed_index, 1)
                self.assertEqual(snapshot.failed_target, "A")
                self.assertEqual(snapshot.failed_outcome, outcome)

    def test_terminal_task_rejects_additional_attempts(self) -> None:
        supervisor = phase9.MultiKeyTypingSupervisor("A")
        supervisor.record_attempt(phase9.SingleKeyAttemptOutcome.SUCCESS)
        with self.assertRaises(RuntimeError):
            supervisor.record_attempt(phase9.SingleKeyAttemptOutcome.SUCCESS)

    def test_v0_rejects_empty_or_non_letter_targets(self) -> None:
        for text in ("", " ", "A1", "A-B", "你好"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    phase9.MultiKeyTypingSupervisor(text)


if __name__ == "__main__":
    unittest.main()
