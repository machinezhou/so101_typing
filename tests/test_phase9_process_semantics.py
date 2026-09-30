from __future__ import annotations

from io import StringIO
from pathlib import Path
import sys
import unittest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import phase9_multi_key_typing as phase9


class Phase9ProcessSemanticsTests(unittest.TestCase):
    def test_handled_task_failure_is_not_a_process_crash(self) -> None:
        self.assertEqual(
            phase9._phase9_process_exit_code(
                phase9.MultiKeyTaskState.FAILED,
            ),
            0,
        )

    def test_success_is_normal_process_completion(self) -> None:
        self.assertEqual(
            phase9._phase9_process_exit_code(
                phase9.MultiKeyTaskState.SUCCEEDED,
            ),
            0,
        )

    def test_session_contract_violation_remains_nonzero(self) -> None:
        self.assertEqual(
            phase9._phase9_process_exit_code(
                phase9.MultiKeyTaskState.SUCCEEDED,
                session_contract_ok=False,
            ),
            2,
        )

    def test_tee_mirrors_output(self) -> None:
        left = StringIO()
        right = StringIO()
        tee = phase9._Tee(left, right)
        tee.write("phase9\n")
        tee.flush()
        self.assertEqual(left.getvalue(), "phase9\n")
        self.assertEqual(right.getvalue(), "phase9\n")


if __name__ == "__main__":
    unittest.main()
