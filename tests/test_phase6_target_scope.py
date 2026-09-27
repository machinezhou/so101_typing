from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase6_target_scope import (  # noqa: E402
    collect_target_episode_sources,
    has_completed_validation,
    target_qc_dir,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload) + "\n",
        encoding="utf-8",
    )


class Phase6TargetScopeTest(unittest.TestCase):
    def _make_run(
        self,
        runs: Path,
        name: str,
        indices: list[int],
        *,
        target: str = "U",
    ) -> Path:
        run = runs / name

        write_json(
            run / "run_summary.json",
            {
                "target": target,
                "repo_id": "local/test",
                "dataset_root": "artifacts/test_dataset",
            },
        )

        for number, idx in enumerate(indices, start=1):
            write_json(
                run
                / f"attempt_{number:03d}_episode_{number:03d}"
                / "summary.json",
                {
                    "accepted": True,
                    "episode_number": number,
                    "dataset_episode_index": idx,
                },
            )

        return run

    def test_target_scope_collects_across_runs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs = Path(td) / "runs"

            self._make_run(
                runs,
                "run_20260927_100000_u",
                [181, 182, 183],
            )
            self._make_run(
                runs,
                "run_20260927_110000_u",
                [184, 185, 186],
            )

            result = collect_target_episode_sources(
                runs,
                "U",
            )

            self.assertEqual(
                [
                    item["dataset_episode_index"]
                    for item in result["episodes"]
                ],
                [181, 182, 183, 184, 185, 186],
            )
            self.assertEqual(
                len(result["run_dirs"]),
                2,
            )

    def test_incomplete_latest_run_without_summary_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs = Path(td) / "runs"

            self._make_run(
                runs,
                "run_20260927_100000_u",
                [181, 182, 183],
            )

            incomplete = runs / "run_20260927_120000_u"
            incomplete.mkdir(parents=True)

            result = collect_target_episode_sources(
                runs,
                "U",
            )

            self.assertEqual(
                result["run_dirs"],
                [
                    str(
                        runs
                        / "run_20260927_100000_u"
                    )
                ],
            )
            self.assertEqual(
                [
                    item["dataset_episode_index"]
                    for item in result["episodes"]
                ],
                [181, 182, 183],
            )

    def test_discarded_attempt_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs = Path(td) / "runs"
            run = self._make_run(
                runs,
                "run_20260927_100000_u",
                [181],
            )

            write_json(
                run
                / "attempt_999_episode_002"
                / "summary.json",
                {
                    "accepted": False,
                    "episode_number": 2,
                },
            )

            result = collect_target_episode_sources(
                runs,
                "U",
            )
            self.assertEqual(
                [
                    item["dataset_episode_index"]
                    for item in result["episodes"]
                ],
                [181],
            )

    def test_duplicate_dataset_index_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs = Path(td) / "runs"

            self._make_run(
                runs,
                "run_20260927_100000_u",
                [181],
            )
            self._make_run(
                runs,
                "run_20260927_110000_u",
                [181],
            )

            with self.assertRaises(RuntimeError):
                collect_target_episode_sources(
                    runs,
                    "U",
                )

    def test_pass_and_fail_complete_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            for status in ("PASS", "FAIL"):
                episode_dir = root / status
                summary = episode_dir / "summary.json"

                write_json(
                    summary,
                    {
                        "accepted": True,
                        "dataset_episode_index": 1,
                    },
                )
                write_json(
                    episode_dir / "takeover_validation.json",
                    {
                        "trials": [
                            {"status": status}
                        ]
                    },
                )

                self.assertTrue(
                    has_completed_validation(summary)
                )

    def test_invalid_or_aborted_remains_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            for status in (
                "INVALID_REPLAY",
                "ABORTED",
                "UNKNOWN",
            ):
                episode_dir = root / status
                summary = episode_dir / "summary.json"

                write_json(
                    summary,
                    {
                        "accepted": True,
                        "dataset_episode_index": 1,
                    },
                )
                write_json(
                    episode_dir / "takeover_validation.json",
                    {
                        "trials": [
                            {"status": status}
                        ]
                    },
                )

                self.assertFalse(
                    has_completed_validation(summary)
                )

    def test_target_qc_directory_is_target_scoped(self) -> None:
        base = Path(
            "artifacts/phase6_act_dataset/keyboard_v1"
        )
        self.assertEqual(
            target_qc_dir(base, "U"),
            base / "target_qc" / "u",
        )

    def test_semantic_lock_error_is_not_hardcoded_to_g(self) -> None:
        text = (
            ROOT
            / "scripts"
            / "validate_phase5_single_key.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn(
            "Could not establish initial semantic G lock",
            text,
        )
        self.assertIn(
            "semantic {TARGET} lock",
            text,
        )


if __name__ == "__main__":
    unittest.main()
