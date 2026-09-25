import json
import shutil
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.phase7_act_training_input import (
    _compute_image_stats,
    _load_verified_manifest,
    _max_stat_difference,
    _resolve_dataset_location,
)


class TestPhase7ActTrainingInput(unittest.TestCase):
    def setUp(self):
        self.work_dir = Path(
            "artifacts/test_phase7_act_training_input"
        )
        shutil.rmtree(
            self.work_dir,
            ignore_errors=True,
        )
        self.work_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def tearDown(self):
        shutil.rmtree(
            self.work_dir,
            ignore_errors=True,
        )

    def _write_json(
        self,
        path: Path,
        payload: dict,
    ) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    def test_verified_manifest_accepts_sorted_unique_indices(self):
        path = self.work_dir / "verified_episodes.json"

        self._write_json(
            path,
            {
                "schema": "phase6.verified_episode_manifest.v4",
                "dataset_repo_id": "local/test_dataset",
                "dataset_root": "artifacts/test_dataset",
                "targets": ["A", "G"],
                "dataset_episode_indices": [0, 2],
                "episodes": [
                    {
                        "dataset_episode_index": 0,
                        "target": "A",
                        "verified": True,
                    },
                    {
                        "dataset_episode_index": 2,
                        "target": "G",
                        "verified": True,
                    },
                ],
            },
        )

        manifest, verified = _load_verified_manifest(
            path
        )

        self.assertEqual(
            manifest["targets"],
            ["A", "G"],
        )
        self.assertEqual(
            verified,
            [0, 2],
        )

    def test_verified_manifest_rejects_duplicate_indices(self):
        path = self.work_dir / "verified_episodes.json"

        self._write_json(
            path,
            {
                "schema": "phase6.verified_episode_manifest.v4",
                "dataset_repo_id": "local/test_dataset",
                "dataset_root": "artifacts/test_dataset",
                "targets": ["A"],
                "dataset_episode_indices": [0, 1, 1, 3],
                "episodes": [],
            },
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Duplicate verified episode indices",
        ):
            _load_verified_manifest(path)

    def test_verified_manifest_rejects_unsorted_indices(self):
        path = self.work_dir / "verified_episodes.json"

        self._write_json(
            path,
            {
                "schema": "phase6.verified_episode_manifest.v4",
                "dataset_repo_id": "local/test_dataset",
                "dataset_root": "artifacts/test_dataset",
                "targets": ["A"],
                "dataset_episode_indices": [0, 2, 1, 3],
                "episodes": [],
            },
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "not sorted",
        ):
            _load_verified_manifest(path)

    def test_resolve_dataset_location_accepts_single_location(self):
        runs_root = self.work_dir / "runs"

        for run_name in (
            "run_001",
            "run_002",
        ):
            self._write_json(
                runs_root
                / run_name
                / "run_summary.json",
                {
                    "repo_id": "local/test_dataset",
                    "dataset_root": "artifacts/test_dataset",
                },
            )

        repo_id, dataset_root = _resolve_dataset_location(
            runs_root
        )

        self.assertEqual(
            repo_id,
            "local/test_dataset",
        )
        self.assertEqual(
            dataset_root,
            Path("artifacts/test_dataset"),
        )

    def test_resolve_dataset_location_rejects_conflict(self):
        runs_root = self.work_dir / "runs"

        self._write_json(
            runs_root
            / "run_001"
            / "run_summary.json",
            {
                "repo_id": "local/test_dataset",
                "dataset_root": "artifacts/test_dataset_a",
            },
        )

        self._write_json(
            runs_root
            / "run_002"
            / "run_summary.json",
            {
                "repo_id": "local/test_dataset",
                "dataset_root": "artifacts/test_dataset_b",
            },
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "exactly one LeRobot dataset location",
        ):
            _resolve_dataset_location(
                runs_root
            )

    def test_embedded_float_image_stats_stay_in_zero_one_scale(self):
        rows = [
            {
                "observation.images.top": torch.full(
                    (3, 4, 5),
                    0.25,
                    dtype=torch.float32,
                )
            },
            {
                "observation.images.top": torch.full(
                    (3, 4, 5),
                    0.75,
                    dtype=torch.float32,
                )
            },
        ]

        stats = _compute_image_stats(
            rows,
            "observation.images.top",
        )

        self.assertEqual(
            stats["count"].tolist(),
            [2],
        )

        np.testing.assert_allclose(
            stats["mean"],
            np.full(
                (3, 1, 1),
                0.5,
                dtype=np.float32,
            ),
            atol=1e-6,
        )

        np.testing.assert_allclose(
            stats["std"],
            np.full(
                (3, 1, 1),
                0.25,
                dtype=np.float32,
            ),
            atol=1e-6,
        )

    def test_embedded_image_stats_reject_out_of_range_tensor(self):
        rows = [
            {
                "observation.images.top": torch.full(
                    (3, 4, 5),
                    255.0,
                    dtype=torch.float32,
                )
            }
        ]

        with self.assertRaisesRegex(
            RuntimeError,
            r"expected decoded float image in \[0,1\]",
        ):
            _compute_image_stats(
                rows,
                "observation.images.top",
            )

    def test_max_stat_difference(self):
        global_stats = {
            "mean": np.array(
                [1.0, 5.0, -2.0],
                dtype=np.float32,
            )
        }

        verified_stats = {
            "mean": np.array(
                [1.5, 2.0, -1.0],
                dtype=np.float32,
            )
        }

        difference = _max_stat_difference(
            global_stats,
            verified_stats,
            "mean",
        )

        self.assertAlmostEqual(
            difference,
            3.0,
        )


if __name__ == "__main__":
    unittest.main()
