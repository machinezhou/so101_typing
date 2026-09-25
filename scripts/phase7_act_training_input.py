from __future__ import annotations

import argparse
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.datasets.compute_stats import (
    aggregate_stats,
    auto_downsample_height_width,
    compute_episode_stats,
    get_feature_stats,
    sample_indices,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset


EXPECTED_LEROBOT_VERSION = "0.6.1"
EXPECTED_FPS = 15

IMAGE_KEYS = (
    "observation.images.top",
    "observation.images.wrist",
)

NUMERIC_KEYS = (
    "observation.state",
    "action",
)

MODEL_KEYS = IMAGE_KEYS + NUMERIC_KEYS

TARGET_VOCAB = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + (
    "SPACE",
    "BACKSPACE",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    return value


def _resolve_dataset_location(
    runs_root: Path,
) -> tuple[str, Path]:
    summaries = sorted(
        runs_root.glob("run_*/run_summary.json")
    )

    if not summaries:
        raise RuntimeError(
            f"No run_summary.json found under {runs_root}"
        )

    locations: set[tuple[str, str]] = set()

    for path in summaries:
        payload = _read_json(path)

        locations.add(
            (
                str(payload["repo_id"]),
                str(payload["dataset_root"]),
            )
        )

    if len(locations) != 1:
        raise RuntimeError(
            "Phase 6 run summaries do not resolve to exactly "
            f"one LeRobot dataset location: {sorted(locations)}"
        )

    repo_id, dataset_root = next(iter(locations))

    return repo_id, Path(dataset_root)


def _load_verified_manifest(
    path: Path,
) -> tuple[dict[str, Any], list[int]]:
    manifest = _read_json(path)

    if manifest.get("schema") != (
        "phase6.verified_episode_manifest.v4"
    ):
        raise RuntimeError(
            "Unexpected verified manifest schema: "
            f"{manifest.get('schema')!r}"
        )

    raw_indices = manifest.get(
        "dataset_episode_indices"
    )

    if not isinstance(raw_indices, list) or not raw_indices:
        raise RuntimeError(
            "verified manifest contains no dataset episodes"
        )

    verified = [int(x) for x in raw_indices]

    if len(set(verified)) != len(verified):
        raise RuntimeError(
            f"Duplicate verified episode indices: {verified}"
        )

    if verified != sorted(verified):
        raise RuntimeError(
            f"Verified episode indices are not sorted: {verified}"
        )

    return manifest, verified


def _compute_image_stats(
    ep_hf,
    key: str,
) -> dict[str, np.ndarray]:
    sampled = sample_indices(len(ep_hf))

    images: list[np.ndarray] = []

    for row_index in sampled:
        value = ep_hf[row_index][key]

        if not isinstance(value, torch.Tensor):
            raise RuntimeError(
                f"{key}: expected torch.Tensor, "
                f"got {type(value)}"
            )

        arr = (
            value
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

        if arr.ndim != 3 or arr.shape[0] != 3:
            raise RuntimeError(
                f"{key}: expected CHW RGB tensor, "
                f"got shape={arr.shape}"
            )

        if not np.isfinite(arr).all():
            raise RuntimeError(
                f"{key}: non-finite image values"
            )

        minimum = float(arr.min())
        maximum = float(arr.max())

        if minimum < -1e-6 or maximum > 1.0 + 1e-6:
            raise RuntimeError(
                f"{key}: expected decoded float image in "
                f"[0,1], got min={minimum}, max={maximum}"
            )

        images.append(
            auto_downsample_height_width(arr)
        )

    batch = np.stack(images, axis=0)

    stats = get_feature_stats(
        batch,
        axis=(0, 2, 3),
        keepdims=True,
    )

    # LeRobot's image episode stats have shape (C,1,1),
    # while get_feature_stats returns (1,C,1,1).
    return {
        stat_name: (
            stat_value
            if stat_name == "count"
            else np.squeeze(stat_value, axis=0)
        )
        for stat_name, stat_value in stats.items()
    }


def _compute_verified_stats(
    ds: LeRobotDataset,
    verified: list[int],
) -> dict[str, dict[str, np.ndarray]]:
    hf = ds.hf_dataset

    episode_indices = np.asarray(
        hf["episode_index"],
        dtype=np.int64,
    )

    per_episode_stats = []

    numeric_features = {
        key: ds.features[key]
        for key in NUMERIC_KEYS
    }

    for episode_index in verified:
        row_indices = np.flatnonzero(
            episode_indices == episode_index
        ).tolist()

        if not row_indices:
            raise RuntimeError(
                f"No rows loaded for verified "
                f"episode {episode_index}"
            )

        ep_hf = hf.select(row_indices)

        numeric_data = {
            key: np.asarray(
                ep_hf[key],
                dtype=np.float32,
            )
            for key in NUMERIC_KEYS
        }

        ep_stats = compute_episode_stats(
            numeric_data,
            numeric_features,
        )

        for key in IMAGE_KEYS:
            ep_stats[key] = _compute_image_stats(
                ep_hf,
                key,
            )

        per_episode_stats.append(ep_stats)

    return aggregate_stats(per_episode_stats)


def _max_stat_difference(
    global_stats: dict[str, Any],
    verified_stats: dict[str, Any],
    stat_name: str,
) -> float:
    global_value = np.asarray(
        global_stats[stat_name],
        dtype=np.float64,
    )

    verified_value = np.asarray(
        verified_stats[stat_name],
        dtype=np.float64,
    )

    return float(
        np.max(
            np.abs(
                global_value - verified_value
            )
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build and validate the Phase 7 ACT training-input "
            "contract from the Phase 6 verified episode manifest."
        )
    )

    parser.add_argument(
        "--base",
        type=Path,
        default=Path(
            "artifacts/phase6_act_dataset/keyboard_v1"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/phase7_act_coarse/keyboard_v1"
        ),
    )

    args = parser.parse_args()

    manifest_path = (
        args.base / "verified_episodes.json"
    )

    runs_root = args.base / "runs"

    manifest, verified = _load_verified_manifest(
        manifest_path
    )

    repo_id, dataset_root = (
        _resolve_dataset_location(runs_root)
    )

    manifest_repo_id = str(
        manifest.get("dataset_repo_id", "")
    )

    manifest_dataset_root = Path(
        str(
            manifest.get(
                "dataset_root",
                "",
            )
        )
    )

    if manifest_repo_id != repo_id:
        raise RuntimeError(
            "Verified manifest repo_id does not match "
            f"collection runs: {manifest_repo_id!r} != {repo_id!r}"
        )

    if manifest_dataset_root != dataset_root:
        raise RuntimeError(
            "Verified manifest dataset_root does not match "
            f"collection runs: {manifest_dataset_root} != {dataset_root}"
        )

    lerobot_version = version("lerobot")

    if lerobot_version != EXPECTED_LEROBOT_VERSION:
        raise RuntimeError(
            f"Expected lerobot {EXPECTED_LEROBOT_VERSION}, "
            f"got {lerobot_version}"
        )

    ds = LeRobotDataset(
        repo_id,
        root=dataset_root,
        episodes=verified,
        return_uint8=True,
    )

    if int(ds.meta.fps) != EXPECTED_FPS:
        raise RuntimeError(
            f"Expected {EXPECTED_FPS} Hz, "
            f"got {ds.meta.fps}"
        )

    hf = ds.hf_dataset

    loaded_episode_indices = [
        int(x)
        for x in hf["episode_index"]
    ]

    episode_counts = Counter(
        loaded_episode_indices
    )

    loaded_unique = sorted(episode_counts)

    if loaded_unique != verified:
        raise RuntimeError(
            f"Loaded episodes {loaded_unique} "
            f"!= verified whitelist {verified}"
        )

    states = np.asarray(
        hf["observation.state"],
        dtype=np.float32,
    )

    actions = np.asarray(
        hf["action"],
        dtype=np.float32,
    )

    if states.shape != (len(ds), 34):
        raise RuntimeError(
            f"Expected state [N,34], got {states.shape}"
        )

    if actions.shape != (len(ds), 6):
        raise RuntimeError(
            f"Expected action [N,6], got {actions.shape}"
        )

    if not np.isfinite(states).all():
        raise RuntimeError(
            "observation.state contains non-finite values"
        )

    if not np.isfinite(actions).all():
        raise RuntimeError(
            "action contains non-finite values"
        )

    one_hot = states[:, 6:]

    if not np.allclose(
        one_hot.sum(axis=1),
        1.0,
        atol=1e-5,
    ):
        raise RuntimeError(
            "Invalid target one-hot rows"
        )

    manifest_episode_targets: dict[int, str] = {}

    for item in manifest.get(
        "episodes",
        [],
    ):
        if not item.get("verified"):
            continue

        episode_index = int(
            item["dataset_episode_index"]
        )

        target = (
            str(item["target"])
            .strip()
            .upper()
        )

        if target not in TARGET_VOCAB:
            raise RuntimeError(
                "Unsupported verified target in manifest: "
                f"{target}"
            )

        if episode_index in manifest_episode_targets:
            raise RuntimeError(
                "Duplicate verified episode metadata: "
                f"{episode_index}"
            )

        manifest_episode_targets[
            episode_index
        ] = target

    if sorted(
        manifest_episode_targets
    ) != verified:
        raise RuntimeError(
            "Verified manifest episode metadata does not "
            "match dataset_episode_indices"
        )

    row_episode_indices = np.asarray(
        hf["episode_index"],
        dtype=np.int64,
    )

    row_target_indices = np.argmax(
        one_hot,
        axis=1,
    )

    for episode_index in verified:
        target = manifest_episode_targets[
            episode_index
        ]

        expected_target_index = (
            TARGET_VOCAB.index(target)
        )

        mask = (
            row_episode_indices
            == episode_index
        )

        actual_target_indices = sorted(
            set(
                row_target_indices[
                    mask
                ].tolist()
            )
        )

        if actual_target_indices != [
            expected_target_index
        ]:
            raise RuntimeError(
                "Unexpected target encoding for "
                f"episode {episode_index}: "
                f"{actual_target_indices}; expected "
                f"[{expected_target_index}] ({target})"
            )

    verified_targets = sorted(
        set(
            manifest_episode_targets.values()
        )
    )

    sample = ds[0]

    for key in IMAGE_KEYS:
        value = sample[key]

        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != (3, 480, 640)
            or value.dtype != torch.float32
        ):
            raise RuntimeError(
                f"Unexpected decoded {key}: "
                f"type={type(value)}, "
                f"shape={getattr(value, 'shape', None)}, "
                f"dtype={getattr(value, 'dtype', None)}"
            )

    verified_stats = _compute_verified_stats(
        ds,
        verified,
    )

    verified_frame_count = len(ds)

    for key in MODEL_KEYS:
        count = int(
            np.asarray(
                verified_stats[key]["count"]
            ).reshape(-1)[0]
        )

        if count != verified_frame_count:
            raise RuntimeError(
                f"{key} verified stats count={count}, "
                f"expected {verified_frame_count}"
            )

    comparison: dict[str, Any] = {}

    for key in MODEL_KEYS:
        global_count = int(
            np.asarray(
                ds.meta.stats[key]["count"]
            ).reshape(-1)[0]
        )

        verified_count = int(
            np.asarray(
                verified_stats[key]["count"]
            ).reshape(-1)[0]
        )

        comparison[key] = {
            "global_count": global_count,
            "verified_count": verified_count,
            "max_abs_mean_diff": (
                _max_stat_difference(
                    ds.meta.stats[key],
                    verified_stats[key],
                    "mean",
                )
            ),
            "max_abs_std_diff": (
                _max_stat_difference(
                    ds.meta.stats[key],
                    verified_stats[key],
                    "std",
                )
            ),
        }

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stats_path = (
        args.output_dir
        / "normalization_stats.json"
    )

    contract_path = (
        args.output_dir
        / "training_input_contract.json"
    )

    stats_payload = {
        "schema": (
            "phase7.act_verified_normalization_stats.v1"
        ),
        "source_manifest": str(manifest_path),
        "repo_id": repo_id,
        "dataset_root": str(dataset_root),
        "dataset_episode_indices": verified,
        "frame_count": verified_frame_count,
        "stats": _jsonable(verified_stats),
    }

    contract_payload = {
        "schema": (
            "phase7.act_training_input_contract.v1"
        ),
        "lerobot_version": lerobot_version,
        "targets": verified_targets,
        "fps": int(ds.meta.fps),
        "repo_id": repo_id,
        "dataset_root": str(dataset_root),
        "source_manifest": str(manifest_path),
        "verified_episode_indices": verified,
        "verified_episode_frame_counts": {
            str(ep): int(episode_counts[ep])
            for ep in verified
        },
        "verified_frame_count": verified_frame_count,
        "raw_total_episodes": int(
            ds.meta.total_episodes
        ),
        "raw_total_frames": int(
            ds.meta.total_frames
        ),
        "features": {
            key: {
                "dtype": ds.features[key]["dtype"],
                "shape": list(
                    ds.features[key]["shape"]
                ),
                "names": ds.features[key].get(
                    "names"
                ),
            }
            for key in MODEL_KEYS
        },
        "target_one_hot": {
            "dimension": len(TARGET_VOCAB),
            "expected_index": (
                expected_target_index
            ),
            "actual_indices": (
                actual_target_indices
            ),
        },
        "normalization": {
            "source": (
                "verified episodes only"
            ),
            "stats_file": str(stats_path),
            "comparison_to_raw_global_stats": (
                comparison
            ),
        },
        "status": "PASS",
    }

    stats_path.write_text(
        json.dumps(
            stats_payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    contract_path.write_text(
        json.dumps(
            contract_payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print("PHASE 7A — ACT TRAINING INPUT CONTRACT")
    print("=" * 72)

    print("lerobot          :", lerobot_version)
    print("target           :", target)
    print("fps              :", ds.meta.fps)
    print("verified episodes:", verified)
    print("verified frames  :", verified_frame_count)
    print("raw frames       :", ds.meta.total_frames)

    print()
    print("episode frame counts:")

    for ep in verified:
        print(
            f"  {ep}: {episode_counts[ep]}"
        )

    print()
    print("normalization counts:")

    for key in MODEL_KEYS:
        item = comparison[key]

        print(
            f"  {key:28s} "
            f"verified={item['verified_count']} "
            f"global={item['global_count']}"
        )

    print()
    print("global → verified stats differences:")

    for key in MODEL_KEYS:
        item = comparison[key]

        print(
            f"  {key}"
        )
        print(
            "    max |mean diff| = "
            f"{item['max_abs_mean_diff']:.9f}"
        )
        print(
            "    max |std  diff| = "
            f"{item['max_abs_std_diff']:.9f}"
        )

    print()
    print("normalization stats:", stats_path)
    print("contract           :", contract_path)

    print()
    print("STATUS: PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()
