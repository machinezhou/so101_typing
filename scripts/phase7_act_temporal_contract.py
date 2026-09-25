from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


FPS = 15
CHUNK_SIZE = 20

EXPECTED_MANIFEST_SCHEMA = (
    "phase6.verified_episode_manifest.v4"
)


def _read_json(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def _verified_indices(
    manifest: dict,
) -> list[int]:
    if manifest.get("schema") != EXPECTED_MANIFEST_SCHEMA:
        raise RuntimeError(
            "Unexpected verified manifest schema: "
            f"{manifest.get('schema')!r}"
        )

    raw = manifest.get(
        "dataset_episode_indices"
    )

    if not isinstance(raw, list) or not raw:
        raise RuntimeError(
            "Verified manifest contains no episodes"
        )

    verified = [
        int(value)
        for value in raw
    ]

    if len(set(verified)) != len(verified):
        raise RuntimeError(
            "Duplicate verified episode indices: "
            f"{verified}"
        )

    if verified != sorted(verified):
        raise RuntimeError(
            "Verified episode indices are not sorted: "
            f"{verified}"
        )

    return verified


def _episode_lengths(
    episode_indices: np.ndarray,
    verified: list[int],
) -> dict[int, int]:
    return {
        ep: int(
            np.sum(
                episode_indices == ep
            )
        )
        for ep in verified
    }


def _valid_steps_for_episode(
    episode_length: int,
    chunk_size: int,
) -> list[int]:
    if episode_length <= 0:
        raise ValueError(
            "episode_length must be positive"
        )

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be positive"
        )

    return [
        min(
            chunk_size,
            episode_length - position,
        )
        for position in range(
            episode_length
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the frozen Phase 7 ACT temporal "
            "training contract for the verified formal dataset."
        )
    )

    parser.add_argument(
        "--base",
        type=Path,
        default=Path(
            "artifacts/phase6_act_dataset/"
            "keyboard_v1"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/phase7_act_coarse/"
            "keyboard_v1/"
            "temporal_contract.json"
        ),
    )

    args = parser.parse_args()

    manifest = _read_json(
        args.base
        / "verified_episodes.json"
    )

    verified = _verified_indices(
        manifest
    )

    repo_id = str(
        manifest["dataset_repo_id"]
    )

    dataset_root = Path(
        manifest["dataset_root"]
    )

    ds = LeRobotDataset(
        repo_id,
        root=dataset_root,
        episodes=verified,
        delta_timestamps={
            "action": [
                index / FPS
                for index in range(
                    CHUNK_SIZE
                )
            ]
        },
        return_uint8=True,
    )

    episode_indices = np.asarray(
        ds.hf_dataset[
            "episode_index"
        ],
        dtype=np.int64,
    )

    lengths = _episode_lengths(
        episode_indices,
        verified,
    )

    missing = [
        ep
        for ep, length
        in lengths.items()
        if length <= 0
    ]

    if missing:
        raise RuntimeError(
            "Verified episodes contain no loaded frames: "
            f"{missing}"
        )

    expected_valid_steps: list[int] = []

    for ep in verified:
        expected_valid_steps.extend(
            _valid_steps_for_episode(
                lengths[ep],
                CHUNK_SIZE,
            )
        )

    reader = ds._ensure_reader()

    actual_valid_steps: list[int] = []

    for row_index in range(
        len(ds.hf_dataset)
    ):
        row = ds.hf_dataset[
            row_index
        ]

        abs_idx = int(
            row["index"]
        )

        ep_idx = int(
            row["episode_index"]
        )

        _, padding = (
            reader._get_query_indices(
                abs_idx,
                ep_idx,
            )
        )

        actual_valid_steps.append(
            int(
                (
                    ~padding[
                        "action_is_pad"
                    ]
                )
                .sum()
                .item()
            )
        )

    if (
        actual_valid_steps
        != expected_valid_steps
    ):
        raise RuntimeError(
            "LeRobot action padding behavior "
            "does not match temporal contract"
        )

    total_slots = (
        len(expected_valid_steps)
        * CHUNK_SIZE
    )

    valid_slots = sum(
        expected_valid_steps
    )

    padded_slots = (
        total_slots - valid_slots
    )

    full_windows = sum(
        value == CHUNK_SIZE
        for value in expected_valid_steps
    )

    frame_count = len(ds)

    payload = {
        "schema": (
            "phase7.act_temporal_contract.v1"
        ),
        "verified_manifest_schema": (
            EXPECTED_MANIFEST_SCHEMA
        ),
        "fps": FPS,
        "n_obs_steps": 1,
        "chunk_size": CHUNK_SIZE,
        "action_delta_indices": list(
            range(CHUNK_SIZE)
        ),
        "action_delta_timestamps_s": [
            i / FPS
            for i in range(
                CHUNK_SIZE
            )
        ],
        "first_to_last_action_span_s": (
            (CHUNK_SIZE - 1) / FPS
        ),
        "command_count_duration_s": (
            CHUNK_SIZE / FPS
        ),
        "verified_episode_indices": (
            verified
        ),
        "episode_frame_counts": {
            str(ep): lengths[ep]
            for ep in verified
        },
        "verified_frame_count": (
            frame_count
        ),
        "padding": {
            "total_action_slots": (
                total_slots
            ),
            "valid_action_slots": (
                valid_slots
            ),
            "padded_action_slots": (
                padded_slots
            ),
            "padded_fraction": (
                padded_slots
                / total_slots
            ),
            "full_window_count": (
                full_windows
            ),
            "full_window_fraction": (
                full_windows
                / frame_count
            ),
        },
        "n_action_steps": {
            "status": "not_frozen",
            "reason": (
                "runtime execution horizon "
                "will be selected after "
                "ACT inference latency "
                "characterization"
            ),
        },
        "status": "PASS",
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print(
        "PHASE 7B — ACT TEMPORAL CONTRACT"
    )
    print("=" * 72)

    print(
        "verified episodes:",
        len(verified),
    )
    print(
        "verified frames  :",
        frame_count,
    )
    print(
        "fps              :",
        FPS,
    )
    print(
        "chunk_size       :",
        CHUNK_SIZE,
    )
    print(
        "prediction span  :",
        f"{(CHUNK_SIZE - 1) / FPS:.3f} s",
    )
    print(
        "command duration :",
        f"{CHUNK_SIZE / FPS:.3f} s",
    )
    print(
        "padding          :",
        f"{100 * padded_slots / total_slots:.2f}%",
    )
    print(
        "full windows     :",
        f"{full_windows}/{frame_count} "
        f"({100 * full_windows / frame_count:.2f}%)",
    )

    print()
    print(
        "n_action_steps   : NOT FROZEN"
    )
    print(
        "output           :",
        args.output,
    )
    print()
    print("STATUS: PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()
