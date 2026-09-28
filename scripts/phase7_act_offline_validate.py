from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy


DEFAULT_BASE = Path(
    "artifacts/phase6_act_dataset/keyboard_v1"
)

DEFAULT_CHECKPOINT = Path(
    "artifacts/phase7_act_coarse/keyboard_v1/"
    "act_train_20k/checkpoints/last"
)

DEFAULT_OUTPUT = Path(
    "artifacts/phase7_act_coarse/keyboard_v1/"
    "offline_validation_20k.json"
)

IMAGE_KEYS = (
    "observation.images.top",
    "observation.images.wrist",
)

STATE_KEY = "observation.state"

TARGET_VOCAB = (
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + ["SPACE", "BACKSPACE"]
)

SAMPLE_FRACTIONS = (
    0.25,
    0.50,
    0.75,
)


def _read_json(path: Path) -> dict:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def _to_inference_observation(
    sample: dict,
) -> dict[str, torch.Tensor]:
    observation = {}

    for key in IMAGE_KEYS:
        value = sample[key]

        if not isinstance(
            value,
            torch.Tensor,
        ):
            raise RuntimeError(
                f"{key}: expected Tensor, "
                f"got {type(value)}"
            )

        if tuple(value.shape) != (
            3,
            480,
            640,
        ):
            raise RuntimeError(
                f"{key}: unexpected shape "
                f"{tuple(value.shape)}"
            )

        if value.dtype == torch.uint8:
            value = (
                value.to(torch.float32)
                / 255.0
            )
        elif value.dtype.is_floating_point:
            value = value.to(
                torch.float32
            )
        else:
            raise RuntimeError(
                f"{key}: unexpected dtype "
                f"{value.dtype}"
            )

        if (
            float(value.min()) < 0.0
            or float(value.max()) > 1.0
        ):
            raise RuntimeError(
                f"{key}: image outside "
                "[0, 1] range"
            )

        observation[key] = value

    state = sample[STATE_KEY]

    if not isinstance(
        state,
        torch.Tensor,
    ):
        raise RuntimeError(
            "observation.state is not Tensor"
        )

    if tuple(state.shape) != (34,):
        raise RuntimeError(
            "Expected state shape (34,), got "
            f"{tuple(state.shape)}"
        )

    observation[STATE_KEY] = (
        state.to(torch.float32)
    )

    return observation


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline Phase 7D ACT inference "
            "validation. No robot motion."
        )
    )

    parser.add_argument(
        "--base",
        type=Path,
        default=DEFAULT_BASE,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    if not args.checkpoint.is_dir():
        raise RuntimeError(
            "Checkpoint directory not found: "
            f"{args.checkpoint}"
        )

    manifest = _read_json(
        args.base
        / "verified_episodes.json"
    )

    verified = [
        int(x)
        for x in manifest[
            "dataset_episode_indices"
        ]
    ]

    if not verified:
        raise RuntimeError(
            "No verified episodes"
        )

    episode_targets = {}

    for item in manifest["episodes"]:
        if not item.get(
            "verified",
            False,
        ):
            continue

        episode_index = int(
            item["dataset_episode_index"]
        )

        target = str(
            item["target"]
        ).strip().upper()

        episode_targets[
            episode_index
        ] = target

    verified_targets = sorted(
        set(
            episode_targets.values()
        )
    )

    expected_targets = list(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )

    if verified_targets != expected_targets:
        raise RuntimeError(
            "Expected verified targets A-Z, "
            f"got {verified_targets}"
        )

    repo_id = str(
        manifest["dataset_repo_id"]
    )

    dataset_root = Path(
        manifest["dataset_root"]
    )

    dataset = LeRobotDataset(
        repo_id,
        root=dataset_root,
        episodes=verified,
        return_uint8=True,
    )

    hf = dataset.hf_dataset
    raw_hf = hf.with_format(None)

    row_episode_indices = np.asarray(
        raw_hf["episode_index"],
        dtype=np.int64,
    )

    print("=" * 78)
    print(
        "PHASE 7D-1 — ACT OFFLINE "
        "INFERENCE VALIDATION"
    )
    print("=" * 78)

    print(
        "checkpoint         :",
        args.checkpoint,
    )
    print(
        "dataset frames     :",
        len(dataset),
    )
    print(
        "verified episodes  :",
        len(verified),
    )
    print(
        "targets            :",
        ",".join(
            verified_targets
        ),
    )
    print(
        "samples/target     :",
        len(SAMPLE_FRACTIONS),
    )

    # ------------------------------------------------------------
    # Load the exact saved policy + saved processors.
    # ------------------------------------------------------------

    policy = ACTPolicy.from_pretrained(
        args.checkpoint,
        local_files_only=True,
        strict=True,
    )

    policy.eval()

    if policy.config.chunk_size != 20:
        raise RuntimeError(
            "Expected chunk_size=20, got "
            f"{policy.config.chunk_size}"
        )

    if policy.config.n_action_steps != 1:
        raise RuntimeError(
            "Expected training checkpoint "
            "n_action_steps=1, got "
            f"{policy.config.n_action_steps}"
        )

    preprocessor, postprocessor = (
        make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=str(
                args.checkpoint
            ),
        )
    )

    print(
        "device             :",
        policy.config.device,
    )
    print(
        "chunk_size         :",
        policy.config.chunk_size,
    )
    print(
        "n_action_steps     :",
        policy.config.n_action_steps,
    )

    print()
    print("-" * 78)

    # ------------------------------------------------------------
    # Pick exactly one verified episode per A-Z target.
    # Then evaluate 25%, 50%, 75% positions in that episode.
    # ------------------------------------------------------------

    selected_episode = {}

    for episode_index in verified:
        target = episode_targets[
            episode_index
        ]

        if target not in selected_episode:
            selected_episode[
                target
            ] = episode_index

    results = []

    for target in expected_targets:
        episode_index = (
            selected_episode[target]
        )

        rows = np.flatnonzero(
            row_episode_indices
            == episode_index
        )

        if len(rows) == 0:
            raise RuntimeError(
                "No rows found for episode "
                f"{episode_index}"
            )

        for fraction in SAMPLE_FRACTIONS:
            offset = int(
                round(
                    fraction
                    * (len(rows) - 1)
                )
            )

            row_index = int(
                rows[offset]
            )

            sample = dataset[
                row_index
            ]

            state = sample[
                STATE_KEY
            ].detach().cpu()

            one_hot = state[
                6:
            ].numpy()

            actual_target_index = int(
                np.argmax(one_hot)
            )

            expected_target_index = (
                TARGET_VOCAB.index(
                    target
                )
            )

            if (
                actual_target_index
                != expected_target_index
            ):
                raise RuntimeError(
                    "Target one-hot mismatch: "
                    f"target={target}, "
                    f"episode={episode_index}, "
                    f"actual={actual_target_index}, "
                    f"expected={expected_target_index}"
                )

            observation = (
                _to_inference_observation(
                    sample
                )
            )

            policy.reset()

            processed = preprocessor(
                observation
            )

            with torch.inference_mode():
                predicted = (
                    policy.select_action(
                        processed
                    )
                )

            predicted = postprocessor(
                predicted
            )

            if not isinstance(
                predicted,
                torch.Tensor,
            ):
                raise RuntimeError(
                    "Postprocessed action "
                    "is not Tensor"
                )

            if tuple(
                predicted.shape
            ) != (1, 6):
                raise RuntimeError(
                    "Expected predicted action "
                    "shape (1, 6), got "
                    f"{tuple(predicted.shape)}"
                )

            predicted_action = (
                predicted[
                    0
                ]
                .detach()
                .cpu()
                .to(torch.float32)
            )

            actual_action = (
                sample["action"]
                .detach()
                .cpu()
                .to(torch.float32)
            )

            if tuple(
                actual_action.shape
            ) != (6,):
                raise RuntimeError(
                    "Expected ground-truth "
                    "action shape (6,), got "
                    f"{tuple(actual_action.shape)}"
                )

            if not torch.isfinite(
                predicted_action
            ).all():
                raise RuntimeError(
                    "Non-finite predicted action"
                )

            abs_error = (
                predicted_action
                - actual_action
            ).abs()

            mae = float(
                abs_error.mean()
            )

            max_error = float(
                abs_error.max()
            )

            results.append(
                {
                    "target": target,
                    "episode_index": (
                        episode_index
                    ),
                    "fraction": (
                        fraction
                    ),
                    "row_index": row_index,
                    "predicted_action": [
                        float(x)
                        for x
                        in predicted_action
                        .tolist()
                    ],
                    "actual_action": [
                        float(x)
                        for x
                        in actual_action
                        .tolist()
                    ],
                    "mae_deg": mae,
                    "max_abs_error_deg": (
                        max_error
                    ),
                }
            )

            print(
                f"{target:>2s} "
                f"ep={episode_index:03d} "
                f"pos={fraction:>4.2f} "
                f"mae={mae:7.3f}deg "
                f"max={max_error:7.3f}deg"
            )

    if len(results) != 78:
        raise RuntimeError(
            "Expected exactly 78 "
            "validation samples, got "
            f"{len(results)}"
        )

    maes = np.asarray(
        [
            item["mae_deg"]
            for item in results
        ],
        dtype=np.float64,
    )

    max_errors = np.asarray(
        [
            item[
                "max_abs_error_deg"
            ]
            for item in results
        ],
        dtype=np.float64,
    )

    per_target = {}

    for target in expected_targets:
        target_rows = [
            item
            for item in results
            if item["target"] == target
        ]

        per_target[target] = {
            "mean_mae_deg": float(
                np.mean(
                    [
                        item["mae_deg"]
                        for item
                        in target_rows
                    ]
                )
            ),
            "max_abs_error_deg": float(
                np.max(
                    [
                        item[
                            "max_abs_error_deg"
                        ]
                        for item
                        in target_rows
                    ]
                )
            ),
        }

    payload = {
        "schema": (
            "phase7.act_offline_"
            "inference_validation.v1"
        ),
        "checkpoint": str(
            args.checkpoint
        ),
        "verified_episode_count": (
            len(verified)
        ),
        "target_count": 26,
        "sample_count": len(
            results
        ),
        "sampling": [
            0.25,
            0.50,
            0.75,
        ],
        "diagnostic_only": (
            "training-set imitation "
            "error is not an acceptance "
            "criterion or generalization "
            "metric"
        ),
        "overall": {
            "mean_mae_deg": float(
                maes.mean()
            ),
            "median_mae_deg": float(
                np.median(maes)
            ),
            "p95_mae_deg": float(
                np.percentile(
                    maes,
                    95,
                )
            ),
            "max_sample_mae_deg": float(
                maes.max()
            ),
            "max_joint_abs_error_deg": (
                float(
                    max_errors.max()
                )
            ),
        },
        "per_target": per_target,
        "samples": results,
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

    print("-" * 78)
    print(
        "mean MAE          :",
        f"{maes.mean():.3f} deg",
    )
    print(
        "median MAE        :",
        f"{np.median(maes):.3f} deg",
    )
    print(
        "p95 MAE           :",
        f"{np.percentile(maes, 95):.3f} deg",
    )
    print(
        "max sample MAE    :",
        f"{maes.max():.3f} deg",
    )
    print(
        "max joint error   :",
        f"{max_errors.max():.3f} deg",
    )
    print(
        "report            :",
        args.output,
    )
    print()
    print(
        "NOTE: imitation error is "
        "diagnostic only because these "
        "are training episodes."
    )
    print("=" * 78)
    print("STATUS: PASS")
    print("=" * 78)


if __name__ == "__main__":
    main()
