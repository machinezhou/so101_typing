from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.configs import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTConfig, ACTPolicy
from lerobot.processor.normalize_processor import NormalizerProcessorStep
from lerobot.utils.feature_utils import dataset_to_policy_features


DEFAULT_BASE = Path(
    "artifacts/phase6_act_dataset/keyboard_v1"
)

DEFAULT_PHASE7_INPUT = Path(
    "artifacts/phase7_act_coarse/keyboard_v1"
)

EXPECTED_MANIFEST_SCHEMA = (
    "phase6.verified_episode_manifest.v4"
)


def _read_json(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalizer_count(
    normalizer: NormalizerProcessorStep,
    key: str,
) -> int:
    return int(
        np.asarray(
            normalizer.stats[key]["count"]
        ).reshape(-1)[0]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train Phase 7 ACT coarse policy using "
            "verified-only episodes and normalization."
        )
    )

    parser.add_argument(
        "--base",
        type=Path,
        default=DEFAULT_BASE,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_PHASE7_INPUT,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/phase7_act_coarse/"
            "keyboard_v1/act_train"
        ),
    )

    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError(
            "--steps must be positive"
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive"
        )

    _seed_everything(
        args.seed
    )

    # ------------------------------------------------------------
    # Frozen Phase 7 contracts.
    # ------------------------------------------------------------

    manifest = _read_json(
        args.base
        / "verified_episodes.json"
    )

    if manifest.get("schema") != (
        EXPECTED_MANIFEST_SCHEMA
    ):
        raise RuntimeError(
            "Unexpected verified manifest schema: "
            f"{manifest.get('schema')!r}"
        )

    verified = [
        int(x)
        for x in manifest.get(
            "dataset_episode_indices",
            [],
        )
    ]

    if not verified:
        raise RuntimeError(
            "Verified manifest contains no episodes"
        )

    if (
        verified != sorted(verified)
        or len(set(verified)) != len(verified)
    ):
        raise RuntimeError(
            "Verified episode indices must be "
            f"sorted and unique: {verified}"
        )

    temporal = _read_json(
        args.input_dir
        / "temporal_contract.json"
    )

    fps = int(
        temporal["fps"]
    )

    chunk_size = int(
        temporal["chunk_size"]
    )

    if fps != 15:
        raise RuntimeError(
            f"Expected 15 Hz, got {fps}"
        )

    if chunk_size != 20:
        raise RuntimeError(
            "Expected frozen chunk_size=20, "
            f"got {chunk_size}"
        )

    if temporal.get(
        "verified_episode_indices"
    ) != verified:
        raise RuntimeError(
            "Temporal contract verified episodes "
            "do not match manifest"
        )

    stats_payload = _read_json(
        args.input_dir
        / "normalization_stats.json"
    )

    verified_frame_count = int(
        stats_payload["frame_count"]
    )

    if verified_frame_count <= 0:
        raise RuntimeError(
            "Normalization stats contain no "
            "verified frames"
        )

    if int(
        temporal["verified_frame_count"]
    ) != verified_frame_count:
        raise RuntimeError(
            "Temporal and normalization frame "
            "counts do not match"
        )

    verified_stats = (
        stats_payload["stats"]
    )

    # ------------------------------------------------------------
    # Resolve immutable Phase 6 raw dataset.
    # ------------------------------------------------------------

    repo_id = str(
        manifest["dataset_repo_id"]
    )

    dataset_root = Path(
        manifest["dataset_root"]
    )

    # ------------------------------------------------------------
    # ACT training target:
    # action[t : t + chunk_size]
    # ------------------------------------------------------------

    delta_timestamps = {
        "action": [
            index / fps
            for index in range(
                chunk_size
            )
        ]
    }

    dataset = LeRobotDataset(
        repo_id,
        root=dataset_root,
        episodes=verified,
        delta_timestamps=delta_timestamps,
        return_uint8=True,
    )

    if len(dataset) != verified_frame_count:
        raise RuntimeError(
            "Training dataset frame count does "
            "not match verified normalization: "
            f"{len(dataset)} != {verified_frame_count}"
        )

    # Raw metadata MUST remain repository-global.
    raw_action_count = int(
        np.asarray(
            dataset.meta.stats[
                "action"
            ]["count"]
        ).reshape(-1)[0]
    )

    if raw_action_count != int(
        dataset.meta.total_frames
    ):
        raise RuntimeError(
            "Unexpected raw metadata frame count: "
            f"stats={raw_action_count}, "
            f"meta={dataset.meta.total_frames}"
        )

    # ------------------------------------------------------------
    # Policy features.
    # ------------------------------------------------------------

    features = dataset_to_policy_features(
        dataset.features
    )

    output_features = {
        key: feature
        for key, feature
        in features.items()
        if feature.type
        is FeatureType.ACTION
    }

    input_features = {
        key: feature
        for key, feature
        in features.items()
        if key not in output_features
    }

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ------------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT override pretrained_backbone_weights.
    #
    # We intentionally use LeRobot ACT's official default:
    # ResNet18 + ImageNet pretrained initialization.
    #
    # n_action_steps=1 remains a TRAINING/CONFIG placeholder.
    # Runtime execution horizon is not frozen yet.
    # ------------------------------------------------------------

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=1,
        chunk_size=chunk_size,
        n_action_steps=1,
        device=device,
    )

    if cfg.action_delta_indices != list(
        range(chunk_size)
    ):
        raise RuntimeError(
            "ACT temporal configuration mismatch"
        )

    # ------------------------------------------------------------
    # Verified-only processor injection.
    # ------------------------------------------------------------

    preprocessor, postprocessor = (
        make_pre_post_processors(
            policy_cfg=cfg,
            dataset_stats=verified_stats,
        )
    )

    normalizer = next(
        step
        for step in preprocessor.steps
        if isinstance(
            step,
            NormalizerProcessorStep,
        )
    )

    for key in (
        "observation.images.top",
        "observation.images.wrist",
        "observation.state",
        "action",
    ):
        count = _normalizer_count(
            normalizer,
            key,
        )

        if count != verified_frame_count:
            raise RuntimeError(
                f"{key}: global normalization "
                f"leak detected: count={count}"
            )

    # ------------------------------------------------------------
    # ACT model.
    # ------------------------------------------------------------

    policy = ACTPolicy(
        cfg
    ).to(device)

    policy.train()

    optimizer_cfg = (
        cfg.get_optimizer_preset()
    )

    optimizer = optimizer_cfg.build(
        policy.get_optim_params()
    )

    # ------------------------------------------------------------
    # DataLoader.
    # ------------------------------------------------------------

    generator = torch.Generator()
    generator.manual_seed(
        args.seed
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=(
            device == "cuda"
        ),
        drop_last=False,
    )

    # ------------------------------------------------------------
    # Training.
    # ------------------------------------------------------------

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    iterator = iter(loader)

    print("=" * 78)
    print(
        "PHASE 7C — ACT TRAINING"
    )
    print("=" * 78)

    print("device            :", device)

    if device == "cuda":
        print(
            "gpu               :",
            torch.cuda.get_device_name(0),
        )

    print("verified episodes :", verified)
    print("training frames   :", len(dataset))
    print("raw meta frames   :", dataset.meta.total_frames)
    print("fps               :", fps)
    print("chunk_size        :", chunk_size)
    print(
        "n_action_steps    :",
        cfg.n_action_steps,
        "(NOT RUNTIME-FROZEN)",
    )
    print(
        "backbone weights  :",
        cfg.pretrained_backbone_weights,
    )
    print("batch_size        :", args.batch_size)
    print("steps             :", args.steps)

    print()
    print("-" * 78)

    last_loss = None
    last_loss_dict = None

    for step in range(
        1,
        args.steps + 1,
    ):
        try:
            raw_batch = next(
                iterator
            )
        except StopIteration:
            iterator = iter(
                loader
            )
            raw_batch = next(
                iterator
            )

        # Match LeRobot v0.6.1 official trainer:
        # uint8 camera tensors must enter the policy as [0, 1] float.
        for camera_key in dataset.meta.camera_keys:
            if (
                camera_key in raw_batch
                and raw_batch[camera_key].dtype
                == torch.uint8
            ):
                raw_batch[camera_key] = (
                    raw_batch[camera_key]
                    .to(dtype=torch.float32)
                    / 255.0
                )

        batch = preprocessor(
            raw_batch
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss, loss_dict = (
            policy.forward(
                batch
            )
        )

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                f"Non-finite loss at step {step}"
            )

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            max_norm=optimizer_cfg.grad_clip_norm,
        )

        if not torch.isfinite(
            grad_norm
        ):
            raise RuntimeError(
                f"Non-finite grad norm "
                f"at step {step}"
            )

        optimizer.step()

        last_loss = float(
            loss.item()
        )

        last_loss_dict = (
            loss_dict
        )

        print(
            f"step={step:6d} "
            f"loss={last_loss:.6f} "
            f"l1={float(loss_dict['l1_loss']):.6f} "
            f"kld={float(loss_dict.get('kld_loss', 0.0)):.6f} "
            f"grad_norm={float(grad_norm):.6f}"
        )

    # ------------------------------------------------------------
    # Save checkpoint + processors.
    #
    # This ensures the VERIFIED normalization stats travel
    # with the policy checkpoint.
    # ------------------------------------------------------------

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    policy.save_pretrained(
        args.output_dir
    )

    preprocessor.save_pretrained(
        args.output_dir
    )

    postprocessor.save_pretrained(
        args.output_dir
    )

    training_contract = {
        "schema": (
            "phase7.act_training_run.v1"
        ),
        "verified_episode_indices": (
            verified
        ),
        "verified_frame_count": (
            len(dataset)
        ),
        "raw_frame_count": (
            int(
                dataset.meta.total_frames
            )
        ),
        "fps": fps,
        "chunk_size": chunk_size,
        "n_action_steps": {
            "value": 1,
            "status": (
                "training_placeholder_not_runtime_frozen"
            ),
        },
        "normalization_frame_count": (
            verified_frame_count
        ),
        "backbone_weights": str(
            cfg.pretrained_backbone_weights
        ),
        "batch_size": (
            args.batch_size
        ),
        "steps": (
            args.steps
        ),
        "seed": (
            args.seed
        ),
        "final_loss": (
            last_loss
        ),
        "final_loss_dict": (
            last_loss_dict
        ),
    }

    (
        args.output_dir
        / "phase7_training_run.json"
    ).write_text(
        json.dumps(
            training_contract,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("-" * 78)

    if device == "cuda":
        print(
            "cuda allocated MB :",
            f"{torch.cuda.memory_allocated() / 1024**2:.1f}",
        )

        print(
            "cuda peak MB      :",
            f"{torch.cuda.max_memory_allocated() / 1024**2:.1f}",
        )

    print(
        "checkpoint         :",
        args.output_dir,
    )

    print()
    print(
        "verified samples   : PASS"
    )
    print(
        "verified stats     : PASS"
    )
    print(
        "forward            : PASS"
    )
    print(
        "backward           : PASS"
    )
    print(
        "optimizer step     : PASS"
    )
    print(
        "checkpoint save    : PASS"
    )

    print("=" * 78)
    print("STATUS: PASS")
    print("=" * 78)


if __name__ == "__main__":
    main()
