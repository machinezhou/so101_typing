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


def _update_last_checkpoint(
    checkpoint_dir: Path,
) -> None:
    last_path = (
        checkpoint_dir.parent
        / "last"
    )

    if last_path.is_symlink():
        last_path.unlink()
    elif last_path.exists():
        raise RuntimeError(
            "Refusing to replace non-symlink "
            f"checkpoint path: {last_path}"
        )

    last_path.symlink_to(
        checkpoint_dir.name,
        target_is_directory=True,
    )


def _save_training_checkpoint(
    *,
    checkpoint_dir: Path,
    policy,
    preprocessor,
    postprocessor,
    optimizer,
    generator: torch.Generator,
    completed_step: int,
    target_steps: int,
    device: str,
    batch_size: int,
    num_workers: int,
    save_freq: int,
    seed: int,
    fps: int,
    chunk_size: int,
    verified_frame_count: int,
    verified_episode_indices: list[int],
    output_dir: Path,
) -> None:
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    policy.save_pretrained(
        checkpoint_dir
    )

    preprocessor.save_pretrained(
        checkpoint_dir
    )

    postprocessor.save_pretrained(
        checkpoint_dir
    )

    training_state = {
        "schema": (
            "phase7.act_resume_state.v1"
        ),
        "completed_step": int(
            completed_step
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "torch_rng_state": (
            torch.get_rng_state()
        ),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all()
            if device == "cuda"
            else None
        ),
        "loader_generator_state": (
            generator.get_state()
        ),
    }

    torch.save(
        training_state,
        checkpoint_dir
        / "training_state.pt",
    )

    resume_config = {
        "schema": (
            "phase7.act_resume_config.v1"
        ),
        "completed_step": int(
            completed_step
        ),
        "target_steps": int(
            target_steps
        ),
        "device": device,
        "batch_size": int(
            batch_size
        ),
        "num_workers": int(
            num_workers
        ),
        "save_freq": int(
            save_freq
        ),
        "seed": int(
            seed
        ),
        "fps": int(
            fps
        ),
        "chunk_size": int(
            chunk_size
        ),
        "verified_frame_count": int(
            verified_frame_count
        ),
        "verified_episode_indices": (
            verified_episode_indices
        ),
        "output_dir": str(
            output_dir
        ),
    }

    (
        checkpoint_dir
        / "resume_config.json"
    ).write_text(
        json.dumps(
            resume_config,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    _update_last_checkpoint(
        checkpoint_dir
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
        default=None,
        help=(
            "Total target training step. "
            "Fresh run defaults to 1. "
            "Resume defaults to the target stored in the checkpoint."
        ),
    )

    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help=(
            "Resume from a Phase 7C checkpoint directory, "
            "including checkpoints/last."
        ),
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--save-freq",
        type=int,
        default=0,
        help=(
            "Save an intermediate policy checkpoint every N steps; "
            "0 disables periodic checkpoints."
        ),
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

    resume_config = None

    if args.resume_from is not None:
        resume_config_path = (
            args.resume_from
            / "resume_config.json"
        )

        if not resume_config_path.is_file():
            raise RuntimeError(
                "Resume checkpoint has no resume_config.json: "
                f"{args.resume_from}"
            )

        resume_config = _read_json(
            resume_config_path
        )

        if resume_config.get("schema") != (
            "phase7.act_resume_config.v1"
        ):
            raise RuntimeError(
                "Unexpected resume config schema: "
                f"{resume_config.get('schema')!r}"
            )

        # Strict resume: training parameters come from
        # the checkpoint rather than CLI defaults.
        args.batch_size = int(
            resume_config["batch_size"]
        )
        args.num_workers = int(
            resume_config["num_workers"]
        )
        args.save_freq = int(
            resume_config["save_freq"]
        )
        args.seed = int(
            resume_config["seed"]
        )
        args.device = str(
            resume_config["device"]
        )
        args.output_dir = Path(
            resume_config["output_dir"]
        )

        if args.steps is None:
            args.steps = int(
                resume_config[
                    "target_steps"
                ]
            )

    elif args.steps is None:
        args.steps = 1

    if args.steps <= 0:
        raise ValueError(
            "--steps must be positive"
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive"
        )

    if args.num_workers < 0:
        raise ValueError(
            "--num-workers must be >= 0"
        )

    if args.save_freq < 0:
        raise ValueError(
            "--save-freq must be >= 0"
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

    if args.device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda requested but CUDA is unavailable"
            )
        device = "cuda"
    else:
        device = "cpu"

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

    resume_state = None
    resume_completed_step = 0

    if resume_config is not None:
        if int(
            resume_config["fps"]
        ) != fps:
            raise RuntimeError(
                "Resume fps does not match current contract"
            )

        if int(
            resume_config["chunk_size"]
        ) != chunk_size:
            raise RuntimeError(
                "Resume chunk_size does not match current contract"
            )

        if int(
            resume_config[
                "verified_frame_count"
            ]
        ) != verified_frame_count:
            raise RuntimeError(
                "Resume verified frame count does not "
                "match current contract"
            )

        if [
            int(x)
            for x in resume_config[
                "verified_episode_indices"
            ]
        ] != verified:
            raise RuntimeError(
                "Resume verified episodes do not "
                "match current manifest"
            )

        policy = ACTPolicy.from_pretrained(
            args.resume_from,
            config=cfg,
            local_files_only=True,
            strict=True,
        ).to(device)

    else:
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

    if resume_config is not None:
        state_path = (
            args.resume_from
            / "training_state.pt"
        )

        if not state_path.is_file():
            raise RuntimeError(
                "Resume checkpoint has no training_state.pt: "
                f"{args.resume_from}"
            )

        # Training/RNG state must be loaded on CPU.
        # In particular, torch.Generator.set_state() requires
        # a CPU ByteTensor. Optimizer state is moved to the
        # parameter device by optimizer.load_state_dict().
        resume_state = torch.load(
            state_path,
            map_location="cpu",
            weights_only=False,
        )

        if resume_state.get("schema") != (
            "phase7.act_resume_state.v1"
        ):
            raise RuntimeError(
                "Unexpected resume state schema: "
                f"{resume_state.get('schema')!r}"
            )

        resume_completed_step = int(
            resume_state[
                "completed_step"
            ]
        )

        if resume_completed_step != int(
            resume_config[
                "completed_step"
            ]
        ):
            raise RuntimeError(
                "Resume state/config step mismatch"
            )

        if resume_completed_step >= args.steps:
            raise RuntimeError(
                "Resume checkpoint already reached "
                f"step {resume_completed_step}, "
                f"target steps={args.steps}"
            )

        optimizer.load_state_dict(
            resume_state[
                "optimizer_state_dict"
            ]
        )

    # ------------------------------------------------------------
    # DataLoader.
    # ------------------------------------------------------------

    generator = torch.Generator()

    if resume_state is not None:
        generator.set_state(
            resume_state[
                "loader_generator_state"
            ]
        )
    else:
        generator.manual_seed(
            args.seed
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
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

    start_step = 1

    if resume_state is not None:
        torch.set_rng_state(
            resume_state[
                "torch_rng_state"
            ].cpu()
        )

        if (
            device == "cuda"
            and resume_state[
                "cuda_rng_state_all"
            ] is not None
        ):
            torch.cuda.set_rng_state_all(
                resume_state[
                    "cuda_rng_state_all"
                ]
            )

        start_step = (
            resume_completed_step + 1
        )

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
    print("num_workers       :", args.num_workers)
    print("steps             :", args.steps)
    print("save_freq         :", args.save_freq)

    if args.resume_from is not None:
        print(
            "resume_from       :",
            args.resume_from,
        )
        print(
            "resume_step       :",
            resume_completed_step,
        )
        print(
            "next_step         :",
            start_step,
        )

    print()
    print("-" * 78)

    last_loss = None
    last_loss_dict = None
    last_saved_step = (
        resume_completed_step
    )

    for step in range(
        start_step,
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

        if (
            args.save_freq > 0
            and step % args.save_freq == 0
        ):
            checkpoint_dir = (
                args.output_dir
                / "checkpoints"
                / f"step_{step:06d}"
            )

            _save_training_checkpoint(
                checkpoint_dir=checkpoint_dir,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                optimizer=optimizer,
                generator=generator,
                completed_step=step,
                target_steps=args.steps,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                save_freq=args.save_freq,
                seed=args.seed,
                fps=fps,
                chunk_size=chunk_size,
                verified_frame_count=verified_frame_count,
                verified_episode_indices=verified,
                output_dir=args.output_dir,
            )

            last_saved_step = step

            print(
                "checkpoint         :",
                checkpoint_dir,
            )
            print(
                "checkpoint last    :",
                args.output_dir
                / "checkpoints"
                / "last",
            )

    # Always leave the final training step as a resumable checkpoint.
    # If save_freq already saved this exact step, do not duplicate it.
    if last_saved_step != args.steps:
        checkpoint_dir = (
            args.output_dir
            / "checkpoints"
            / f"step_{args.steps:06d}"
        )

        _save_training_checkpoint(
            checkpoint_dir=checkpoint_dir,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            optimizer=optimizer,
            generator=generator,
            completed_step=args.steps,
            target_steps=args.steps,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            save_freq=args.save_freq,
            seed=args.seed,
            fps=fps,
            chunk_size=chunk_size,
            verified_frame_count=verified_frame_count,
            verified_episode_indices=verified,
            output_dir=args.output_dir,
        )

        last_saved_step = args.steps

        print(
            "checkpoint         :",
            checkpoint_dir,
        )
        print(
            "checkpoint last    :",
            args.output_dir
            / "checkpoints"
            / "last",
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
        "device": device,
        "batch_size": (
            args.batch_size
        ),
        "num_workers": (
            args.num_workers
        ),
        "steps": (
            args.steps
        ),
        "save_freq": (
            args.save_freq
        ),
        "completed_steps": (
            args.steps
        ),
        "resume": {
            "source": (
                str(args.resume_from)
                if args.resume_from is not None
                else None
            ),
            "resumed_from_step": (
                resume_completed_step
            ),
            "last_checkpoint": str(
                args.output_dir
                / "checkpoints"
                / "last"
            ),
        },
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
