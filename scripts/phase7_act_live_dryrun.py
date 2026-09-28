from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy
from lerobot.robots.so_follower import (
    SO101Follower,
    SO101FollowerConfig,
)

from so101_typing.adapters.cameras import (
    CameraSpec,
    ThreadedOpenCVCamera,
)


ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "lawson_follower_arm"

TOP_CAMERA_CONFIG = Path(
    "configs/cameras/top.yaml"
)
WRIST_CAMERA_CONFIG = Path(
    "configs/cameras/wrist.yaml"
)

DEFAULT_CHECKPOINT = Path(
    "artifacts/phase7_act_coarse/keyboard_v1/"
    "act_train_20k/checkpoints/last"
)

TARGET_VOCAB = (
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + ["SPACE", "BACKSPACE"]
)


def _action_key(name: str) -> str:
    return (
        name
        if name.endswith(".pos")
        else f"{name}.pos"
    )


def _present_state(
    observation: dict,
    motor_names: list[str],
) -> np.ndarray:
    values = []

    for name in motor_names:
        key = _action_key(name)

        if key not in observation:
            raise RuntimeError(
                f"Robot observation missing {key}"
            )

        values.append(
            float(observation[key])
        )

    return np.asarray(
        values,
        dtype=np.float32,
    )


def _target_one_hot(
    target: str,
) -> np.ndarray:
    target = target.strip().upper()

    if target not in TARGET_VOCAB:
        raise ValueError(
            f"Unsupported target: {target}"
        )

    vector = np.zeros(
        len(TARGET_VOCAB),
        dtype=np.float32,
    )

    vector[
        TARGET_VOCAB.index(target)
    ] = 1.0

    return vector


def _wait_initial_frames(
    top: ThreadedOpenCVCamera,
    wrist: ThreadedOpenCVCamera,
    timeout_s: float = 5.0,
) -> None:
    deadline = (
        time.monotonic() + timeout_s
    )

    while (
        time.monotonic()
        < deadline
    ):
        if (
            top.latest() is not None
            and wrist.latest() is not None
        ):
            return

        time.sleep(0.02)

    raise RuntimeError(
        "TOP/WRIST did not both "
        "produce an initial frame"
    )


def _image_tensor(
    bgr: np.ndarray,
) -> torch.Tensor:
    if bgr.shape != (
        480,
        640,
        3,
    ):
        raise RuntimeError(
            "Unexpected image shape: "
            f"{bgr.shape}"
        )

    rgb = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB,
    )

    return (
        torch.from_numpy(rgb)
        .permute(2, 0, 1)
        .contiguous()
        .to(torch.float32)
        / 255.0
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 7D live ACT dry-run. "
            "Inference only. "
            "NO robot action is sent."
        )
    )

    parser.add_argument(
        "--target",
        default="G",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--robot-port",
        default=ROBOT_PORT,
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=0.20,
    )

    args = parser.parse_args()

    target = (
        str(args.target)
        .strip()
        .upper()
    )

    if target not in TARGET_VOCAB[:26]:
        raise ValueError(
            "Dry-run currently supports A-Z"
        )

    if args.samples < 1:
        raise ValueError(
            "--samples must be >= 1"
        )

    if not args.checkpoint.is_dir():
        raise RuntimeError(
            "Checkpoint not found: "
            f"{args.checkpoint}"
        )

    print("=" * 78)
    print(
        "PHASE 7D-2 — ACT LIVE DRY-RUN"
    )
    print("=" * 78)
    print(
        "target             :",
        target,
    )
    print(
        "checkpoint         :",
        args.checkpoint,
    )
    print(
        "samples            :",
        args.samples,
    )
    print()
    print(
        "AUTONOMOUS MOTION  : DISABLED"
    )
    print(
        "robot.send_action  : NEVER CALLED"
    )
    print(
        "SIDE / OCR / Z     : DISABLED"
    )
    print("=" * 78)

    policy = ACTPolicy.from_pretrained(
        args.checkpoint,
        local_files_only=True,
        strict=True,
    )

    policy.eval()

    preprocessor, postprocessor = (
        make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=str(
                args.checkpoint
            ),
        )
    )

    if policy.config.chunk_size != 20:
        raise RuntimeError(
            "Expected chunk_size=20"
        )

    top_spec = CameraSpec.from_yaml(
        TOP_CAMERA_CONFIG
    )

    wrist_spec = CameraSpec.from_yaml(
        WRIST_CAMERA_CONFIG
    )

    top = ThreadedOpenCVCamera(
        top_spec
    )

    wrist = ThreadedOpenCVCamera(
        wrist_spec
    )

    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.robot_port,
            id=ROBOT_ID,
            use_degrees=True,
            max_relative_target=10.0,
            cameras={},
        )
    )

    predictions = []
    deltas = []

    try:
        top.start()
        wrist.start()

        _wait_initial_frames(
            top,
            wrist,
        )

        robot.connect()

        if not robot.is_connected:
            raise RuntimeError(
                "Follower connection failed"
            )

        motor_names = list(
            robot.bus.motors.keys()
        )

        if len(motor_names) != 6:
            raise RuntimeError(
                "Expected 6 motors, got "
                f"{motor_names}"
            )

        print()
        print(
            "motor order        :",
            motor_names,
        )
        print(
            "TOP measured fps   :",
            f"{top.measured_fps:.1f}",
        )
        print(
            "WRIST measured fps :",
            f"{wrist.measured_fps:.1f}",
        )
        print("-" * 78)

        target_vector = (
            _target_one_hot(target)
        )

        for index in range(
            1,
            args.samples + 1,
        ):
            top_frame = top.latest(
                copy_image=True
            )

            wrist_frame = wrist.latest(
                copy_image=True
            )

            if (
                top_frame is None
                or wrist_frame is None
            ):
                raise RuntimeError(
                    "Camera frame missing"
                )

            robot_obs = (
                robot.get_observation()
            )

            present = _present_state(
                robot_obs,
                motor_names,
            )

            state = np.concatenate(
                [
                    present,
                    target_vector,
                ]
            ).astype(
                np.float32
            )

            if state.shape != (34,):
                raise RuntimeError(
                    "Expected state shape "
                    "(34,), got "
                    f"{state.shape}"
                )

            observation = {
                "observation.images.top":
                    _image_tensor(
                        top_frame.image
                    ),
                "observation.images.wrist":
                    _image_tensor(
                        wrist_frame.image
                    ),
                "observation.state":
                    torch.from_numpy(
                        state
                    ),
            }

            policy.reset()

            processed = preprocessor(
                observation
            )

            started = (
                time.perf_counter()
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

            inference_ms = (
                time.perf_counter()
                - started
            ) * 1000.0

            if (
                not isinstance(
                    predicted,
                    torch.Tensor,
                )
                or tuple(
                    predicted.shape
                ) != (1, 6)
            ):
                raise RuntimeError(
                    "Unexpected ACT output: "
                    f"{type(predicted)}, "
                    f"{getattr(predicted, 'shape', None)}"
                )

            action = (
                predicted[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            if not np.isfinite(
                action
            ).all():
                raise RuntimeError(
                    "ACT produced non-finite action"
                )

            delta = action - present

            predictions.append(
                action.copy()
            )

            deltas.append(
                delta.copy()
            )

            print()
            print(
                f"[{index:02d}/{args.samples:02d}] "
                f"inference={inference_ms:6.1f} ms"
            )

            for joint_index, name in enumerate(
                motor_names
            ):
                print(
                    f"  {name:16s} "
                    f"present={present[joint_index]:8.3f}  "
                    f"pred={action[joint_index]:8.3f}  "
                    f"delta={delta[joint_index]:+8.3f}"
                )

            # IMPORTANT:
            # This script intentionally contains
            # NO robot.send_action() call.

            time.sleep(
                max(
                    0.0,
                    args.interval,
                )
            )

        prediction_array = np.stack(
            predictions
        )

        delta_array = np.stack(
            deltas
        )

        print()
        print("=" * 78)
        print(
            "LIVE DRY-RUN SUMMARY"
        )
        print("=" * 78)

        print(
            "samples            :",
            len(predictions),
        )

        print(
            "prediction std deg :",
            np.round(
                prediction_array.std(
                    axis=0
                ),
                3,
            ).tolist(),
        )

        print(
            "mean |delta| deg   :",
            np.round(
                np.mean(
                    np.abs(
                        delta_array
                    ),
                    axis=0,
                ),
                3,
            ).tolist(),
        )

        print(
            "max |delta| deg    :",
            f"{np.max(np.abs(delta_array)):.3f}",
        )

        print()
        print(
            "ROBOT COMMANDS SENT: 0"
        )
        print(
            "STATUS: PASS"
        )

    finally:
        if robot.is_connected:
            robot.disconnect()

        top.stop()
        wrist.stop()


if __name__ == "__main__":
    main()
