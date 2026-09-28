from __future__ import annotations

import argparse
import json
import queue
import signal
import time
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread

import cv2
import numpy as np
import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTConfig, ACTPolicy
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.robot_utils import precise_sleep

import validate_phase5_single_key as phase5
from phase6_replay_takeover_validate import (
    GLYPH_MODEL,
    MAX_RELATIVE_TARGET_DEG,
    RECOVERY_HOME_CONFIG,
    TOOL_REFERENCE,
    ROBOT_ID,
    ROBOT_PORT,
    _auto_restore_recorded_start,
    _emergency_hold_until_operator,
    _load_recovery_pose,
    _run_takeover,
    _synchronize_goal_to_present,
    _verify_home,
)
from so101_typing.adapters.cameras import CameraSpec, ThreadedOpenCVCamera
from so101_typing.control.tool_reference import ToolReferenceCalibration
from so101_typing.perception.glyph_runtime import RuntimeHOGGlyphRecognizer
from so101_typing.perception.wrist_target import observe_target

TOP_CAMERA_CONFIG = Path("configs/cameras/top.yaml")
WRIST_CAMERA_CONFIG = Path("configs/cameras/wrist.yaml")
DEFAULT_CHECKPOINT = Path(
    "artifacts/phase7_act_coarse/keyboard_v1/act_train_20k/checkpoints/last"
)
ARTIFACT_ROOT = Path("artifacts/phase7_act_coarse/keyboard_v1/rollouts")

TARGET_VOCAB = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ("SPACE", "BACKSPACE")
ROLLOUT_HZ = 15.0
CHUNK_SIZE = 20
DEFAULT_DURATION_S = 30.0
DEFAULT_N_ACTION_STEPS = CHUNK_SIZE
STARTUP_SENT_DIFF_DEG = 0.25
HOME_STEP_DEG = 2.0
CAMERA_WARMUP_S = 2.0


class OperatorStop(RuntimeError):
    pass


class HomeStop(RuntimeError):
    pass


def _action_key(name: str) -> str:
    return name if name.endswith(".pos") else f"{name}.pos"


def _numeric(action: dict) -> dict[str, float]:
    return {
        str(k): float(v)
        for k, v in action.items()
        if isinstance(v, (int, float, np.integer, np.floating))
    }


def _max_action_delta(a: dict, b: dict, keys: list[str]) -> float:
    values = [
        abs(float(a[key]) - float(b[key]))
        for key in keys
        if key in a and key in b
    ]
    return max(values) if values else float("inf")


def _present_state(observation: dict, motor_names: list[str]) -> np.ndarray:
    values = []
    for name in motor_names:
        key = _action_key(name)
        if key not in observation:
            raise RuntimeError(f"Robot observation missing {key}")
        values.append(float(observation[key]))
    return np.asarray(values, dtype=np.float32)


def _target_one_hot(target: str) -> np.ndarray:
    target = str(target).strip().upper()
    if target not in TARGET_VOCAB:
        raise ValueError(f"Unsupported target {target!r}")
    vector = np.zeros(len(TARGET_VOCAB), dtype=np.float32)
    vector[TARGET_VOCAB.index(target)] = 1.0
    return vector


def _image_tensor(bgr: np.ndarray) -> torch.Tensor:
    if bgr.shape != (480, 640, 3):
        raise RuntimeError(f"Unexpected camera image shape: {bgr.shape}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return (
        torch.from_numpy(rgb)
        .permute(2, 0, 1)
        .contiguous()
        .to(torch.float32)
        / 255.0
    )


def _wait_initial_frames(top, wrist, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if top.latest() is not None and wrist.latest() is not None:
            return
        time.sleep(0.02)
    raise RuntimeError("TOP/WRIST did not both produce an initial frame")


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _percentile(values: list[float], q: float):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


class AsyncFrameWriter:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.queue = queue.Queue(maxsize=128)
        self.thread = Thread(target=self._run, daemon=False)
        self.saved_pairs = 0
        self.dropped_pairs = 0
        self.error = None

    def start(self) -> None:
        self.thread.start()

    def submit(self, tick: int, top_bgr: np.ndarray, wrist_bgr: np.ndarray) -> bool:
        try:
            self.queue.put_nowait((int(tick), top_bgr, wrist_bgr))
            return True
        except queue.Full:
            self.dropped_pairs += 1
            return False

    def _run(self) -> None:
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    return
                tick, top_bgr, wrist_bgr = item
                top_path = self.root / f"tick_{tick:04d}_top.jpg"
                wrist_path = self.root / f"tick_{tick:04d}_wrist.jpg"
                if not cv2.imwrite(str(top_path), top_bgr):
                    raise RuntimeError(f"Failed to save {top_path}")
                if not cv2.imwrite(str(wrist_path), wrist_bgr):
                    raise RuntimeError(f"Failed to save {wrist_path}")
                self.saved_pairs += 1
        except BaseException as exc:
            self.error = exc

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join()
        if self.error is not None:
            raise RuntimeError("Frame writer failed") from self.error


def _capture_observation(*, robot, top, wrist, motor_names, target_vector) -> dict:
    top_frame = top.latest(copy_image=True)
    wrist_frame = wrist.latest(copy_image=True)
    if top_frame is None or wrist_frame is None:
        raise RuntimeError("TOP/WRIST frame missing")

    read_started = time.monotonic()
    robot_observation = robot.get_observation()
    read_finished = time.monotonic()
    present = _present_state(robot_observation, motor_names)

    state = np.concatenate([present, target_vector]).astype(np.float32)
    if state.shape != (34,):
        raise RuntimeError(f"Expected state shape (34,), got {state.shape}")

    observation = {
        "observation.images.top": _image_tensor(top_frame.image),
        "observation.images.wrist": _image_tensor(wrist_frame.image),
        "observation.state": torch.from_numpy(state),
    }
    return {
        "top": top_frame,
        "wrist": wrist_frame,
        "present": present,
        "observation": observation,
        "state_read_ms": (read_finished - read_started) * 1000.0,
    }


def _processor_reset(processor) -> None:
    reset = getattr(processor, "reset", None)
    if callable(reset):
        reset()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 7 ACT rollout evaluation: official ACT select_action queue "
            "semantics + deterministic WRIST takeover + canonical HOME."
        )
    )
    parser.add_argument("--target", default="G")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=DEFAULT_N_ACTION_STEPS,
        help="ACT execution horizon before policy replans; Phase 7 runtime value is not frozen yet.",
    )
    parser.add_argument("--robot-port", default=ROBOT_PORT)
    parser.add_argument("--recovery-home", type=Path, default=RECOVERY_HOME_CONFIG)
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()

    target = str(args.target).strip().upper()
    if target not in TARGET_VOCAB[:26]:
        raise ValueError("Formal V1 rollout currently supports A-Z targets")
    if not np.isfinite(args.duration) or args.duration <= 0.0:
        raise ValueError("--duration must be finite and > 0")
    if not 1 <= args.n_action_steps <= CHUNK_SIZE:
        raise ValueError(f"--n-action-steps must be in [1, {CHUNK_SIZE}]")

    print("=" * 78)
    print("PHASE 7 — ACT ROLLOUT EVALUATION v4")
    print("=" * 78)
    print("target               :", target)
    print("checkpoint           :", args.checkpoint)
    print("control rate         :", f"{ROLLOUT_HZ:.1f} Hz")
    print("duration limit       :", f"{args.duration:.1f} s")
    print("chunk_size           :", CHUNK_SIZE)
    print("n_action_steps       :", args.n_action_steps)
    print("execution horizon    :", f"{args.n_action_steps / ROLLOUT_HZ:.3f} s")
    print("ACT queue            : OFFICIAL policy.select_action()")
    print("joint safety limit   :", f"{MAX_RELATIVE_TARGET_DEG:.1f} deg/send")
    print("servo-ready gate     :", f"WRIST error <= {phase5.INITIAL_CAPTURE_THRESHOLD_PX:.1f} px")
    print("success              : WRIST takeover PASS -> AUTO HOME")
    print("timeout / Ctrl+C     : AUTO HOME")
    print("Z / press / SIDE/OCR : DISABLED")
    print("HOME                 :", args.recovery_home)
    print("=" * 78)

    if args.print_config:
        return

    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    if not GLYPH_MODEL.exists():
        raise FileNotFoundError(GLYPH_MODEL)

    saved_cfg = ACTConfig.from_pretrained(
        args.checkpoint,
        local_files_only=True,
    )
    if int(saved_cfg.chunk_size) != CHUNK_SIZE:
        raise RuntimeError(
            f"Checkpoint chunk_size={saved_cfg.chunk_size}, expected {CHUNK_SIZE}"
        )
    if saved_cfg.temporal_ensemble_coeff is not None:
        raise RuntimeError(
            "This rollout contract expects ACT action-queue semantics, "
            "but checkpoint temporal ensembling is enabled."
        )

    runtime_cfg = replace(
        saved_cfg,
        n_action_steps=int(args.n_action_steps),
    )

    policy = ACTPolicy.from_pretrained(
        args.checkpoint,
        config=runtime_cfg,
        local_files_only=True,
        strict=True,
    )
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=runtime_cfg,
        pretrained_path=str(args.checkpoint),
    )

    home_baseline = _load_recovery_pose(args.recovery_home)
    recognizer = RuntimeHOGGlyphRecognizer.load(GLYPH_MODEL)
    tool_reference = ToolReferenceCalibration.load(TOOL_REFERENCE)
    if tool_reference is None:
        raise RuntimeError("tool reference is missing")
    tool_reference.require_direct_tool_tip()
    target_vector = _target_one_hot(target)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = ARTIFACT_ROOT / f"run_{timestamp}_{target.lower()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    steps_path = run_dir / "steps.jsonl"
    chunks_path = run_dir / "chunks.jsonl"
    events_path = run_dir / "events.jsonl"
    summary_path = run_dir / "summary.json"

    writer = AsyncFrameWriter(run_dir / "frames")
    writer_started = False

    top = ThreadedOpenCVCamera(CameraSpec.from_yaml(TOP_CAMERA_CONFIG))
    wrist = ThreadedOpenCVCamera(CameraSpec.from_yaml(WRIST_CAMERA_CONFIG))
    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.robot_port,
            id=ROBOT_ID,
            use_degrees=True,
            max_relative_target=MAX_RELATIVE_TARGET_DEG,
            cameras={},
        )
    )

    autonomy_stop = Event()
    home_stop = Event()
    phase = {"name": "STARTUP"}
    previous_sigint = signal.getsignal(signal.SIGINT)
    signal_installed = False

    original_raise_if_screen_event = phase5.raise_if_screen_event
    original_wait_motion_stable = phase5.wait_motion_stable

    def on_sigint(signum, frame):
        del signum, frame
        current = phase["name"]
        if current == "HOME":
            if not home_stop.is_set():
                home_stop.set()
                print("\n[CTRL+C] HOME emergency stop requested; current bus operation will finish first.")
            return
        if current in {"ACT", "TAKEOVER", "STARTUP"}:
            if not autonomy_stop.is_set():
                autonomy_stop.set()
                print("\n[CTRL+C] autonomy stop requested; current bus operation will finish first.")
            return

    def guarded_raise_if_screen_event():
        original_raise_if_screen_event()
        if phase["name"] == "TAKEOVER" and autonomy_stop.is_set():
            raise OperatorStop("operator stop during WRIST takeover")

    def guarded_wait_motion_stable(robot_obj, motor_names_obj):
        settled = original_wait_motion_stable(robot_obj, motor_names_obj)
        if phase["name"] == "TAKEOVER" and autonomy_stop.is_set():
            raise OperatorStop("operator stop during WRIST takeover")
        if phase["name"] == "HOME" and home_stop.is_set():
            raise HomeStop("operator emergency stop during HOME recovery")
        return settled

    phase5.raise_if_screen_event = guarded_raise_if_screen_event
    phase5.wait_motion_stable = guarded_wait_motion_stable

    motor_names: list[str] = []
    keys: list[str] = []
    task_status = "STARTING"
    task_error = None
    first_target_seen = None
    target_trigger = None
    takeover_result = None
    home_result = {"status": "NOT_ATTEMPTED"}

    rollout_start = None
    rollout_stop = None
    command_count = 0
    chunk_count = 0
    action_index_in_chunk = 0
    chunk_source_timestamp = None
    chunk_source_top_frame_id = None
    chunk_source_wrist_frame_id = None

    present_initial = None
    present_previous = None
    present_history: list[np.ndarray] = []
    sent_delta_history: list[np.ndarray] = []
    pipeline_ms_values: list[float] = []
    replan_ms_values: list[float] = []
    obs_age_ms_values: list[float] = []
    send_latency_ms_values: list[float] = []
    clip_values: list[float] = []
    send_timestamps: list[float] = []

    try:
        top.start()
        wrist.start()
        _wait_initial_frames(top, wrist)

        robot.connect()
        motor_names = list(robot.bus.motors.keys())
        if len(motor_names) != 6:
            raise RuntimeError(f"Expected 6 motors, got {motor_names}")
        keys = [_action_key(name) for name in motor_names]

        missing_home = [key for key in keys if key not in home_baseline]
        if missing_home:
            raise RuntimeError(f"Recovery HOME missing keys: {missing_home}")

        expected_order = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]
        if keys != expected_order:
            raise RuntimeError(
                "Follower motor order no longer matches the Phase-6 dataset action order: "
                f"{keys} != {expected_order}"
            )

        signal.signal(signal.SIGINT, on_sigint)
        signal_installed = True

        print("motor order         :", motor_names)
        print("runtime policy n    :", runtime_cfg.n_action_steps)

        _synchronize_goal_to_present(
            robot=robot,
            motor_names=motor_names,
            max_sent_diff_deg=STARTUP_SENT_DIFF_DEG,
        )

        if autonomy_stop.is_set():
            raise OperatorStop("operator stop during startup")

        time.sleep(CAMERA_WARMUP_S)

        warm = _capture_observation(
            robot=robot,
            top=top,
            wrist=wrist,
            motor_names=motor_names,
            target_vector=target_vector,
        )
        warm_processed = preprocessor(warm["observation"])
        with torch.inference_mode():
            warm_action = policy.select_action(warm_processed)
            _ = postprocessor(warm_action)

        # Important: warmup must not leak a precomputed ACT queue into rollout.
        policy.reset()
        _processor_reset(preprocessor)
        _processor_reset(postprocessor)

        writer.start()
        writer_started = True

        phase["name"] = "ACT"
        rollout_start = time.perf_counter()
        next_tick = rollout_start
        period_s = 1.0 / ROLLOUT_HZ
        task_status = "RUNNING"

        print()
        print("=" * 78)
        print("ACT APPROACH START")
        print("=" * 78)

        while True:
            precise_sleep(max(0.0, next_tick - time.perf_counter()))
            cycle_start = time.perf_counter()
            elapsed = cycle_start - rollout_start

            if autonomy_stop.is_set():
                task_status = "OPERATOR_ABORT"
                rollout_stop = cycle_start
                policy.reset()
                break

            if elapsed >= args.duration:
                task_status = "TIMEOUT_NO_SERVO_READY"
                rollout_stop = cycle_start
                policy.reset()
                print(f"\n[TIMEOUT] ACT approach reached {args.duration:.1f}s")
                break

            live = _capture_observation(
                robot=robot,
                top=top,
                wrist=wrist,
                motor_names=motor_names,
                target_vector=target_vector,
            )
            top_frame = live["top"]
            wrist_frame = live["wrist"]
            present = live["present"]

            if present_initial is None:
                present_initial = present.copy()
            move_delta = (
                0.0
                if present_previous is None
                else float(np.max(np.abs(present - present_previous)))
            )
            net_delta = float(np.max(np.abs(present - present_initial)))
            present_previous = present.copy()
            present_history.append(present.copy())

            tick = command_count + 1
            frame_queued = writer.submit(
                tick,
                top_frame.image,
                wrist_frame.image,
            )

            perception_started = time.perf_counter()
            target_observation = observe_target(
                wrist_frame.image,
                target,
                recognizer,
            )
            perception_ms = (time.perf_counter() - perception_started) * 1000.0

            target_error_px = None
            target_error_norm_px = None

            if target_observation.found:
                target_center = np.asarray(
                    target_observation.center_px,
                    dtype=np.float64,
                )
                tool_center = np.asarray(
                    tool_reference.center,
                    dtype=np.float64,
                )
                target_error_px = target_center - tool_center
                target_error_norm_px = float(
                    np.linalg.norm(target_error_px)
                )

                if first_target_seen is None:
                    first_target_seen = {
                        "tick": tick,
                        "commands_sent": command_count,
                        "act_elapsed_s": float(
                            time.perf_counter() - rollout_start
                        ),
                        "top_frame_id": int(top_frame.frame_id),
                        "wrist_frame_id": int(wrist_frame.frame_id),
                        "center_px": target_center.tolist(),
                        "error_px": target_error_px.tolist(),
                        "error_norm_px": target_error_norm_px,
                        "observation": target_observation.to_dict(),
                    }
                    _append_jsonl(
                        events_path,
                        {"event": "TARGET_ACQUIRED", **first_target_seen},
                    )
                    print(
                        f"\n[TARGET ACQUIRED] "
                        f"t={first_target_seen['act_elapsed_s']:.3f}s "
                        f"commands={command_count} "
                        f"error={target_error_norm_px:.2f}px; "
                        "ACT CONTINUES until servo-ready."
                    )

                if (
                    target_error_norm_px
                    <= phase5.INITIAL_CAPTURE_THRESHOLD_PX
                ):
                    rollout_stop = time.perf_counter()
                    policy.reset()
                    target_trigger = {
                        "tick": tick,
                        "commands_sent": command_count,
                        "act_elapsed_s": float(rollout_stop - rollout_start),
                        "top_frame_id": int(top_frame.frame_id),
                        "wrist_frame_id": int(wrist_frame.frame_id),
                        "center_px": target_center.tolist(),
                        "error_px": target_error_px.tolist(),
                        "error_norm_px": target_error_norm_px,
                        "threshold_px": float(
                            phase5.INITIAL_CAPTURE_THRESHOLD_PX
                        ),
                        "observation": target_observation.to_dict(),
                    }
                    _append_jsonl(
                        events_path,
                        {"event": "SERVO_READY", **target_trigger},
                    )
                    task_status = "SERVO_READY"
                    print(
                        f"\n[SERVO READY] "
                        f"t={target_trigger['act_elapsed_s']:.3f}s "
                        f"commands={command_count} "
                        f"error={target_error_norm_px:.2f}px "
                        f"<= {phase5.INITIAL_CAPTURE_THRESHOLD_PX:.1f}px"
                    )
                    print(
                        "[ACT STOP] official ACT queue reset; "
                        "deterministic WRIST takeover begins."
                    )
                    break

            replanned = action_index_in_chunk == 0
            if replanned:
                chunk_count += 1
                chunk_source_timestamp = min(
                    float(top_frame.capture_timestamp),
                    float(wrist_frame.capture_timestamp),
                )
                chunk_source_top_frame_id = int(top_frame.frame_id)
                chunk_source_wrist_frame_id = int(wrist_frame.frame_id)
                _append_jsonl(
                    chunks_path,
                    {
                        "chunk_id": chunk_count,
                        "command_index_start": command_count + 1,
                        "act_elapsed_s": float(time.perf_counter() - rollout_start),
                        "present_state": present.tolist(),
                        "top_frame_id": chunk_source_top_frame_id,
                        "wrist_frame_id": chunk_source_wrist_frame_id,
                    },
                )

            pipeline_started = time.perf_counter()
            processed = preprocessor(live["observation"])
            with torch.inference_mode():
                action_tensor = policy.select_action(processed)
                action_tensor = postprocessor(action_tensor)
            action = (
                action_tensor[0]
                .detach()
                .cpu()
                .to(torch.float32)
                .numpy()
            )
            pipeline_ms = (time.perf_counter() - pipeline_started) * 1000.0
            pipeline_ms_values.append(pipeline_ms)
            if replanned:
                replan_ms_values.append(pipeline_ms)

            if action.shape != (6,) or not np.isfinite(action).all():
                raise RuntimeError(f"Invalid ACT action shape/value: {action}")

            if autonomy_stop.is_set():
                task_status = "OPERATOR_ABORT"
                rollout_stop = time.perf_counter()
                policy.reset()
                break

            requested = {
                key: float(value)
                for key, value in zip(keys, action)
            }
            present_action = {
                key: float(value)
                for key, value in zip(keys, present)
            }

            send_started = time.monotonic()
            actual = _numeric(robot.send_action(requested))
            send_finished = time.monotonic()

            command_count += 1
            send_timestamps.append(send_finished)
            send_latency_ms = (send_finished - send_started) * 1000.0
            send_latency_ms_values.append(send_latency_ms)

            clip_diff = _max_action_delta(requested, actual, keys)
            clip_values.append(clip_diff)
            actual_vector = np.asarray(
                [actual[key] for key in keys],
                dtype=np.float32,
            )
            sent_delta = actual_vector - present
            sent_delta_history.append(sent_delta.copy())

            pred_delta_max = float(np.max(np.abs(action - present)))
            sent_delta_max = float(np.max(np.abs(sent_delta)))
            obs_age_ms = (
                send_started - float(chunk_source_timestamp)
            ) * 1000.0
            obs_age_ms_values.append(obs_age_ms)

            _append_jsonl(
                steps_path,
                {
                    "command_index": command_count,
                    "act_elapsed_s": float(time.perf_counter() - rollout_start),
                    "chunk_id": chunk_count,
                    "chunk_action_index": action_index_in_chunk,
                    "replanned": replanned,
                    "present_state": present.tolist(),
                    "predicted_action": action.tolist(),
                    "actual_sent_action": actual,
                    "sent_delta": sent_delta.tolist(),
                    "physical_step_max_deg": move_delta,
                    "physical_net_max_deg": net_delta,
                    "predicted_delta_max_deg": pred_delta_max,
                    "sent_delta_max_deg": sent_delta_max,
                    "requested_to_sent_max_diff_deg": clip_diff,
                    "policy_pipeline_ms": pipeline_ms,
                    "observation_to_action_age_ms": obs_age_ms,
                    "send_latency_ms": send_latency_ms,
                    "state_read_ms": live["state_read_ms"],
                    "perception_ms": perception_ms,
                    "chunk_source_top_frame_id": chunk_source_top_frame_id,
                    "chunk_source_wrist_frame_id": chunk_source_wrist_frame_id,
                    "current_top_frame_id": int(top_frame.frame_id),
                    "current_wrist_frame_id": int(wrist_frame.frame_id),
                    "target_observation": target_observation.to_dict(),
                    "target_error_px": (
                        None
                        if target_error_px is None
                        else target_error_px.tolist()
                    ),
                    "target_error_norm_px": target_error_norm_px,
                    "servo_ready": bool(
                        target_error_norm_px is not None
                        and target_error_norm_px
                        <= phase5.INITIAL_CAPTURE_THRESHOLD_PX
                    ),
                    "frame_save_queued": frame_queued,
                },
            )

            marker = "R" if replanned else "-"
            print(
                f"[{command_count:04d}] {marker} "
                f"t={time.perf_counter() - rollout_start:6.2f}s "
                f"chunk={chunk_count:03d}:{action_index_in_chunk:02d} "
                f"pipe={pipeline_ms:5.1f}ms "
                f"age={obs_age_ms:6.1f}ms "
                f"sentΔ={sent_delta_max:5.2f}° "
                f"moveΔ={move_delta:5.2f}° "
                f"netΔ={net_delta:6.2f}° "
                f"clip={clip_diff:4.2f}°"
            )

            action_index_in_chunk += 1
            if action_index_in_chunk >= args.n_action_steps:
                action_index_in_chunk = 0

            if autonomy_stop.is_set():
                task_status = "OPERATOR_ABORT"
                rollout_stop = time.perf_counter()
                policy.reset()
                break

            next_tick += period_s

        if task_status == "SERVO_READY":
            phase["name"] = "TAKEOVER"
            settled_timestamp = phase5.wait_motion_stable(robot, motor_names)
            if autonomy_stop.is_set():
                raise OperatorStop("operator stop before WRIST takeover")

            print()
            print("=" * 78)
            print("DETERMINISTIC WRIST TAKEOVER")
            print("=" * 78)

            try:
                takeover_result = _run_takeover(
                    robot=robot,
                    wrist_camera=wrist,
                    target=target,
                    motor_names=motor_names,
                    endpoint_settled_timestamp=settled_timestamp,
                    episode_dir=run_dir,
                )
            except OperatorStop:
                raise
            except Exception as exc:
                task_status = "FAIL_TAKEOVER"
                task_error = f"{type(exc).__name__}: {exc}"
                _append_jsonl(
                    events_path,
                    {"event": "TAKEOVER_FAIL", "error": task_error},
                )
                print("\n[TAKEOVER FAIL]", task_error)
            else:
                if autonomy_stop.is_set():
                    raise OperatorStop("operator stop during WRIST takeover")
                task_status = "PASS"
                _append_jsonl(
                    events_path,
                    {"event": "TAKEOVER_PASS", **takeover_result},
                )

    except OperatorStop as exc:
        task_status = "OPERATOR_ABORT"
        task_error = str(exc)
        if rollout_start is not None and rollout_stop is None:
            rollout_stop = time.perf_counter()

    except KeyboardInterrupt:
        # Only possible before the non-throwing SIGINT handler is installed.
        task_status = "OPERATOR_ABORT"
        task_error = "KeyboardInterrupt during early startup"
        if rollout_start is not None and rollout_stop is None:
            rollout_stop = time.perf_counter()

    except Exception as exc:
        task_status = "FAIL"
        task_error = f"{type(exc).__name__}: {exc}"
        if rollout_start is not None and rollout_stop is None:
            rollout_stop = time.perf_counter()
        print("\n[ROLLOUT FAILURE]", task_error)

    finally:
        if writer_started:
            try:
                writer.close()
            except Exception as exc:
                print("[FRAME WRITER ERROR]", exc)

        if robot.is_connected and motor_names:
            phase["name"] = "HOME"
            print()
            print("=" * 78)
            print("FINAL AUTOMATIC HOME RECOVERY")
            print("=" * 78)
            print("task status :", task_status)

            try:
                home_hold = _synchronize_goal_to_present(
                    robot=robot,
                    motor_names=motor_names,
                    max_sent_diff_deg=STARTUP_SENT_DIFF_DEG,
                )
                if home_stop.is_set():
                    raise HomeStop("operator emergency stop during HOME recovery")

                home_restore = _auto_restore_recorded_start(
                    robot=robot,
                    baseline=home_baseline,
                    motor_names=motor_names,
                    step_deg=HOME_STEP_DEG,
                    max_sent_diff_deg=STARTUP_SENT_DIFF_DEG,
                    label="RECOVERY HOME",
                )
                home_diff = _verify_home(
                    robot=robot,
                    motor_names=motor_names,
                    home_baseline=home_baseline,
                    max_diff_deg=STARTUP_SENT_DIFF_DEG,
                )
                if home_stop.is_set():
                    raise HomeStop("operator emergency stop during HOME verification")
                home_result = {
                    "status": "PASS",
                    "present_hold": home_hold,
                    "restore": home_restore,
                    "final_goal_diff_deg": float(home_diff),
                }
                print(f"[HOME RESTORED] final Goal diff={home_diff:.3f}deg")

            except BaseException as exc:
                home_result = {
                    "status": "FAIL",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print("\n[HOME RECOVERY FAILURE]", home_result["error"])

                # Emergency hold performs no bus I/O; restore ordinary Ctrl+C
                # so the operator can explicitly authorize disconnect.
                phase5.raise_if_screen_event = original_raise_if_screen_event
                phase5.wait_motion_stable = original_wait_motion_stable
                if signal_installed:
                    signal.signal(signal.SIGINT, previous_sigint)
                    signal_installed = False
                _emergency_hold_until_operator(robot, home_result["error"])

        top.stop()
        wrist.stop()

        phase5.raise_if_screen_event = original_raise_if_screen_event
        phase5.wait_motion_stable = original_wait_motion_stable
        if signal_installed:
            signal.signal(signal.SIGINT, previous_sigint)

        if robot.is_connected:
            robot.disconnect()

        if rollout_start is not None and rollout_stop is None:
            rollout_stop = time.perf_counter()
        act_duration_s = (
            None
            if rollout_start is None
            else float(rollout_stop - rollout_start)
        )

        if len(send_timestamps) >= 2:
            achieved_send_hz = float(
                (len(send_timestamps) - 1)
                / (send_timestamps[-1] - send_timestamps[0])
            )
        else:
            achieved_send_hz = 0.0

        if present_history:
            present_array = np.asarray(present_history, dtype=np.float64)
            physical_net = present_array[-1] - present_array[0]
            physical_range = np.ptp(present_array, axis=0)
        else:
            physical_net = np.zeros(6, dtype=np.float64)
            physical_range = np.zeros(6, dtype=np.float64)

        summary = {
            "schema": "phase7.act_rollout_eval.v5",
            "target": target,
            "checkpoint": str(args.checkpoint),
            "task_status": task_status,
            "task_error": task_error,
            "control_hz": ROLLOUT_HZ,
            "duration_limit_s": float(args.duration),
            "act_duration_s": act_duration_s,
            "chunk_size": CHUNK_SIZE,
            "n_action_steps": int(args.n_action_steps),
            "commands_sent": int(command_count),
            "chunks_generated": int(chunk_count),
            "achieved_send_hz": achieved_send_hz,
            "first_target_seen": first_target_seen,
            "servo_ready_trigger": target_trigger,
            "takeover": takeover_result,
            "physical_motion": {
                "joint_names": motor_names,
                "net_deg": physical_net.tolist(),
                "range_deg": physical_range.tolist(),
            },
            "timing": {
                "policy_pipeline_ms_p50": _percentile(pipeline_ms_values, 50),
                "policy_pipeline_ms_p95": _percentile(pipeline_ms_values, 95),
                "replan_pipeline_ms_p50": _percentile(replan_ms_values, 50),
                "replan_pipeline_ms_p95": _percentile(replan_ms_values, 95),
                "observation_to_action_age_ms_p50": _percentile(obs_age_ms_values, 50),
                "observation_to_action_age_ms_p95": _percentile(obs_age_ms_values, 95),
                "send_latency_ms_p95": _percentile(send_latency_ms_values, 95),
            },
            "clipping": {
                "count": int(sum(value > 1e-3 for value in clip_values)),
                "max_diff_deg": max(clip_values) if clip_values else None,
            },
            "frame_writer": {
                "saved_pairs": int(writer.saved_pairs),
                "dropped_pairs": int(writer.dropped_pairs),
            },
            "home_recovery": home_result,
            "steps_jsonl": str(steps_path),
            "chunks_jsonl": str(chunks_path),
            "events_jsonl": str(events_path),
            "frames_dir": str(run_dir / "frames"),
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        print()
        print("=" * 78)
        print("PHASE 7 ROLLOUT SUMMARY")
        print("=" * 78)
        print("task status         :", task_status)
        print("ACT duration        :", act_duration_s)
        print("commands sent       :", command_count)
        print("chunks generated    :", chunk_count)
        print("achieved send Hz    :", f"{achieved_send_hz:.2f}")
        print("physical net deg    :", np.round(physical_net, 3).tolist())
        print("physical range deg  :", np.round(physical_range, 3).tolist())
        print("replan ms p50/p95   :", _percentile(replan_ms_values, 50), "/", _percentile(replan_ms_values, 95))
        print("obs age ms p50/p95  :", _percentile(obs_age_ms_values, 50), "/", _percentile(obs_age_ms_values, 95))
        if first_target_seen is not None:
            print(
                "first target seen   :",
                f"{first_target_seen['act_elapsed_s']:.3f}s",
                f"error={first_target_seen['error_norm_px']:.2f}px",
            )
        if target_trigger is not None:
            print(
                "servo ready         :",
                f"{target_trigger['act_elapsed_s']:.3f}s",
                f"error={target_trigger['error_norm_px']:.2f}px",
            )
        print("HOME recovery       :", home_result["status"])
        print("summary             :", summary_path)
        print("=" * 78)


if __name__ == "__main__":
    main()
