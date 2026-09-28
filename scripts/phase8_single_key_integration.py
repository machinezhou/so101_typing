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
    TOOL_REFERENCE,
    MAX_RELATIVE_TARGET_DEG,
    RECOVERY_HOME_CONFIG,
    ROBOT_ID,
    ROBOT_PORT,
    _auto_restore_recorded_start,
    _emergency_hold_until_operator,
    _load_recovery_pose,
    _synchronize_goal_to_present,
    _verify_home,
)
from so101_typing.adapters.cameras import CameraSpec, ThreadedOpenCVCamera
from so101_typing.control.tool_reference import ToolReferenceCalibration
from so101_typing.perception.glyph_runtime import RuntimeHOGGlyphRecognizer
from so101_typing.perception.semantic_target_lock import SemanticTargetLock
from so101_typing.perception.wrist_target import (
    observe_glyph_candidates,
    select_target_observation,
)

TOP_CAMERA_CONFIG = Path("configs/cameras/top.yaml")
WRIST_CAMERA_CONFIG = Path("configs/cameras/wrist.yaml")
DEFAULT_CHECKPOINT = Path(
    "artifacts/phase7_act_coarse/keyboard_v1/act_train_20k/checkpoints/last"
)
ARTIFACT_ROOT = Path("artifacts/phase8_single_key_integration")

TARGET_VOCAB = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ("SPACE", "BACKSPACE")
ROLLOUT_HZ = 15.0
CHUNK_SIZE = 20
DEFAULT_DURATION_S = 30.0
DEFAULT_N_ACTION_STEPS = CHUNK_SIZE
STARTUP_SENT_DIFF_DEG = 0.25
HOME_STEP_DEG = 2.0
CAMERA_WARMUP_S = 2.0
HANDOFF_CANDIDATE_THRESHOLD_PX = float(phase5.INITIAL_CAPTURE_THRESHOLD_PX)
HANDOFF_CANDIDATE_MIN_KEYCAPS = 4


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


def _capture_settled_endpoint(
    *,
    wrist_camera,
    recognizer,
    target: str,
    tool_reference,
    settled_timestamp: float,
) -> dict:
    """Measure ACT's settled endpoint without making an acceptance decision.

    Reuse the Phase-5 initial semantic-target capture contract so this
    diagnostic does not invent a second target-acquisition implementation.
    The 80 px threshold is reported only as a historical Phase-5 capture
    reference; deterministic WRIST convergence remains the ground truth.
    """
    phase5.TARGET = target
    semantic_lock = SemanticTargetLock(target)
    min_timestamp = float(settled_timestamp) + float(phase5.POST_MOTION_GUARD_S)

    target_center, error, error_norm, frame = (
        phase5.capture_initial_semantic_target(
            wrist_camera,
            recognizer,
            semantic_lock,
            tool_reference,
            min_timestamp=min_timestamp,
        )
    )

    return {
        "status": "MEASURED",
        "frame_id": int(frame.frame_id),
        "capture_timestamp": float(frame.capture_timestamp),
        "settled_timestamp": float(settled_timestamp),
        "post_motion_guard_s": float(phase5.POST_MOTION_GUARD_S),
        "min_timestamp": min_timestamp,
        "target_center_px": [float(target_center[0]), float(target_center[1])],
        "tool_tip_px": [float(tool_reference.center[0]), float(tool_reference.center[1])],
        "error_px": [float(error[0]), float(error[1])],
        "error_norm_px": float(error_norm),
        "phase5_capture_threshold_px": HANDOFF_CANDIDATE_THRESHOLD_PX,
        "within_phase5_capture_threshold": bool(
            error_norm <= HANDOFF_CANDIDATE_THRESHOLD_PX
        ),
        "acceptance_role": "diagnostic_only",
    }


def _retract_to_z0(
    *,
    robot,
    planner,
    state,
    motor_names,
    sent_state_tracker,
    events: list[dict],
):
    retract_targets = phase5.stepped_z_targets_to_zero(
        state.z_mm,
        max_step_mm=phase5.RETRACT_Z_STEP_MM,
    )

    print()
    print(
        "[RETRACT PLAN] "
        f"start_z={state.z_mm:+.2f}mm "
        f"step<={phase5.RETRACT_Z_STEP_MM:.2f}mm "
        f"segments={len(retract_targets)}"
    )

    for retract_index, retract_z in enumerate(retract_targets, start=1):
        state = state.with_z_level(retract_z)
        _, plan = phase5.send_command_state(
            robot,
            planner,
            state,
            motor_names,
            label=f"RETRACT {retract_index}/{len(retract_targets)}",
            sent_state_tracker=sent_state_tracker,
        )
        events.append(
            {
                "event": "retract",
                "segment": int(retract_index),
                "segment_count": int(len(retract_targets)),
                "xyz_mm": [float(v) for v in state.xyz_mm],
                "predicted_delta_mm": [float(v) for v in plan.predicted_delta_mm],
            }
        )

    if abs(float(state.z_mm)) > 1e-12:
        raise RuntimeError("bounded retract did not finish at Z=0")

    return state


def _run_full_deterministic_press(
    *,
    robot,
    wrist_camera,
    side_camera,
    target: str,
    motor_names: list[str],
    recognizer,
    tool_reference,
    screen_calibration,
    screen_ocr,
    screen_char_ocr,
    baseline_side_frame_id: int,
    endpoint_settled_timestamp: float,
    endpoint_wrist_frame_id: int,
    run_dir: Path,
    autonomy_stop: Event,
    phase: dict,
) -> dict:
    """Run the accepted Phase-5 deterministic primitive from one Goal anchor.

    This is orchestration only.  Perception, visual servoing, Cartesian planning,
    SIDE event detection, OCR, release, and retract all reuse the accepted
    Phase-5 implementation.
    """
    phase5.TARGET = target
    phase5.ARTIFACT_DIR = run_dir / "deterministic"
    phase5.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    phase5.SEMANTIC_REENTRY_GUARDED = False
    phase5.ACTIVE_SCREEN_WATCHER = None

    jacobian = phase5.ImageJacobianCalibration.load(phase5.IMAGE_JACOBIAN)
    if jacobian is None:
        raise RuntimeError("image Jacobian is missing")

    semantic_lock = phase5.SemanticTargetLock(target)

    present_observation = phase5.ordered_joint_observation(
        robot.get_observation(),
        motor_names,
    )
    goal_positions = robot.bus.sync_read(
        "Goal_Position",
        num_retry=robot.config.num_read_retries,
    )
    anchor = phase5.FixedGoalAnchor.from_goal_positions(
        goal_positions,
        motor_names,
    )

    print(
        "[COMMAND ANCHOR] Goal-Present: "
        + ", ".join(
            (
                f"{name}="
                f"{anchor.position_deg(name) - float(present_observation[f'{name}.pos']):+.3f}deg"
            )
            for name in motor_names
            if name != "gripper"
        )
    )

    kinematics = phase5.build_so101_kinematics(
        phase5.URDF,
        motor_names=motor_names,
    )
    anchor_xyz_mm = phase5.end_effector_xyz_mm(
        kinematics,
        anchor.as_observation(),
        motor_names,
    )
    pipeline = phase5.build_official_cartesian_pipeline(
        phase5.URDF,
        motor_names=motor_names,
        end_effector_bounds={
            "min": [-1.0, -1.0, -1.0],
            "max": [1.0, 1.0, 1.0],
        },
        max_ee_step_m=0.035,
        orientation_weight=0.01,
        raise_on_jump=True,
        use_latched_reference=True,
    )
    planner = phase5.FixedAnchorCartesianPlanner(
        pipeline=pipeline,
        validation_kinematics=kinematics,
        motor_names=motor_names,
        anchor=anchor,
        anchor_xyz_mm=anchor_xyz_mm,
        config=phase5.FixedAnchorPlannerConfig(
            max_command_norm_mm=phase5.MAX_COMMAND_NORM_MM,
            max_model_xy_error_mm=1.0,
            max_model_z_error_mm=1.0,
            max_relative_target_deg=MAX_RELATIVE_TARGET_DEG,
            max_zero_joint_shift_deg=0.5,
        ),
    )
    zero_shift = planner.latch_zero_delta()
    print(f"[ANCHOR] zero-delta max joint shift={zero_shift:.3f}deg")

    state = phase5.FixedAnchorXYZCommandState.at_anchor(
        anchor,
        max_xy_norm_mm=phase5.MAX_XY_CORRECTION_MM,
        max_xyz_norm_mm=phase5.MAX_COMMAND_NORM_MM,
    )
    sent_state_tracker = phase5.LatestSentXYZCommandState(state)
    controller = phase5.SingleKeyPressController(
        phase5.SingleKeyPressConfig(
            z_levels_mm=None,
            z_step_mm=phase5.Z_STEP_MM,
            max_descent_mm=None,
            max_uncertain_reobservations=1,
        )
    )

    wrist_frame_id = int(endpoint_wrist_frame_id)
    min_wrist_timestamp = (
        float(endpoint_settled_timestamp)
        + float(phase5.POST_MOTION_GUARD_S)
    )
    side_frame_id = int(baseline_side_frame_id)

    print()
    print("=" * 78)
    print("PHASE 8 DETERMINISTIC OWNERSHIP")
    print("=" * 78)
    print("ACT queue/processors : RESET; no further ACT command is allowed")
    print("command anchor       : existing Goal_Position")
    print("WRIST alignment      : Phase-5 accepted controller")
    print("Z / SIDE / OCR       : Phase-5 accepted single-key primitive")
    print("absolute Z guard     : operator Ctrl+C (unchanged Phase-5 contract)")
    print()
    answer = input(
        "Type GO then ENTER to arm deterministic alignment + press: "
    ).strip().upper()
    if answer != "GO":
        raise RuntimeError("Operator did not arm Phase-8 deterministic press")
    if autonomy_stop.is_set():
        raise OperatorStop("operator stop before deterministic press")

    phase["name"] = "PRESS"
    screen_event_watcher = None
    events: list[dict] = []
    screen_records: list[dict] = []

    try:
        print()
        print("===== FAST SIDE EVENT BASELINE =====")
        screen_event_watcher = phase5.FastSidePressEventWatcher(
            side_camera=side_camera,
            calibration=screen_calibration,
            artifact_dir=(phase5.ARTIFACT_DIR / "side_event"),
        )
        event_baseline = screen_event_watcher.arm(
            baseline_duration_s=phase5.SCREEN_EVENT_BASELINE_S
        )
        phase5.ACTIVE_SCREEN_WATCHER = screen_event_watcher

        print(
            "[SIDE EVENT] ARMED "
            f"baseline_frames={event_baseline['baseline_frames']} "
            f"continuation_x0={event_baseline['continuation_x0']}"
        )
        events.append(
            {
                "event": "side_event_armed",
                **event_baseline,
            }
        )

        (
            state,
            directive,
            wrist_frame_id,
            min_wrist_timestamp,
        ) = phase5.align_wrist(
            robot=robot,
            wrist_camera=wrist_camera,
            recognizer=recognizer,
            semantic_lock=semantic_lock,
            tool_reference=tool_reference,
            jacobian=jacobian,
            planner=planner,
            state=state,
            controller=controller,
            motor_names=motor_names,
            last_frame_id=wrist_frame_id,
            min_timestamp=min_wrist_timestamp,
            threshold_px=phase5.INITIAL_ALIGNMENT_THRESHOLD_PX,
            max_commands=phase5.MAX_INITIAL_XY_COMMANDS,
            sent_state_tracker=sent_state_tracker,
        )

        while True:
            if autonomy_stop.is_set():
                raise OperatorStop("operator stop during deterministic press")

            print()
            print(
                "[SUPERVISOR] directive="
                f"{directive.kind} reason={directive.reason!r}"
            )

            if directive.kind is phase5.PressDirectiveKind.DESCEND_TO_Z:
                state = state.with_z_level(directive.z_target_mm)
                settled_timestamp, plan = phase5.send_command_state(
                    robot,
                    planner,
                    state,
                    motor_names,
                    label="Z",
                    sent_state_tracker=sent_state_tracker,
                )
                events.append(
                    {
                        "event": "z_command",
                        "xyz_mm": [float(v) for v in state.xyz_mm],
                        "predicted_delta_mm": [
                            float(v) for v in plan.predicted_delta_mm
                        ],
                    }
                )
                min_wrist_timestamp = (
                    float(settled_timestamp)
                    + float(phase5.POST_MOTION_GUARD_S)
                )
                directive = controller.on_z_motion_stable()
                continue

            if directive.kind is phase5.PressDirectiveKind.OBSERVE_WRIST:
                (
                    state,
                    directive,
                    wrist_frame_id,
                    min_wrist_timestamp,
                ) = phase5.align_wrist(
                    robot=robot,
                    wrist_camera=wrist_camera,
                    recognizer=recognizer,
                    semantic_lock=semantic_lock,
                    tool_reference=tool_reference,
                    jacobian=jacobian,
                    planner=planner,
                    state=state,
                    controller=controller,
                    motor_names=motor_names,
                    last_frame_id=wrist_frame_id,
                    min_timestamp=min_wrist_timestamp,
                    threshold_px=phase5.Z_ALIGNMENT_THRESHOLD_PX,
                    max_commands=phase5.MAX_REALIGN_COMMANDS_PER_Z,
                    sent_state_tracker=sent_state_tracker,
                )
                continue

            if directive.kind in {
                phase5.PressDirectiveKind.VERIFY_SCREEN,
                phase5.PressDirectiveKind.REOBSERVE_SCREEN,
            }:
                result, side_frame_id, records = (
                    phase5.capture_screen_verification(
                        side_camera=side_camera,
                        calibration=screen_calibration,
                        ocr=screen_ocr,
                        after_frame_id=side_frame_id,
                        min_timestamp=min_wrist_timestamp,
                    )
                )
                screen_records.append(
                    {
                        "stage": f"z_{state.z_mm:+.1f}",
                        "status": str(result.status),
                        "wrong_character": result.wrong_character,
                        "records": records,
                    }
                )
                directive = controller.on_verification(result.status)
                continue

            if directive.kind is phase5.PressDirectiveKind.RETRACT_TO_Z0:
                state = _retract_to_z0(
                    robot=robot,
                    planner=planner,
                    state=state,
                    motor_names=motor_names,
                    sent_state_tracker=sent_state_tracker,
                    events=events,
                )
                directive = controller.on_retraction_complete()
                continue

            if directive.kind is phase5.PressDirectiveKind.COMPLETE:
                break

            raise RuntimeError(
                f"Unhandled supervisor directive: {directive.kind}"
            )

    except phase5.ScreenPressEventDetected:
        phase5.ACTIVE_SCREEN_WATCHER = None
        if screen_event_watcher is None:
            raise RuntimeError("SIDE event fired without an active watcher")

        screen_event_watcher.stop()
        event_details = screen_event_watcher.snapshot()

        print()
        print("!" * 78)
        print("SIDE PRESS EVENT LATCHED — RELEASE HAS PRIORITY")
        print("!" * 78)
        print(f"[SIDE EVENT] {event_details}")

        outer_state_before_event = state
        state = sent_state_tracker.state
        print(
            "[SIDE EVENT STATE] "
            "outer="
            f"({outer_state_before_event.x_mm:+.2f},"
            f"{outer_state_before_event.y_mm:+.2f},"
            f"{outer_state_before_event.z_mm:+.2f})mm "
            "latest_sent="
            f"({state.x_mm:+.2f},"
            f"{state.y_mm:+.2f},"
            f"{state.z_mm:+.2f})mm"
        )
        events.append(
            {
                "event": "side_press_event",
                **event_details,
                "outer_xyz_mm": [float(v) for v in outer_state_before_event.xyz_mm],
                "latest_sent_xyz_mm": [float(v) for v in state.xyz_mm],
            }
        )

        release_target_z = min(
            0.0,
            float(state.z_mm) + float(phase5.SCREEN_EVENT_RELEASE_MM),
        )
        release_directive = controller.on_screen_event(
            release_z_target_mm=release_target_z,
        )
        if (
            release_directive.kind
            is not phase5.PressDirectiveKind.RELEASE_TO_Z
        ):
            raise RuntimeError("SIDE event did not enter release state")

        release_index = 0
        last_release_timestamp = time.monotonic()
        while state.z_mm < release_target_z - 1e-12:
            release_index += 1
            next_z = min(
                release_target_z,
                state.z_mm + float(phase5.SCREEN_EVENT_RELEASE_STEP_MM),
            )
            state = state.with_z_level(next_z)
            last_release_timestamp, plan = phase5.send_command_state(
                robot,
                planner,
                state,
                motor_names,
                label=f"RELEASE {release_index}",
                sent_state_tracker=sent_state_tracker,
                event_capture_timestamp=(
                    event_details.get("capture_timestamp")
                    if release_index == 1
                    else None
                ),
            )
            events.append(
                {
                    "event": "release",
                    "segment": int(release_index),
                    "xyz_mm": [float(v) for v in state.xyz_mm],
                    "predicted_delta_mm": [
                        float(v) for v in plan.predicted_delta_mm
                    ],
                }
            )

        directive = controller.on_release_complete()
        result, side_frame_id, records = (
            phase5.capture_event_character_verification(
                side_camera=side_camera,
                calibration=screen_calibration,
                change_model=screen_event_watcher.model,
                char_ocr=screen_char_ocr,
                after_frame_id=int(
                    event_details.get("frame_id", side_frame_id)
                ),
                min_timestamp=last_release_timestamp,
            )
        )
        screen_records.append(
            {
                "stage": "released_press_event",
                "status": str(result.status),
                "wrong_character": result.wrong_character,
                "records": records,
            }
        )
        directive = controller.on_verification(result.status)

        if directive.kind is phase5.PressDirectiveKind.REOBSERVE_SCREEN:
            result, side_frame_id, records = (
                phase5.capture_event_character_verification(
                    side_camera=side_camera,
                    calibration=screen_calibration,
                    change_model=screen_event_watcher.model,
                    char_ocr=screen_char_ocr,
                    after_frame_id=side_frame_id,
                    min_timestamp=last_release_timestamp,
                )
            )
            screen_records.append(
                {
                    "stage": "released_press_event_reobserve",
                    "status": str(result.status),
                    "wrong_character": result.wrong_character,
                    "records": records,
                }
            )
            directive = controller.on_verification(result.status)

        if directive.kind is not phase5.PressDirectiveKind.RETRACT_TO_Z0:
            raise RuntimeError(
                "latched SIDE event must end in retract after released-screen verification"
            )

        state = _retract_to_z0(
            robot=robot,
            planner=planner,
            state=state,
            motor_names=motor_names,
            sent_state_tracker=sent_state_tracker,
            events=events,
        )
        directive = controller.on_retraction_complete()
        if directive.kind is not phase5.PressDirectiveKind.COMPLETE:
            raise RuntimeError("event retract did not complete the controller")

    finally:
        phase5.ACTIVE_SCREEN_WATCHER = None
        if screen_event_watcher is not None:
            screen_event_watcher.stop()

    outcome = (
        None
        if controller.pending_outcome is None
        else str(controller.pending_outcome)
    )
    success = outcome == "SUCCESS"

    result = {
        "success": bool(success),
        "controller_state": str(controller.state),
        "outcome": outcome,
        "final_xyz_mm": [float(v) for v in state.xyz_mm],
        "zero_delta_joint_shift_deg": float(zero_shift),
        "anchor_goal_positions_deg": {
            name: float(anchor.position_deg(name))
            for name in motor_names
        },
        "events": events,
        "screen": screen_records,
    }

    result_path = run_dir / "deterministic_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("PHASE 8 DETERMINISTIC RESULT")
    print("=" * 78)
    print("controller state :", controller.state)
    print("outcome          :", controller.pending_outcome)
    print("final XYZ        :", state.xyz_mm)
    print("saved            :", result_path)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 8 single-key integration pilot: Phase-7 ACT approach + "
            "accepted Phase-5 deterministic align/press/SIDE/OCR/retract + canonical HOME."
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
    parser.add_argument("--confirmed-prefix", default=phase5.CONFIRMED_PREFIX)
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()

    target = str(args.target).strip().upper()
    if target != "G":
        raise ValueError(
            "Phase-8 v1 pilot is intentionally restricted to target G"
        )
    if not np.isfinite(args.duration) or args.duration <= 0.0:
        raise ValueError("--duration must be finite and > 0")
    if not 1 <= args.n_action_steps <= CHUNK_SIZE:
        raise ValueError(f"--n-action-steps must be in [1, {CHUNK_SIZE}]")

    print("=" * 78)
    print("PHASE 8 — END-TO-END SINGLE-KEY INTEGRATION PILOT v1")
    print("=" * 78)
    print("target               :", target)
    print("checkpoint           :", args.checkpoint)
    print("control rate         :", f"{ROLLOUT_HZ:.1f} Hz")
    print("duration limit       :", f"{args.duration:.1f} s")
    print("chunk_size           :", CHUNK_SIZE)
    print("n_action_steps       :", args.n_action_steps)
    print("execution horizon    :", f"{args.n_action_steps / ROLLOUT_HZ:.3f} s")
    print("ACT queue            : OFFICIAL policy.select_action()")
    print(
        "handoff candidate   :",
        f"moving target + >= {HANDOFF_CANDIDATE_MIN_KEYCAPS} keycaps "
        f"+ error <= {HANDOFF_CANDIDATE_THRESHOLD_PX:.1f} px",
    )
    print(
        "settled endpoint    :",
        "fresh WRIST after motion-stable + POST_MOTION_GUARD (diagnostic only)",
    )
    print("deterministic phase  : SAME Goal anchor through align + Z + release + retract")
    print("success authority    : SIDE released-character verification")
    print("confirmed prefix     :", args.confirmed_prefix)
    print("Z policy             :", f"iterative -{phase5.Z_STEP_MM:g} mm command-space steps")
    print("absolute Z guard     : OPERATOR Ctrl+C (unchanged Phase-5 contract)")
    print("joint safety limit   :", f"{MAX_RELATIVE_TARGET_DEG:.1f} deg/send")
    print("success              : expected character -> retract -> AUTO HOME")
    print("timeout / Ctrl+C     : AUTO HOME")
    print("HOME                 :", args.recovery_home)
    print("=" * 78)

    if args.print_config:
        return

    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    if not GLYPH_MODEL.exists():
        raise FileNotFoundError(GLYPH_MODEL)
    if not phase5.IMAGE_JACOBIAN.exists():
        raise FileNotFoundError(phase5.IMAGE_JACOBIAN)
    if not phase5.SCREEN_CALIBRATION.exists():
        raise FileNotFoundError(phase5.SCREEN_CALIBRATION)

    phase5.TARGET = target
    phase5.CONFIRMED_PREFIX = str(args.confirmed_prefix)
    phase5.ACTIVE_SCREEN_WATCHER = None
    phase5.SEMANTIC_REENTRY_GUARDED = False

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

    screen_calibration = phase5.ScreenCalibration.load(
        phase5.SCREEN_CALIBRATION
    )
    if screen_calibration is None:
        raise RuntimeError("screen calibration is missing")
    screen_ocr = phase5.TesseractScreenLineOCR(
        language="eng",
        scale=3.0,
        clahe_clip_limit=2.0,
        dark_threshold=185,
    )
    screen_char_ocr = phase5.TesseractSingleCharacterOCR(
        language="eng",
        scale=4.0,
        clahe_clip_limit=2.0,
        dark_threshold=185,
    )

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
    side = ThreadedOpenCVCamera(CameraSpec.from_yaml(phase5.SIDE_CAMERA_CONFIG))
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
        if current in {"ACT", "DETERMINISTIC", "PRESS", "STARTUP"}:
            if not autonomy_stop.is_set():
                autonomy_stop.set()
                print("\n[CTRL+C] autonomy stop requested; current bus operation will finish first.")
            return

    def guarded_raise_if_screen_event():
        original_raise_if_screen_event()
        if phase["name"] in {"DETERMINISTIC", "PRESS"} and autonomy_stop.is_set():
            raise OperatorStop("operator stop during deterministic phase")

    def guarded_wait_motion_stable(robot_obj, motor_names_obj):
        settled = original_wait_motion_stable(robot_obj, motor_names_obj)
        if phase["name"] in {"DETERMINISTIC", "PRESS"} and autonomy_stop.is_set():
            raise OperatorStop("operator stop during deterministic phase")
        if phase["name"] == "HOME" and home_stop.is_set():
            raise HomeStop("operator emergency stop during HOME recovery")
        return settled

    phase5.raise_if_screen_event = guarded_raise_if_screen_event
    phase5.wait_motion_stable = guarded_wait_motion_stable

    motor_names: list[str] = []
    keys: list[str] = []
    task_status = "STARTING"
    task_error = None
    target_trigger = None
    handoff_candidate_trigger = None
    settled_endpoint = None
    deterministic_result = None
    screen_baseline = None
    baseline_side_frame_id = -1
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
        print()
        print("Before continuing, the Mac screen must contain exactly:")
        print()
        print(f"    {args.confirmed_prefix}")
        print()
        print(
            "with the caret immediately after the final character. "
            "Keep key-repeat delay at the previously accepted Phase-5 setting."
        )
        input("\nPress ENTER when the screen is ready: ")

        top.start()
        wrist.start()
        side.start()
        _wait_initial_frames(top, wrist)
        side_deadline = time.monotonic() + 5.0
        while time.monotonic() < side_deadline:
            if side.latest() is not None:
                break
            time.sleep(0.02)
        else:
            raise RuntimeError("SIDE did not produce an initial frame")

        time.sleep(0.5)
        print()
        print("===== SCREEN BASELINE =====")
        baseline, baseline_side_frame_id, baseline_records = (
            phase5.capture_screen_verification(
                side_camera=side,
                calibration=screen_calibration,
                ocr=screen_ocr,
                after_frame_id=-1,
                min_timestamp=None,
            )
        )
        screen_baseline = {
            "status": str(baseline.status),
            "records": baseline_records,
        }
        if (
            baseline.status
            is not phase5.ScreenVerificationStatus.CONFIRMED_NO_CHANGE
        ):
            raise RuntimeError(
                "SIDE baseline is not CONFIRMED_NO_CHANGE. Do not move the robot."
            )

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
                task_status = "TIMEOUT_NO_TAKEOVER"
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
            glyph_observations = observe_glyph_candidates(
                wrist_frame.image,
                recognizer,
            )
            target_observation = select_target_observation(
                target,
                glyph_observations,
            )
            perception_ms = (time.perf_counter() - perception_started) * 1000.0

            keycap_count = len(glyph_observations)
            handoff_error_px = None
            handoff_error_norm_px = None
            handoff_candidate_now = False

            if target_observation.found and target_observation.center_px is not None:
                handoff_error_px = np.asarray(
                    tool_reference.error_px(target_observation.center_px),
                    dtype=np.float64,
                )
                handoff_error_norm_px = float(np.linalg.norm(handoff_error_px))

                # TARGET_ACQUIRED is diagnostic only.  Seeing the glyph does NOT
                # transfer controller ownership.
                if target_trigger is None:
                    target_trigger = {
                        "tick": tick,
                        "commands_sent": command_count,
                        "act_elapsed_s": float(time.perf_counter() - rollout_start),
                        "top_frame_id": int(top_frame.frame_id),
                        "wrist_frame_id": int(wrist_frame.frame_id),
                        "keycap_count": int(keycap_count),
                        "tool_tip_px": list(tool_reference.center),
                        "error_px": handoff_error_px.tolist(),
                        "error_norm_px": handoff_error_norm_px,
                        "observation": target_observation.to_dict(),
                    }
                    _append_jsonl(
                        events_path,
                        {"event": "TARGET_ACQUIRED", **target_trigger},
                    )
                    print(
                        f"\n[TARGET ACQUIRED] t={target_trigger['act_elapsed_s']:.3f}s "
                        f"commands={command_count} center={target_observation.center_px} "
                        f"error={handoff_error_norm_px:.2f}px keycaps={keycap_count}"
                    )
                    print(
                        "[ACT CONTINUES] target is visible, but controller ownership stays with ACT "
                        "until the moving-frame handoff-candidate condition is reached."
                    )

                handoff_candidate_now = (
                    keycap_count >= HANDOFF_CANDIDATE_MIN_KEYCAPS
                    and handoff_error_norm_px <= HANDOFF_CANDIDATE_THRESHOLD_PX
                )

                if handoff_candidate_now:
                    rollout_stop = time.perf_counter()
                    policy.reset()
                    _processor_reset(preprocessor)
                    _processor_reset(postprocessor)
                    handoff_candidate_trigger = {
                        "tick": tick,
                        "commands_sent": command_count,
                        "act_elapsed_s": float(rollout_stop - rollout_start),
                        "top_frame_id": int(top_frame.frame_id),
                        "wrist_frame_id": int(wrist_frame.frame_id),
                        "keycap_count": int(keycap_count),
                        "tool_tip_px": list(tool_reference.center),
                        "error_px": handoff_error_px.tolist(),
                        "error_norm_px": handoff_error_norm_px,
                        "threshold_px": HANDOFF_CANDIDATE_THRESHOLD_PX,
                        "observation": target_observation.to_dict(),
                    }
                    _append_jsonl(
                        events_path,
                        {"event": "HANDOFF_CANDIDATE", **handoff_candidate_trigger},
                    )
                    task_status = "HANDOFF_CANDIDATE"
                    print(
                        f"\n[HANDOFF CANDIDATE] "
                        f"t={handoff_candidate_trigger['act_elapsed_s']:.3f}s "
                        f"commands={command_count} error={handoff_error_norm_px:.2f}px "
                        f"<= {HANDOFF_CANDIDATE_THRESHOLD_PX:.1f}px keycaps={keycap_count}"
                    )
                    print(
                        "[ACT STOP] candidate only; settled endpoint will be measured "
                        "before ground-truth takeover."
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
                    "keycap_count": int(keycap_count),
                    "handoff_error_px": None if handoff_error_px is None else handoff_error_px.tolist(),
                    "handoff_error_norm_px": handoff_error_norm_px,
                    "handoff_candidate": bool(handoff_candidate_now),
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

        if task_status == "HANDOFF_CANDIDATE":
            phase["name"] = "DETERMINISTIC"
            settled_timestamp = phase5.wait_motion_stable(robot, motor_names)
            if autonomy_stop.is_set():
                raise OperatorStop("operator stop before settled endpoint measurement")

            print()
            print("=" * 78)
            print("SETTLED ACT ENDPOINT DIAGNOSTIC")
            print("=" * 78)

            try:
                settled_endpoint = _capture_settled_endpoint(
                    wrist_camera=wrist,
                    recognizer=recognizer,
                    target=target,
                    tool_reference=tool_reference,
                    settled_timestamp=settled_timestamp,
                )
            except OperatorStop:
                raise
            except Exception as exc:
                settled_endpoint = {
                    "status": "UNAVAILABLE",
                    "error": f"{type(exc).__name__}: {exc}",
                    "settled_timestamp": float(settled_timestamp),
                    "post_motion_guard_s": float(phase5.POST_MOTION_GUARD_S),
                    "acceptance_role": "diagnostic_only",
                }
                _append_jsonl(
                    events_path,
                    {"event": "SETTLED_ENDPOINT_UNAVAILABLE", **settled_endpoint},
                )
                print(
                    "[SETTLED ENDPOINT] measurement unavailable; "
                    "ground-truth takeover will still run:",
                    settled_endpoint["error"],
                )
            else:
                candidate_error = np.asarray(
                    handoff_candidate_trigger["error_px"],
                    dtype=np.float64,
                )
                endpoint_error = np.asarray(
                    settled_endpoint["error_px"],
                    dtype=np.float64,
                )
                endpoint_drift = endpoint_error - candidate_error
                settled_endpoint["moving_candidate_error_px"] = (
                    candidate_error.tolist()
                )
                settled_endpoint["moving_candidate_error_norm_px"] = float(
                    handoff_candidate_trigger["error_norm_px"]
                )
                settled_endpoint["error_drift_px"] = endpoint_drift.tolist()
                settled_endpoint["error_drift_norm_px"] = float(
                    np.linalg.norm(endpoint_drift)
                )
                settled_endpoint["error_norm_change_px"] = float(
                    settled_endpoint["error_norm_px"]
                    - handoff_candidate_trigger["error_norm_px"]
                )

                _append_jsonl(
                    events_path,
                    {"event": "SETTLED_ENDPOINT", **settled_endpoint},
                )
                print(
                    f"[SETTLED ENDPOINT] frame={settled_endpoint['frame_id']} "
                    f"error={settled_endpoint['error_norm_px']:.2f}px "
                    f"norm_change={settled_endpoint['error_norm_change_px']:+.2f}px "
                    f"drift={settled_endpoint['error_drift_norm_px']:.2f}px "
                    f"phase5<=80={settled_endpoint['within_phase5_capture_threshold']}"
                )
                print(
                    "[DIAGNOSTIC ONLY] settled residual does not gate takeover; "
                    "deterministic convergence is the ground truth."
                )

            if autonomy_stop.is_set():
                raise OperatorStop("operator stop before deterministic ownership")

            print()
            print("=" * 78)
            print("END-TO-END DETERMINISTIC SINGLE-KEY PHASE")
            print("=" * 78)

            endpoint_wrist_frame_id = int(
                (
                    settled_endpoint.get("frame_id")
                    if isinstance(settled_endpoint, dict)
                    and settled_endpoint.get("frame_id") is not None
                    else handoff_candidate_trigger["wrist_frame_id"]
                )
            )

            deterministic_result = _run_full_deterministic_press(
                robot=robot,
                wrist_camera=wrist,
                side_camera=side,
                target=target,
                motor_names=motor_names,
                recognizer=recognizer,
                tool_reference=tool_reference,
                screen_calibration=screen_calibration,
                screen_ocr=screen_ocr,
                screen_char_ocr=screen_char_ocr,
                baseline_side_frame_id=baseline_side_frame_id,
                endpoint_settled_timestamp=settled_timestamp,
                endpoint_wrist_frame_id=endpoint_wrist_frame_id,
                run_dir=run_dir,
                autonomy_stop=autonomy_stop,
                phase=phase,
            )

            _append_jsonl(
                events_path,
                {
                    "event": (
                        "END_TO_END_PASS"
                        if deterministic_result["success"]
                        else "END_TO_END_FAIL"
                    ),
                    "outcome": deterministic_result["outcome"],
                    "controller_state": deterministic_result["controller_state"],
                    "final_xyz_mm": deterministic_result["final_xyz_mm"],
                },
            )

            if deterministic_result["success"]:
                task_status = "PASS"
                print("\n[END-TO-END PASS] SIDE confirmed expected character.")
            else:
                task_status = "FAIL_PRESS_OUTCOME"
                task_error = (
                    "deterministic press completed with outcome "
                    f"{deterministic_result['outcome']}"
                )
                print("\n[END-TO-END FAIL]", task_error)

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

        phase5.ACTIVE_SCREEN_WATCHER = None
        top.stop()
        wrist.stop()
        side.stop()

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
            "schema": "phase8.single_key_integration.v1",
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
            "target_acquisition": target_trigger,
            "handoff_candidate": handoff_candidate_trigger,
            "handoff_candidate_contract": {
                "threshold_px": HANDOFF_CANDIDATE_THRESHOLD_PX,
                "min_keycaps": HANDOFF_CANDIDATE_MIN_KEYCAPS,
                "reference": "phase5.INITIAL_CAPTURE_THRESHOLD_PX",
                "meaning": "moving_frame_candidate_only_not_servo_ready",
            },
            "settled_endpoint": settled_endpoint,
            "screen_baseline": screen_baseline,
            "deterministic_result": deterministic_result,
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
        print("PHASE 8 END-TO-END SUMMARY")
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
        print("HOME recovery       :", home_result["status"])
        print("summary             :", summary_path)
        print("=" * 78)


if __name__ == "__main__":
    main()
