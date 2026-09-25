from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
import numpy as np

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.robot_utils import precise_sleep

import validate_phase5_single_key as phase5

from so101_typing.adapters.cameras import CameraSpec, ThreadedOpenCVCamera
from so101_typing.control.cartesian_kinematics import (
    build_official_cartesian_pipeline,
    build_so101_kinematics,
    end_effector_xyz_mm,
    ordered_joint_observation,
)
from so101_typing.control.fixed_anchor_planner import (
    FixedAnchorCartesianPlanner,
    FixedAnchorPlannerConfig,
)
from so101_typing.control.fixed_anchor_xyz import (
    FixedAnchorXYZCommandState,
    FixedGoalAnchor,
    LatestSentXYZCommandState,
)
from so101_typing.control.image_jacobian import ImageJacobianCalibration
from so101_typing.control.tool_reference import ToolReferenceCalibration
from so101_typing.perception.glyph_runtime import RuntimeHOGGlyphRecognizer
from so101_typing.perception.semantic_target_lock import SemanticTargetLock
from so101_typing.supervisor.press_controller import (
    PressDirectiveKind,
    SingleKeyPressConfig,
    SingleKeyPressController,
)


ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "lawson_follower_arm"
MAX_RELATIVE_TARGET_DEG = 10.0
ARTIFACT_ROOT = Path("artifacts/phase6_act_dataset")
FORMAL_DATASET_BASE = ARTIFACT_ROOT / "keyboard_v1"
FORMAL_RUNS_ROOT = FORMAL_DATASET_BASE / "runs"

WRIST_CAMERA_CONFIG = Path("configs/cameras/wrist.yaml")
GLYPH_MODEL = Path("artifacts/models/glyph_hog_svm")
TOOL_REFERENCE = Path("calibration/tool_reference.json")
IMAGE_JACOBIAN = Path("calibration/image_jacobian.json")
URDF = Path.home() / ".cache/huggingface/lerobot/robot-urdfs/so101/so101_new_calib.urdf"
RECOVERY_HOME_CONFIG = Path("configs/robot/recovery_home.json")


def _latest_run(target: str) -> Path:
    target = str(target).strip().upper()

    root = FORMAL_RUNS_ROOT
    runs = sorted(
        p
        for p in root.glob(
            f"run_*_{target.lower()}"
        )
        if p.is_dir()
    )

    if not runs:
        raise FileNotFoundError(
            f"No formal Phase 6 runs for target={target} "
            f"found under {root}"
        )

    return runs[-1]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _numeric(action: dict) -> dict[str, float]:
    return {
        str(k): float(v)
        for k, v in action.items()
        if isinstance(v, (int, float, np.integer, np.floating))
    }


def _arm_keys(motor_names: list[str]) -> list[str]:
    return [name if name.endswith(".pos") else f"{name}.pos" for name in motor_names]


def _max_action_delta(a: dict, b: dict, names: list[str]) -> float:
    values = [abs(float(a[name]) - float(b[name])) for name in names if name in a and name in b]
    return max(values) if values else float("inf")


def _find_episode_summary(run_dir: Path, dataset_episode_index: int) -> tuple[Path, dict]:
    for path in sorted(run_dir.glob("attempt_*/summary.json")):
        summary = _read_json(path)
        if summary.get("accepted") and int(summary.get("dataset_episode_index", -1)) == dataset_episode_index:
            return path, summary
    raise FileNotFoundError(
        f"No accepted episode with dataset_episode_index={dataset_episode_index} under {run_dir}"
    )


def _require_qc_pass(run_dir: Path, dataset_episode_index: int) -> None:
    path = run_dir / "qc_report.json"
    if not path.exists():
        raise RuntimeError(
            f"QC report not found: {path}. Run scripts/phase6_episode_qc.py first, "
            "or pass --allow-unchecked only for debugging."
        )
    report = _read_json(path)
    match = next(
        (x for x in report.get("episodes", []) if int(x["dataset_episode_index"]) == dataset_episode_index),
        None,
    )
    if match is None:
        raise RuntimeError(f"Episode {dataset_episode_index} is not present in {path}")
    if match.get("status") != "PASS":
        raise RuntimeError(
            f"Episode {dataset_episode_index} QC status is {match.get('status')}, not PASS. "
            f"Reasons: {match.get('fail_reasons') or match.get('review_reasons')}"
        )


def _auto_restore_recorded_start(
    *,
    robot: SO101Follower,
    baseline: dict[str, float],
    motor_names: list[str],
    step_deg: float = 4.0,
    max_sent_diff_deg: float = 0.25,
    label: str = "RECORDED START",
) -> dict:
    """Automatically move from the current synchronized Goal state to a recorded baseline.

    This transport is NOT part of the recorded episode. The caller must ensure the
    Goal register is synchronized with the physical Present pose before using this
    helper at process startup. Every intermediate command is bounded in joint space
    and validated against the actual action returned by robot.send_action().
    """
    if step_deg <= 0.0 or step_deg > MAX_RELATIVE_TARGET_DEG:
        raise ValueError(
            f"restore step_deg must be in (0, {MAX_RELATIVE_TARGET_DEG}], got {step_deg}"
        )

    keys = _arm_keys(motor_names)
    goal_positions = robot.bus.sync_read(
        "Goal_Position",
        num_retry=robot.config.num_read_retries,
    )
    current = {f"{name}.pos": float(goal_positions[name]) for name in motor_names}
    missing = [name for name in keys if name not in baseline]
    if missing:
        raise ReplayInvalid(f"Recorded ARM baseline is missing joints: {missing}")

    delta0 = _max_action_delta(current, baseline, keys)
    steps = max(1, int(math.ceil(delta0 / step_deg)))
    print()
    print(f"===== AUTO RESTORE {label} =====")
    print(f"initial max Goal diff : {delta0:.2f}deg")
    print(f"joint step bound      : {step_deg:.2f}deg")
    print(f"planned restore steps : {steps}")
    print("operator input         : NONE (Ctrl+C remains emergency stop)")

    max_sent_diff = 0.0
    for i in range(1, steps + 1):
        alpha = i / steps
        requested = dict(current)
        for name in keys:
            requested[name] = float(current[name] + alpha * (baseline[name] - current[name]))

        actual = _numeric(robot.send_action(requested))
        sent_diff = _max_action_delta(actual, requested, keys)
        max_sent_diff = max(max_sent_diff, sent_diff)
        if sent_diff > max_sent_diff_deg:
            raise ReplayInvalid(
                f"Automatic start restore command was changed/clipped by {sent_diff:.3f}deg "
                f"at step {i}/{steps}; limit={max_sent_diff_deg:.3f}deg"
            )
        # Restore is deliberately slower than the recorded replay. Each bounded
        # transport step must become motion-stable before the next one so a large
        # initial pose difference cannot accumulate servo lag or trigger hidden
        # max-relative-target clipping. Goal==Present is still NOT required.
        phase5.wait_motion_stable(robot, motor_names)

    exact = _numeric(robot.send_action(baseline))
    exact_delta = _max_action_delta(exact, baseline, keys)
    if exact_delta > max_sent_diff_deg:
        raise ReplayInvalid(
            f"Exact recorded baseline was changed/clipped by {exact_delta:.3f}deg; "
            f"limit={max_sent_diff_deg:.3f}deg"
        )

    settled_timestamp = phase5.wait_motion_stable(robot, motor_names)
    goal_after = robot.bus.sync_read(
        "Goal_Position",
        num_retry=robot.config.num_read_retries,
    )
    restored = {f"{name}.pos": float(goal_after[name]) for name in motor_names}
    final_goal_diff = _max_action_delta(restored, baseline, keys)
    if final_goal_diff > max_sent_diff_deg:
        raise ReplayInvalid(
            f"Recorded baseline restore ended with Goal diff={final_goal_diff:.3f}deg; "
            f"limit={max_sent_diff_deg:.3f}deg"
        )

    print(
        f"[{label} RESTORED] final Goal diff={final_goal_diff:.3f}deg; "
        "motion stable."
    )
    return {
        "initial_max_goal_diff_deg": float(delta0),
        "restore_steps": int(steps),
        "step_bound_deg": float(step_deg),
        "max_sent_diff_deg": float(max_sent_diff),
        "final_goal_diff_deg": float(final_goal_diff),
        "settled_timestamp": float(settled_timestamp),
    }



def _synchronize_goal_to_present(
    *,
    robot: SO101Follower,
    motor_names: list[str],
    max_sent_diff_deg: float,
) -> dict:
    """Establish a safe startup hold from the robot's actual physical Present pose.

    After torque-off/reconnect, Goal_Position can be stale or unrelated to the
    physical arm pose.  Using that stale Goal as the origin for an automatic HOME
    transport can create a very large first physical command even when the planner
    thinks it is taking a small step.  For startup/recovery transport only, copy the
    measured Present positions into Goal once, validate the actual sent command, and
    wait for motion stability.  This is deliberately separate from the deterministic
    WRIST controller's fixed-Goal anchor rule.
    """
    keys = _arm_keys(motor_names)
    present_positions = robot.bus.sync_read(
        "Present_Position",
        num_retry=robot.config.num_read_retries,
    )
    present_hold = {
        f"{name}.pos": float(present_positions[name])
        for name in motor_names
    }

    goal_before_raw = robot.bus.sync_read(
        "Goal_Position",
        num_retry=robot.config.num_read_retries,
    )
    goal_before = {
        f"{name}.pos": float(goal_before_raw[name])
        for name in motor_names
    }
    stale_goal_gap = _max_action_delta(goal_before, present_hold, keys)

    print()
    print("===== SYNCHRONIZE STARTUP HOLD TO PHYSICAL PRESENT =====")
    print(f"Goal↔Present gap before hold : {stale_goal_gap:.2f}deg")
    print("startup transport origin     : Present_Position (safety initialization only)")

    actual = _numeric(robot.send_action(present_hold))
    sent_diff = _max_action_delta(actual, present_hold, keys)
    if sent_diff > max_sent_diff_deg:
        raise ReplayInvalid(
            f"Present-hold command was changed/clipped by {sent_diff:.3f}deg; "
            f"limit={max_sent_diff_deg:.3f}deg"
        )

    settled_timestamp = phase5.wait_motion_stable(robot, motor_names)
    goal_after_raw = robot.bus.sync_read(
        "Goal_Position",
        num_retry=robot.config.num_read_retries,
    )
    present_after_raw = robot.bus.sync_read(
        "Present_Position",
        num_retry=robot.config.num_read_retries,
    )
    goal_after = {
        f"{name}.pos": float(goal_after_raw[name])
        for name in motor_names
    }
    present_after = {
        f"{name}.pos": float(present_after_raw[name])
        for name in motor_names
    }
    final_gap = _max_action_delta(goal_after, present_after, keys)

    print(
        f"[STARTUP HOLD READY] sent diff={sent_diff:.3f}deg; "
        f"Goal↔Present={final_gap:.3f}deg; motion stable."
    )
    return {
        "initial_goal_present_gap_deg": float(stale_goal_gap),
        "sent_diff_deg": float(sent_diff),
        "final_goal_present_gap_deg": float(final_gap),
        "settled_timestamp": float(settled_timestamp),
    }

def _replay_trace(
    *,
    robot: SO101Follower,
    trace: list[dict],
    motor_names: list[str],
    max_sent_diff_deg: float,
) -> dict:
    if len(trace) < 2:
        raise RuntimeError("Replay trace is too short")
    keys = _arm_keys(motor_names)
    original_t0 = float(trace[0]["timestamp"])
    replay_t0 = time.perf_counter()
    max_sent_diff = 0.0
    max_lateness_ms = 0.0
    send_latencies_ms: list[float] = []

    print()
    print("===== FULL EPISODE REPLAY =====")
    print(f"control samples: {len(trace)}")
    print("Timing source   : original per-command monotonic intervals")

    for i, item in enumerate(trace):
        target_elapsed = float(item["timestamp"]) - original_t0
        elapsed = time.perf_counter() - replay_t0
        precise_sleep(max(0.0, target_elapsed - elapsed))

        before = time.perf_counter()
        requested = _numeric(item["sent_action"])
        actual = _numeric(robot.send_action(requested))
        after = time.perf_counter()
        send_latencies_ms.append((after - before) * 1000.0)
        diff = _max_action_delta(actual, requested, keys)
        max_sent_diff = max(max_sent_diff, diff)
        lateness_ms = max(0.0, ((after - replay_t0) - target_elapsed) * 1000.0)
        max_lateness_ms = max(max_lateness_ms, lateness_ms)

        if diff > max_sent_diff_deg:
            raise ReplayInvalid(
                f"Replay command was changed/clipped by {diff:.3f}deg at sample {i}; "
                f"limit={max_sent_diff_deg:.3f}deg"
            )

    settled_timestamp = phase5.wait_motion_stable(robot, motor_names)
    print(
        f"[REPLAY COMPLETE] max sent diff={max_sent_diff:.3f}deg "
        f"max schedule lateness={max_lateness_ms:.1f}ms"
    )
    return {
        "control_samples": len(trace),
        "original_duration_s": float(trace[-1]["timestamp"] - trace[0]["timestamp"]),
        "max_sent_diff_deg": float(max_sent_diff),
        "max_schedule_lateness_ms": float(max_lateness_ms),
        "send_latency_ms_p95": float(np.percentile(send_latencies_ms, 95)) if send_latencies_ms else None,
        "endpoint_settled_timestamp": float(settled_timestamp),
    }


def _load_recovery_pose(path: Path) -> dict[str, float]:
    """Load the explicit robot recovery pose. Dataset contents never define HOME."""
    if not path.exists():
        raise FileNotFoundError(
            f"Recovery HOME config not found: {path}. "
            "Create it once with scripts/export_episode_start_pose.py or another explicit pose-capture tool."
        )
    payload = _read_json(path)
    action = payload.get("action")
    if not isinstance(action, dict) or not action:
        raise RuntimeError(f"Recovery pose config has no non-empty 'action' mapping: {path}")
    out = {str(k): float(v) for k, v in action.items()}
    if not all(math.isfinite(v) for v in out.values()):
        raise RuntimeError(f"Recovery pose config contains non-finite values: {path}")
    print(f"recovery HOME config  : {path}")
    return out


def _emergency_hold_until_operator(robot: SO101Follower, reason: str) -> None:
    """Fail safe: never torque-off automatically from an unknown suspended pose."""
    print()
    print("!" * 72)
    print("AUTOMATIC HOME RECOVERY FAILED — TORQUE REMAINS ENABLED")
    print("!" * 72)
    print(reason)
    print("The validator will NOT disconnect the robot from this unknown pose.")
    print("Support the arm / make the workspace safe, then press Ctrl+C once to allow disconnect.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[EMERGENCY DISCONNECT AUTHORIZED] Operator requested disconnect.")


class ReplayInvalid(RuntimeError):
    pass


def _run_takeover(
    *,
    robot: SO101Follower,
    wrist_camera: ThreadedOpenCVCamera,
    target: str,
    motor_names: list[str],
    endpoint_settled_timestamp: float,
    episode_dir: Path,
) -> dict:
    phase5.TARGET = target
    phase5.ACTIVE_SCREEN_WATCHER = None
    phase5.SEMANTIC_REENTRY_GUARDED = False
    phase5.ARTIFACT_DIR = episode_dir / "takeover_debug"

    tool_reference = ToolReferenceCalibration.load(TOOL_REFERENCE)
    if tool_reference is None:
        raise RuntimeError("tool reference is missing")
    tool_reference.require_direct_tool_tip()
    jacobian = ImageJacobianCalibration.load(IMAGE_JACOBIAN)
    if jacobian is None:
        raise RuntimeError("image Jacobian is missing")
    recognizer = RuntimeHOGGlyphRecognizer.load(GLYPH_MODEL)
    semantic_lock = SemanticTargetLock(target)

    print()
    print("===== SEAMLESS WRIST TAKEOVER =====")
    print("No teleop command is sent after replay endpoint. No 65px/80px gate is applied.")
    print("Ground truth = deterministic WRIST servo actually converges to stable XY alignment.")

    target_center, error, error_norm, pre_frame = phase5.capture_initial_semantic_target(
        wrist_camera,
        recognizer,
        semantic_lock,
        tool_reference,
        min_timestamp=endpoint_settled_timestamp,
    )
    print(
        f"[TAKEOVER INITIAL] frame={pre_frame.frame_id} target={target_center} "
        f"error=({error[0]:+.2f},{error[1]:+.2f})px norm={error_norm:.2f}px"
    )

    present_observation = ordered_joint_observation(robot.get_observation(), motor_names)
    goal_positions = robot.bus.sync_read(
        "Goal_Position",
        num_retry=robot.config.num_read_retries,
    )
    anchor = FixedGoalAnchor.from_goal_positions(goal_positions, motor_names)
    kinematics = build_so101_kinematics(URDF, motor_names=motor_names)
    anchor_xyz_mm = end_effector_xyz_mm(kinematics, anchor.as_observation(), motor_names)
    pipeline = build_official_cartesian_pipeline(
        URDF,
        motor_names=motor_names,
        end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
        max_ee_step_m=0.035,
        orientation_weight=0.01,
        raise_on_jump=True,
        use_latched_reference=True,
    )
    planner = FixedAnchorCartesianPlanner(
        pipeline=pipeline,
        validation_kinematics=kinematics,
        motor_names=motor_names,
        anchor=anchor,
        anchor_xyz_mm=anchor_xyz_mm,
        config=FixedAnchorPlannerConfig(
            max_command_norm_mm=phase5.MAX_COMMAND_NORM_MM,
            max_model_xy_error_mm=1.0,
            max_model_z_error_mm=1.0,
            max_relative_target_deg=MAX_RELATIVE_TARGET_DEG,
            max_zero_joint_shift_deg=0.5,
        ),
    )
    zero_shift = planner.latch_zero_delta()
    print(f"[ANCHOR] zero-delta max joint shift={zero_shift:.3f}deg")

    state = FixedAnchorXYZCommandState.at_anchor(
        anchor,
        max_xy_norm_mm=phase5.MAX_XY_CORRECTION_MM,
        max_xyz_norm_mm=phase5.MAX_COMMAND_NORM_MM,
    )
    sent_state_tracker = LatestSentXYZCommandState(state)
    controller = SingleKeyPressController(
        SingleKeyPressConfig(
            z_levels_mm=None,
            z_step_mm=phase5.Z_STEP_MM,
            max_descent_mm=None,
            max_uncertain_reobservations=1,
        )
    )

    state, directive, wrist_frame_id, min_wrist_timestamp = phase5.align_wrist(
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
        last_frame_id=int(pre_frame.frame_id),
        min_timestamp=endpoint_settled_timestamp + phase5.POST_MOTION_GUARD_S,
        threshold_px=phase5.INITIAL_ALIGNMENT_THRESHOLD_PX,
        max_commands=phase5.MAX_INITIAL_XY_COMMANDS,
        sent_state_tracker=sent_state_tracker,
    )

    if directive.kind is not PressDirectiveKind.DESCEND_TO_Z:
        raise RuntimeError(
            f"WRIST servo returned unexpected directive after alignment: {directive.kind} {directive.reason!r}"
        )
    if abs(float(state.z_mm)) > 1e-9:
        raise RuntimeError(f"Takeover validation changed Z unexpectedly: z={state.z_mm}")

    print()
    print("[TAKEOVER PASS] WRIST servo reached stable XY alignment. Z-down/press is intentionally NOT executed.")
    return {
        "initial_target_center_px": [float(target_center[0]), float(target_center[1])],
        "initial_error_px": [float(error[0]), float(error[1])],
        "initial_error_norm_px": float(error_norm),
        "final_command_xyz_mm": [float(v) for v in state.xyz_mm],
        "next_directive": str(directive.kind),
        "wrist_frame_id": int(wrist_frame_id),
        "min_wrist_timestamp": float(min_wrist_timestamp),
    }


def _append_validation(episode_dir: Path, record: dict) -> Path:
    path = episode_dir / "takeover_validation.json"
    payload = {
        "schema": "phase6.takeover_validation.v1",
        "dataset_episode_index": int(record["dataset_episode_index"]),
        "target": record["target"],
        "trials": [],
    }
    if path.exists():
        payload = _read_json(path)
    payload.setdefault("trials", []).append(record)
    trials = payload["trials"]
    payload["pass_trials"] = sum(x["status"] == "PASS" for x in trials)
    payload["fail_trials"] = sum(x["status"] == "FAIL" for x in trials)
    payload["invalid_trials"] = sum(x["status"] == "INVALID_REPLAY" for x in trials)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _refresh_verified_manifest(
    run_dir: Path,
    required_passes: int,
) -> Path:
    """Build one verified manifest for the shared formal dataset.

    Collection runs remain per-target audit/session boundaries, but every
    accepted episode belongs to the same appendable LeRobot dataset.
    Eligibility is therefore keyed by global dataset_episode_index.

    The manifest aggregates all completed takeover validations under the
    shared keyboard_v1/runs directory and preserves each episode's target.
    """

    runs_root = (
        run_dir.parent
        if run_dir.parent.name == "runs"
        else FORMAL_RUNS_ROOT
    )

    run_dirs = sorted(
        p
        for p in runs_root.glob("run_*")
        if p.is_dir()
    )

    if not run_dirs:
        raise RuntimeError(
            f"No formal collection runs found under {runs_root}"
        )

    locations: set[tuple[str, str]] = set()
    by_index: dict[int, dict] = {}

    for source_run in run_dirs:
        run_summary_path = (
            source_run / "run_summary.json"
        )

        if not run_summary_path.exists():
            continue

        run_summary = _read_json(
            run_summary_path
        )

        source_target = (
            str(run_summary["target"])
            .strip()
            .upper()
        )

        locations.add(
            (
                str(run_summary["repo_id"]),
                str(run_summary["dataset_root"]),
            )
        )

        for summary_path in sorted(
            source_run.glob(
                "attempt_*/summary.json"
            )
        ):
            summary = _read_json(
                summary_path
            )

            if (
                not summary.get("accepted")
                or "dataset_episode_index"
                not in summary
            ):
                continue

            validation_path = (
                summary_path.parent
                / "takeover_validation.json"
            )

            if not validation_path.exists():
                continue

            validation = _read_json(
                validation_path
            )

            episode_target = (
                str(
                    validation.get(
                        "target",
                        source_target,
                    )
                )
                .strip()
                .upper()
            )

            if episode_target != source_target:
                raise RuntimeError(
                    "Validation target does not match collection run: "
                    f"run={source_target}, validation={episode_target}, "
                    f"path={validation_path}"
                )

            idx = int(
                summary[
                    "dataset_episode_index"
                ]
            )

            entry = by_index.get(idx)

            if entry is None:
                entry = {
                    "dataset_episode_index": idx,
                    "target": episode_target,
                    "pass_trials": 0,
                    "fail_trials": 0,
                    "invalid_trials": 0,
                    "sources": [],
                }

                by_index[idx] = entry

            elif entry["target"] != episode_target:
                raise RuntimeError(
                    "One dataset_episode_index is associated with "
                    "multiple targets: "
                    f"episode={idx}, "
                    f"{entry['target']} vs {episode_target}"
                )

            entry["pass_trials"] += int(
                validation.get(
                    "pass_trials",
                    0,
                )
            )

            entry["fail_trials"] += int(
                validation.get(
                    "fail_trials",
                    0,
                )
            )

            entry["invalid_trials"] += int(
                validation.get(
                    "invalid_trials",
                    0,
                )
            )

            entry["sources"].append(
                {
                    "run_dir": str(
                        source_run
                    ),
                    "summary_json": str(
                        summary_path
                    ),
                    "validation_json": str(
                        validation_path
                    ),
                }
            )

    if len(locations) != 1:
        raise RuntimeError(
            "Formal runs do not resolve to exactly one "
            f"shared LeRobot dataset: {sorted(locations)}"
        )

    repo_id, dataset_root = next(
        iter(locations)
    )

    details: list[dict] = []
    verified: list[int] = []

    for idx in sorted(by_index):
        entry = by_index[idx]

        ok = (
            int(entry["pass_trials"])
            >= required_passes
            and int(entry["fail_trials"])
            == 0
        )

        item = dict(entry)
        item["verified"] = bool(ok)

        details.append(item)

        if ok:
            verified.append(idx)

    verified_targets = sorted(
        {
            item["target"]
            for item in details
            if item["verified"]
        }
    )

    path = (
        runs_root.parent
        / "verified_episodes.json"
    )

    path.write_text(
        json.dumps(
            {
                "schema": (
                    "phase6.verified_episode_manifest.v4"
                ),
                "dataset_repo_id": repo_id,
                "dataset_root": dataset_root,
                "targets": verified_targets,
                "required_pass_trials": required_passes,
                "policy": (
                    "episode-level QC PASS + physical full replay + "
                    "seamless WRIST stable XY alignment; "
                    "press/SIDE/OCR excluded; aggregated across all "
                    "formal A-Z collection runs"
                ),
                "dataset_episode_indices": verified,
                "episodes": details,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return path

def _qc_candidate_indices(run_dir: Path) -> list[int]:
    candidates_path = run_dir / "qc_candidates.json"
    if not candidates_path.exists():
        raise RuntimeError(
            f"Replay candidate list not found: {candidates_path}. "
            "Run scripts/phase6_episode_qc.py first."
        )
    candidates = [
        int(x)
        for x in _read_json(candidates_path).get("dataset_episode_indices", [])
    ]
    if not candidates:
        raise RuntimeError("QC produced no PASS replay candidates")
    return candidates


def _has_completed_ground_truth(run_dir: Path, dataset_episode_index: int) -> bool:
    """PASS/FAIL completes the first ground-truth trial; INVALID/ABORTED does not."""
    summary_path, _ = _find_episode_summary(run_dir, dataset_episode_index)
    validation_path = summary_path.parent / "takeover_validation.json"
    if not validation_path.exists():
        return False
    payload = _read_json(validation_path)
    for trial in reversed(payload.get("trials", [])):
        status = str(trial.get("status", "UNKNOWN"))
        if status in {"PASS", "FAIL"}:
            return True
        if status in {"INVALID_REPLAY", "ABORTED", "UNKNOWN"}:
            continue
    return False


def _batch_candidates(
    run_dir: Path,
    *,
    episode: int | None,
    include_completed: bool,
) -> list[int]:
    candidates = _qc_candidate_indices(run_dir)

    if episode is not None:
        if episode not in candidates:
            raise RuntimeError(
                f"Episode {episode} is not in the QC-PASS candidate list: {candidates}"
            )
        return [episode]

    if include_completed:
        return candidates

    return [
        idx
        for idx in candidates
        if not _has_completed_ground_truth(run_dir, idx)
    ]


def _verify_home(
    *,
    robot: SO101Follower,
    motor_names: list[str],
    home_baseline: dict[str, float],
    max_diff_deg: float,
) -> float:
    final_goal = robot.bus.sync_read(
        "Goal_Position",
        num_retry=robot.config.num_read_retries,
    )
    final_goal_action = {
        f"{name}.pos": float(final_goal[name])
        for name in motor_names
    }
    final_home_diff = _max_action_delta(
        final_goal_action,
        home_baseline,
        _arm_keys(motor_names),
    )
    if final_home_diff > max_diff_deg:
        raise ReplayInvalid(
            "Automatic recovery ended outside the configured HOME tolerance: "
            f"Goal diff={final_home_diff:.3f}deg; "
            f"limit={max_diff_deg:.3f}deg"
        )
    return float(final_home_diff)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 6E batch physical validation. Offline QC is already episode-batch. "
            "This validator connects once, normalizes to canonical HOME once, then validates "
            "all pending QC-PASS episodes continuously: current pose -> next recorded start -> "
            "full-rate replay -> deterministic WRIST XY takeover. It does NOT return HOME "
            "between episodes. Canonical HOME recovery happens exactly once after the batch."
        )
    )
    parser.add_argument(
        "--target",
        default=None,
        help=(
            "A-Z target used only to locate the latest formal run "
            "when --run-dir is omitted. The authoritative target "
            "is read from run_summary.json."
        ),
    )
    parser.add_argument(
        "--episode",
        type=int,
        help="Debug/repeat exactly one QC-PASS dataset_episode_index.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Re-run all QC-PASS episodes, including episodes that already have PASS/FAIL trials.",
    )
    parser.add_argument(
        "--next",
        action="store_true",
        help=(
            "Backward-compatible alias. v7 is batch mode, so --next means "
            "'validate all remaining QC-PASS episodes', not one episode."
        ),
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--recovery-home", type=Path, default=RECOVERY_HOME_CONFIG, help="Explicit robot recovery HOME pose JSON. Dataset episodes are never used to infer HOME.")
    parser.add_argument("--robot-port", default=ROBOT_PORT)
    parser.add_argument("--restore-step-deg", type=float, default=4.0)
    parser.add_argument("--max-replay-sent-diff-deg", type=float, default=0.25)
    parser.add_argument("--required-passes", type=int, default=1)
    parser.add_argument("--allow-unchecked", action="store_true")
    args = parser.parse_args()

    requested_target = None

    if args.target is not None:
        requested_target = (
            str(args.target)
            .strip()
            .upper()
        )

        if requested_target not in tuple(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ):
            raise ValueError(
                "Formal V1 replay validation currently "
                "supports A-Z targets"
            )

    if args.run_dir is not None:
        run_dir = args.run_dir
    else:
        if requested_target is None:
            raise RuntimeError(
                "Provide --target A-Z or an explicit --run-dir"
            )

        run_dir = _latest_run(
            requested_target
        )

    run_summary_path = (
        run_dir / "run_summary.json"
    )

    if not run_summary_path.exists():
        raise FileNotFoundError(
            run_summary_path
        )

    run_summary = _read_json(
        run_summary_path
    )

    target = (
        str(run_summary["target"])
        .strip()
        .upper()
    )

    if target not in tuple(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ):
        raise ValueError(
            f"Run target is not supported by formal V1: {target}"
        )

    if (
        requested_target is not None
        and requested_target != target
    ):
        raise RuntimeError(
            "--target does not match run_summary.json: "
            f"requested={requested_target}, run={target}"
        )

    if args.episode is not None and args.all:
        raise RuntimeError("--episode and --all are mutually exclusive")

    candidates = _batch_candidates(
        run_dir,
        episode=args.episode,
        include_completed=bool(args.all),
    )

    if not candidates:
        manifest_path = _refresh_verified_manifest(
            run_dir,
            max(
                1,
                int(args.required_passes),
            ),
        )
        print("=" * 72)
        print("PHASE 6E — BATCH VALIDATION v7")
        print("=" * 72)
        print("No remaining QC-PASS episode needs a first valid takeover trial.")
        print("verified manifest :", manifest_path)
        print("Use --all to repeat every QC-PASS episode, or --episode N for one repeat.")
        return

    for idx in candidates:
        if not args.allow_unchecked:
            _require_qc_pass(run_dir, idx)

    for path, label in (
        (WRIST_CAMERA_CONFIG, "WRIST camera config"),
        (GLYPH_MODEL, "glyph model"),
        (TOOL_REFERENCE, "tool reference"),
        (IMAGE_JACOBIAN, "image Jacobian"),
        (URDF, "SO101 URDF"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    home_baseline = _load_recovery_pose(args.recovery_home)

    print("=" * 72)
    print("PHASE 6E — GENERIC BATCH REPLAY -> WRIST TAKEOVER -> FINAL AUTO HOME v7")
    print("=" * 72)
    print("target               :", target)
    print("batch candidates     :", candidates)
    print("episodes this batch  :", len(candidates))
    print("between episodes     : current physical pose -> next recorded start")
    print("inter-episode HOME   : DISABLED")
    print("final HOME           : ONCE, from explicit robot config after entire batch")
    print("human positioning    : NONE; Ctrl+C is emergency stop only")
    print("acceptance endpoint  : stable WRIST XY alignment only")
    print("Z-down / SIDE / OCR  : DISABLED")
    print()

    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.robot_port,
            id=ROBOT_ID,
            use_degrees=True,
            max_relative_target=MAX_RELATIVE_TARGET_DEG,
            cameras={},
        )
    )
    wrist = ThreadedOpenCVCamera(CameraSpec.from_yaml(WRIST_CAMERA_CONFIG))

    motor_names: list[str] = []
    batch_records: list[dict] = []
    recovery_ok = False
    aborted = False

    try:
        robot.connect()
        motor_names = list(robot.bus.motors.keys())
        if len(motor_names) != 6:
            raise RuntimeError(f"Expected 6 motors, got {motor_names}")
        missing_home = [key for key in _arm_keys(motor_names) if key not in home_baseline]
        if missing_home:
            raise RuntimeError(
                f"Recovery HOME config is missing motor keys {missing_home}: {args.recovery_home}"
            )

        # One startup normalization only. This establishes a deterministic and
        # safe batch origin. No HOME reset occurs between validated episodes.
        print()
        print("===== BATCH START — NORMALIZE TO EXPLICIT RECOVERY HOME ONCE =====")
        startup_hold = _synchronize_goal_to_present(
            robot=robot,
            motor_names=motor_names,
            max_sent_diff_deg=float(args.max_replay_sent_diff_deg),
        )
        startup_home = _auto_restore_recorded_start(
            robot=robot,
            baseline=home_baseline,
            motor_names=motor_names,
            step_deg=min(2.0, float(args.restore_step_deg)),
            max_sent_diff_deg=float(args.max_replay_sent_diff_deg),
            label="RECOVERY HOME",
        )

        wrist.start()
        deadline = time.monotonic() + 5.0
        while wrist.latest() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if wrist.latest() is None:
            raise RuntimeError("No WRIST frame arrived")

        for batch_index, episode_index in enumerate(candidates, start=1):
            summary_path, summary = _find_episode_summary(run_dir, episode_index)
            episode_dir = summary_path.parent
            replay_path = Path(
                summary.get("replay_trace_jsonl")
                or episode_dir / "replay_trace.jsonl"
            )
            trace = _read_jsonl(replay_path)
            baseline = {
                str(k): float(v)
                for k, v in summary["arm_baseline_sent_action"].items()
            }

            print()
            print("=" * 72)
            print(
                f"BATCH EPISODE {batch_index}/{len(candidates)} — "
                f"dataset_ep={episode_index}"
            )
            print("=" * 72)
            print("replay trace         :", replay_path)
            print("replay samples       :", len(trace))
            print("return HOME after it : NO")
            print(
                "next transition      : current physical pose -> "
                "next episode start"
                if batch_index < len(candidates)
                else "next transition      : final batch HOME"
            )

            record = {
                "trial_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "batch_mode": True,
                "batch_index": int(batch_index),
                "batch_size": int(len(candidates)),
                "dataset_episode_index": int(episode_index),
                "episode_number": int(summary["episode_number"]),
                "target": target,
                "status": "UNKNOWN",
            }
            if batch_index == 1:
                record["batch_startup_present_hold"] = startup_hold
                record["batch_startup_home_normalize"] = startup_home

            try:
                # After the previous takeover, Goal and Present may differ due to
                # normal SO101 lag/backlash. For inter-episode transport only,
                # establish a hold at the *current physical pose*. This does not
                # return HOME and does not become a visual-servo precision anchor.
                if batch_index > 1:
                    print()
                    print("===== INTER-EPISODE HANDOFF =====")
                    print("HOME reset      : NO")
                    print("transport origin: current physical Present_Position")
                    print("destination     : next recorded episode start")
                    record["inter_episode_present_hold"] = _synchronize_goal_to_present(
                        robot=robot,
                        motor_names=motor_names,
                        max_sent_diff_deg=float(args.max_replay_sent_diff_deg),
                    )

                record["start_restore"] = _auto_restore_recorded_start(
                    robot=robot,
                    baseline=baseline,
                    motor_names=motor_names,
                    step_deg=float(args.restore_step_deg),
                    max_sent_diff_deg=float(args.max_replay_sent_diff_deg),
                    label=f"EPISODE {episode_index} START",
                )

                replay_stats = _replay_trace(
                    robot=robot,
                    trace=trace,
                    motor_names=motor_names,
                    max_sent_diff_deg=float(args.max_replay_sent_diff_deg),
                )
                record["replay"] = replay_stats

                record["takeover"] = _run_takeover(
                    robot=robot,
                    wrist_camera=wrist,
                    target=target,
                    motor_names=motor_names,
                    endpoint_settled_timestamp=float(
                        replay_stats["endpoint_settled_timestamp"]
                    ),
                    episode_dir=episode_dir,
                )
                record["status"] = "PASS"

            except ReplayInvalid as exc:
                record["status"] = "INVALID_REPLAY"
                record["error"] = str(exc)
                print("\n[REPLAY INVALID]", exc)
            except KeyboardInterrupt:
                record["status"] = "ABORTED"
                record["error"] = "operator KeyboardInterrupt"
                aborted = True
                print("\n[ABORTED] Operator interrupted batch validation.")
            except Exception as exc:
                record["status"] = "FAIL"
                record["error"] = f"{type(exc).__name__}: {exc}"
                print("\n[TAKEOVER FAIL]", record["error"])
            finally:
                validation_path = _append_validation(episode_dir, record)
                batch_records.append(record)
                manifest_path = _refresh_verified_manifest(
                    run_dir,
                    target,
                    max(1, int(args.required_passes)),
                )
                print("validation record :", validation_path)
                print("verified manifest :", manifest_path)
                print(
                    f"[BATCH RESULT] dataset_ep={episode_index} "
                    f"status={record['status']}"
                )

            if aborted:
                break

            # No HOME here. The next loop iteration synchronizes a hold at the
            # current physical pose and transports directly to the next episode start.
            if batch_index < len(candidates):
                print()
                print(
                    "[CONTINUE] No HOME reset. "
                    "Proceeding automatically to the next episode start."
                )

    except KeyboardInterrupt:
        aborted = True
        print("\n[ABORTED] Operator interrupted batch validation.")
    finally:
        if robot.is_connected and motor_names:
            if aborted:
                _emergency_hold_until_operator(
                    robot,
                    "Emergency abort occurred before normal final HOME recovery.",
                )
                recovery_ok = True
            else:
                try:
                    # Exactly one HOME recovery, after the complete batch.
                    print()
                    print("=" * 72)
                    print("BATCH COMPLETE — FINAL AUTOMATIC RETURN TO HOME")
                    print("=" * 72)
                    print("inter-episode HOME resets : 0")
                    print("final HOME recovery       : 1")
                    print(
                        "recovery path             : current physical pose "
                        "-> bounded canonical HOME"
                    )
                    print("operator input            : NONE")

                    final_hold = _synchronize_goal_to_present(
                        robot=robot,
                        motor_names=motor_names,
                        max_sent_diff_deg=float(args.max_replay_sent_diff_deg),
                    )
                    final_home = _auto_restore_recorded_start(
                        robot=robot,
                        baseline=home_baseline,
                        motor_names=motor_names,
                        step_deg=min(2.0, float(args.restore_step_deg)),
                        max_sent_diff_deg=float(args.max_replay_sent_diff_deg),
                        label="RECOVERY HOME",
                    )
                    final_home_diff = _verify_home(
                        robot=robot,
                        motor_names=motor_names,
                        home_baseline=home_baseline,
                        max_diff_deg=float(args.max_replay_sent_diff_deg),
                    )
                    print(
                        f"[HOME RESTORED] final Goal diff={final_home_diff:.3f}deg; "
                        "motion stable. Safe disconnect is now allowed."
                    )

                    batch_summary_path = run_dir / "takeover_batch_summary.json"
                    batch_summary_path.write_text(
                        json.dumps(
                            {
                                "schema": "phase6.takeover_batch_summary.v2",
                                "target": target,
                                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                "candidate_indices": candidates,
                                "results": [
                                    {
                                        "dataset_episode_index": int(
                                            r["dataset_episode_index"]
                                        ),
                                        "episode_number": int(r["episode_number"]),
                                        "status": r["status"],
                                        "error": r.get("error"),
                                    }
                                    for r in batch_records
                                ],
                                "final_recovery": {
                                    "present_hold": final_hold,
                                    "home_restore": final_home,
                                    "final_home_diff_deg": final_home_diff,
                                },
                            },
                            indent=2,
                            ensure_ascii=False,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    print("batch summary       :", batch_summary_path)
                    recovery_ok = True
                except Exception as recovery_exc:
                    print(
                        "\n[FINAL RECOVERY FAILURE] "
                        f"{type(recovery_exc).__name__}: {recovery_exc}"
                    )
                    _emergency_hold_until_operator(
                        robot,
                        f"{type(recovery_exc).__name__}: {recovery_exc}",
                    )
                    recovery_ok = True

        wrist.stop()
        if robot.is_connected and recovery_ok:
            robot.disconnect()

    print()
    print("=" * 72)
    print("BATCH VALIDATION SUMMARY")
    print("=" * 72)
    counts: dict[str, int] = {}
    for record in batch_records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
        print(
            f"dataset_ep={record['dataset_episode_index']:03d} "
            f"status={record['status']}"
        )
    print("counts:", counts)

    manifest_path = _refresh_verified_manifest(
        run_dir,
        target,
        max(1, int(args.required_passes)),
    )
    print("verified manifest:", manifest_path)

    if aborted or any(r["status"] != "PASS" for r in batch_records):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
