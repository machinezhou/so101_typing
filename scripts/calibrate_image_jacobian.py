from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from threading import Event, Thread

import cv2
import numpy as np

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.constants import HF_LEROBOT_HOME

from so101_typing.adapters.cameras import CameraSpec, ThreadedOpenCVCamera
from so101_typing.control.cartesian_kinematics import (
    MOTION_FRAME,
    MOTION_UNIT,
    build_official_cartesian_pipeline,
    build_so101_kinematics,
    end_effector_xyz_mm,
    make_cartesian_delta_action,
    ordered_joint_observation,
)
from so101_typing.control.image_jacobian import ImageJacobianCalibration
from so101_typing.control.target_measurement import (
    TargetBurstConfig,
    robust_target_center,
)
from so101_typing.control.tool_reference import ToolReferenceCalibration
from so101_typing.perception.glyph_runtime import RuntimeHOGGlyphRecognizer
from so101_typing.perception.wrist_target import observe_target


DEFAULT_CAMERA_CONFIG = Path("configs/cameras/wrist.yaml")
DEFAULT_MODEL_DIR = Path("artifacts/models/glyph_hog_svm")
DEFAULT_TOOL_REFERENCE = Path("calibration/tool_reference.json")
DEFAULT_ARTIFACT_DIR = Path("artifacts/image_jacobian_calibration")
DEFAULT_CANDIDATE_OUTPUT = DEFAULT_ARTIFACT_DIR / "candidate_image_jacobian.json"
DEFAULT_URDF = HF_LEROBOT_HOME / "robot-urdfs" / "so101" / "so101_new_calib.urdf"

# Phase-3 calibration deliberately uses visible, centimetre-scale motion rather
# than pretending the SO-101 is a sub-millimetre open-loop positioning system.
DEFAULT_STEP_MM = 10.0
DEFAULT_PROBE_STEP_MM = 5.0
DEFAULT_CYCLES = 2
DEFAULT_MAX_DELTA_NORM_MM = 12.0
DEFAULT_MAX_EE_STEP_M = 0.015
DEFAULT_MAX_RELATIVE_TARGET_DEG = 10.0
DEFAULT_MAX_CONDITION_NUMBER = 20.0
DEFAULT_BURST_FRAMES = 5
DEFAULT_BURST_MIN_INLIERS = 4
DEFAULT_BURST_CLUSTER_RADIUS_PX = 5.0
DEFAULT_POST_SETTLE_GUARD_MS = 100.0
DEFAULT_TELEOP_HZ = 60.0
DEFAULT_MAX_MODEL_XY_ERROR_MM = 1.0
DEFAULT_MAX_MODEL_Z_ERROR_MM = 0.5
DEFAULT_MAX_MEASURED_XY_ERROR_MM = 2.0
DEFAULT_MAX_MEASURED_Z_ERROR_MM = 1.0
DEFAULT_MAX_RESIDUAL_RMS_PX = 5.0
DEFAULT_MAX_OPPOSITION_COSINE = -0.5
DEFAULT_MAX_ZERO_JOINT_SHIFT_DEG = 0.5
DEFAULT_CONDITIONING_MIN_CYCLES = 2
DEFAULT_CONDITIONING_MAX_CYCLES = 4
DEFAULT_CONDITIONING_MAX_BEFORE_DRIFT_PX = 10.0
DEFAULT_CONDITIONING_MAX_RESPONSE_DRIFT_PX = 8.0


def _finite_positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return value


def _wait_for_initial_frame(camera: ThreadedOpenCVCamera, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if camera.latest() is not None:
            return
        time.sleep(0.01)
    raise RuntimeError("No initial WRIST frame arrived before timeout")


def _run_positioning_teleop(
    robot: SO101Follower,
    leader: SO101Leader,
    *,
    teleop_hz: float,
) -> None:
    """Keep follower/leader connected and teleoperate until the operator presses Enter."""
    ready = Event()

    def _wait_for_enter() -> None:
        try:
            input()
        except EOFError:
            return
        ready.set()

    print()
    print("===== MANUAL POSITIONING =====")
    print(
        "Teleoperation is live inside this process; "
        "the follower will NOT disconnect at handoff."
    )
    print("Move to the perception-safe hover with target visible and keep delta-Z clearance.")
    print("Press ENTER once the pose is ready. Ctrl+C aborts autonomous work and enters recovery.")
    Thread(target=_wait_for_enter, name="phase3-position-ready", daemon=True).start()

    period_s = 1.0 / teleop_hz
    while not ready.is_set():
        started = time.monotonic()
        robot.send_action(leader.get_action())
        time.sleep(max(0.0, period_s - (time.monotonic() - started)))

    # One final mirror action makes the handoff pose explicit before autonomy starts.
    robot.send_action(leader.get_action())
    print("[POSITIONED] Manual teleoperation paused; follower remains connected and holding pose.")


def _run_recovery_teleop(
    robot: SO101Follower,
    leader: SO101Leader,
    *,
    teleop_hz: float,
) -> None:
    """Operator-controlled recovery. Ctrl+C exits only after the operator returns home."""
    print()
    print("===== OPERATOR RECOVERY =====")
    print("Teleoperation is live again. Return the follower to the normal zero/home pose.")
    print("When safely at zero/home, press Ctrl+C to disconnect using normal LeRobot torque-off.")
    period_s = 1.0 / teleop_hz
    try:
        while True:
            started = time.monotonic()
            robot.send_action(leader.get_action())
            time.sleep(max(0.0, period_s - (time.monotonic() - started)))
    except KeyboardInterrupt:
        print("\n[RECOVERY COMPLETE] Disconnecting at operator-selected zero/home pose.")


def _capture_target_center(
    camera: ThreadedOpenCVCamera,
    recognizer: RuntimeHOGGlyphRecognizer,
    target_label: str,
    *,
    frame_count: int,
    max_frame_age_ms: float,
    timeout_s: float,
    after_frame_id: int | None,
    min_capture_timestamp: float | None,
    burst_config: TargetBurstConfig,
) -> tuple[np.ndarray, list[dict], np.ndarray, int]:
    centers: list[tuple[float, float]] = []
    records: list[dict] = []
    last_image: np.ndarray | None = None
    last_frame_id = -1 if after_frame_id is None else int(after_frame_id)
    deadline = time.monotonic() + timeout_s

    while len(centers) < frame_count and time.monotonic() < deadline:
        frame = camera.latest(copy_image=True)
        if frame is None or frame.frame_id <= last_frame_id:
            time.sleep(0.005)
            continue

        last_frame_id = int(frame.frame_id)
        if (
            min_capture_timestamp is not None
            and float(frame.capture_timestamp) <= min_capture_timestamp
        ):
            continue
        now = time.monotonic()
        age_ms = max(
            float(frame.frame_age_ms),
            max(0.0, (now - float(frame.capture_timestamp)) * 1000.0),
        )
        if age_ms > max_frame_age_ms:
            continue

        observation = observe_target(frame.image, target_label, recognizer)
        if not observation.found or observation.center_px is None:
            continue

        center = (
            float(observation.center_px[0]),
            float(observation.center_px[1]),
        )
        centers.append(center)
        records.append(
            {
                "frame_id": int(frame.frame_id),
                "capture_timestamp": float(frame.capture_timestamp),
                "frame_age_ms": float(age_ms),
                "center_px": [center[0], center[1]],
                "vote_fraction": float(observation.vote_fraction),
                "similarity": float(observation.similarity),
                "similarity_threshold": float(observation.similarity_threshold),
                "quality_score": float(observation.quality_score),
            }
        )
        last_image = frame.image

    if len(centers) < frame_count or last_image is None:
        raise RuntimeError(
            f"Only acquired {len(centers)}/{frame_count} accepted fresh observations "
            f"of target {target_label!r} within {timeout_s:.1f}s"
        )

    array = np.asarray(centers, dtype=np.float64)
    burst = robust_target_center(array, config=burst_config)
    inlier_set = set(burst.inlier_indices)
    for index, record in enumerate(records):
        record["burst_inlier"] = index in inlier_set

    if not burst.accepted or burst.center_px is None:
        raise RuntimeError(
            "Target burst did not contain a stable image-space consensus: "
            f"inliers={burst.inlier_count}/{burst.total_count}, "
            f"cluster_radius_px={burst_config.cluster_radius_px:.1f}"
        )

    center = np.asarray(burst.center_px, dtype=np.float64)
    return center, records, last_image, last_frame_id


def _joint_positions(observation: dict) -> dict[str, float]:
    return {
        key.removesuffix(".pos"): float(value)
        for key, value in observation.items()
        if isinstance(key, str) and key.endswith(".pos")
    }



def _wait_until_settled(
    robot: SO101Follower,
    sent_action: dict,
    *,
    tolerance_deg: float,
    stable_reads: int,
    timeout_s: float,
) -> tuple[dict, float]:
    """Wait for measured joint motion to stabilize.

    SETTLE_GATE_V2_MOTION_STABILITY

    `tolerance_deg` is retained for CLI/backward compatibility and is printed
    for diagnostics, but it no longer defines "settled". The SO-101 can hold
    a repeatable non-zero servo residual, so settling is detected from a short
    rolling window of measured joint positions.

    Safety remains fail-closed: a motion-stable pose whose largest target
    residual exceeds 5 deg is rejected.
    """
    target = {
        key.removesuffix(".pos"): float(value)
        for key, value in sent_action.items()
        if isinstance(key, str)
        and key.endswith(".pos")
        and key != "gripper.pos"
    }
    if not target:
        raise RuntimeError("No arm joint targets were produced by Cartesian pipeline")

    # Phase-3 hardware gate.  Five reads at the existing ~30 ms cadence gives
    # a short stability window.  A 0.20 deg span tolerates encoder quantization
    # while still rejecting visibly moving joints.
    min_wait_s = 0.25
    motion_window_span_deg = 0.20
    max_target_error_deg = 5.0

    started = time.monotonic()
    deadline = started + timeout_s
    window: list[tuple[dict, dict[str, float]]] = []
    last_obs: dict | None = None
    last_max_error = float("inf")
    last_max_error_joint = "unknown"
    last_window_span = float("inf")

    while time.monotonic() < deadline:
        obs = robot.get_observation()
        last_obs = obs
        present = _joint_positions(obs)
        if not all(name in present for name in target):
            missing = sorted(set(target) - set(present))
            raise RuntimeError(f"Robot observation is missing joints: {missing}")

        errors = {name: abs(present[name] - value) for name, value in target.items()}
        max_error_joint = max(errors, key=errors.get)
        max_error = errors[max_error_joint]
        last_max_error = max_error
        last_max_error_joint = max_error_joint

        window.append((obs, present))
        if len(window) > stable_reads:
            window.pop(0)

        elapsed = time.monotonic() - started
        if elapsed >= min_wait_s and len(window) == stable_reads:
            spans = {
                name: max(item[1][name] for item in window)
                - min(item[1][name] for item in window)
                for name in target
            }
            max_span_joint = max(spans, key=spans.get)
            max_span = spans[max_span_joint]
            last_window_span = max_span

            if max_span <= motion_window_span_deg:
                if max_error > max_target_error_deg:
                    raise RuntimeError(
                        "Robot stopped moving but settled too far from the sent target: "
                        f"max target error={max_error:.3f} deg at {max_error_joint}; "
                        f"safety cap={max_target_error_deg:.3f} deg"
                    )
                settled_timestamp = time.monotonic()
                print(
                    "[SETTLED] mode=motion-stable "
                    f"elapsed={elapsed:.3f}s "
                    f"max_target_error={max_error:.3f}deg "
                    f"joint={max_error_joint} "
                    f"window_span={max_span:.3f}deg "
                    f"span_joint={max_span_joint} "
                    f"legacy_target_tolerance={tolerance_deg:.3f}deg"
                )
                return obs, settled_timestamp

        time.sleep(0.03)

    if last_obs is None:
        raise RuntimeError("No robot observation received while waiting for settle")
    raise RuntimeError(
        f"Robot motion did not stabilize within {timeout_s:.1f}s; "
        f"last max target error={last_max_error:.3f} deg at {last_max_error_joint}; "
        f"last {stable_reads}-read window span={last_window_span:.3f} deg "
        f"(required <= {motion_window_span_deg:.3f} deg)"
    )

def _draw_preview(
    image: np.ndarray,
    *,
    target_label: str,
    before: np.ndarray,
    after: np.ndarray,
    motion: tuple[float, float],
    sample_index: int,
) -> np.ndarray:
    preview = image.copy()
    b = tuple(int(v) for v in np.rint(before))
    a = tuple(int(v) for v in np.rint(after))
    cv2.drawMarker(preview, b, (0, 255, 255), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
    cv2.drawMarker(preview, a, (0, 0, 255), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
    cv2.arrowedLine(preview, b, a, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.15)
    cv2.putText(
        preview,
        (
            f"sample={sample_index} target={target_label} "
            f"dXY=({motion[0]:+.1f},{motion[1]:+.1f}) mm"
        ),
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return preview


def _validate_assets(args: argparse.Namespace) -> tuple[
    CameraSpec,
    RuntimeHOGGlyphRecognizer,
    ToolReferenceCalibration,
]:
    for path, label in (
        (args.camera_config, "WRIST camera config"),
        (args.model_dir, "glyph model directory"),
        (args.tool_reference, "tool reference calibration"),
        (args.urdf, "SO-101 URDF"),
    ):
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    spec = CameraSpec.from_yaml(args.camera_config)
    if spec.name != "wrist":
        raise ValueError(f"Expected WRIST camera config, got {spec.name!r}")

    recognizer = RuntimeHOGGlyphRecognizer.load(args.model_dir)
    tool_reference = ToolReferenceCalibration.load(args.tool_reference)
    if tool_reference is None:
        raise RuntimeError("tool_reference.json is still uncalibrated")
    if tool_reference.camera_name != spec.name:
        raise ValueError(
            f"Tool reference camera {tool_reference.camera_name!r} != {spec.name!r}"
        )
    if tuple(tool_reference.image_size) != (spec.width, spec.height):
        raise ValueError(
            "Tool reference image size does not match WRIST config: "
            f"{tool_reference.image_size} != {(spec.width, spec.height)}"
        )

    return spec, recognizer, tool_reference


def _build_pipeline(args: argparse.Namespace, motor_names: list[str]):
    return build_official_cartesian_pipeline(
        args.urdf,
        motor_names=motor_names,
        end_effector_bounds={
            "min": [-1.0, -1.0, -1.0],
            "max": [1.0, 1.0, 1.0],
        },
        max_ee_step_m=float(args.max_ee_step_m),
        orientation_weight=float(args.orientation_weight),
        raise_on_jump=True,
        # H3.2 is a perturbation experiment around ONE operating point.
        # Never rebase +/- samples on the previous sample's measured pose.
        use_latched_reference=True,
    )


def _plan_delta(
    pipeline,
    observation: dict,
    dx_mm: float,
    dy_mm: float,
    *,
    motor_names: list[str],
    validation_kinematics,
    anchor_xyz_mm: np.ndarray,
    max_delta_norm_mm: float,
    max_model_xy_error_mm: float,
    max_model_z_error_mm: float,
) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    """Plan one fixed-anchor Cartesian target and verify it before send_action."""
    ordered_observation = ordered_joint_observation(observation, motor_names)
    delta_action = make_cartesian_delta_action(
        dx_mm,
        dy_mm,
        max_delta_norm_mm=max_delta_norm_mm,
    )
    joint_action = pipeline((delta_action, ordered_observation))

    predicted_xyz_mm = end_effector_xyz_mm(
        validation_kinematics,
        joint_action,
        motor_names,
    )
    predicted_delta_mm = predicted_xyz_mm - anchor_xyz_mm
    expected_delta_mm = np.asarray([dx_mm, dy_mm, 0.0], dtype=np.float64)
    model_error_mm = predicted_delta_mm - expected_delta_mm
    model_xy_error_mm = float(np.linalg.norm(model_error_mm[:2]))
    model_z_error_mm = abs(float(model_error_mm[2]))

    print(
        "[PLAN] "
        f"request=({dx_mm:+.2f},{dy_mm:+.2f},+0.00)mm "
        f"fk=({predicted_delta_mm[0]:+.2f},{predicted_delta_mm[1]:+.2f},"
        f"{predicted_delta_mm[2]:+.2f})mm "
        f"error_xy={model_xy_error_mm:.2f}mm error_z={model_z_error_mm:.2f}mm"
    )

    if model_xy_error_mm > max_model_xy_error_mm:
        raise RuntimeError(
            "Refusing send_action: project-side FK of the planned joint target "
            f"misses requested XY by {model_xy_error_mm:.3f} mm "
            f"(limit {max_model_xy_error_mm:.3f} mm)"
        )
    if model_z_error_mm > max_model_z_error_mm:
        raise RuntimeError(
            "Refusing send_action: project-side FK of the planned joint target "
            f"changes Z by {predicted_delta_mm[2]:+.3f} mm relative to the anchor "
            f"(limit {max_model_z_error_mm:.3f} mm)"
        )

    diagnostics = {
        "anchor_xyz_mm": anchor_xyz_mm.tolist(),
        "predicted_xyz_mm": predicted_xyz_mm.tolist(),
        "predicted_delta_from_anchor_mm": predicted_delta_mm.tolist(),
        "expected_delta_from_anchor_mm": expected_delta_mm.tolist(),
        "model_error_mm": model_error_mm.tolist(),
        "model_xy_error_mm": model_xy_error_mm,
        "model_z_error_mm": model_z_error_mm,
    }
    return joint_action, diagnostics, predicted_xyz_mm, predicted_delta_mm


def _command_delta(
    robot: SO101Follower,
    pipeline,
    observation: dict,
    dx_mm: float,
    dy_mm: float,
    *,
    motor_names: list[str],
    validation_kinematics,
    anchor_xyz_mm: np.ndarray,
    max_delta_norm_mm: float,
    max_model_xy_error_mm: float,
    max_model_z_error_mm: float,
    max_measured_xy_error_mm: float,
    max_measured_z_error_mm: float,
    settle_tolerance_deg: float,
    settle_stable_reads: int,
    settle_timeout_s: float,
) -> tuple[dict, dict, dict, float, dict]:
    joint_action, diagnostics, _, _ = _plan_delta(
        pipeline,
        observation,
        dx_mm,
        dy_mm,
        motor_names=motor_names,
        validation_kinematics=validation_kinematics,
        anchor_xyz_mm=anchor_xyz_mm,
        max_delta_norm_mm=max_delta_norm_mm,
        max_model_xy_error_mm=max_model_xy_error_mm,
        max_model_z_error_mm=max_model_z_error_mm,
    )

    # Nothing reaches hardware until the project-side FK check above passes.
    sent_action = robot.send_action(joint_action)

    clipped = {
        key: (float(joint_action[key]), float(sent_action[key]))
        for key in joint_action
        if key.endswith(".pos")
        and key in sent_action
        and key != "gripper.pos"
        and abs(float(joint_action[key]) - float(sent_action[key])) > 1e-6
    }
    if clipped:
        details = ", ".join(
            f"{key}: requested={requested:.3f}, sent={sent:.3f}"
            for key, (requested, sent) in clipped.items()
        )
        raise RuntimeError(
            "Joint target was clipped; do not use this sample. "
            f"{details}"
        )

    settled_obs, settled_timestamp = _wait_until_settled(
        robot,
        sent_action,
        tolerance_deg=settle_tolerance_deg,
        stable_reads=settle_stable_reads,
        timeout_s=settle_timeout_s,
    )
    settled_obs = ordered_joint_observation(settled_obs, motor_names)
    measured_xyz_mm = end_effector_xyz_mm(
        validation_kinematics,
        settled_obs,
        motor_names,
    )
    measured_delta_mm = measured_xyz_mm - anchor_xyz_mm
    expected_delta_mm = np.asarray([dx_mm, dy_mm, 0.0], dtype=np.float64)
    measured_error_mm = measured_delta_mm - expected_delta_mm
    measured_xy_error_mm = float(np.linalg.norm(measured_error_mm[:2]))
    measured_z_error_mm = abs(float(measured_error_mm[2]))

    diagnostics.update(
        {
            "measured_xyz_mm": measured_xyz_mm.tolist(),
            "measured_delta_from_anchor_mm": measured_delta_mm.tolist(),
            "measured_error_mm": measured_error_mm.tolist(),
            "measured_xy_error_mm": measured_xy_error_mm,
            "measured_z_error_mm": measured_z_error_mm,
        }
    )
    print(
        "[MEASURED-FK] "
        f"delta=({measured_delta_mm[0]:+.2f},{measured_delta_mm[1]:+.2f},"
        f"{measured_delta_mm[2]:+.2f})mm "
        f"error_xy={measured_xy_error_mm:.2f}mm error_z={measured_z_error_mm:.2f}mm"
    )

    # H3.2 calibrates command-space image response:
    # requested Cartesian XY command -> observed WRIST pixel delta.
    # Measured-joint FK is diagnostic only: the known steady servo residual,
    # backlash and compliance must not be reinterpreted as a millimetre-accuracy
    # acceptance gate. Hardware safety is still fail-closed before/around motion
    # via command norm, EE-step/model-FK checks, joint clipping/slew limits and
    # the motion-stability settle gate.
    if (
        measured_z_error_mm > max_measured_z_error_mm
        or measured_xy_error_mm > max_measured_xy_error_mm
    ):
        print(
            "[MEASURED-FK NOTE] diagnostic only; not an H3.2 acceptance gate "
            f"(reference thresholds: XY {max_measured_xy_error_mm:.2f} mm, "
            f"Z {max_measured_z_error_mm:.2f} mm)"
        )

    return joint_action, sent_action, settled_obs, settled_timestamp, diagnostics


def _probe_motion(axis: str, step_mm: float) -> tuple[float, float]:
    motions = {
        "+x": (step_mm, 0.0),
        "-x": (-step_mm, 0.0),
        "+y": (0.0, step_mm),
        "-y": (0.0, -step_mm),
    }
    try:
        return motions[axis]
    except KeyError as exc:
        raise ValueError(f"unsupported probe axis: {axis!r}") from exc


def _direction_consistency(
    motions: list[list[float]],
    image_deltas: list[list[float]],
) -> dict[str, dict[str, object]]:
    """Report whether positive/negative commands produce opposing image motion."""
    motion_array = np.asarray(motions, dtype=np.float64)
    image_array = np.asarray(image_deltas, dtype=np.float64)
    report: dict[str, dict[str, object]] = {}

    for axis_name, axis_index in (("x", 0), ("y", 1)):
        positive = image_array[motion_array[:, axis_index] > 0.0]
        negative = image_array[motion_array[:, axis_index] < 0.0]
        positive_mean = np.mean(positive, axis=0)
        negative_mean = np.mean(negative, axis=0)
        positive_norm = float(np.linalg.norm(positive_mean))
        negative_norm = float(np.linalg.norm(negative_mean))
        denominator = positive_norm * negative_norm
        cosine = None
        if denominator > 0.0:
            cosine = float(np.dot(positive_mean, negative_mean) / denominator)

        report[axis_name] = {
            "positive_mean_image_delta_px": positive_mean.tolist(),
            "negative_mean_image_delta_px": negative_mean.tolist(),
            "positive_mean_norm_px": positive_norm,
            "negative_mean_norm_px": negative_norm,
            "opposition_cosine": cosine,
        }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase-3 H3.2: automatically estimate the local WRIST image Jacobian "
            "using LeRobot's official Cartesian processor pipeline. No Z motion and no key press."
        )
    )
    parser.add_argument("--robot-port", default=None)
    parser.add_argument("--robot-id", default="lawson_follower_arm")
    parser.add_argument("--leader-port", default="/dev/ttyACM1")
    parser.add_argument("--leader-id", default="lawson_leader_arm")
    parser.add_argument("--teleop-hz", type=float, default=DEFAULT_TELEOP_HZ)
    parser.add_argument("--target", default="G")
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--tool-reference", type=Path, default=DEFAULT_TOOL_REFERENCE)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CANDIDATE_OUTPUT,
        help=(
            "Candidate Jacobian output. This script deliberately does not overwrite "
            "calibration/image_jacobian.json; promote only after reviewing the session."
        ),
    )
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--step-mm", type=float, default=DEFAULT_STEP_MM)
    parser.add_argument(
        "--probe-axis",
        choices=("+x", "-x", "+y", "-y"),
        default=None,
        help="Run exactly one controlled XY probe and exit without fitting/saving a Jacobian.",
    )
    parser.add_argument(
        "--probe-step-mm",
        type=float,
        default=DEFAULT_PROBE_STEP_MM,
        help="Requested Cartesian command magnitude used only with --probe-axis.",
    )
    parser.add_argument("--cycles", type=int, default=DEFAULT_CYCLES)
    parser.add_argument(
        "--conditioning-min-cycles",
        type=int,
        default=DEFAULT_CONDITIONING_MIN_CYCLES,
        help="Minimum full +X/-X/+Y/-Y conditioning cycles before readiness can pass.",
    )
    parser.add_argument(
        "--conditioning-max-cycles",
        type=int,
        default=DEFAULT_CONDITIONING_MAX_CYCLES,
        help="Maximum conditioning cycles before failing closed without formal samples.",
    )
    parser.add_argument(
        "--conditioning-max-before-drift-px",
        type=float,
        default=DEFAULT_CONDITIONING_MAX_BEFORE_DRIFT_PX,
        help="Maximum same-phase BEFORE-center drift between adjacent conditioning cycles.",
    )
    parser.add_argument(
        "--conditioning-max-response-drift-px",
        type=float,
        default=DEFAULT_CONDITIONING_MAX_RESPONSE_DRIFT_PX,
        help="Maximum same-phase dUV drift between adjacent conditioning cycles.",
    )
    parser.add_argument("--burst-frames", type=int, default=DEFAULT_BURST_FRAMES)
    parser.add_argument(
        "--burst-min-inliers",
        type=int,
        default=DEFAULT_BURST_MIN_INLIERS,
    )
    parser.add_argument(
        "--burst-cluster-radius-px",
        type=float,
        default=DEFAULT_BURST_CLUSTER_RADIUS_PX,
    )
    parser.add_argument("--max-frame-age-ms", type=float, default=100.0)
    parser.add_argument("--capture-timeout-s", type=float, default=5.0)
    parser.add_argument("--initial-timeout-s", type=float, default=5.0)
    parser.add_argument("--settle-tolerance-deg", type=float, default=1.5)
    parser.add_argument("--settle-stable-reads", type=int, default=3)
    parser.add_argument("--settle-timeout-s", type=float, default=3.0)
    parser.add_argument(
        "--post-settle-guard-ms",
        type=float,
        default=DEFAULT_POST_SETTLE_GUARD_MS,
        help=(
            "Require after-frames to have capture timestamps later than settle completion "
            "plus this guard interval."
        ),
    )
    parser.add_argument(
        "--max-relative-target-deg",
        type=float,
        default=DEFAULT_MAX_RELATIVE_TARGET_DEG,
    )
    parser.add_argument(
        "--max-delta-norm-mm",
        type=float,
        default=DEFAULT_MAX_DELTA_NORM_MM,
    )
    parser.add_argument("--max-ee-step-m", type=float, default=DEFAULT_MAX_EE_STEP_M)
    parser.add_argument("--orientation-weight", type=float, default=0.01)
    parser.add_argument(
        "--max-zero-joint-shift-deg",
        type=float,
        default=DEFAULT_MAX_ZERO_JOINT_SHIFT_DEG,
        help=(
            "Fail before any autonomous motion if a zero Cartesian delta "
            "changes planned arm joints."
        ),
    )
    parser.add_argument(
        "--max-model-xy-error-mm",
        type=float,
        default=DEFAULT_MAX_MODEL_XY_ERROR_MM,
        help="Fail before send_action if FK(planned joints) misses requested anchor-relative XY.",
    )
    parser.add_argument(
        "--max-model-z-error-mm",
        type=float,
        default=DEFAULT_MAX_MODEL_Z_ERROR_MM,
        help="Fail before send_action if FK(planned joints) leaves the fixed anchor Z plane.",
    )
    parser.add_argument(
        "--max-measured-xy-error-mm",
        type=float,
        default=DEFAULT_MAX_MEASURED_XY_ERROR_MM,
        help="Diagnostic reference only; FK(measured joints) is not an H3.2 accuracy gate.",
    )
    parser.add_argument(
        "--max-measured-z-error-mm",
        type=float,
        default=DEFAULT_MAX_MEASURED_Z_ERROR_MM,
        help="Diagnostic reference only; FK(measured joints) is not an H3.2 accuracy gate.",
    )
    parser.add_argument(
        "--max-residual-rms-px",
        type=float,
        default=DEFAULT_MAX_RESIDUAL_RMS_PX,
    )
    parser.add_argument(
        "--max-opposition-cosine",
        type=float,
        default=DEFAULT_MAX_OPPOSITION_COSINE,
        help="Both +/- axis opposition cosines must be <= this value (ideal -1).",
    )
    parser.add_argument(
        "--max-condition-number",
        type=float,
        default=DEFAULT_MAX_CONDITION_NUMBER,
    )
    parser.add_argument("--damping", type=float, default=1e-6)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help=(
            "After manual positioning, compute fixed-anchor zero/+X/-X/+Y/-Y joint targets "
            "and FK checks but never call robot.send_action for Cartesian motion."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate software/assets and construct the official pipeline "
            "without camera or robot I/O."
        ),
    )
    args = parser.parse_args()

    target_label = args.target.strip().upper()
    if len(target_label) != 1 or target_label < "A" or target_label > "Z":
        raise ValueError("--target must be one ASCII A-Z letter")
    if args.cycles < 1:
        raise ValueError("--cycles must be >= 1")
    if args.conditioning_min_cycles < 2:
        raise ValueError("--conditioning-min-cycles must be >= 2")
    if args.conditioning_max_cycles < args.conditioning_min_cycles:
        raise ValueError(
            "--conditioning-max-cycles must be >= --conditioning-min-cycles"
        )
    if args.burst_frames < 1:
        raise ValueError("--burst-frames must be >= 1")
    if not 1 <= args.burst_min_inliers <= args.burst_frames:
        raise ValueError(
            "--burst-min-inliers must satisfy 1 <= min_inliers <= --burst-frames"
        )
    if args.settle_stable_reads < 1:
        raise ValueError("--settle-stable-reads must be >= 1")

    for name in (
        "step_mm",
        "probe_step_mm",
        "burst_cluster_radius_px",
        "max_frame_age_ms",
        "capture_timeout_s",
        "initial_timeout_s",
        "settle_tolerance_deg",
        "settle_timeout_s",
        "post_settle_guard_ms",
        "teleop_hz",
        "max_relative_target_deg",
        "max_delta_norm_mm",
        "max_ee_step_m",
        "max_zero_joint_shift_deg",
        "max_model_xy_error_mm",
        "max_model_z_error_mm",
        "max_measured_xy_error_mm",
        "max_measured_z_error_mm",
        "max_residual_rms_px",
        "max_condition_number",
        "conditioning_max_before_drift_px",
        "conditioning_max_response_drift_px",
    ):
        _finite_positive(getattr(args, name), f"--{name.replace('_', '-')}")

    active_step_mm = args.probe_step_mm if args.probe_axis else args.step_mm
    if active_step_mm > args.max_delta_norm_mm:
        raise ValueError("--step-mm must be <= --max-delta-norm-mm")
    if active_step_mm / 1000.0 > args.max_ee_step_m:
        raise ValueError("--step-mm must be <= --max-ee-step-m expressed in millimetres")
    if not math.isfinite(float(args.orientation_weight)) or args.orientation_weight < 0.0:
        raise ValueError("--orientation-weight must be finite and >= 0")
    if not math.isfinite(float(args.damping)) or args.damping < 0.0:
        raise ValueError("--damping must be finite and >= 0")
    if (
        not math.isfinite(float(args.max_opposition_cosine))
        or not -1.0 <= float(args.max_opposition_cosine) <= 1.0
    ):
        raise ValueError("--max-opposition-cosine must be finite and within [-1, 1]")

    spec, recognizer, tool_reference = _validate_assets(args)
    burst_config = TargetBurstConfig(
        frame_count=args.burst_frames,
        min_inliers=args.burst_min_inliers,
        cluster_radius_px=args.burst_cluster_radius_px,
    )

    # Construct the same motor ordering as SOFollower without connecting hardware.
    robot_cfg = SO101FollowerConfig(
        port=args.robot_port or "/dev/null",
        id=args.robot_id,
        use_degrees=True,
        max_relative_target=float(args.max_relative_target_deg),
        cameras={},
    )
    robot = SO101Follower(robot_cfg)
    motor_names = list(robot.bus.motors.keys())
    pipeline = _build_pipeline(args, motor_names)

    print("===== PHASE 3 H3.2 PREFLIGHT =====")
    print(f"target             = {target_label}")
    print(f"wrist              = {spec.width}x{spec.height} {spec.fourcc}")
    print(f"glyph model        = {Path(args.model_dir).resolve()}")
    print(f"tool reference p*  = ({tool_reference.u:.3f}, {tool_reference.v:.3f}) px")
    print(f"URDF               = {Path(args.urdf).resolve()}")
    print(f"motor order        = {motor_names}")
    print(f"motion contract    = {MOTION_FRAME} / {MOTION_UNIT}")
    if args.plan_only:
        print("mode               = PLAN ONLY (no Cartesian send_action)")
    elif args.probe_axis:
        print(
            "mode               = controlled probe "
            f"{args.probe_axis} / {args.probe_step_mm:.1f} commanded-mm"
        )
    else:
        print(f"calibration step   = {args.step_mm:.1f} commanded-mm")
        print(f"cycles             = {args.cycles} ({4 * args.cycles} formal motion samples)")
        print(
            "conditioning       = adaptive "
            f"{args.conditioning_min_cycles}..{args.conditioning_max_cycles} cycles; "
            f"same-phase BEFORE <= {args.conditioning_max_before_drift_px:.1f}px, "
            f"dUV <= {args.conditioning_max_response_drift_px:.1f}px, "
            f"X/Y opposition <= {args.max_opposition_cosine:.2f}"
        )
        print(f"candidate output   = {args.output.resolve()}")
    print(
        "burst consensus    = "
        f"{args.burst_min_inliers}/{args.burst_frames} within "
        f"{args.burst_cluster_radius_px:.1f} px"
    )
    print(f"joint slew limit   = {args.max_relative_target_deg:.1f} deg/send_action")
    print("reference mode     = fixed existing Goal_Position anchor")
    print(f"zero-delta joint gate = <= {args.max_zero_joint_shift_deg:.2f} deg")
    print(
        "model FK gates      = "
        f"XY <= {args.max_model_xy_error_mm:.2f} mm, "
        f"|Z| <= {args.max_model_z_error_mm:.2f} mm before send"
    )
    print(
        "measured FK diag    = non-gating; reference thresholds "
        f"XY {args.max_measured_xy_error_mm:.2f} mm, "
        f"Z {args.max_measured_z_error_mm:.2f} mm"
    )
    print(
        "pipeline            = EEReferenceAndDelta(fixed anchor) -> EEBoundsAndSafety -> "
        "GripperVelocityToJoint -> IK -> project FK self-check"
    )

    if args.dry_run:
        print("DRY RUN PASS: no camera opened, no serial port opened, no robot command sent.")
        return

    if not args.robot_port:
        raise ValueError("--robot-port is required unless --dry-run is used")
    if not args.leader_port:
        raise ValueError("--leader-port is required unless --dry-run is used")

    # Recreate with the real port so calibration lookup uses the intended robot config.
    robot_cfg = SO101FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        use_degrees=True,
        max_relative_target=float(args.max_relative_target_deg),
        cameras={},
    )
    robot = SO101Follower(robot_cfg)
    leader = SO101Leader(
        SO101LeaderConfig(
            port=args.leader_port,
            id=args.leader_id,
            use_degrees=True,
        )
    )
    motor_names = list(robot.bus.motors.keys())
    pipeline = _build_pipeline(args, motor_names)
    camera = ThreadedOpenCVCamera(spec)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    if not args.probe_axis:
        # Never leave a stale candidate that could be mistaken for this run.
        args.output.unlink(missing_ok=True)

    motions: list[list[float]] = []
    image_deltas: list[list[float]] = []
    samples: list[dict] = []
    last_frame_id: int | None = None

    if args.probe_axis:
        sequence = [_probe_motion(args.probe_axis, args.probe_step_mm)]
    else:
        # Alternating signs reduce drift and give paired + / - measurements on both axes.
        sequence: list[tuple[float, float]] = []
        for _ in range(args.cycles):
            sequence.extend(
                [
                    (args.step_mm, 0.0),
                    (-args.step_mm, 0.0),
                    (0.0, args.step_mm),
                    (0.0, -args.step_mm),
                ]
            )

    print()
    print("SAFETY:")
    print("  - Start the script with follower AND leader at the normal zero/home pose.")
    print("  - This process connects once at zero, then owns teleop -> autonomy -> recovery.")
    print("  - Do not use a separate lerobot-teleoperate process for this calibration.")
    print(
        "  - XY calibration stays at this hover: delta_z = 0; "
        "do not enter the <1 cm occlusion zone."
    )
    print("  - This calibration commands XY only: delta_z = 0 and it never presses a key.")
    print(
        "  - Keep a hand on power/stop. Autonomous failure never launches "
        "a blind home trajectory."
    )
    print(
        "  - After completion/failure, operator teleop resumes; "
        "return home, then Ctrl+C torque-off."
    )
    print()

    try:
        if not args.plan_only:
            camera.start()
            _wait_for_initial_frame(camera, args.initial_timeout_s)
        robot.connect()
        if not robot.is_connected:
            raise RuntimeError("SO-101 follower did not connect")
        leader.connect()
        if not leader.is_connected:
            raise RuntimeError("SO-101 leader did not connect")

        _run_positioning_teleop(
            robot,
            leader,
            teleop_hz=args.teleop_hz,
        )

        # Freeze one command-space operating point for the entire experiment.
        #
        # IMPORTANT:
        # Preserve the servo preload already holding the arm.  Phase-3 hardware
        # validation showed that Present_Position can differ from Goal_Position
        # while the arm is motion-stable.  Re-basing Cartesian command space on
        # Present_Position therefore changes the servo goal even for a nominal
        # zero Cartesian command.
        #
        # H3.2 calibration must use the EXISTING Goal_Position as its fixed
        # command-space anchor. Present_Position remains diagnostic/safety state.
        validation_kinematics = build_so101_kinematics(
            args.urdf,
            motor_names=motor_names,
        )

        goal_positions = robot.bus.sync_read(
            "Goal_Position",
            num_retry=robot.config.num_read_retries,
        )
        anchor_observation = {
            f"{name}.pos": float(goal_positions[name])
            for name in motor_names
        }
        anchor_observation = ordered_joint_observation(
            anchor_observation,
            motor_names,
        )

        # Wait for physical motion to stop without changing the Goal-space anchor.
        _, handoff_settled_timestamp = _wait_until_settled(
            robot,
            anchor_observation,
            tolerance_deg=args.settle_tolerance_deg,
            stable_reads=args.settle_stable_reads,
            timeout_s=args.settle_timeout_s,
        )

        present_observation = ordered_joint_observation(
            robot.get_observation(),
            motor_names,
        )

        print(
            "[COMMAND-ANCHOR] Goal - Present before H3.2: "
            + ", ".join(
                f"{name}="
                f"{float(anchor_observation[name + '.pos']) - float(present_observation[name + '.pos']):+.3f}deg"
                for name in motor_names
                if name != "gripper"
            )
        )

        anchor_xyz_mm = end_effector_xyz_mm(
            validation_kinematics,
            anchor_observation,
            motor_names,
        )

        handoff_guard_timestamp = (
            handoff_settled_timestamp
            + args.post_settle_guard_ms / 1000.0
        )
        zero_joint_action, zero_plan, _, _ = _plan_delta(
            pipeline,
            anchor_observation,
            0.0,
            0.0,
            motor_names=motor_names,
            validation_kinematics=validation_kinematics,
            anchor_xyz_mm=anchor_xyz_mm,
            max_delta_norm_mm=args.max_delta_norm_mm,
            max_model_xy_error_mm=args.max_model_xy_error_mm,
            max_model_z_error_mm=args.max_model_z_error_mm,
        )
        zero_joint_shifts = {
            name: abs(
                float(zero_joint_action[f"{name}.pos"])
                - float(anchor_observation[f"{name}.pos"])
            )
            for name in motor_names
            if name != "gripper"
        }
        max_zero_joint = max(zero_joint_shifts, key=zero_joint_shifts.get)
        max_zero_joint_shift = float(zero_joint_shifts[max_zero_joint])
        print(
            "[ANCHOR] "
            f"XYZ=({anchor_xyz_mm[0]:+.2f},{anchor_xyz_mm[1]:+.2f},"
            f"{anchor_xyz_mm[2]:+.2f})mm "
            f"zero-delta max joint shift={max_zero_joint_shift:.3f}deg "
            f"({max_zero_joint})"
        )
        if max_zero_joint_shift > args.max_zero_joint_shift_deg:
            raise RuntimeError(
                "Refusing autonomous motion: zero Cartesian delta does not preserve "
                f"the anchor joint pose; max planned shift={max_zero_joint_shift:.3f} deg "
                f"at {max_zero_joint} (limit {args.max_zero_joint_shift_deg:.3f} deg)"
            )

        # PLAN ONLY: pure fixed-anchor kinematics verification.
        # Return before any camera/glyph observation is requested.
        if args.plan_only:
            print("===== FIXED-ANCHOR PLAN CHECK =====")

            for index, (dx_mm, dy_mm) in enumerate(sequence, start=1):
                joint_action, _, _, _ = _plan_delta(
                    pipeline,
                    anchor_observation,
                    dx_mm,
                    dy_mm,
                    motor_names=motor_names,
                    validation_kinematics=validation_kinematics,
                    anchor_xyz_mm=anchor_xyz_mm,
                    max_delta_norm_mm=args.max_delta_norm_mm,
                    max_model_xy_error_mm=args.max_model_xy_error_mm,
                    max_model_z_error_mm=args.max_model_z_error_mm,
                )

                joint_plan_parts = []
                for name in motor_names:
                    if name == "gripper":
                        continue

                    delta_deg = (
                        float(joint_action[f"{name}.pos"])
                        - float(anchor_observation[f"{name}.pos"])
                    )
                    joint_plan_parts.append(
                        f"{name}={delta_deg:+.3f}deg"
                    )

                print(
                    "[JOINT-PLAN] "
                    + " ".join(joint_plan_parts)
                )
                print(
                    f"[PLAN {index:02d}/{len(sequence):02d}] PASS; "
                    "nothing sent to robot"
                )

            print(
                "PLAN CHECK PASS: Cartesian joint targets were computed "
                "but never sent."
            )

            _run_recovery_teleop(
                robot,
                leader,
                teleop_hz=args.teleop_hz,
            )
            return

        # Normal H3.2 only: vision starts here.
        ready_center, _, _, last_frame_id = _capture_target_center(
            camera,
            recognizer,
            target_label,
            frame_count=args.burst_frames,
            max_frame_age_ms=args.max_frame_age_ms,
            timeout_s=args.capture_timeout_s,
            after_frame_id=last_frame_id,
            min_capture_timestamp=handoff_guard_timestamp,
            burst_config=burst_config,
        )

        print(
            f"[READY] target {target_label} center="
            f"({ready_center[0]:.2f}, {ready_center[1]:.2f}) px"
        )

        def _execute_motion_sample(
            dx_mm: float,
            dy_mm: float,
            *,
            sample_index: int,
            sample_role: str,
            preview_filename: str,
            skip_anchor_return: bool,
        ) -> dict:
            nonlocal last_frame_id

            if not skip_anchor_return:
                return_observation = robot.get_observation()
                _, _, _, anchor_settled_timestamp, anchor_return_fk = _command_delta(
                    robot,
                    pipeline,
                    return_observation,
                    0.0,
                    0.0,
                    motor_names=motor_names,
                    validation_kinematics=validation_kinematics,
                    anchor_xyz_mm=anchor_xyz_mm,
                    max_delta_norm_mm=args.max_delta_norm_mm,
                    max_model_xy_error_mm=args.max_model_xy_error_mm,
                    max_model_z_error_mm=args.max_model_z_error_mm,
                    max_measured_xy_error_mm=args.max_measured_xy_error_mm,
                    max_measured_z_error_mm=args.max_measured_z_error_mm,
                    settle_tolerance_deg=args.settle_tolerance_deg,
                    settle_stable_reads=args.settle_stable_reads,
                    settle_timeout_s=args.settle_timeout_s,
                )
                anchor_guard_timestamp = (
                    anchor_settled_timestamp
                    + args.post_settle_guard_ms / 1000.0
                )
            else:
                anchor_return_fk = zero_plan
                anchor_guard_timestamp = None

            before, before_records, _, last_frame_id = _capture_target_center(
                camera,
                recognizer,
                target_label,
                frame_count=args.burst_frames,
                max_frame_age_ms=args.max_frame_age_ms,
                timeout_s=args.capture_timeout_s,
                after_frame_id=last_frame_id,
                min_capture_timestamp=anchor_guard_timestamp,
                burst_config=burst_config,
            )

            observation = robot.get_observation()
            (
                joint_action,
                sent_action,
                settled_obs,
                settled_timestamp,
                fk_diagnostics,
            ) = _command_delta(
                robot,
                pipeline,
                observation,
                dx_mm,
                dy_mm,
                motor_names=motor_names,
                validation_kinematics=validation_kinematics,
                anchor_xyz_mm=anchor_xyz_mm,
                max_delta_norm_mm=args.max_delta_norm_mm,
                max_model_xy_error_mm=args.max_model_xy_error_mm,
                max_model_z_error_mm=args.max_model_z_error_mm,
                max_measured_xy_error_mm=args.max_measured_xy_error_mm,
                max_measured_z_error_mm=args.max_measured_z_error_mm,
                settle_tolerance_deg=args.settle_tolerance_deg,
                settle_stable_reads=args.settle_stable_reads,
                settle_timeout_s=args.settle_timeout_s,
            )

            min_after_capture_timestamp = (
                settled_timestamp
                + args.post_settle_guard_ms / 1000.0
            )

            after, after_records, image, last_frame_id = _capture_target_center(
                camera,
                recognizer,
                target_label,
                frame_count=args.burst_frames,
                max_frame_age_ms=args.max_frame_age_ms,
                timeout_s=args.capture_timeout_s,
                after_frame_id=last_frame_id,
                min_capture_timestamp=min_after_capture_timestamp,
                burst_config=burst_config,
            )

            image_delta = after - before
            motion = [float(dx_mm), float(dy_mm)]
            delta_px = [
                float(image_delta[0]),
                float(image_delta[1]),
            ]

            preview = _draw_preview(
                image,
                target_label=target_label,
                before=before,
                after=after,
                motion=(dx_mm, dy_mm),
                sample_index=sample_index,
            )
            preview_path = args.artifact_dir / preview_filename
            if not cv2.imwrite(str(preview_path), preview):
                raise RuntimeError(
                    f"Failed to write preview: {preview_path}"
                )

            return {
                "sample_index": int(sample_index),
                "sample_role": sample_role,
                "requested_cartesian_delta_mm": motion,
                "input_semantics": "requested_cartesian_delta",
                "reference_semantics": "fixed_anchor",
                "anchor_xyz_mm": anchor_xyz_mm.tolist(),
                "anchor_return_fk": anchor_return_fk,
                "fk_diagnostics": fk_diagnostics,
                "before_center_px": [
                    float(before[0]),
                    float(before[1]),
                ],
                "after_center_px": [
                    float(after[0]),
                    float(after[1]),
                ],
                "image_delta_px": delta_px,
                "before_observations": before_records,
                "after_observations": after_records,
                "joint_action_requested": {
                    key: float(value)
                    for key, value in joint_action.items()
                    if isinstance(
                        value,
                        (int, float, np.integer, np.floating),
                    )
                },
                "joint_action_sent": {
                    key: float(value)
                    for key, value in sent_action.items()
                    if isinstance(
                        value,
                        (int, float, np.integer, np.floating),
                    )
                },
                "settled_joint_observation": {
                    key: float(value)
                    for key, value in settled_obs.items()
                    if isinstance(
                        value,
                        (int, float, np.integer, np.floating),
                    )
                },
                "settled_timestamp": float(settled_timestamp),
                "min_after_capture_timestamp": float(
                    min_after_capture_timestamp
                ),
            }

        conditioning_samples: list[dict] = []
        conditioning_reports: list[dict] = []
        conditioning_converged = False
        conditioning_cycles_executed = 0

        if not args.probe_axis:
            conditioning_sequence = [
                (args.step_mm, 0.0),
                (-args.step_mm, 0.0),
                (0.0, args.step_mm),
                (0.0, -args.step_mm),
            ]
            previous_cycle: list[dict] | None = None

            print()
            print("===== H3.2 ADAPTIVE CONDITIONING =====")
            print(
                "Conditioning samples are readiness-only "
                "and are NEVER used to fit J_cmd."
            )

            for cycle_index in range(
                1,
                args.conditioning_max_cycles + 1,
            ):
                cycle_records: list[dict] = []

                for phase_index, (dx_mm, dy_mm) in enumerate(
                    conditioning_sequence,
                    start=1,
                ):
                    conditioning_index = (
                        (cycle_index - 1) * 4 + phase_index
                    )

                    record = _execute_motion_sample(
                        dx_mm,
                        dy_mm,
                        sample_index=conditioning_index,
                        sample_role="conditioning",
                        preview_filename=(
                            f"conditioning_c{cycle_index:02d}"
                            f"_p{phase_index:02d}.jpg"
                        ),
                        skip_anchor_return=(
                            cycle_index == 1
                            and phase_index == 1
                        ),
                    )

                    record["conditioning_cycle"] = int(
                        cycle_index
                    )
                    record["conditioning_phase"] = int(
                        phase_index
                    )

                    conditioning_samples.append(record)
                    cycle_records.append(record)

                    duv = record["image_delta_px"]
                    print(
                        f"[COND {cycle_index:02d}.{phase_index}] "
                        f"anchor->dXY="
                        f"({dx_mm:+.1f},{dy_mm:+.1f}) mm  "
                        f"dUV=({duv[0]:+.2f},{duv[1]:+.2f}) px"
                    )

                cycle_motions = [
                    record["requested_cartesian_delta_mm"]
                    for record in cycle_records
                ]
                cycle_duv = [
                    record["image_delta_px"]
                    for record in cycle_records
                ]

                cycle_direction = _direction_consistency(
                    cycle_motions,
                    cycle_duv,
                )

                x_cos = cycle_direction["x"][
                    "opposition_cosine"
                ]
                y_cos = cycle_direction["y"][
                    "opposition_cosine"
                ]

                report = {
                    "cycle": int(cycle_index),
                    "direction_consistency": cycle_direction,
                    "compared_to_cycle": None,
                    "before_phase_drift_px": None,
                    "response_phase_drift_px": None,
                    "max_before_phase_drift_px": None,
                    "max_response_phase_drift_px": None,
                    "converged": False,
                }

                if previous_cycle is not None:
                    before_drifts = []
                    response_drifts = []

                    for previous, current in zip(
                        previous_cycle,
                        cycle_records,
                        strict=True,
                    ):
                        previous_before = np.asarray(
                            previous["before_center_px"],
                            dtype=np.float64,
                        )
                        current_before = np.asarray(
                            current["before_center_px"],
                            dtype=np.float64,
                        )

                        previous_duv = np.asarray(
                            previous["image_delta_px"],
                            dtype=np.float64,
                        )
                        current_duv = np.asarray(
                            current["image_delta_px"],
                            dtype=np.float64,
                        )

                        before_drifts.append(
                            float(
                                np.linalg.norm(
                                    current_before
                                    - previous_before
                                )
                            )
                        )

                        response_drifts.append(
                            float(
                                np.linalg.norm(
                                    current_duv
                                    - previous_duv
                                )
                            )
                        )

                    max_before_drift = max(before_drifts)
                    max_response_drift = max(
                        response_drifts
                    )

                    direction_ok = (
                        x_cos is not None
                        and y_cos is not None
                        and float(x_cos)
                        <= args.max_opposition_cosine
                        and float(y_cos)
                        <= args.max_opposition_cosine
                    )

                    drift_ok = (
                        max_before_drift
                        <= args.conditioning_max_before_drift_px
                        and max_response_drift
                        <= args.conditioning_max_response_drift_px
                    )

                    enough_cycles = (
                        cycle_index
                        >= args.conditioning_min_cycles
                    )

                    conditioning_converged = bool(
                        enough_cycles
                        and direction_ok
                        and drift_ok
                    )

                    report.update(
                        {
                            "compared_to_cycle": int(
                                cycle_index - 1
                            ),
                            "before_phase_drift_px": before_drifts,
                            "response_phase_drift_px": response_drifts,
                            "max_before_phase_drift_px": float(
                                max_before_drift
                            ),
                            "max_response_phase_drift_px": float(
                                max_response_drift
                            ),
                            "converged": conditioning_converged,
                        }
                    )

                    print(
                        f"[COND CHECK "
                        f"{cycle_index - 1}->{cycle_index}] "
                        f"max BEFORE drift="
                        f"{max_before_drift:.2f}px "
                        f"(<= "
                        f"{args.conditioning_max_before_drift_px:.2f})  "
                        f"max dUV drift="
                        f"{max_response_drift:.2f}px "
                        f"(<= "
                        f"{args.conditioning_max_response_drift_px:.2f})  "
                        f"opposition X={x_cos!r} "
                        f"Y={y_cos!r} "
                        f"(<= "
                        f"{args.max_opposition_cosine:.2f})"
                    )

                else:
                    print(
                        f"[COND CHECK {cycle_index}] "
                        "baseline cycle recorded; "
                        "one adjacent cycle is required "
                        "for convergence."
                    )

                conditioning_reports.append(report)
                conditioning_cycles_executed = cycle_index

                if conditioning_converged:
                    print(
                        f"[CONDITIONING PASS] stable after "
                        f"{cycle_index} cycles; "
                        "formal samples start fresh and "
                        "conditioning data remain excluded."
                    )
                    break

                previous_cycle = cycle_records

            if not conditioning_converged:
                conditioning_failure = {
                    "schema_version": 1,
                    "protocol": "fixed_anchor_conditioning_v1",
                    "input_semantics":
                        "requested_cartesian_delta",
                    "reference_semantics": "fixed_anchor",
                    "measured_fk_policy": "diagnostic_only",
                    "anchor_xyz_mm": anchor_xyz_mm.tolist(),
                    "target_label": target_label,
                    "step_commanded_mm": float(
                        args.step_mm
                    ),
                    "thresholds": {
                        "min_cycles": int(
                            args.conditioning_min_cycles
                        ),
                        "max_cycles": int(
                            args.conditioning_max_cycles
                        ),
                        "max_before_phase_drift_px": float(
                            args.conditioning_max_before_drift_px
                        ),
                        "max_response_phase_drift_px": float(
                            args.conditioning_max_response_drift_px
                        ),
                        "max_opposition_cosine": float(
                            args.max_opposition_cosine
                        ),
                    },
                    "cycles_executed": int(
                        conditioning_cycles_executed
                    ),
                    "converged": False,
                    "cycle_reports": conditioning_reports,
                    "samples": conditioning_samples,
                }

                failure_path = (
                    args.artifact_dir
                    / "conditioning_failure_session.json"
                )

                failure_path.write_text(
                    json.dumps(
                        conditioning_failure,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )

                raise RuntimeError(
                    "Adaptive conditioning did not converge "
                    f"within {args.conditioning_max_cycles} cycles; "
                    "formal H3.2 samples were NOT started. "
                    f"See {failure_path}"
                )

            print("===== FORMAL H3.2 SAMPLING =====")

        # Probe bypasses conditioning.
        # Full H3.2 starts a completely fresh formal dataset.
        for index, (dx_mm, dy_mm) in enumerate(
            sequence,
            start=1,
        ):
            record = _execute_motion_sample(
                dx_mm,
                dy_mm,
                sample_index=index,
                sample_role=(
                    "probe"
                    if args.probe_axis
                    else "calibration"
                ),
                preview_filename=f"sample_{index:02d}.jpg",
                skip_anchor_return=bool(
                    args.probe_axis
                    and index == 1
                ),
            )

            motion = record[
                "requested_cartesian_delta_mm"
            ]
            delta_px = record["image_delta_px"]

            motions.append(motion)
            image_deltas.append(delta_px)
            samples.append(record)

            print(
                f"[{index:02d}/{len(sequence):02d}] "
                f"anchor->dXY="
                f"({dx_mm:+.1f},{dy_mm:+.1f}) mm  "
                f"dUV=({delta_px[0]:+.2f},"
                f"{delta_px[1]:+.2f}) px"
            )

        # End autonomous sampling by returning to the same fixed anchor.
        final_observation = robot.get_observation()
        _, _, _, _, final_anchor_fk = _command_delta(
            robot,
            pipeline,
            final_observation,
            0.0,
            0.0,
            motor_names=motor_names,
            validation_kinematics=validation_kinematics,
            anchor_xyz_mm=anchor_xyz_mm,
            max_delta_norm_mm=args.max_delta_norm_mm,
            max_model_xy_error_mm=args.max_model_xy_error_mm,
            max_model_z_error_mm=args.max_model_z_error_mm,
            max_measured_xy_error_mm=args.max_measured_xy_error_mm,
            max_measured_z_error_mm=args.max_measured_z_error_mm,
            settle_tolerance_deg=args.settle_tolerance_deg,
            settle_stable_reads=args.settle_stable_reads,
            settle_timeout_s=args.settle_timeout_s,
        )
        print("[ANCHOR RESTORED] fixed calibration anchor reached before recovery.")

        if args.probe_axis:
            response_norm_px = float(np.linalg.norm(np.asarray(image_deltas[0])))
            session = {
                "schema_version": 1,
                "protocol": "fixed_anchor_cartesian_single_xy_probe_v2",
                "input_semantics": "requested_cartesian_delta",
                "reference_semantics": "fixed_anchor",
                "measured_fk_policy": "diagnostic_only",
                "anchor_xyz_mm": anchor_xyz_mm.tolist(),
                "final_anchor_fk": final_anchor_fk,
                "target_label": target_label,
                "robot_id": args.robot_id,
                "robot_port": args.robot_port,
                "probe_axis": args.probe_axis,
                "probe_step_commanded_mm": float(args.probe_step_mm),
                "response_norm_px": response_norm_px,
                "burst_consensus": {
                    "frame_count": int(args.burst_frames),
                    "min_inliers": int(args.burst_min_inliers),
                    "cluster_radius_px": float(args.burst_cluster_radius_px),
                },
                "post_settle_guard_ms": float(args.post_settle_guard_ms),
                "samples": samples,
                "camera": {
                    "measured_fps": float(camera.measured_fps),
                    "read_errors": int(camera.read_errors),
                    "actual_properties": camera.actual_properties(),
                },
            }
            session_path = args.artifact_dir / "probe_session.json"
            session_path.write_text(json.dumps(session, indent=2) + "\n", encoding="utf-8")
            print()
            print("===== CONTROLLED PROBE RESULT =====")
            print(
                f"requested command  = {args.probe_axis} "
                f"{args.probe_step_mm:.1f} commanded-mm"
            )
            print(
                "observed image dUV = "
                f"({image_deltas[0][0]:+.2f}, {image_deltas[0][1]:+.2f}) px"
            )
            print(f"response norm      = {response_norm_px:.2f} px")
            print(f"session            = {session_path.resolve()}")
            print(
                "PROBE COMPLETE: no Jacobian was fitted and canonical "
                "calibration was not modified."
            )
            _run_recovery_teleop(
                robot,
                leader,
                teleop_hz=args.teleop_hz,
            )
            return

        calibration = ImageJacobianCalibration.from_samples(
            motions,
            image_deltas,
            motion_frame=MOTION_FRAME,
            motion_unit=MOTION_UNIT,
            damping=float(args.damping),
        )

        predicted = np.asarray(motions, dtype=np.float64) @ calibration.array.T
        measured = np.asarray(image_deltas, dtype=np.float64)
        residual = measured - predicted
        direction_consistency = _direction_consistency(motions, image_deltas)
        acceptance_errors: list[str] = []
        if calibration.condition_number > args.max_condition_number:
            acceptance_errors.append(
                f"condition_number={calibration.condition_number:.3f} > "
                f"{args.max_condition_number:.3f}"
            )
        if calibration.residual_rms_px > args.max_residual_rms_px:
            acceptance_errors.append(
                f"residual_rms_px={calibration.residual_rms_px:.3f} > "
                f"{args.max_residual_rms_px:.3f}"
            )
        for axis in ("x", "y"):
            cosine = direction_consistency[axis]["opposition_cosine"]
            if cosine is None or float(cosine) > args.max_opposition_cosine:
                acceptance_errors.append(
                    f"{axis.upper()} opposition_cosine={cosine!r} > "
                    f"{args.max_opposition_cosine:.3f}"
                )
        accepted = not acceptance_errors

        session = {
            "schema_version": 1,
            "protocol": "fixed_anchor_conditioned_cartesian_paired_xy_perturbations_v3",
            "input_semantics": "requested_cartesian_delta",
            "reference_semantics": "fixed_anchor",
            "measured_fk_policy": "diagnostic_only",
            "anchor_xyz_mm": anchor_xyz_mm.tolist(),
            "final_anchor_fk": final_anchor_fk,
            "target_label": target_label,
            "robot_id": args.robot_id,
            "robot_port": args.robot_port,
            "urdf": str(Path(args.urdf).resolve()),
            "camera_config": str(Path(args.camera_config).resolve()),
            "model_dir": str(Path(args.model_dir).resolve()),
            "tool_reference": tool_reference.to_dict(),
            "step_commanded_mm": float(args.step_mm),
            "cycles": int(args.cycles),
            "max_relative_target_deg": float(args.max_relative_target_deg),
            "max_ee_step_m": float(args.max_ee_step_m),
            "orientation_weight": float(args.orientation_weight),
            "burst_consensus": {
                "frame_count": int(args.burst_frames),
                "min_inliers": int(args.burst_min_inliers),
                "cluster_radius_px": float(args.burst_cluster_radius_px),
            },
            "post_settle_guard_ms": float(args.post_settle_guard_ms),
            "conditioning": {
                "policy": "phase_matched_adjacent_cycle_convergence_v1",
                "min_cycles": int(args.conditioning_min_cycles),
                "max_cycles": int(args.conditioning_max_cycles),
                "max_before_phase_drift_px": float(
                    args.conditioning_max_before_drift_px
                ),
                "max_response_phase_drift_px": float(
                    args.conditioning_max_response_drift_px
                ),
                "max_opposition_cosine": float(
                    args.max_opposition_cosine
                ),
                "cycles_executed": int(
                    conditioning_cycles_executed
                ),
                "converged": bool(
                    conditioning_converged
                ),
                "cycle_reports": conditioning_reports,
                "samples": conditioning_samples,
                "excluded_from_jacobian_fit": True,
            },
            "samples": samples,
            "image_jacobian": calibration.to_dict(),
            "residual_vectors_px": residual.tolist(),
            "direction_consistency": direction_consistency,
            "acceptance": {
                "accepted": accepted,
                "errors": acceptance_errors,
                "max_condition_number": float(args.max_condition_number),
                "max_residual_rms_px": float(args.max_residual_rms_px),
                "max_opposition_cosine": float(args.max_opposition_cosine),
            },
            "camera": {
                "measured_fps": float(camera.measured_fps),
                "read_errors": int(camera.read_errors),
                "actual_properties": camera.actual_properties(),
            },
        }
        session_path = args.artifact_dir / "session.json"
        session_path.write_text(json.dumps(session, indent=2) + "\n", encoding="utf-8")

        if not accepted:
            args.output.unlink(missing_ok=True)
            print()
            print("===== H3.2 REJECTED =====")
            for error in acceptance_errors:
                print(f"  - {error}")
            print(f"session              = {session_path.resolve()}")
            raise RuntimeError(
                "H3.2 acceptance gates failed; candidate was NOT written"
            )

        calibration.save(args.output)
        print()
        print("===== H3.2 CANDIDATE RESULT =====")
        print("J_cmd [px/commanded-mm] =")
        print(np.asarray(calibration.matrix))
        print(f"samples              = {calibration.sample_count}")
        print(f"residual_rms_px      = {calibration.residual_rms_px:.3f}")
        print(f"singular_values      = {calibration.singular_values}")
        print(f"condition_number     = {calibration.condition_number:.3f}")
        print(
            "opposition cosine   = "
            f"X {direction_consistency['x']['opposition_cosine']!r}, "
            f"Y {direction_consistency['y']['opposition_cosine']!r} "
            "(ideal -1)"
        )
        print(f"candidate            = {args.output.resolve()}")
        print(f"session              = {session_path.resolve()}")
        print(
            "CANDIDATE ONLY: review direction consistency, residuals, conditioning, "
            "and artifacts before promoting to calibration/image_jacobian.json."
        )
        _run_recovery_teleop(
            robot,
            leader,
            teleop_hz=args.teleop_hz,
        )

    except KeyboardInterrupt:
        print("\n[AUTONOMY ABORTED] No further Cartesian command will be issued.")
        if robot.is_connected and leader.is_connected:
            _run_recovery_teleop(
                robot,
                leader,
                teleop_hz=args.teleop_hz,
            )
    except Exception as exc:
        print(f"\n[CONTROLLED FAILURE] {type(exc).__name__}: {exc}")
        print("No further Cartesian command will be issued.")
        if robot.is_connected and leader.is_connected:
            try:
                _run_recovery_teleop(
                    robot,
                    leader,
                    teleop_hz=args.teleop_hz,
                )
            except Exception as recovery_exc:
                print(
                    "[RECOVERY UNAVAILABLE] Operator teleop could not continue: "
                    f"{type(recovery_exc).__name__}: {recovery_exc}"
                )
        raise
    finally:
        camera.stop()
        if leader.is_connected:
            leader.disconnect()
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
