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
    make_cartesian_delta_action,
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
    target = {
        key.removesuffix(".pos"): float(value)
        for key, value in sent_action.items()
        if isinstance(key, str)
        and key.endswith(".pos")
        and key != "gripper.pos"
    }
    if not target:
        raise RuntimeError("No arm joint targets were produced by Cartesian pipeline")

    deadline = time.monotonic() + timeout_s
    consecutive = 0
    last_obs: dict | None = None

    while time.monotonic() < deadline:
        obs = robot.get_observation()
        last_obs = obs
        present = _joint_positions(obs)
        if not all(name in present for name in target):
            missing = sorted(set(target) - set(present))
            raise RuntimeError(f"Robot observation is missing joints: {missing}")

        max_error = max(abs(present[name] - value) for name, value in target.items())
        if max_error <= tolerance_deg:
            consecutive += 1
            if consecutive >= stable_reads:
                return obs, time.monotonic()
        else:
            consecutive = 0

        time.sleep(0.03)

    if last_obs is None:
        raise RuntimeError("No robot observation received while waiting for settle")
    present = _joint_positions(last_obs)
    max_error = max(abs(present[name] - value) for name, value in target.items())
    raise RuntimeError(
        f"Robot did not settle within {timeout_s:.1f}s; max joint error={max_error:.3f} deg"
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
    )


def _command_delta(
    robot: SO101Follower,
    pipeline,
    observation: dict,
    dx_mm: float,
    dy_mm: float,
    *,
    max_delta_norm_mm: float,
    settle_tolerance_deg: float,
    settle_stable_reads: int,
    settle_timeout_s: float,
) -> tuple[dict, dict, dict, float]:
    delta_action = make_cartesian_delta_action(
        dx_mm,
        dy_mm,
        max_delta_norm_mm=max_delta_norm_mm,
    )
    joint_action = pipeline((delta_action, observation))
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
            "LeRobot max_relative_target clipped this Cartesian calibration step; "
            "do not fit a Jacobian from a clipped step. Reduce --step-mm (for example 5) "
            f"or deliberately raise the joint slew limit after review. {details}"
        )

    settled_obs, settled_timestamp = _wait_until_settled(
        robot,
        sent_action,
        tolerance_deg=settle_tolerance_deg,
        stable_reads=settle_stable_reads,
        timeout_s=settle_timeout_s,
    )
    return joint_action, sent_action, settled_obs, settled_timestamp


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
        "--max-condition-number",
        type=float,
        default=DEFAULT_MAX_CONDITION_NUMBER,
    )
    parser.add_argument("--damping", type=float, default=1e-6)
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
        "max_condition_number",
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
    if args.probe_axis:
        print(
            "mode               = controlled probe "
            f"{args.probe_axis} / {args.probe_step_mm:.1f} commanded-mm"
        )
    else:
        print(f"calibration step   = {args.step_mm:.1f} commanded-mm")
        print(f"cycles             = {args.cycles} ({4 * args.cycles} motion samples)")
        print(f"candidate output   = {args.output.resolve()}")
    print(
        "burst consensus    = "
        f"{args.burst_min_inliers}/{args.burst_frames} within "
        f"{args.burst_cluster_radius_px:.1f} px"
    )
    print(f"joint slew limit   = {args.max_relative_target_deg:.1f} deg/send_action")
    print(
        "pipeline            = EEReferenceAndDelta -> EEBoundsAndSafety -> "
        "GripperVelocityToJoint -> IK"
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

        # Establish that the target is visible before any motion.
        before, before_records, _, last_frame_id = _capture_target_center(
            camera,
            recognizer,
            target_label,
            frame_count=args.burst_frames,
            max_frame_age_ms=args.max_frame_age_ms,
            timeout_s=args.capture_timeout_s,
            after_frame_id=last_frame_id,
            min_capture_timestamp=None,
            burst_config=burst_config,
        )
        print(f"[READY] target {target_label} center=({before[0]:.2f}, {before[1]:.2f}) px")

        for index, (dx_mm, dy_mm) in enumerate(sequence, start=1):
            before, before_records, _, last_frame_id = _capture_target_center(
                camera,
                recognizer,
                target_label,
                frame_count=args.burst_frames,
                max_frame_age_ms=args.max_frame_age_ms,
                timeout_s=args.capture_timeout_s,
                after_frame_id=last_frame_id,
                min_capture_timestamp=None,
                burst_config=burst_config,
            )

            # Read the robot state immediately before constructing the command.
            # Do not reuse an observation acquired before the camera burst.
            observation = robot.get_observation()

            joint_action, sent_action, settled_obs, settled_timestamp = _command_delta(
                robot,
                pipeline,
                observation,
                dx_mm,
                dy_mm,
                max_delta_norm_mm=args.max_delta_norm_mm,
                settle_tolerance_deg=args.settle_tolerance_deg,
                settle_stable_reads=args.settle_stable_reads,
                settle_timeout_s=args.settle_timeout_s,
            )

            min_after_capture_timestamp = (
                settled_timestamp + args.post_settle_guard_ms / 1000.0
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
            delta_px = [float(image_delta[0]), float(image_delta[1])]
            motions.append(motion)
            image_deltas.append(delta_px)

            preview = _draw_preview(
                image,
                target_label=target_label,
                before=before,
                after=after,
                motion=(dx_mm, dy_mm),
                sample_index=index,
            )
            preview_path = args.artifact_dir / f"sample_{index:02d}.jpg"
            if not cv2.imwrite(str(preview_path), preview):
                raise RuntimeError(f"Failed to write preview: {preview_path}")

            samples.append(
                {
                    "sample_index": index,
                    "requested_cartesian_delta_mm": motion,
                    "input_semantics": "requested_cartesian_delta",
                    "before_center_px": [float(before[0]), float(before[1])],
                    "after_center_px": [float(after[0]), float(after[1])],
                    "image_delta_px": delta_px,
                    "before_observations": before_records,
                    "after_observations": after_records,
                    "joint_action_requested": {
                        key: float(value)
                        for key, value in joint_action.items()
                        if isinstance(value, (int, float, np.integer, np.floating))
                    },
                    "joint_action_sent": {
                        key: float(value)
                        for key, value in sent_action.items()
                        if isinstance(value, (int, float, np.integer, np.floating))
                    },
                    "settled_joint_observation": {
                        key: float(value)
                        for key, value in settled_obs.items()
                        if isinstance(value, (int, float, np.integer, np.floating))
                    },
                    "settled_timestamp": float(settled_timestamp),
                    "min_after_capture_timestamp": float(min_after_capture_timestamp),
                }
            )

            print(
                f"[{index:02d}/{len(sequence):02d}] "
                f"dXY_cmd=({dx_mm:+.1f},{dy_mm:+.1f}) mm  "
                f"dUV=({image_delta[0]:+.2f},{image_delta[1]:+.2f}) px"
            )

        if args.probe_axis:
            response_norm_px = float(np.linalg.norm(np.asarray(image_deltas[0])))
            session = {
                "schema_version": 1,
                "protocol": "official_lerobot_cartesian_single_xy_probe",
                "input_semantics": "requested_cartesian_delta",
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
            max_condition_number=float(args.max_condition_number),
        )
        calibration.save(args.output)

        predicted = np.asarray(motions, dtype=np.float64) @ calibration.array.T
        measured = np.asarray(image_deltas, dtype=np.float64)
        residual = measured - predicted
        direction_consistency = _direction_consistency(motions, image_deltas)

        session = {
            "schema_version": 1,
            "protocol": "official_lerobot_cartesian_paired_xy_perturbations",
            "input_semantics": "requested_cartesian_delta",
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
            "samples": samples,
            "image_jacobian": calibration.to_dict(),
            "residual_vectors_px": residual.tolist(),
            "direction_consistency": direction_consistency,
            "camera": {
                "measured_fps": float(camera.measured_fps),
                "read_errors": int(camera.read_errors),
                "actual_properties": camera.actual_properties(),
            },
        }
        session_path = args.artifact_dir / "session.json"
        session_path.write_text(json.dumps(session, indent=2) + "\n", encoding="utf-8")

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
