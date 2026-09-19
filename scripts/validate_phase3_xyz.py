from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

from calibrate_image_jacobian import (
    _plan_delta,
    _run_positioning_teleop,
    _run_recovery_teleop,
)
from so101_typing.adapters.cameras import CameraSpec, ThreadedOpenCVCamera
from so101_typing.control.cartesian_kinematics import (
    build_official_cartesian_pipeline,
    build_so101_kinematics,
    end_effector_xyz_mm,
    make_cartesian_delta_action,
    ordered_joint_observation,
)
from so101_typing.control.image_jacobian import ImageJacobianCalibration
from so101_typing.control.tool_reference import ToolReferenceCalibration
from so101_typing.control.visual_servo import (
    VisualServoConfig,
    VisualServoStepStatus,
    compute_visual_servo_step,
)
from so101_typing.perception.glyph_runtime import RuntimeHOGGlyphRecognizer
from so101_typing.perception.wrist_target import observe_target
from so101_typing.perception.keycaps import detect_keycaps


ROBOT_PORT = "/dev/ttyACM0"
LEADER_PORT = "/dev/ttyACM1"
TARGET = "G"

URDF = (
    Path.home()
    / ".cache/huggingface/lerobot/robot-urdfs/so101/so101_new_calib.urdf"
)
TOOL_REFERENCE = Path("calibration/tool_reference.json")
IMAGE_JACOBIAN = Path("calibration/image_jacobian.json")
GLYPH_MODEL = Path("artifacts/models/glyph_hog_svm")
CAMERA_CONFIG = Path("configs/cameras/wrist.yaml")

# Phase-3 hardware acceptance / regression limits.
# J_cmd was calibrated in command space with 5 mm Cartesian perturbations.
# Runtime corrections are accumulated against one fixed existing Goal_Position
# anchor instead of rebasing on measured joint/FK pose each cycle.
MAX_COMMANDS = 8
MAX_STEP_NORM_MM = 5.0
MAX_TOTAL_CORRECTION_MM = 35.0
CONVERGENCE_THRESHOLD_PX = 4.0
STABLE_FRAMES_REQUIRED = 2
INITIAL_CAPTURE_THRESHOLD_PX = 80.0

# Safety gates only; these do not decide whether XY succeeded.
MAX_RELATIVE_TARGET_DEG = 10.0
MAX_MODEL_XY_ERROR_MM = 1.0
MAX_MODEL_Z_ERROR_MM = 1.0
MOTION_STABLE_DELTA_DEG = 0.5
MOTION_STABLE_READS = 3
MOTION_STABLE_TIMEOUT_S = 4.0
POST_MOTION_GUARD_S = 0.15

# XYZ cumulative-command validation.
# All XYZ commands remain relative to ONE fixed existing Goal_Position anchor.
XY_TO_Z_HANDOFF_PX = 6.0
Z_LEVEL_XY_TOLERANCE_PX = 6.0
Z_LEVELS_MM = (-3.0, -6.0)
MAX_XY_REALIGN_COMMANDS_PER_Z = 3

# Semantic-lock tracking state.
# Glyph recognition establishes key identity; guarded whole-keyboard geometry
# may propagate that already-established target during autonomous occlusion.
_TRACK_PREV_CENTERS = None
_TRACK_PREV_TARGET = None



def _estimate_keyboard_translation(previous_centers, current_centers):
    previous = np.asarray(previous_centers, dtype=np.float64)
    current = np.asarray(current_centers, dtype=np.float64)

    if previous.ndim != 2 or current.ndim != 2:
        return None
    if len(previous) < 4 or len(current) < 4:
        return None

    distances = np.linalg.norm(
        previous[:, None, :] - current[None, :, :],
        axis=2,
    )

    prev_to_curr = np.argmin(distances, axis=1)
    curr_to_prev = np.argmin(distances, axis=0)

    shifts = []
    for previous_index, current_index in enumerate(prev_to_curr):
        current_index = int(current_index)

        # Mutual nearest-neighbour matching protects against accidentally
        # pairing adjacent keys in the repetitive keyboard grid.
        if int(curr_to_prev[current_index]) != previous_index:
            continue

        pair_distance = float(distances[previous_index, current_index])
        if pair_distance > 18.0:
            continue

        shifts.append(
            current[current_index] - previous[previous_index]
        )

    if len(shifts) < 4:
        return None

    shifts = np.asarray(shifts, dtype=np.float64)

    first_median = np.median(shifts, axis=0)
    residuals = np.linalg.norm(shifts - first_median, axis=1)
    inlier_mask = residuals <= 4.0

    if int(np.count_nonzero(inlier_mask)) < 4:
        return None

    inlier_shifts = shifts[inlier_mask]
    translation = np.median(inlier_shifts, axis=0)
    final_residuals = np.linalg.norm(
        inlier_shifts - translation,
        axis=1,
    )

    return {
        "translation": translation,
        "matches": int(len(shifts)),
        "inliers": int(len(inlier_shifts)),
        "median_residual": float(np.median(final_residuals)),
        "max_residual": float(np.max(final_residuals)),
    }


def capture_target(camera, recognizer, *, after_frame_id=-1, min_timestamp=None):
    global _TRACK_PREV_CENTERS, _TRACK_PREV_TARGET

    deadline = time.monotonic() + 5.0
    last_track_retry_log = 0.0
    last_seen = int(after_frame_id)

    while time.monotonic() < deadline:
        frame = camera.latest(copy_image=True)
        if frame is None or frame.frame_id <= last_seen:
            time.sleep(0.01)
            continue

        last_seen = int(frame.frame_id)
        if min_timestamp is not None and frame.capture_timestamp <= min_timestamp:
            continue

        obs = observe_target(frame.image, TARGET, recognizer)
        if obs.found and obs.center_px is not None:
            candidates = detect_keycaps(frame.image)
            _TRACK_PREV_CENTERS = np.asarray(
                [candidate.center for candidate in candidates],
                dtype=np.float64,
            )
            _TRACK_PREV_TARGET = np.asarray(
                obs.center_px,
                dtype=np.float64,
            )
            return obs, frame

        # Geometry fallback is valid only after autonomy has started and
        # semantic identity has already been established from the glyph.
        # PRECHECK/manual positioning never uses this fallback.
        if (
            after_frame_id >= 0
            and _TRACK_PREV_CENTERS is not None
            and _TRACK_PREV_TARGET is not None
        ):
            candidates = detect_keycaps(frame.image)
            current_centers = np.asarray(
                [candidate.center for candidate in candidates],
                dtype=np.float64,
            )

            estimate = _estimate_keyboard_translation(
                _TRACK_PREV_CENTERS,
                current_centers,
            )

            if estimate is not None:
                shift = estimate["translation"]
                predicted_target = _TRACK_PREV_TARGET + shift

                # After autonomous motion, a high-confidence whole-keyboard
                # translation may carry the already-established semantic G lock
                # through temporary glyph occlusion.  This is still a visual
                # measurement; FK/joint state does not decide XY success.
                if (
                    after_frame_id >= 0
                    and estimate["inliers"] >= 8
                    and estimate["median_residual"] <= 2.5
                    and float(np.linalg.norm(shift)) <= 25.0
                ):
                    _TRACK_PREV_CENTERS = current_centers
                    _TRACK_PREV_TARGET = predicted_target

                    print(
                        "[TRACK] G glyph hidden; accepted keyboard-geometry "
                        "observation: "
                        f"translation=({shift[0]:+.2f},{shift[1]:+.2f})px, "
                        f"matches={estimate['matches']}, "
                        f"inliers={estimate['inliers']}, "
                        f"median_residual={estimate['median_residual']:.2f}px, "
                        f"tracked_G=({predicted_target[0]:.2f},"
                        f"{predicted_target[1]:.2f})"
                    )

                    return (
                        SimpleNamespace(
                            center_px=(
                                float(predicted_target[0]),
                                float(predicted_target[1]),
                            )
                        ),
                        frame,
                    )

                # One poor geometry frame is not a control failure.
                # At low Z the pencil occludes more of the keyboard and the
                # detected keycap set varies frame-to-frame. Keep the strict
                # acceptance gate, but use the remaining capture timeout to
                # wait for another fresh visual frame.
                now = time.monotonic()
                if now - last_track_retry_log >= 0.5:
                    print(
                        "[TRACK-RETRY] G hidden and this geometry frame is "
                        "below the safe tracking gate; waiting for another "
                        "fresh frame: "
                        f"translation=({shift[0]:+.2f},{shift[1]:+.2f})px, "
                        f"matches={estimate['matches']}, "
                        f"inliers={estimate['inliers']}, "
                        f"median_residual={estimate['median_residual']:.2f}px, "
                        f"max_residual={estimate['max_residual']:.2f}px"
                    )
                    last_track_retry_log = now

                time.sleep(0.02)
                continue

    # Diagnostic only: if semantic recognition disappears, report whether
    # keyboard geometry is still visible.  Do NOT silently substitute another
    # keycap here yet; autonomous tracking will only be added after we verify it.
    frame = camera.latest(copy_image=True)
    if frame is not None:
        candidates = detect_keycaps(frame.image)
        centers = [candidate.center for candidate in candidates]

        if centers:
            diag_tool_reference = ToolReferenceCalibration.load(TOOL_REFERENCE)
            if diag_tool_reference is None:
                raise RuntimeError("tool_reference.json is not calibrated")
            p_tip = np.asarray(diag_tool_reference.center, dtype=np.float64)
            ranked = sorted(
                (
                    (
                        float(np.linalg.norm(np.asarray(center) - p_tip)),
                        center,
                    )
                    for center in centers
                ),
                key=lambda item: item[0],
            )

            nearest = ranked[:5]
            detail = ", ".join(
                f"{center} dist_to_p_tip={distance:.1f}px"
                for distance, center in nearest
            )

            raise RuntimeError(
                f"Target {TARGET!r} not found by glyph recognition in a fresh WRIST frame; "
                f"geometry still detected {len(candidates)} keycaps. "
                f"Nearest keycaps to p_tip: {detail}"
            )

        raise RuntimeError(
            f"Target {TARGET!r} not found by glyph recognition; "
            "geometry detector also found 0 keycaps."
        )

    raise RuntimeError(
        f"Target {TARGET!r} not found in a fresh WRIST frame; camera.latest() returned None."
    )


def visual_error(tool_reference, obs):
    error = np.asarray(tool_reference.error_px(obs.center_px), dtype=np.float64)
    return error, float(np.linalg.norm(error))


def wait_motion_stable(robot, motor_names):
    arm_names = [name for name in motor_names if name != "gripper"]
    deadline = time.monotonic() + MOTION_STABLE_TIMEOUT_S
    previous = None
    stable_reads = 0

    while time.monotonic() < deadline:
        obs = robot.get_observation()
        current = np.asarray(
            [float(obs[f"{name}.pos"]) for name in arm_names],
            dtype=np.float64,
        )

        if previous is not None:
            max_change = float(np.max(np.abs(current - previous)))
            if max_change <= MOTION_STABLE_DELTA_DEG:
                stable_reads += 1
                if stable_reads >= MOTION_STABLE_READS:
                    return time.monotonic()
            else:
                stable_reads = 0

        previous = current
        time.sleep(0.05)

    raise RuntimeError(
        "Robot did not become motion-stable before timeout. "
        "This gate checks only that motion stopped, not that a commanded pose was reached."
    )


def check_relative_joint_target(target_action, observation, motor_names):
    violations = []
    for name in motor_names:
        if name == "gripper":
            continue
        key = f"{name}.pos"
        delta = abs(float(target_action[key]) - float(observation[key]))
        if delta > MAX_RELATIVE_TARGET_DEG:
            violations.append((key, delta))

    if violations:
        detail = ", ".join(f"{key}={delta:.2f}deg" for key, delta in violations)
        raise RuntimeError(
            f"Refusing send_action: planned joint change exceeds {MAX_RELATIVE_TARGET_DEG:.1f} deg safety gate: "
            + detail
        )


def check_zero_delta_latch(pipeline, anchor_observation, motor_names):
    ordered = ordered_joint_observation(anchor_observation, motor_names)
    zero_action = make_cartesian_delta_action(
        0.0,
        0.0,
        delta_z_mm=0.0,
        max_delta_norm_mm=MAX_TOTAL_CORRECTION_MM,
    )
    zero_target = pipeline((zero_action, ordered))

    shifts = [
        abs(float(zero_target[f"{name}.pos"]) - float(ordered[f"{name}.pos"]))
        for name in motor_names
        if name != "gripper"
    ]
    max_shift = max(shifts, default=0.0)
    print(f"[ANCHOR] zero-delta planned joint shift: {max_shift:.3f} deg")
    if max_shift > 0.5:
        raise RuntimeError(
            "Refusing autonomy: zero-delta latch unexpectedly changes arm joints by "
            f"{max_shift:.3f} deg"
        )



def plan_fixed_anchor_xyz(
    pipeline,
    observation,
    *,
    cumulative_xy,
    cumulative_z_mm,
    motor_names,
    validation_kinematics,
    anchor_xyz_mm,
):
    ordered = ordered_joint_observation(
        observation,
        motor_names,
    )

    dx_mm = float(cumulative_xy[0])
    dy_mm = float(cumulative_xy[1])
    dz_mm = float(cumulative_z_mm)

    expected = np.asarray(
        [dx_mm, dy_mm, dz_mm],
        dtype=np.float64,
    )

    command_norm = float(np.linalg.norm(expected))
    if command_norm > MAX_TOTAL_CORRECTION_MM:
        raise RuntimeError(
            "Refusing XYZ command: fixed-anchor cumulative norm "
            f"{command_norm:.2f} mm exceeds "
            f"{MAX_TOTAL_CORRECTION_MM:.2f} mm."
        )

    delta_action = make_cartesian_delta_action(
        dx_mm,
        dy_mm,
        delta_z_mm=dz_mm,
        max_delta_norm_mm=MAX_TOTAL_CORRECTION_MM,
    )

    target_action = pipeline(
        (delta_action, ordered)
    )

    predicted_xyz = end_effector_xyz_mm(
        validation_kinematics,
        target_action,
        motor_names,
    )
    predicted_delta = predicted_xyz - anchor_xyz_mm
    model_error = predicted_delta - expected

    xy_error_mm = float(
        np.linalg.norm(model_error[:2])
    )
    z_error_mm = abs(float(model_error[2]))

    print(
        "[XYZ PLAN] request="
        f"({dx_mm:+.2f},{dy_mm:+.2f},{dz_mm:+.2f})mm "
        "fk="
        f"({predicted_delta[0]:+.2f},"
        f"{predicted_delta[1]:+.2f},"
        f"{predicted_delta[2]:+.2f})mm "
        f"model_error_xy={xy_error_mm:.2f}mm "
        f"model_error_z={z_error_mm:.2f}mm"
    )

    # These are PLAN sanity checks only.
    # They do NOT decide whether physical XYZ motion succeeded.
    if xy_error_mm > MAX_MODEL_XY_ERROR_MM:
        raise RuntimeError(
            "Refusing XYZ send: planned XY model error "
            f"{xy_error_mm:.3f} mm exceeds "
            f"{MAX_MODEL_XY_ERROR_MM:.3f} mm."
        )

    if z_error_mm > MAX_MODEL_Z_ERROR_MM:
        raise RuntimeError(
            "Refusing XYZ send: planned Z model error "
            f"{z_error_mm:.3f} mm exceeds "
            f"{MAX_MODEL_Z_ERROR_MM:.3f} mm."
        )

    # Direction sanity only.  This is NOT a measured-motion requirement.
    if dz_mm < 0.0 and predicted_delta[2] >= -1e-6:
        raise RuntimeError(
            "Refusing downward XYZ send: planner does not predict "
            "negative Z direction."
        )

    return target_action



def main():
    robot = SO101Follower(
        SO101FollowerConfig(
            port=ROBOT_PORT,
            id="lawson_follower_arm",
            use_degrees=True,
            max_relative_target=MAX_RELATIVE_TARGET_DEG,
            cameras={},
        )
    )
    leader = SO101Leader(
        SO101LeaderConfig(
            port=LEADER_PORT,
            id="lawson_leader_arm",
            use_degrees=True,
        )
    )
    camera = None

    try:
        robot.connect()
        leader.connect()
        motor_names = list(robot.bus.motors.keys())

        tool_reference = ToolReferenceCalibration.load(TOOL_REFERENCE)
        if tool_reference is None:
            raise RuntimeError("tool_reference.json is not calibrated")
        tool_reference.require_direct_tool_tip()

        jacobian = ImageJacobianCalibration.load(IMAGE_JACOBIAN)
        if jacobian is None:
            raise RuntimeError("image_jacobian.json is not calibrated")

        recognizer = RuntimeHOGGlyphRecognizer.load(GLYPH_MODEL)
        camera = ThreadedOpenCVCamera(CameraSpec.from_yaml(CAMERA_CONFIG))
        camera.start()

        pipeline = build_official_cartesian_pipeline(
            URDF,
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
        validation_kinematics = build_so101_kinematics(
            URDF,
            motor_names=motor_names,
        )

        print("=" * 68)
        print("PHASE 3 XYZ VALIDATION — FIXED GOAL ANCHOR / CUMULATIVE COMMAND")
        print("=" * 68)
        print("target              :", TARGET)
        print("p_tip               :", tool_reference.center)
        print("J_cmd               :")
        print(np.asarray(jacobian.matrix))
        print("step limit          :", MAX_STEP_NORM_MM, "mm")
        print("anchor command cap  :", MAX_TOTAL_CORRECTION_MM, "mm")
        print("Z command           : 0 during XY; cumulative -3/-6 mm in staged Z")
        print("XY success authority: WRIST visual error only")

        # Stay in manual teleoperation until a real glyph observation is
        # inside the Phase-3 precheck gate. Geometry propagation is intentionally
        # disabled here; semantic identity must be visible before autonomy starts.
        while True:
            _run_positioning_teleop(robot, leader, teleop_hz=60.0)
            positioned_timestamp = time.monotonic()

            pre_obs, pre_frame = capture_target(
                camera,
                recognizer,
                min_timestamp=positioned_timestamp,
            )
            pre_error_px, pre_error_norm = visual_error(
                tool_reference,
                pre_obs,
            )

            print(
                f"[PRECHECK] frame={pre_frame.frame_id} "
                f"target={pre_obs.center_px} "
                f"p_tip={tool_reference.center} "
                f"error=({pre_error_px[0]:+.2f},{pre_error_px[1]:+.2f})px "
                f"norm={pre_error_norm:.2f}px"
            )

            if pre_error_norm <= INITIAL_CAPTURE_THRESHOLD_PX:
                print(
                    f"[PRECHECK] PASS: {pre_error_norm:.2f}px <= "
                    f"{INITIAL_CAPTURE_THRESHOLD_PX:.2f}px; "
                    "locking autonomy anchor."
                )
                break

            print(
                f"[PRECHECK] HOLD: {pre_error_norm:.2f}px > "
                f"{INITIAL_CAPTURE_THRESHOLD_PX:.2f}px. "
                "No XY command was sent."
            )
            print(
                "[PRECHECK] Teleoperation will resume. "
                "Move the target center closer to "
                f"p_tip={tool_reference.center}, then press ENTER again."
            )

        print(
            "[HANDOFF] Waiting for follower to become motion-stable..."
        )
        handoff_settled_timestamp = wait_motion_stable(
            robot,
            motor_names,
        )

        # Physical encoder state is diagnostic/safety state only.
        present_observation = ordered_joint_observation(
            robot.get_observation(),
            motor_names,
        )

        # Preserve the command preload already holding the arm.
        # This is the true command-space anchor.
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

        print(
            "[COMMAND-ANCHOR] Goal - Present: "
            + ", ".join(
                f"{name}="
                f"{float(anchor_observation[f'{name}.pos']) - float(present_observation[f'{name}.pos']):+.3f}deg"
                for name in motor_names
                if name != "gripper"
            )
        )

        anchor_xyz_mm = end_effector_xyz_mm(
            validation_kinematics,
            anchor_observation,
            motor_names,
        )
        print(
            "[ANCHOR] FK xyz = "
            f"({anchor_xyz_mm[0]:.2f}, {anchor_xyz_mm[1]:.2f}, {anchor_xyz_mm[2]:.2f}) mm"
        )

        # The first pipeline call is a zero-delta PLAN ONLY. It latches the
        # autonomy handoff pose so all later commands are anchor-relative.
        check_zero_delta_latch(pipeline, anchor_observation, motor_names)

        servo_config = VisualServoConfig(
            gain=1.0,
            convergence_threshold_px=CONVERGENCE_THRESHOLD_PX,
            max_step_norm=MAX_STEP_NORM_MM,
            max_total_correction_norm=MAX_TOTAL_CORRECTION_MM,
        )

        cumulative = np.zeros(2, dtype=np.float64)
        commands_sent = 0
        stable_frames = 0
        last_frame_id = -1
        min_timestamp = (
            handoff_settled_timestamp + POST_MOTION_GUARD_S
        )

        while True:
            obs, frame = capture_target(
                camera,
                recognizer,
                after_frame_id=last_frame_id,
                min_timestamp=min_timestamp,
            )
            last_frame_id = int(frame.frame_id)
            error_px, error_norm = visual_error(tool_reference, obs)

            print()
            print(
                f"[VISION] frame={frame.frame_id} target={obs.center_px} "
                f"error=({error_px[0]:+.2f},{error_px[1]:+.2f})px "
                f"norm={error_norm:.2f}px "
                f"cumulative=({cumulative[0]:+.2f},{cumulative[1]:+.2f})mm"
            )

            step = compute_visual_servo_step(
                error_px,
                jacobian,
                config=servo_config,
                cumulative_correction=cumulative,
            )

            if step.status == VisualServoStepStatus.WITHIN_TOLERANCE:
                stable_frames += 1
                print(
                    f"[VISION] inside tolerance: "
                    f"{stable_frames}/{STABLE_FRAMES_REQUIRED} fresh frames"
                )
                if stable_frames >= STABLE_FRAMES_REQUIRED:
                    print("\n[PASS] XY alignment is visually stable.")
                    break
                time.sleep(0.15)
                continue

            stable_frames = 0
            if step.status == VisualServoStepStatus.BUDGET_EXHAUSTED:
                print(f"\n[STOP] {MAX_TOTAL_CORRECTION_MM:.0f} mm anchor-relative command budget exhausted.")
                break

            if commands_sent >= MAX_COMMANDS:
                print(
                    "\n[STOP] Maximum command count reached. "
                    "Final fresh visual observation was captured; "
                    "no additional command will be sent."
                )
                break

            next_cumulative = np.asarray(step.cumulative_after, dtype=np.float64)
            print(
                "[SERVO] increment="
                f"({step.correction[0]:+.2f},{step.correction[1]:+.2f})mm -> "
                "anchor-relative target="
                f"({next_cumulative[0]:+.2f},{next_cumulative[1]:+.2f})mm"
            )

            current_observation = ordered_joint_observation(
                robot.get_observation(),
                motor_names,
            )
            target_action, _, _, _ = _plan_delta(
                pipeline,
                current_observation,
                float(next_cumulative[0]),
                float(next_cumulative[1]),
                motor_names=motor_names,
                validation_kinematics=validation_kinematics,
                anchor_xyz_mm=anchor_xyz_mm,
                max_delta_norm_mm=MAX_TOTAL_CORRECTION_MM,
                max_model_xy_error_mm=MAX_MODEL_XY_ERROR_MM,
                max_model_z_error_mm=MAX_MODEL_Z_ERROR_MM,
            )
            check_relative_joint_target(
                target_action,
                current_observation,
                motor_names,
            )

            answer = input(
                f"Press ENTER to send XY command #{commands_sent + 1}; "
                "type q then ENTER to stop: "
            ).strip().lower()
            if answer == "q":
                print("[STOP] Operator stopped before send_action.")
                break

            sent_action = robot.send_action(target_action)
            for key, requested in target_action.items():
                if (
                    key.endswith(".pos")
                    and key != "gripper.pos"
                    and key in sent_action
                    and abs(float(sent_action[key]) - float(requested)) > 1e-6
                ):
                    raise RuntimeError(
                        f"Joint target clipped after send: {key} "
                        f"requested={float(requested):.3f} "
                        f"sent={float(sent_action[key]):.3f}"
                    )

            cumulative = next_cumulative
            commands_sent += 1
            settled_timestamp = wait_motion_stable(robot, motor_names)
            min_timestamp = settled_timestamp + POST_MOTION_GUARD_S
            print(
                f"[MOTION] command #{commands_sent} stopped; "
                "next control decision comes from a fresh WRIST frame"
            )

        # ========================================================
        # CUMULATIVE Z STAGE
        #
        # Critical semantics:
        #   * DO NOT relatch from Present_Position.
        #   * cumulative remains the current fixed-anchor XY command.
        #   * Z goes 0 -> -3 -> -6 mm against the SAME Goal anchor.
        #   * FK is planning/safety diagnostics only.
        #   * Lack of small measured Z motion is NOT a failure.
        # ========================================================
        z_commands_sent = 0
        cumulative_z_mm = 0.0
        z_stage_status = "not_started"
        xyz_abort = False

        print()
        print("=" * 68)
        print("CUMULATIVE Z STAGE")
        print("=" * 68)
        print(
            f"current XY visual error : {error_norm:.2f}px"
        )
        print(
            "fixed-anchor XY command  : "
            f"({cumulative[0]:+.2f},{cumulative[1]:+.2f})mm"
        )
        print(
            "planned Z levels         : "
            + ", ".join(f"{z:+.1f}mm" for z in Z_LEVELS_MM)
        )

        if error_norm > XY_TO_Z_HANDOFF_PX:
            z_stage_status = "skipped_xy_not_ready"
            print(
                "[Z SKIP] XY is not sufficiently aligned for staged Z: "
                f"{error_norm:.2f}px > {XY_TO_Z_HANDOFF_PX:.2f}px."
            )
        else:
            z_stage_status = "running"
            print(
                "[Z READY] XY is inside the Z-handoff region. "
                "The Goal_Position anchor remains unchanged."
            )

            for z_index, z_target_mm in enumerate(Z_LEVELS_MM, start=1):
                print()
                print(
                    f"===== Z LEVEL {z_index}/{len(Z_LEVELS_MM)}: "
                    f"cumulative Z={z_target_mm:+.2f} mm ====="
                )

                current_observation = ordered_joint_observation(
                    robot.get_observation(),
                    motor_names,
                )

                target_action = plan_fixed_anchor_xyz(
                    pipeline,
                    current_observation,
                    cumulative_xy=cumulative,
                    cumulative_z_mm=z_target_mm,
                    motor_names=motor_names,
                    validation_kinematics=validation_kinematics,
                    anchor_xyz_mm=anchor_xyz_mm,
                )

                check_relative_joint_target(
                    target_action,
                    current_observation,
                    motor_names,
                )

                answer = input(
                    f"Press ENTER to send cumulative XYZ command "
                    f"Z={z_target_mm:+.2f}mm; "
                    "type q then ENTER to stop: "
                ).strip().lower()

                if answer == "q":
                    print(
                        "[STOP] Operator stopped before staged Z send."
                    )
                    z_stage_status = "operator_stop"
                    xyz_abort = True
                    break

                sent_action = robot.send_action(target_action)

                for key, requested in target_action.items():
                    if (
                        key.endswith(".pos")
                        and key != "gripper.pos"
                        and key in sent_action
                        and abs(
                            float(sent_action[key]) - float(requested)
                        ) > 1e-6
                    ):
                        raise RuntimeError(
                            "Joint target clipped during staged XYZ send: "
                            f"{key} requested={float(requested):.3f} "
                            f"sent={float(sent_action[key]):.3f}"
                        )

                cumulative_z_mm = float(z_target_mm)
                z_commands_sent += 1

                settled_timestamp = wait_motion_stable(
                    robot,
                    motor_names,
                )
                min_timestamp = (
                    settled_timestamp + POST_MOTION_GUARD_S
                )

                # Fresh WRIST observation after the Z level.
                obs, frame = capture_target(
                    camera,
                    recognizer,
                    after_frame_id=last_frame_id,
                    min_timestamp=min_timestamp,
                )
                last_frame_id = int(frame.frame_id)

                error_px, error_norm = visual_error(
                    tool_reference,
                    obs,
                )

                print(
                    "[Z VISION] "
                    f"frame={frame.frame_id} "
                    f"target={obs.center_px} "
                    f"error=({error_px[0]:+.2f},"
                    f"{error_px[1]:+.2f})px "
                    f"norm={error_norm:.2f}px "
                    f"command="
                    f"({cumulative[0]:+.2f},"
                    f"{cumulative[1]:+.2f},"
                    f"{cumulative_z_mm:+.2f})mm"
                )

                # Diagnostic only: expose servo preload / dead zone.
                goal_now = robot.bus.sync_read(
                    "Goal_Position",
                    num_retry=robot.config.num_read_retries,
                )
                present_now = robot.bus.sync_read(
                    "Present_Position",
                    num_retry=robot.config.num_read_retries,
                )

                print(
                    "[XYZ DIAG] Goal-Present after settle: "
                    + ", ".join(
                        f"{name}="
                        f"{float(goal_now[name]) - float(present_now[name]):+.3f}deg"
                        for name in motor_names
                        if name != "gripper"
                    )
                )

                # ------------------------------------------------
                # If changing Z changes image alignment, re-align XY
                # AT THIS SAME CUMULATIVE Z LEVEL.
                #
                # Never relatch. Never reset Z. Never use Present as
                # a new command origin.
                # ------------------------------------------------
                realign_sent = 0

                while (
                    error_norm > Z_LEVEL_XY_TOLERANCE_PX
                    and realign_sent < MAX_XY_REALIGN_COMMANDS_PER_Z
                ):
                    print(
                        "[Z-LEVEL XY] visual alignment changed at "
                        f"Z={cumulative_z_mm:+.2f}mm: "
                        f"{error_norm:.2f}px > "
                        f"{Z_LEVEL_XY_TOLERANCE_PX:.2f}px."
                    )

                    step = compute_visual_servo_step(
                        error_px,
                        jacobian,
                        config=servo_config,
                        cumulative_correction=cumulative,
                    )

                    if step.status == VisualServoStepStatus.BUDGET_EXHAUSTED:
                        print(
                            "[Z-LEVEL XY STOP] XY command budget exhausted."
                        )
                        xyz_abort = True
                        z_stage_status = "xy_budget_exhausted_at_z"
                        break

                    if step.status == VisualServoStepStatus.WITHIN_TOLERANCE:
                        break

                    next_cumulative = np.asarray(
                        step.cumulative_after,
                        dtype=np.float64,
                    )

                    print(
                        "[Z-LEVEL XY] correction="
                        f"({step.correction[0]:+.2f},"
                        f"{step.correction[1]:+.2f})mm -> "
                        "fixed-anchor XYZ="
                        f"({next_cumulative[0]:+.2f},"
                        f"{next_cumulative[1]:+.2f},"
                        f"{cumulative_z_mm:+.2f})mm"
                    )

                    current_observation = ordered_joint_observation(
                        robot.get_observation(),
                        motor_names,
                    )

                    target_action = plan_fixed_anchor_xyz(
                        pipeline,
                        current_observation,
                        cumulative_xy=next_cumulative,
                        cumulative_z_mm=cumulative_z_mm,
                        motor_names=motor_names,
                        validation_kinematics=validation_kinematics,
                        anchor_xyz_mm=anchor_xyz_mm,
                    )

                    check_relative_joint_target(
                        target_action,
                        current_observation,
                        motor_names,
                    )

                    answer = input(
                        "Press ENTER to send XY re-alignment at "
                        f"Z={cumulative_z_mm:+.2f}mm; "
                        "type q then ENTER to stop: "
                    ).strip().lower()

                    if answer == "q":
                        print(
                            "[STOP] Operator stopped during Z-level XY re-alignment."
                        )
                        xyz_abort = True
                        z_stage_status = "operator_stop"
                        break

                    sent_action = robot.send_action(target_action)

                    for key, requested in target_action.items():
                        if (
                            key.endswith(".pos")
                            and key != "gripper.pos"
                            and key in sent_action
                            and abs(
                                float(sent_action[key])
                                - float(requested)
                            ) > 1e-6
                        ):
                            raise RuntimeError(
                                "Joint target clipped during Z-level "
                                f"XY re-alignment: {key}"
                            )

                    cumulative = next_cumulative
                    realign_sent += 1

                    settled_timestamp = wait_motion_stable(
                        robot,
                        motor_names,
                    )
                    min_timestamp = (
                        settled_timestamp + POST_MOTION_GUARD_S
                    )

                    obs, frame = capture_target(
                        camera,
                        recognizer,
                        after_frame_id=last_frame_id,
                        min_timestamp=min_timestamp,
                    )
                    last_frame_id = int(frame.frame_id)

                    error_px, error_norm = visual_error(
                        tool_reference,
                        obs,
                    )

                    print(
                        "[Z-LEVEL XY VISION] "
                        f"frame={frame.frame_id} "
                        f"error=({error_px[0]:+.2f},"
                        f"{error_px[1]:+.2f})px "
                        f"norm={error_norm:.2f}px"
                    )

                if xyz_abort:
                    break

                if error_norm > Z_LEVEL_XY_TOLERANCE_PX:
                    print(
                        "[Z STOP] Could not restore visual XY alignment "
                        f"within {MAX_XY_REALIGN_COMMANDS_PER_Z} commands "
                        f"at Z={cumulative_z_mm:+.2f}mm."
                    )
                    z_stage_status = "xy_not_restored_at_z"
                    break

                print(
                    f"[Z LEVEL PASS] cumulative Z={cumulative_z_mm:+.2f}mm; "
                    f"XY visual error={error_norm:.2f}px."
                )

            else:
                z_stage_status = "complete"
                print()
                print(
                    "[Z STAGE COMPLETE] Reached cumulative "
                    f"Z={cumulative_z_mm:+.2f}mm from the ORIGINAL "
                    "Goal_Position anchor."
                )
                print(
                    "[Z STAGE COMPLETE] This validates the staged XYZ "
                    "primitive only; it is NOT yet a keypress-success claim."
                )

        print()
        print("=" * 68)
        print("RESULT")
        print("commands sent      :", commands_sent)
        print(
            "cumulative command : "
            f"({cumulative[0]:+.2f}, {cumulative[1]:+.2f}) mm from fixed anchor"
        )
        print("Z commands sent    :", z_commands_sent)
        print(
            "cumulative Z       : "
            f"{cumulative_z_mm:+.2f} mm from fixed Goal anchor"
        )
        print("Z stage status     :", z_stage_status)
        print("=" * 68)

    finally:
        if camera is not None:
            camera.stop()

        if robot.is_connected and leader.is_connected:
            _run_recovery_teleop(robot, leader, teleop_hz=60.0)

        if leader.is_connected:
            leader.disconnect()
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
