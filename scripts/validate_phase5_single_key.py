from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from lerobot.robots.so_follower import (
    SO101Follower,
    SO101FollowerConfig,
)
from lerobot.teleoperators.so_leader import (
    SO101Leader,
    SO101LeaderConfig,
)

from calibrate_image_jacobian import (
    _run_positioning_teleop,
    _run_recovery_teleop,
)

from so101_typing.adapters.cameras import (
    CameraSpec,
    ThreadedOpenCVCamera,
)
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
    stepped_z_targets_to_zero,
)
from so101_typing.control.image_jacobian import (
    ImageJacobianCalibration,
)
from so101_typing.control.tool_reference import (
    ToolReferenceCalibration,
)
from so101_typing.control.visual_servo import (
    VisualServoConfig,
    VisualServoStepStatus,
    compute_visual_servo_step,
)
from so101_typing.perception.glyph_runtime import (
    RuntimeHOGGlyphRecognizer,
)
from so101_typing.perception.keycaps import (
    detect_keycaps,
)
from so101_typing.perception.screen_change import (
    FastSidePressEventWatcher,
)
from so101_typing.perception.screen_ocr import (
    TesseractScreenLineOCR,
    TesseractSingleCharacterOCR,
)
from so101_typing.perception.screen_rectify import (
    ScreenCalibration,
)
from so101_typing.perception.semantic_target_lock import (
    SemanticTargetLock,
)
from so101_typing.perception.wrist_target import (
    observe_target,
)
from so101_typing.supervisor.press_controller import (
    PressDirectiveKind,
    SingleKeyPressConfig,
    SingleKeyPressController,
)
from so101_typing.supervisor.verification import (
    ScreenVerificationStatus,
    verify_screen_texts,
)


ROBOT_PORT = "/dev/ttyACM0"
LEADER_PORT = "/dev/ttyACM1"

DEFAULT_TARGET = "G"
TARGET = DEFAULT_TARGET
CONFIRMED_PREFIX = "KEYPRESS"

URDF = (
    Path.home()
    / ".cache/huggingface/lerobot/robot-urdfs/so101/so101_new_calib.urdf"
)

TOOL_REFERENCE = Path(
    "calibration/tool_reference.json"
)
IMAGE_JACOBIAN = Path(
    "calibration/image_jacobian.json"
)
SCREEN_CALIBRATION = Path(
    "calibration/screen_homography.json"
)

GLYPH_MODEL = Path(
    "artifacts/models/glyph_hog_svm"
)

WRIST_CAMERA_CONFIG = Path(
    "configs/cameras/wrist.yaml"
)
SIDE_CAMERA_CONFIG = Path(
    "configs/cameras/screen.yaml"
)

ARTIFACT_DIR = Path(
    "artifacts/phase5_single_key"
)

MAX_RELATIVE_TARGET_DEG = 10.0

MAX_STEP_NORM_MM = 5.0
GEOMETRY_MAX_STEP_NORM_MM = 2.0

# Once geometry fallback has taken over, a single sporadic glyph
# recognition is not allowed to immediately replace the geometry track.
SEMANTIC_REENTRY_MAX_DISTANCE_PX = 8.0
SEMANTIC_REENTRY_MAX_PAIR_PX = 4.0
SEMANTIC_REENTRY_REQUIRED_FRAMES = 2

# Runtime state for this one-key validation process.
# False initially; becomes True once geometry tracking is used.
SEMANTIC_REENTRY_GUARDED = False
MAX_XY_CORRECTION_MM = 35.0

# Broad planner envelope for supervised Phase-5 validation.
# This is NOT the press depth and NOT the normal stopping condition.
# Absolute downward safety is currently provided by the operator.
MAX_COMMAND_NORM_MM = 80.0

INITIAL_CAPTURE_THRESHOLD_PX = 80.0
INITIAL_ALIGNMENT_THRESHOLD_PX = 4.0
Z_ALIGNMENT_THRESHOLD_PX = 6.0

STABLE_FRAMES_REQUIRED = 2

MAX_INITIAL_XY_COMMANDS = None
MAX_REALIGN_COMMANDS_PER_Z = None

MOTION_STABLE_DELTA_DEG = 0.5
MOTION_STABLE_READS = 3
MOTION_STABLE_TIMEOUT_S = 4.0

POST_MOTION_GUARD_S = 0.40

# Freeze the accepted semantic/geometry reference while the robot settles.
# Frames during this interval are observation-only and must not advance
# SemanticTargetLock. After the hold, the first accepted observation
# commits exactly one new reference.
WRIST_REFERENCE_HOLD_S = 0.60

# A joint-space "motion stopped" decision does not guarantee that the
# wrist camera image has stopped oscillating.  Require several accepted
# target observations to agree before using one for visual servoing.
WRIST_TARGET_TIMEOUT_S = 15.0

# Visual settling is evaluated over a time window, not just a few
# consecutive frames.  This prevents an oscillation turning point from
# looking falsely "stable".
WRIST_SETTLE_WINDOW_S = 0.75
WRIST_SETTLE_MIN_SPAN_S = 0.55
WRIST_SETTLE_MIN_SAMPLES = 10
WRIST_SETTLE_P90_PX = 3.0

# After descent has started, one 2 mm Z step cannot legitimately move
# an already aligned target tens of pixels.  Reject such observations
# and re-observe without commanding motion.
LOCAL_WRIST_OUTLIER_PX = 40.0

SCREEN_FRAME_COUNT = 9
SCREEN_MIN_VOTE_FRACTION = 0.60
SCREEN_MAX_PREFIX_DISTANCE = 2
SCREEN_MAX_FRAME_AGE_MS = 100.0

SCREEN_EVENT_BASELINE_S = 1.4
SCREEN_EVENT_CHAR_FRAMES = 5
# Hardware evidence:
# - +2/+2 commanded-mm release was too small under load/backlash.
# - one +10 commanded-mm release reduced repeats, but immediate Present-FK
#   motion was only about +3.3 mm and the key still repeated.
#
# Use one +20 commanded-mm upward escape command. This remains below the
# LeRobot 35 mm EE-jump guard. Latest-sent XY is preserved exactly.
SCREEN_EVENT_RELEASE_MM = 20.0
SCREEN_EVENT_RELEASE_STEP_MM = 20.0
# Release is an upward safety retreat. Give it the same bounded
# planner-model XY tolerance used by segmented retract. Normal press motion
# keeps the existing stricter default planner gate.
RELEASE_MAX_MODEL_XY_ERROR_MM = 3.0
RETRACT_MAX_MODEL_XY_ERROR_MM = 3.0

# The watcher only latches an event. Robot commands always remain on the
# foreground control thread.
ACTIVE_SCREEN_WATCHER = None


class ScreenPressEventDetected(RuntimeError):
    def __init__(self, details):
        super().__init__("SIDE detected a persistent press event")
        self.details = dict(details or {})


def raise_if_screen_event():
    watcher = ACTIVE_SCREEN_WATCHER
    if watcher is not None and watcher.triggered:
        raise ScreenPressEventDetected(watcher.snapshot())


# Coarse approach while clearly above the key, then 1 mm increments near
# expected contact. SIDE verification remains the only success authority.
Z_STEP_MM = 2.0

# Retract is deliberately much faster than the 2 mm press descent,
# but remains comfortably below the LeRobot 35 mm EE-jump guard.
# Every target is cumulative from the same fixed Goal-space anchor.
RETRACT_Z_STEP_MM = 10.0

# Phase-5 supervised validation:
# no automatic cumulative-depth limit.
#
# Absolute Z safety is temporarily provided by the operator.
# Press Ctrl+C immediately if the tool approaches an unsafe depth.


def save_wrist_debug_image(
    image,
    *,
    name,
    centers=None,
    note=None,
):
    debug_dir = (
        ARTIFACT_DIR
        / "wrist_debug"
    )
    debug_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        debug_dir
        / f"{name}.png"
    )

    cv2.imwrite(
        str(raw_path),
        image,
    )

    overlay = image.copy()

    if centers is not None:
        for center in centers:
            x = int(round(float(center[0])))
            y = int(round(float(center[1])))

            cv2.circle(
                overlay,
                (x, y),
                4,
                (0, 255, 0),
                1,
            )

    if note:
        cv2.putText(
            overlay,
            str(note),
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    overlay_path = (
        debug_dir
        / f"{name}_overlay.png"
    )

    cv2.imwrite(
        str(overlay_path),
        overlay,
    )

    print(
        "[WRIST DEBUG] saved "
        f"{raw_path}"
    )
    print(
        "[WRIST DEBUG] saved "
        f"{overlay_path}"
    )

    return (
        raw_path,
        overlay_path,
    )

def wait_motion_stable(
    robot,
    motor_names,
):
    arm_names = [
        name
        for name in motor_names
        if name != "gripper"
    ]

    deadline = (
        time.monotonic()
        + MOTION_STABLE_TIMEOUT_S
    )

    previous = None
    stable_reads = 0

    while time.monotonic() < deadline:
        raise_if_screen_event()
        observation = (
            robot.get_observation()
        )

        current = np.asarray(
            [
                float(
                    observation[
                        f"{name}.pos"
                    ]
                )
                for name in arm_names
            ],
            dtype=np.float64,
        )

        if previous is not None:
            max_change = float(
                np.max(
                    np.abs(
                        current - previous
                    )
                )
            )

            if (
                max_change
                <= MOTION_STABLE_DELTA_DEG
            ):
                stable_reads += 1

                if (
                    stable_reads
                    >= MOTION_STABLE_READS
                ):
                    return time.monotonic()

            else:
                stable_reads = 0

        previous = current

        time.sleep(0.05)

    raise RuntimeError(
        "Robot did not become motion-stable before timeout. "
        "This checks stopped motion, not Goal==Present."
    )


def fresh_camera_frame(
    camera,
    *,
    after_frame_id,
    min_timestamp=None,
    timeout_s=5.0,
):
    deadline = (
        time.monotonic()
        + timeout_s
    )

    last_seen = int(
        after_frame_id
    )

    while time.monotonic() < deadline:
        raise_if_screen_event()
        frame = camera.latest(
            copy_image=True
        )

        if (
            frame is None
            or frame.frame_id
            <= last_seen
        ):
            time.sleep(0.005)
            continue

        last_seen = int(
            frame.frame_id
        )

        if (
            min_timestamp is not None
            and frame.capture_timestamp
            <= min_timestamp
        ):
            continue

        return frame

    raise RuntimeError(
        f"Timed out waiting for fresh "
        f"{camera.spec.name} frame"
    )


def capture_initial_semantic_target(
    camera,
    recognizer,
    semantic_lock,
    tool_reference,
    *,
    min_timestamp,
):
    last_frame_id = -1
    deadline = (
        time.monotonic() + 5.0
    )

    while time.monotonic() < deadline:
        frame = fresh_camera_frame(
            camera,
            after_frame_id=last_frame_id,
            min_timestamp=min_timestamp,
        )

        last_frame_id = int(
            frame.frame_id
        )

        observation = observe_target(
            frame.image,
            TARGET,
            recognizer,
        )

        if (
            not observation.found
            or observation.center_px is None
        ):
            continue

        candidates = detect_keycaps(
            frame.image
        )

        centers = [
            candidate.center
            for candidate in candidates
        ]

        if len(centers) < 4:
            continue

        semantic_lock.observe_semantic(
            target_center_px=(
                observation.center_px
            ),
            keycap_centers=centers,
        )

        error = np.asarray(
            tool_reference.error_px(
                observation.center_px
            ),
            dtype=np.float64,
        )

        error_norm = float(
            np.linalg.norm(error)
        )

        return (
            observation.center_px,
            error,
            error_norm,
            frame,
        )

    raise RuntimeError(
        "Could not establish initial semantic G lock "
        "from a fresh WRIST frame."
    )


def capture_locked_target(
    camera,
    recognizer,
    semantic_lock,
    *,
    after_frame_id,
    min_timestamp,
    commit_reference=True,
):
    """Acquire one authoritative WRIST target observation.

    Rules:

    1. Reference stays frozen through the mechanical-settle hold.
    2. One robot motion may commit at most one visual reference.
    3. Same-pose stability CHECK never mutates the live reference.
    4. Once geometry fallback is active, one sporadic semantic G is not
       allowed to immediately seize authority.
    5. Guarded semantic re-entry requires repeated consistent semantic
       observations and, when available, agreement with geometry.
    """

    global SEMANTIC_REENTRY_GUARDED

    deadline = (
        time.monotonic()
        + WRIST_TARGET_TIMEOUT_S
    )

    last_seen = int(
        after_frame_id
    )

    first_capture_timestamp = None

    raw_frames = 0
    raw_semantic = 0
    geometry_attempts = 0
    rejected_geometry = 0
    last_keycaps = 0
    last_diag_time = time.monotonic()

    last_frame = None
    last_centers = []

    semantic_reentry_streak = 0
    semantic_reentry_last_center = None

    while time.monotonic() < deadline:
        raise_if_screen_event()
        frame = fresh_camera_frame(
            camera,
            after_frame_id=last_seen,
            min_timestamp=min_timestamp,
        )

        last_seen = int(
            frame.frame_id
        )

        last_frame = frame

        if first_capture_timestamp is None:
            first_capture_timestamp = float(
                frame.capture_timestamp
            )

        raw_frames += 1

        observation = observe_target(
            frame.image,
            TARGET,
            recognizer,
        )

        candidates = detect_keycaps(
            frame.image
        )

        centers = [
            candidate.center
            for candidate in candidates
        ]

        last_centers = centers
        last_keycaps = len(centers)

        # Save the FIRST fresh image following the most recent motion.
        # Because the filename is overwritten, after a failure it
        # corresponds to the final motion that preceded the failure.
        if (
            raw_frames == 1
            and commit_reference
        ):
            save_wrist_debug_image(
                frame.image,
                name="last_motion_after_first",
                centers=centers,
                note=(
                    f"FIRST AFTER MOTION "
                    f"frame={frame.frame_id} "
                    f"keycaps={len(centers)}"
                ),
            )

        semantic_found = (
            observation.found
            and observation.center_px
            is not None
        )

        if semantic_found:
            raw_semantic += 1

        observed_span_s = (
            float(frame.capture_timestamp)
            - first_capture_timestamp
        )

        # ----------------------------------------------------
        # Settle period: observe only.
        # ----------------------------------------------------
        if (
            observed_span_s
            < WRIST_REFERENCE_HOLD_S
        ):
            now = time.monotonic()

            if (
                now - last_diag_time
                >= 0.50
            ):
                print(
                    "[WRIST HOLD] "
                    f"span={observed_span_s:.2f}/"
                    f"{WRIST_REFERENCE_HOLD_S:.2f}s "
                    f"frames={raw_frames} "
                    f"semantic={raw_semantic} "
                    f"keycaps={last_keycaps} "
                    "reference=FROZEN"
                )

                last_diag_time = now

            continue

        # ----------------------------------------------------
        # Semantic path.
        # ----------------------------------------------------
        if semantic_found:
            semantic_center = np.asarray(
                observation.center_px,
                dtype=np.float64,
            )

            # Normal semantic authority before geometry fallback
            # has ever taken over.
            if not SEMANTIC_REENTRY_GUARDED:
                if commit_reference:
                    if len(centers) >= 4:
                        accepted = (
                            semantic_lock.observe_semantic(
                                target_center_px=(
                                    observation.center_px
                                ),
                                keycap_centers=centers,
                            )
                        )
                    else:
                        accepted = observation

                    mode = "COMMIT"

                else:
                    accepted = observation
                    mode = "CHECK"

                print(
                    f"[TRACK {mode}] "
                    "source=semantic "
                    f"frame={frame.frame_id} "
                    f"center={accepted.center_px} "
                    f"keycaps={len(centers)} "
                    f"hold={observed_span_s:.2f}s "
                    f"reference_mutated="
                    f"{commit_reference}"
                )

                return (
                    accepted,
                    frame,
                )

            # ------------------------------------------------
            # Guarded semantic re-entry.
            #
            # Probe geometry WITHOUT mutating the live lock.
            # ------------------------------------------------
            geometry_probe = None
            geometry_gap_px = None

            if (
                semantic_lock.locked
                and len(centers) >= 4
            ):
                probe_tracker = copy.deepcopy(
                    semantic_lock
                )

                probe = (
                    probe_tracker.propagate_geometry(
                        centers
                    )
                )

                if (
                    probe.found
                    and probe.center_px
                    is not None
                ):
                    geometry_probe = probe

                    geometry_center = np.asarray(
                        probe.center_px,
                        dtype=np.float64,
                    )

                    geometry_gap_px = float(
                        np.linalg.norm(
                            semantic_center
                            - geometry_center
                        )
                    )

            # If geometry is available and semantic strongly
            # disagrees with it, reject the semantic candidate.
            if (
                geometry_gap_px is not None
                and geometry_gap_px
                > SEMANTIC_REENTRY_MAX_DISTANCE_PX
            ):
                print(
                    "[SEMANTIC REENTRY REJECT] "
                    f"frame={frame.frame_id} "
                    f"semantic="
                    f"{observation.center_px} "
                    f"geometry="
                    f"{geometry_probe.center_px} "
                    f"gap={geometry_gap_px:.2f}px "
                    f"> "
                    f"{SEMANTIC_REENTRY_MAX_DISTANCE_PX:.2f}px"
                )

                semantic_reentry_streak = 0
                semantic_reentry_last_center = None

                # Geometry was valid, so use it instead.
                if geometry_probe is not None:
                    if commit_reference:
                        tracked = (
                            semantic_lock.propagate_geometry(
                                centers
                            )
                        )

                        if (
                            not tracked.found
                            or tracked.center_px
                            is None
                        ):
                            continue

                        SEMANTIC_REENTRY_GUARDED = True
                        mode = "COMMIT"

                    else:
                        tracked = geometry_probe
                        mode = "CHECK"

                    print(
                        f"[TRACK {mode}] "
                        "source=geometry "
                        f"frame={frame.frame_id} "
                        f"center={tracked.center_px} "
                        f"keycaps={len(centers)} "
                        "semantic_reentry=REJECTED "
                        f"reference_mutated="
                        f"{commit_reference}"
                    )

                    return (
                        tracked,
                        frame,
                    )

                continue

            # Semantic agrees with geometry, or geometry is
            # temporarily unavailable. Require repeated semantic
            # observations before semantic can regain authority.
            if (
                semantic_reentry_last_center
                is not None
                and float(
                    np.linalg.norm(
                        semantic_center
                        - semantic_reentry_last_center
                    )
                )
                <= SEMANTIC_REENTRY_MAX_PAIR_PX
            ):
                semantic_reentry_streak += 1
            else:
                semantic_reentry_streak = 1

            semantic_reentry_last_center = (
                semantic_center
            )

            print(
                "[SEMANTIC REENTRY] "
                f"frame={frame.frame_id} "
                f"streak={semantic_reentry_streak}/"
                f"{SEMANTIC_REENTRY_REQUIRED_FRAMES} "
                f"center={observation.center_px} "
                f"geometry_gap="
                f"{geometry_gap_px if geometry_gap_px is not None else 'NA'}"
            )

            if (
                semantic_reentry_streak
                >= SEMANTIC_REENTRY_REQUIRED_FRAMES
            ):
                if commit_reference:
                    if len(centers) >= 4:
                        accepted = (
                            semantic_lock.observe_semantic(
                                target_center_px=(
                                    observation.center_px
                                ),
                                keycap_centers=centers,
                            )
                        )
                    else:
                        accepted = observation

                    SEMANTIC_REENTRY_GUARDED = False
                    mode = "COMMIT"

                else:
                    accepted = observation
                    mode = "CHECK"

                print(
                    f"[TRACK {mode}] "
                    "source=semantic "
                    f"frame={frame.frame_id} "
                    f"center={accepted.center_px} "
                    "semantic_reentry=CONFIRMED "
                    f"reference_mutated="
                    f"{commit_reference}"
                )

                return (
                    accepted,
                    frame,
                )

            # Need another semantic frame before takeover.
            continue

        # Semantic disappeared again.
        semantic_reentry_streak = 0
        semantic_reentry_last_center = None

        # ----------------------------------------------------
        # Geometry fallback.
        # ----------------------------------------------------
        if (
            semantic_lock.locked
            and len(centers) >= 4
        ):
            geometry_attempts += 1

            tracker = (
                semantic_lock
                if commit_reference
                else copy.deepcopy(
                    semantic_lock
                )
            )

            tracked = (
                tracker.propagate_geometry(
                    centers
                )
            )

            if (
                tracked.found
                and tracked.center_px
                is not None
            ):
                SEMANTIC_REENTRY_GUARDED = True

                mode = (
                    "COMMIT"
                    if commit_reference
                    else "CHECK"
                )

                print(
                    f"[TRACK {mode}] "
                    "source=geometry "
                    f"frame={frame.frame_id} "
                    f"center={tracked.center_px} "
                    f"keycaps={len(centers)} "
                    f"hold={observed_span_s:.2f}s "
                    f"attempt={geometry_attempts} "
                    f"reference_mutated="
                    f"{commit_reference}"
                )

                return (
                    tracked,
                    frame,
                )

            rejected_geometry += 1

            if rejected_geometry == 1:
                save_wrist_debug_image(
                    frame.image,
                    name="reacquire_first_reject",
                    centers=centers,
                    note=(
                        f"FIRST GEOMETRY REJECT "
                        f"frame={frame.frame_id} "
                        f"keycaps={len(centers)}"
                    ),
                )

        now = time.monotonic()

        if (
            now - last_diag_time
            >= 1.0
        ):
            print(
                "[WRIST REACQUIRE] "
                f"frames={raw_frames} "
                f"semantic={raw_semantic} "
                f"keycaps={last_keycaps} "
                f"geometry_attempts="
                f"{geometry_attempts} "
                f"geometry_rejected="
                f"{rejected_geometry} "
                f"locked={semantic_lock.locked} "
                f"hold={observed_span_s:.2f}s "
                f"semantic_guard="
                f"{SEMANTIC_REENTRY_GUARDED} "
                f"commit_reference="
                f"{commit_reference}"
            )

            last_diag_time = now

        time.sleep(0.01)

    if last_frame is not None:
        save_wrist_debug_image(
            last_frame.image,
            name=(
                "reacquire_failure_"
                f"frame_{last_frame.frame_id}"
            ),
            centers=last_centers,
            note=(
                f"FAIL frame={last_frame.frame_id} "
                f"keycaps={last_keycaps} "
                f"semantic={raw_semantic} "
                f"geometry_rejected="
                f"{rejected_geometry}"
            ),
        )

    raise RuntimeError(
        "Target G could not be reacquired after the "
        "frozen-reference observation window. "
        f"frames={raw_frames}, "
        f"semantic={raw_semantic}, "
        f"geometry_attempts={geometry_attempts}, "
        f"geometry_rejected={rejected_geometry}, "
        f"last_keycaps={last_keycaps}, "
        f"locked={semantic_lock.locked}, "
        f"semantic_guard="
        f"{SEMANTIC_REENTRY_GUARDED}, "
        f"commit_reference={commit_reference}"
    )


def send_command_state(
    robot,
    planner,
    state,
    motor_names,
    *,
    label,
    sent_state_tracker,
    event_capture_timestamp=None,
):
    observation = (
        ordered_joint_observation(
            robot.get_observation(),
            motor_names,
        )
    )

    if str(label).startswith("RELEASE"):
        xy_error_limit = RELEASE_MAX_MODEL_XY_ERROR_MM
    elif str(label).startswith("RETRACT"):
        xy_error_limit = RETRACT_MAX_MODEL_XY_ERROR_MM
    else:
        xy_error_limit = None

    plan = planner.plan(
        state,
        observation,
        max_model_xy_error_mm=xy_error_limit,
    )

    print()
    print(
        f"[{label}] cumulative XYZ = "
        f"({state.x_mm:+.2f}, "
        f"{state.y_mm:+.2f}, "
        f"{state.z_mm:+.2f}) mm"
    )

    print(
        "[PLAN] predicted delta = "
        f"({plan.predicted_delta_mm[0]:+.2f}, "
        f"{plan.predicted_delta_mm[1]:+.2f}, "
        f"{plan.predicted_delta_mm[2]:+.2f}) mm "
        f"model_error_xy={plan.model_xy_error_mm:.3f} "
        f"model_error_z={plan.model_z_error_mm:.3f}"
    )

    action_send_timestamp = time.monotonic()

    if event_capture_timestamp is not None:
        event_to_send_ms = (
            action_send_timestamp
            - float(event_capture_timestamp)
        ) * 1000.0

        print(
            "[RELEASE LATENCY] "
            f"event_capture_to_send={event_to_send_ms:.1f}ms"
        )

    sent_action = robot.send_action(
        plan.joint_action
    )

    planner.validate_sent_action(
        plan,
        sent_action,
    )

    # The command has now been accepted by the robot interface. Record it
    # BEFORE waiting for motion stability because the asynchronous SIDE watcher
    # may interrupt from inside wait_motion_stable().
    sent_state_tracker.record_sent(
        state
    )

    settled_timestamp = (
        wait_motion_stable(
            robot,
            motor_names,
        )
    )

    # Diagnostic only.
    # Goal_Position remains the fixed command-space authority.
    # Present_Position is NEVER used to rebase the command anchor.
    goal_positions = robot.bus.sync_read(
        "Goal_Position",
        num_retry=robot.config.num_read_retries,
    )

    present_positions = robot.bus.sync_read(
        "Present_Position",
        num_retry=robot.config.num_read_retries,
    )

    goal_observation = {
        f"{name}.pos": float(goal_positions[name])
        for name in motor_names
    }

    present_observation = {
        f"{name}.pos": float(present_positions[name])
        for name in motor_names
    }

    goal_xyz_mm = end_effector_xyz_mm(
        planner.validation_kinematics,
        goal_observation,
        motor_names,
    )

    present_xyz_mm = end_effector_xyz_mm(
        planner.validation_kinematics,
        present_observation,
        motor_names,
    )

    goal_delta_mm = (
        goal_xyz_mm
        - planner.anchor_xyz_mm
    )

    present_delta_mm = (
        present_xyz_mm
        - planner.anchor_xyz_mm
    )

    goal_present_residuals = {
        name: (
            float(goal_positions[name])
            - float(present_positions[name])
        )
        for name in motor_names
        if name != "gripper"
    }

    max_residual_name = max(
        goal_present_residuals,
        key=lambda name: abs(
            goal_present_residuals[name]
        ),
    )

    print(
        f"[{label}] motion stopped."
    )

    print(
        "[PHYSICAL DIAG] "
        f"requested="
        f"({state.x_mm:+.2f},"
        f"{state.y_mm:+.2f},"
        f"{state.z_mm:+.2f})mm "
        f"Goal-FK="
        f"({goal_delta_mm[0]:+.2f},"
        f"{goal_delta_mm[1]:+.2f},"
        f"{goal_delta_mm[2]:+.2f})mm "
        f"Present-FK="
        f"({present_delta_mm[0]:+.2f},"
        f"{present_delta_mm[1]:+.2f},"
        f"{present_delta_mm[2]:+.2f})mm"
    )

    print(
        "[SERVO RESIDUAL] max Goal-Present = "
        f"{max_residual_name} "
        f"{goal_present_residuals[max_residual_name]:+.3f}deg"
    )

    return (
        settled_timestamp,
        plan,
    )


def align_wrist(
    *,
    robot,
    wrist_camera,
    recognizer,
    semantic_lock,
    tool_reference,
    jacobian,
    planner,
    state,
    controller,
    motor_names,
    last_frame_id,
    min_timestamp,
    threshold_px,
    max_commands,
    sent_state_tracker,
):
    semantic_servo_config = VisualServoConfig(
        gain=1.0,
        convergence_threshold_px=(
            threshold_px
        ),
        max_step_norm=(
            MAX_STEP_NORM_MM
        ),
        max_total_correction_norm=(
            MAX_XY_CORRECTION_MM
        ),
    )

    geometry_servo_config = VisualServoConfig(
        gain=1.0,
        convergence_threshold_px=(
            threshold_px
        ),
        max_step_norm=(
            GEOMETRY_MAX_STEP_NORM_MM
        ),
        max_total_correction_norm=(
            MAX_XY_CORRECTION_MM
        ),
    )

    stable_frames = 0
    commands_sent = 0

    while True:
        observation, frame = (
            capture_locked_target(
                wrist_camera,
                recognizer,
                semantic_lock,
                after_frame_id=(
                    last_frame_id
                ),
                min_timestamp=(
                    min_timestamp
                ),
                commit_reference=(
                    stable_frames == 0
                ),
            )
        )

        last_frame_id = int(
            frame.frame_id
        )

        error = np.asarray(
            tool_reference.error_px(
                observation.center_px
            ),
            dtype=np.float64,
        )

        error_norm = float(
            np.linalg.norm(error)
        )

        source = getattr(
            observation,
            "source",
            "semantic",
        )

        if (
            source == "semantic"
            and abs(float(state.z_mm)) < 1e-9
            and not SEMANTIC_REENTRY_GUARDED
        ):
            # Large steps are useful only during the initial,
            # clearly visible semantic acquisition phase.
            step_config = (
                semantic_servo_config
            )
            step_limit_mm = (
                MAX_STEP_NORM_MM
            )
        else:
            # Once geometry has taken over, or once descent has
            # started, keep visual-servo corrections conservative.
            step_config = (
                geometry_servo_config
            )
            step_limit_mm = (
                GEOMETRY_MAX_STEP_NORM_MM
            )

        if (
            state.z_mm < 0.0
            and source == "geometry"
            and error_norm
            > LOCAL_WRIST_OUTLIER_PX
        ):
            print(
                "[WRIST OUTLIER] "
                f"geometry error={error_norm:.2f}px "
                f"at Z={state.z_mm:+.2f}mm; "
                "holding position and re-observing"
            )

            stable_frames = 0
            last_frame_id = int(
                frame.frame_id
            )
            continue

        print(
            "[WRIST] "
            f"frame={frame.frame_id} "
            f"source={source} "
            f"error=({error[0]:+.2f},"
            f"{error[1]:+.2f})px "
            f"norm={error_norm:.2f}px "
            f"step_limit={step_limit_mm:.1f}mm "
            f"XYZ=({state.x_mm:+.2f},"
            f"{state.y_mm:+.2f},"
            f"{state.z_mm:+.2f})mm"
        )

        step = (
            compute_visual_servo_step(
                error,
                jacobian,
                config=step_config,
                cumulative_correction=(
                    state.xy_mm
                ),
            )
        )

        if (
            step.status
            is VisualServoStepStatus.WITHIN_TOLERANCE
        ):
            stable_frames += 1

            print(
                "[WRIST] inside tolerance "
                f"{stable_frames}/"
                f"{STABLE_FRAMES_REQUIRED}"
            )

            if (
                stable_frames
                >= STABLE_FRAMES_REQUIRED
            ):
                directive = (
                    controller.on_alignment(
                        aligned=True
                    )
                )

                return (
                    state,
                    directive,
                    last_frame_id,
                    min_timestamp,
                )

            continue

        stable_frames = 0

        directive = (
            controller.on_alignment(
                aligned=False
            )
        )

        if (
            directive.kind
            is not PressDirectiveKind.ALIGN_XY
        ):
            raise RuntimeError(
                "press controller did not "
                "authorize XY correction"
            )

        if (
            step.status
            is VisualServoStepStatus.BUDGET_EXHAUSTED
        ):
            raise RuntimeError(
                "XY cumulative command budget exhausted"
            )

        if (
            max_commands is not None
            and commands_sent >= max_commands
        ):
            raise RuntimeError(
                "XY command-count budget exhausted"
            )

        next_state = (
            state.with_xy_target(
                step.cumulative_after[0],
                step.cumulative_after[1],
            )
        )

        print(
            "[SERVO] correction="
            f"({step.correction[0]:+.2f},"
            f"{step.correction[1]:+.2f})mm "
            "-> cumulative XY="
            f"({next_state.x_mm:+.2f},"
            f"{next_state.y_mm:+.2f})mm "
            f"at Z={next_state.z_mm:+.2f}"
        )

        if source == "geometry":
            save_wrist_debug_image(
                frame.image,
                name=(
                    "last_geometry_command_before"
                ),
                note=(
                    f"geometry XY before motion "
                    f"frame={frame.frame_id} "
                    f"error={error_norm:.2f}px "
                    f"Z={state.z_mm:+.2f}mm"
                ),
            )

        debug_candidates = detect_keycaps(
            frame.image
        )
        debug_centers = [
            candidate.center
            for candidate in debug_candidates
        ]

        save_wrist_debug_image(
            frame.image,
            name="last_xy_command_before",
            centers=debug_centers,
            note=(
                f"BEFORE XY "
                f"frame={frame.frame_id} "
                f"source={source} "
                f"error={error_norm:.2f}px "
                f"cmd=("
                f"{step.correction[0]:+.2f},"
                f"{step.correction[1]:+.2f})mm "
                f"Z={state.z_mm:+.2f}mm"
            ),
        )

        debug_dir = (
            ARTIFACT_DIR
            / "wrist_debug"
        )
        debug_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            debug_dir
            / "last_xy_command_meta.json"
        ).write_text(
            json.dumps(
                {
                    "frame_id": int(
                        frame.frame_id
                    ),
                    "source": str(source),
                    "error_px": [
                        float(error[0]),
                        float(error[1]),
                    ],
                    "error_norm_px": float(
                        error_norm
                    ),
                    "correction_mm": [
                        float(
                            step.correction[0]
                        ),
                        float(
                            step.correction[1]
                        ),
                    ],
                    "xyz_before_mm": [
                        float(state.x_mm),
                        float(state.y_mm),
                        float(state.z_mm),
                    ],
                    "step_limit_mm": float(
                        step_limit_mm
                    ),
                },
                indent=2,
            )
        )

        settled_timestamp, _ = (
            send_command_state(
                robot,
                planner,
                next_state,
                motor_names,
                label="XY",
                sent_state_tracker=(
                    sent_state_tracker
                ),
            )
        )

        state = next_state

        commands_sent += 1

        min_timestamp = (
            settled_timestamp
            + POST_MOTION_GUARD_S
        )


def capture_screen_verification(
    *,
    side_camera,
    calibration,
    ocr,
    after_frame_id,
    min_timestamp,
):
    texts = []
    records = []

    last_frame_id = int(
        after_frame_id
    )

    print()
    print(
        "===== SIDE SCREEN VERIFY ====="
    )

    for index in range(
        1,
        SCREEN_FRAME_COUNT + 1,
    ):
        frame = fresh_camera_frame(
            side_camera,
            after_frame_id=(
                last_frame_id
            ),
            min_timestamp=(
                min_timestamp
            ),
        )

        last_frame_id = int(
            frame.frame_id
        )

        if (
            frame.frame_age_ms
            > SCREEN_MAX_FRAME_AGE_MS
        ):
            continue

        rectified = (
            calibration.rectify(
                frame.image
            )
        )

        roi = (
            calibration.crop_text_roi(
                rectified
            )
        )

        if roi is None:
            raise RuntimeError(
                "screen text ROI is missing"
            )

        observation = ocr.recognize(
            roi
        )

        texts.append(
            observation.normalized_text
        )

        records.append(
            {
                "frame_id": int(
                    frame.frame_id
                ),
                "confidence": float(
                    observation.mean_confidence
                ),
                "text": (
                    observation.normalized_text
                ),
            }
        )

        compact = (
            observation.normalized_text
            .replace("\n", " | ")
        )

        print(
            f"[SIDE] {index}/"
            f"{SCREEN_FRAME_COUNT} "
            f"frame={frame.frame_id} "
            f"conf="
            f"{observation.mean_confidence:.1f} "
            f"text={compact!r}"
        )

    if len(texts) < SCREEN_FRAME_COUNT:
        raise RuntimeError(
            "Could not acquire enough fresh "
            "SIDE OCR frames"
        )

    result = verify_screen_texts(
        texts,
        confirmed_prefix=(
            CONFIRMED_PREFIX
        ),
        expected_char=TARGET,
        min_vote_fraction=(
            SCREEN_MIN_VOTE_FRACTION
        ),
        max_prefix_distance=(
            SCREEN_MAX_PREFIX_DISTANCE
        ),
    )

    print()
    print(
        "[SCREEN RESULT] "
        f"{result.status} "
        f"success={result.success_votes}/"
        f"{result.total_frames} "
        f"no_change="
        f"{result.no_change_votes}/"
        f"{result.total_frames} "
        f"wrong={result.wrong_votes}/"
        f"{result.total_frames} "
        f"uncertain="
        f"{result.uncertain_votes}/"
        f"{result.total_frames} "
        f"wrong_char="
        f"{result.wrong_character}"
    )

    return (
        result,
        last_frame_id,
        records,
    )



def capture_event_character_verification(
    *,
    side_camera,
    calibration,
    change_model,
    char_ocr,
    after_frame_id,
    min_timestamp,
):
    texts = []
    records = []
    last_frame_id = int(after_frame_id)

    print()
    print("===== SIDE RELEASED-CHAR VERIFY =====")

    debug_dir = ARTIFACT_DIR / "side_event"
    debug_dir.mkdir(parents=True, exist_ok=True)

    for index in range(1, SCREEN_EVENT_CHAR_FRAMES + 1):
        frame = fresh_camera_frame(
            side_camera,
            after_frame_id=last_frame_id,
            min_timestamp=min_timestamp,
        )
        last_frame_id = int(frame.frame_id)

        rectified = calibration.rectify(frame.image)
        roi = calibration.crop_text_roi(rectified)
        if roi is None:
            raise RuntimeError("screen text ROI is missing during event verify")

        crop, change = change_model.extract_novel_crop(roi)
        character = None
        confidence = 0.0

        if crop is not None:
            cv2.imwrite(
                str(debug_dir / f"released_char_{index:02d}.png"),
                crop,
            )
            character, confidence = char_ocr.recognize_character(crop)

        synthetic = (
            CONFIRMED_PREFIX + character
            if character is not None
            else ""
        )
        texts.append(synthetic)
        records.append(
            {
                "frame_id": int(frame.frame_id),
                "character": character,
                "confidence": float(confidence),
                "novel_pixels": int(change.novel_pixels),
                "max_component_area_px": int(change.max_component_area_px),
            }
        )

        print(
            f"[SIDE EVENT OCR] {index}/{SCREEN_EVENT_CHAR_FRAMES} "
            f"frame={frame.frame_id} char={character!r} "
            f"conf={confidence:.1f} novel={change.novel_pixels}"
        )

    result = verify_screen_texts(
        texts,
        confirmed_prefix=CONFIRMED_PREFIX,
        expected_char=TARGET,
        min_vote_fraction=SCREEN_MIN_VOTE_FRACTION,
        max_prefix_distance=SCREEN_MAX_PREFIX_DISTANCE,
    )

    print()
    print(
        "[SIDE EVENT RESULT] "
        f"{result.status} "
        f"success={result.success_votes}/{result.total_frames} "
        f"wrong={result.wrong_votes}/{result.total_frames} "
        f"uncertain={result.uncertain_votes}/{result.total_frames} "
        f"wrong_char={result.wrong_character}"
    )

    return result, last_frame_id, records

def main():
    global ACTIVE_SCREEN_WATCHER

    robot = SO101Follower(
        SO101FollowerConfig(
            port=ROBOT_PORT,
            id="lawson_follower_arm",
            use_degrees=True,
            max_relative_target=(
                MAX_RELATIVE_TARGET_DEG
            ),
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

    wrist_camera = None
    side_camera = None
    screen_event_watcher = None

    payload = {
        "target": TARGET,
        "confirmed_prefix": (
            CONFIRMED_PREFIX
        ),
        "events": [],
        "screen": [],
    }

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        ARTIFACT_DIR
        / "validation.json"
    )

    try:
        print("=" * 72)
        print(
            "PHASE 5.3 — DETERMINISTIC "
            "SINGLE-KEY CLOSED LOOP"
        )
        print("=" * 72)
        print(
            f"target            : {TARGET}"
        )
        print(
            f"confirmed prefix  : "
            f"{CONFIRMED_PREFIX}"
        )
        print(
            "ACT               : DISABLED"
        )
        print(
            "success authority : SIDE screen only"
        )
        print(
            f"Z policy          : iterative "
            f"-{Z_STEP_MM:g} mm command-space steps"
        )
        print(
            "absolute Z guard  : OPERATOR / Ctrl+C"
        )
        print()

        print(
            "Before continuing, Mac screen "
            "must contain exactly:"
        )
        print()
        print(
            f"    {CONFIRMED_PREFIX}"
        )
        print()
        print(
            "with the caret immediately after "
            "the final S."
        )
        print(
            "Set Mac key-repeat delay to the "
            "longest value / disable repeat "
            "for this first validation."
        )

        input(
            "\nPress ENTER when the screen "
            "is ready: "
        )

        robot.connect()
        leader.connect()

        motor_names = list(
            robot.bus.motors.keys()
        )

        tool_reference = (
            ToolReferenceCalibration.load(
                TOOL_REFERENCE
            )
        )

        if tool_reference is None:
            raise RuntimeError(
                "tool reference is missing"
            )

        tool_reference.require_direct_tool_tip()

        jacobian = (
            ImageJacobianCalibration.load(
                IMAGE_JACOBIAN
            )
        )

        if jacobian is None:
            raise RuntimeError(
                "image Jacobian is missing"
            )

        recognizer = (
            RuntimeHOGGlyphRecognizer.load(
                GLYPH_MODEL
            )
        )

        screen_calibration = (
            ScreenCalibration.load(
                SCREEN_CALIBRATION
            )
        )

        if screen_calibration is None:
            raise RuntimeError(
                "screen calibration is missing"
            )

        screen_ocr = (
            TesseractScreenLineOCR(
                language="eng",
                scale=3.0,
                clahe_clip_limit=2.0,
                dark_threshold=185,
            )
        )

        screen_char_ocr = (
            TesseractSingleCharacterOCR(
                language="eng",
                scale=4.0,
                clahe_clip_limit=2.0,
                dark_threshold=185,
            )
        )

        wrist_camera = (
            ThreadedOpenCVCamera(
                CameraSpec.from_yaml(
                    WRIST_CAMERA_CONFIG
                )
            )
        )

        side_camera = (
            ThreadedOpenCVCamera(
                CameraSpec.from_yaml(
                    SIDE_CAMERA_CONFIG
                )
            )
        )

        wrist_camera.start()
        side_camera.start()

        time.sleep(0.5)

        print()
        print(
            "===== SCREEN BASELINE ====="
        )

        baseline, side_frame_id, records = (
            capture_screen_verification(
                side_camera=side_camera,
                calibration=(
                    screen_calibration
                ),
                ocr=screen_ocr,
                after_frame_id=-1,
                min_timestamp=None,
            )
        )

        payload["screen"].append(
            {
                "stage": "baseline",
                "status": str(
                    baseline.status
                ),
                "records": records,
            }
        )

        if (
            baseline.status
            is not ScreenVerificationStatus.CONFIRMED_NO_CHANGE
        ):
            raise RuntimeError(
                "SIDE baseline is not "
                "CONFIRMED_NO_CHANGE. "
                "Do not move the robot."
            )

        semantic_lock = (
            SemanticTargetLock(
                TARGET
            )
        )

        while True:
            _run_positioning_teleop(
                robot,
                leader,
                teleop_hz=60.0,
            )

            positioned_timestamp = (
                time.monotonic()
            )

            (
                target_center,
                error,
                error_norm,
                pre_frame,
            ) = (
                capture_initial_semantic_target(
                    wrist_camera,
                    recognizer,
                    semantic_lock,
                    tool_reference,
                    min_timestamp=(
                        positioned_timestamp
                    ),
                )
            )

            print()
            print(
                "[PRECHECK] "
                f"frame={pre_frame.frame_id} "
                f"target={target_center} "
                f"p_tip="
                f"{tool_reference.center} "
                f"error="
                f"({error[0]:+.2f},"
                f"{error[1]:+.2f})px "
                f"norm={error_norm:.2f}px"
            )

            if (
                error_norm
                <= INITIAL_CAPTURE_THRESHOLD_PX
            ):
                print(
                    "[PRECHECK] PASS."
                )
                break

            print(
                "[PRECHECK] target is too far "
                "from p_tip for deterministic "
                "handoff."
            )
            print(
                "Teleoperation will resume."
            )

        print()
        print(
            "[HANDOFF] Waiting for robot "
            "motion to stop..."
        )

        handoff_settled_timestamp = (
            wait_motion_stable(
                robot,
                motor_names,
            )
        )

        present_observation = (
            ordered_joint_observation(
                robot.get_observation(),
                motor_names,
            )
        )

        goal_positions = (
            robot.bus.sync_read(
                "Goal_Position",
                num_retry=(
                    robot.config
                    .num_read_retries
                ),
            )
        )

        anchor = (
            FixedGoalAnchor
            .from_goal_positions(
                goal_positions,
                motor_names,
            )
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

        kinematics = (
            build_so101_kinematics(
                URDF,
                motor_names=motor_names,
            )
        )

        anchor_xyz_mm = (
            end_effector_xyz_mm(
                kinematics,
                anchor.as_observation(),
                motor_names,
            )
        )

        pipeline = (
            build_official_cartesian_pipeline(
                URDF,
                motor_names=motor_names,
                end_effector_bounds={
                    "min": [
                        -1.0,
                        -1.0,
                        -1.0,
                    ],
                    "max": [
                        1.0,
                        1.0,
                        1.0,
                    ],
                },
                max_ee_step_m=0.035,
                orientation_weight=0.01,
                raise_on_jump=True,
                use_latched_reference=True,
            )
        )

        planner = (
            FixedAnchorCartesianPlanner(
                pipeline=pipeline,
                validation_kinematics=(
                    kinematics
                ),
                motor_names=motor_names,
                anchor=anchor,
                anchor_xyz_mm=(
                    anchor_xyz_mm
                ),
                config=(
                    FixedAnchorPlannerConfig(
                        max_command_norm_mm=(
                            MAX_COMMAND_NORM_MM
                        ),
                        max_model_xy_error_mm=1.0,
                        max_model_z_error_mm=1.0,
                        max_relative_target_deg=(
                            MAX_RELATIVE_TARGET_DEG
                        ),
                        max_zero_joint_shift_deg=0.5,
                    )
                ),
            )
        )

        zero_shift = (
            planner.latch_zero_delta()
        )

        print(
            "[ANCHOR] zero-delta max "
            f"joint shift={zero_shift:.3f}deg"
        )

        state = (
            FixedAnchorXYZCommandState
            .at_anchor(
                anchor,
                max_xy_norm_mm=(
                    MAX_XY_CORRECTION_MM
                ),
                max_xyz_norm_mm=(
                    MAX_COMMAND_NORM_MM
                ),
            )
        )

        sent_state_tracker = (
            LatestSentXYZCommandState(
                state
            )
        )

        controller = (
            SingleKeyPressController(
                SingleKeyPressConfig(
                    z_levels_mm=None,
                    z_step_mm=Z_STEP_MM,
                    max_descent_mm=None,
                    max_uncertain_reobservations=1,
                )
            )
        )

        wrist_frame_id = int(
            pre_frame.frame_id
        )

        min_wrist_timestamp = (
            handoff_settled_timestamp
            + POST_MOTION_GUARD_S
        )

        print()
        print("=" * 72)
        print(
            "AUTONOMY ARMED"
        )
        print("=" * 72)
        print(
            "From here the controller may "
            "perform bounded XY correction "
            "and iterative downward Z steps automatically."
        )
        print(
            "Ctrl+C aborts autonomy and "
            "returns to operator recovery."
        )

        answer = input(
            "Type GO then ENTER to start "
            "the closed loop: "
        ).strip().upper()

        if answer != "GO":
            raise RuntimeError(
                "Operator did not arm autonomy."
            )

        print()
        print("===== FAST SIDE EVENT BASELINE =====")

        screen_event_watcher = FastSidePressEventWatcher(
            side_camera=side_camera,
            calibration=screen_calibration,
            artifact_dir=(ARTIFACT_DIR / "side_event"),
        )

        event_baseline = screen_event_watcher.arm(
            baseline_duration_s=SCREEN_EVENT_BASELINE_S
        )

        ACTIVE_SCREEN_WATCHER = screen_event_watcher

        print(
            "[SIDE EVENT] ARMED "
            f"baseline_frames={event_baseline['baseline_frames']} "
            f"continuation_x0={event_baseline['continuation_x0']}"
        )

        payload["events"].append(
            {
                "event": "side_event_armed",
                **event_baseline,
            }
        )

        try:
            (
                state,
                directive,
                wrist_frame_id,
                min_wrist_timestamp,
            ) = align_wrist(
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
                last_frame_id=(
                    wrist_frame_id
                ),
                min_timestamp=(
                    min_wrist_timestamp
                ),
                threshold_px=(
                    INITIAL_ALIGNMENT_THRESHOLD_PX
                ),
                max_commands=(
                    MAX_INITIAL_XY_COMMANDS
                ),
                sent_state_tracker=(
                    sent_state_tracker
                ),
            )

            while True:
                print()
                print(
                    "[SUPERVISOR] directive="
                    f"{directive.kind} "
                    f"reason={directive.reason!r}"
                )

                if (
                    directive.kind
                    is PressDirectiveKind.DESCEND_TO_Z
                ):
                    state = (
                        state.with_z_level(
                            directive.z_target_mm
                        )
                    )

                    settled_timestamp, plan = (
                        send_command_state(
                            robot,
                            planner,
                            state,
                            motor_names,
                            label="Z",
                            sent_state_tracker=(
                                sent_state_tracker
                            ),
                        )
                    )

                    payload["events"].append(
                        {
                            "event": "z_command",
                            "xyz_mm": list(
                                state.xyz_mm
                            ),
                            "predicted_delta_mm": list(
                                plan.predicted_delta_mm
                            ),
                        }
                    )

                    min_wrist_timestamp = (
                        settled_timestamp
                        + POST_MOTION_GUARD_S
                    )

                    directive = (
                        controller
                        .on_z_motion_stable()
                    )

                    continue

                if (
                    directive.kind
                    is PressDirectiveKind.OBSERVE_WRIST
                ):
                    (
                        state,
                        directive,
                        wrist_frame_id,
                        min_wrist_timestamp,
                    ) = align_wrist(
                        robot=robot,
                        wrist_camera=(
                            wrist_camera
                        ),
                        recognizer=recognizer,
                        semantic_lock=(
                            semantic_lock
                        ),
                        tool_reference=(
                            tool_reference
                        ),
                        jacobian=jacobian,
                        planner=planner,
                        state=state,
                        controller=controller,
                        motor_names=motor_names,
                        last_frame_id=(
                            wrist_frame_id
                        ),
                        min_timestamp=(
                            min_wrist_timestamp
                        ),
                        threshold_px=(
                            Z_ALIGNMENT_THRESHOLD_PX
                        ),
                        max_commands=(
                            MAX_REALIGN_COMMANDS_PER_Z
                        ),
                        sent_state_tracker=(
                            sent_state_tracker
                        ),
                    )

                    continue

                if directive.kind in {
                    PressDirectiveKind.VERIFY_SCREEN,
                    PressDirectiveKind.REOBSERVE_SCREEN,
                }:
                    result, side_frame_id, records = (
                        capture_screen_verification(
                            side_camera=side_camera,
                            calibration=(
                                screen_calibration
                            ),
                            ocr=screen_ocr,
                            after_frame_id=(
                                side_frame_id
                            ),
                            min_timestamp=(
                                min_wrist_timestamp
                            ),
                        )
                    )

                    payload["screen"].append(
                        {
                            "stage": (
                                f"z_{state.z_mm:+.1f}"
                            ),
                            "status": str(
                                result.status
                            ),
                            "wrong_character": (
                                result.wrong_character
                            ),
                            "records": records,
                        }
                    )

                    directive = (
                        controller.on_verification(
                            result.status
                        )
                    )

                    continue

                if (
                    directive.kind
                    is PressDirectiveKind.RETRACT_TO_Z0
                ):
                    retract_targets = (
                        stepped_z_targets_to_zero(
                            state.z_mm,
                            max_step_mm=(
                                RETRACT_Z_STEP_MM
                            ),
                        )
                    )

                    print()
                    print(
                        "[RETRACT PLAN] "
                        f"start_z={state.z_mm:+.2f}mm "
                        f"step<={RETRACT_Z_STEP_MM:.2f}mm "
                        f"segments={len(retract_targets)}"
                    )

                    for (
                        retract_index,
                        retract_z,
                    ) in enumerate(
                        retract_targets,
                        start=1,
                    ):
                        state = (
                            state.with_z_level(
                                retract_z
                            )
                        )

                        (
                            settled_timestamp,
                            plan,
                        ) = send_command_state(
                            robot,
                            planner,
                            state,
                            motor_names,
                            label=(
                                "RETRACT "
                                f"{retract_index}/"
                                f"{len(retract_targets)}"
                            ),
                            sent_state_tracker=(
                                sent_state_tracker
                            ),
                        )

                        payload["events"].append(
                            {
                                "event": "retract",
                                "segment": (
                                    retract_index
                                ),
                                "segment_count": (
                                    len(
                                        retract_targets
                                    )
                                ),
                                "xyz_mm": list(
                                    state.xyz_mm
                                ),
                                "predicted_delta_mm": list(
                                    plan.predicted_delta_mm
                                ),
                            }
                        )

                    if abs(state.z_mm) > 1e-12:
                        raise RuntimeError(
                            "bounded retract did not "
                            "finish at Z=0"
                        )

                    directive = (
                        controller
                        .on_retraction_complete()
                    )

                    continue

                if (
                    directive.kind
                    is PressDirectiveKind.COMPLETE
                ):
                    break

                raise RuntimeError(
                    "Unhandled supervisor directive: "
                    f"{directive.kind}"
                )

        except ScreenPressEventDetected as event_exc:
            # Disable the interrupt hook before release/verification commands.
            ACTIVE_SCREEN_WATCHER = None

            if screen_event_watcher is None:
                raise RuntimeError("SIDE event fired without an active watcher")

            screen_event_watcher.stop()
            event_details = screen_event_watcher.snapshot()

            print()
            print("!" * 72)
            print("SIDE PRESS EVENT LATCHED — RELEASE HAS PRIORITY")
            print("!" * 72)
            print(f"[SIDE EVENT] {event_details}")

            # The interrupt may have escaped from inside align_wrist() after one
            # or more local XY commands were already sent. The outer `state`
            # assignment does not happen until align_wrist() returns, so it can
            # legitimately be stale here. Always recover from the latest command
            # state that actually passed send_action validation.
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

            payload["events"].append(
                {
                    "event": "side_press_event",
                    **event_details,
                    "outer_xyz_mm": list(
                        outer_state_before_event.xyz_mm
                    ),
                    "latest_sent_xyz_mm": list(
                        state.xyz_mm
                    ),
                }
            )

            release_target_z = min(
                0.0,
                float(state.z_mm) + SCREEN_EVENT_RELEASE_MM,
            )

            release_directive = controller.on_screen_event(
                release_z_target_mm=release_target_z,
            )

            if release_directive.kind is not PressDirectiveKind.RELEASE_TO_Z:
                raise RuntimeError("SIDE event did not enter release state")

            release_index = 0
            last_release_timestamp = time.monotonic()

            while state.z_mm < release_target_z - 1e-12:
                release_index += 1
                next_z = min(
                    release_target_z,
                    state.z_mm + SCREEN_EVENT_RELEASE_STEP_MM,
                )
                state = state.with_z_level(next_z)

                last_release_timestamp, plan = send_command_state(
                    robot,
                    planner,
                    state,
                    motor_names,
                    label=f"RELEASE {release_index}",
                    sent_state_tracker=(
                        sent_state_tracker
                    ),
                    event_capture_timestamp=(
                        event_details.get(
                            "capture_timestamp"
                        )
                        if release_index == 1
                        else None
                    ),
                )

                payload["events"].append(
                    {
                        "event": "release",
                        "segment": release_index,
                        "xyz_mm": list(state.xyz_mm),
                        "predicted_delta_mm": list(plan.predicted_delta_mm),
                    }
                )

            directive = controller.on_release_complete()

            result, side_frame_id, records = capture_event_character_verification(
                side_camera=side_camera,
                calibration=screen_calibration,
                change_model=screen_event_watcher.model,
                char_ocr=screen_char_ocr,
                after_frame_id=int(event_details.get("frame_id", side_frame_id)),
                min_timestamp=last_release_timestamp,
            )

            payload["screen"].append(
                {
                    "stage": "released_press_event",
                    "status": str(result.status),
                    "wrong_character": result.wrong_character,
                    "records": records,
                }
            )

            directive = controller.on_verification(result.status)

            if directive.kind is PressDirectiveKind.REOBSERVE_SCREEN:
                result, side_frame_id, records = capture_event_character_verification(
                    side_camera=side_camera,
                    calibration=screen_calibration,
                    change_model=screen_event_watcher.model,
                    char_ocr=screen_char_ocr,
                    after_frame_id=side_frame_id,
                    min_timestamp=last_release_timestamp,
                )
                payload["screen"].append(
                    {
                        "stage": "released_press_event_reobserve",
                        "status": str(result.status),
                        "wrong_character": result.wrong_character,
                        "records": records,
                    }
                )
                directive = controller.on_verification(result.status)

            if directive.kind is not PressDirectiveKind.RETRACT_TO_Z0:
                raise RuntimeError(
                    "latched SIDE event must end in retract after released-screen verification"
                )

            retract_targets = stepped_z_targets_to_zero(
                state.z_mm,
                max_step_mm=RETRACT_Z_STEP_MM,
            )

            print()
            print(
                "[RETRACT PLAN] "
                f"start_z={state.z_mm:+.2f}mm "
                f"step<={RETRACT_Z_STEP_MM:.2f}mm "
                f"segments={len(retract_targets)}"
            )

            for retract_index, retract_z in enumerate(retract_targets, start=1):
                state = state.with_z_level(retract_z)
                _, plan = send_command_state(
                    robot,
                    planner,
                    state,
                    motor_names,
                    label=f"RETRACT {retract_index}/{len(retract_targets)}",
                    sent_state_tracker=(
                        sent_state_tracker
                    ),
                )
                payload["events"].append(
                    {
                        "event": "retract",
                        "segment": retract_index,
                        "segment_count": len(retract_targets),
                        "xyz_mm": list(state.xyz_mm),
                        "predicted_delta_mm": list(plan.predicted_delta_mm),
                    }
                )

            directive = controller.on_retraction_complete()
            if directive.kind is not PressDirectiveKind.COMPLETE:
                raise RuntimeError("event retract did not complete the controller")

        payload["final"] = {
            "controller_state": str(
                controller.state
            ),
            "outcome": (
                None
                if controller.pending_outcome
                is None
                else str(
                    controller.pending_outcome
                )
            ),
            "final_xyz_mm": list(
                state.xyz_mm
            ),
        }

        output_path.write_text(
            json.dumps(
                payload,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        print()
        print("=" * 72)
        print(
            "PHASE 5.3 RESULT"
        )
        print("=" * 72)
        print(
            "controller state :",
            controller.state,
        )
        print(
            "outcome          :",
            controller.pending_outcome,
        )
        print(
            "final XYZ        :",
            state.xyz_mm,
        )
        print(
            f"saved            : "
            f"{output_path}"
        )

    finally:
        ACTIVE_SCREEN_WATCHER = None

        if screen_event_watcher is not None:
            screen_event_watcher.stop()

        if wrist_camera is not None:
            wrist_camera.stop()

        if side_camera is not None:
            side_camera.stop()

        if (
            robot.is_connected
            and leader.is_connected
        ):
            _run_recovery_teleop(
                robot,
                leader,
                teleop_hz=60.0,
            )

        if leader.is_connected:
            leader.disconnect()

        if robot.is_connected:
            robot.disconnect()


def parse_cli_target() -> str:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the Phase-5 deterministic "
            "single-key closed loop."
        ),
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help=(
            "Single A-Z target key. "
            f"Default: {DEFAULT_TARGET}"
        ),
    )

    args = parser.parse_args()
    target = str(args.target).strip().upper()

    if (
        len(target) != 1
        or target < "A"
        or target > "Z"
    ):
        parser.error(
            "--target must be exactly one letter A-Z"
        )

    return target


if __name__ == "__main__":
    TARGET = parse_cli_target()
    main()
