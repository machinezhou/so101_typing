from __future__ import annotations

import argparse
import json
import queue
import signal
import time
from collections import Counter
from dataclasses import dataclass, replace
import math
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
from so101_typing.control.fixed_anchor_planner import (
    RelativeJointSafetyError,
)
from so101_typing.control.handoff_endpoint import (
    HandoffEndpointContract,
    HandoffFrameConfirmation,
)
from so101_typing.control.tool_reference import ToolReferenceCalibration
from so101_typing.perception.glyph_runtime import RuntimeHOGGlyphRecognizer
from so101_typing.perception.semantic_target_lock import SemanticTargetLock
from so101_typing.perception.released_text import (
    build_temporal_clean_text,
)
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
HANDOFF_ENDPOINT_CONTRACT = Path(
    "calibration/handoff_endpoint_contract.json"
)

# Historical Phase-5 capture reference only.
# This value MUST NOT transfer controller ownership.
PHASE5_CAPTURE_REFERENCE_THRESHOLD_PX = float(
    phase5.INITIAL_CAPTURE_THRESHOLD_PX
)

HANDOFF_CANDIDATE_MIN_KEYCAPS = 4
SCREEN_NO_EVENT_CONFIRM_S = 0.35
SCREEN_NO_EVENT_MIN_FRESH_FRAMES = 4

RELEASED_TEXT_GUARD_S = 0.20
RELEASED_TEXT_CAPTURE_S = 1.20
RELEASED_TEXT_MIN_FRESH_FRAMES = 12
RELEASED_TEXT_PERSISTENCE_FRACTION = 0.70
RELEASED_TEXT_CROP_MARGIN_PX = 10

# Phase-8 autonomous cumulative-depth fuse.
#
# Hardware evidence includes successful contact as deep as -64 commanded-mm.
# The deterministic controller uses 2 mm steps, so 70 mm preserves three
# additional steps of margin while remaining inside the existing 80 mm total
# cumulative XYZ planner envelope.
#
# Phase 8 has no separate 35 mm cumulative XY hard stop.  At every fixed Z,
# usable XY is the cross-section that remains inside the same 80 mm XYZ sphere.
#
# Commanded millimetres remain command-space units, not physical TCP accuracy.
AUTONOMOUS_MAX_DESCENT_MM = 70.0


class OperatorStop(RuntimeError):
    pass


class HomeStop(RuntimeError):
    pass


class HandoffRejected(RuntimeError):
    pass


def _validate_autonomous_safety_contract() -> float:
    """Validate the Phase-8 cumulative command-space safety envelope.

    The existing total cumulative XYZ planner envelope is the workspace fuse.
    Phase 8 deliberately has no independent 35 mm cumulative XY hard stop.
    """
    maximum = float(AUTONOMOUS_MAX_DESCENT_MM)
    z_step = float(phase5.Z_STEP_MM)
    xyz_budget = float(phase5.MAX_COMMAND_NORM_MM)

    if not math.isfinite(maximum) or maximum <= 0.0:
        raise RuntimeError("autonomous max descent must be finite and positive")
    if not math.isfinite(z_step) or z_step <= 0.0:
        raise RuntimeError("Z step must be finite and positive")

    steps = maximum / z_step
    if abs(steps - round(steps)) > 1e-9:
        raise RuntimeError(
            "autonomous max descent must align exactly with the iterative Z step"
        )

    if maximum >= xyz_budget:
        raise RuntimeError(
            "autonomous Z fuse must remain inside total XYZ planner envelope: "
            f"{maximum:.1f} >= {xyz_budget:.1f} commanded-mm"
        )

    return float(xyz_budget)


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


def _reset_single_key_attempt_state(
    *,
    target: str,
    policy,
    preprocessor,
    postprocessor,
) -> str:
    """Reset state that must never leak from one key attempt to the next."""
    normalized = str(target).strip().upper()
    if normalized not in TARGET_VOCAB[:26]:
        raise ValueError("single-key attempt target must be A-Z")

    phase5.TARGET = normalized
    phase5.ACTIVE_SCREEN_WATCHER = None
    phase5.SEMANTIC_REENTRY_GUARDED = False

    policy.reset()
    _processor_reset(preprocessor)
    _processor_reset(postprocessor)

    return normalized


def _capture_settled_endpoint(
    *,
    wrist_camera,
    recognizer,
    target: str,
    tool_reference,
    settled_timestamp: float,
    semantic_lock,
    after_frame_id: int,
) -> dict:
    """Measure the settled ACT endpoint using the already-locked identity.

    Target identity is established during ACT motion. The settled gate must
    preserve that same physical key; it must not create a fresh empty lock and
    require HOG to rediscover the glyph after the arm stops.
    """
    phase5.TARGET = target

    if (
        semantic_lock is None
        or not semantic_lock.locked
    ):
        raise RuntimeError(
            "settled endpoint requires an already-established "
            "moving semantic target lock"
        )

    # Geometry remains authoritative across transient glyph dropout.
    # Sporadic later semantic observations cannot silently replace the
    # already-established physical target identity.
    phase5.SEMANTIC_REENTRY_GUARDED = True

    min_timestamp = (
        float(settled_timestamp)
        + float(phase5.POST_MOTION_GUARD_S)
    )

    observation, frame = phase5.capture_locked_target(
        wrist_camera,
        recognizer,
        semantic_lock,
        after_frame_id=int(after_frame_id),
        min_timestamp=min_timestamp,
        commit_reference=True,
    )

    if (
        not observation.found
        or observation.center_px is None
    ):
        raise RuntimeError(
            "locked target unavailable at settled endpoint"
        )

    target_center = observation.center_px

    error = np.asarray(
        tool_reference.error_px(
            target_center
        ),
        dtype=np.float64,
    )

    error_norm = float(
        np.linalg.norm(error)
    )

    return {
        "status": "MEASURED",
        "frame_id": int(frame.frame_id),
        "capture_timestamp": float(frame.capture_timestamp),
        "settled_timestamp": float(settled_timestamp),
        "post_motion_guard_s": float(phase5.POST_MOTION_GUARD_S),
        "min_timestamp": min_timestamp,
        "target_center_px": [
            float(target_center[0]),
            float(target_center[1]),
        ],
        "tool_tip_px": [
            float(tool_reference.center[0]),
            float(tool_reference.center[1]),
        ],
        "error_px": [
            float(error[0]),
            float(error[1]),
        ],
        "error_norm_px": float(error_norm),
        "tracking_source": str(observation.source),
        "phase5_capture_threshold_px": (
            PHASE5_CAPTURE_REFERENCE_THRESHOLD_PX
        ),
        "within_phase5_capture_threshold": bool(
            error_norm
            <= PHASE5_CAPTURE_REFERENCE_THRESHOLD_PX
        ),
        "acceptance_role": "handoff_gate",
    }

def _retract_to_z0(
    *,
    robot,
    planner,
    state,
    motor_names,
    sent_state_tracker,
    events: list[dict],
    label_prefix: str = "RETRACT",
    nominal_step_mm: float | None = None,
    event_capture_timestamp: float | None = None,
):
    """Retreat to fixed-anchor Z=0 with safety-aware adaptive segmentation.

    RELEASE and RETRACT share this exact safety path. The caller may choose the
    nominal command-space Z segment size, but any segment that violates the
    retained per-send joint-slew gate is subdivided before a command is sent.
    All other planner/runtime failures remain fail-closed and propagate.
    """
    motion_label = str(label_prefix).strip().upper()
    if motion_label not in {"RELEASE", "RETRACT"}:
        raise ValueError("label_prefix must be RELEASE or RETRACT")

    event_name = motion_label.lower()
    step_mm = (
        float(phase5.RETRACT_Z_STEP_MM)
        if nominal_step_mm is None
        else float(nominal_step_mm)
    )
    if step_mm <= 0.0:
        raise ValueError("nominal_step_mm must be positive")

    retract_targets = phase5.stepped_z_targets_to_zero(
        state.z_mm,
        max_step_mm=step_mm,
    )

    print()
    print(
        f"[{motion_label} PLAN] "
        f"start_z={state.z_mm:+.2f}mm "
        f"nominal_step<={step_mm:.2f}mm "
        f"nominal_segments={len(retract_targets)} "
        "adaptive_joint_slew=ON"
    )

    actual_segment = 0
    nominal_segment_count = len(retract_targets)

    for nominal_index, nominal_target_z in enumerate(
        retract_targets,
        start=1,
    ):
        pending_targets = [
            (float(nominal_target_z), 0)
        ]

        while pending_targets:
            target_z, subdivision_depth = pending_targets.pop()
            from_z = float(state.z_mm)
            candidate_state = state.with_z_level(target_z)

            try:
                _, plan = phase5.send_command_state(
                    robot,
                    planner,
                    candidate_state,
                    motor_names,
                    label=(
                        f"{motion_label} "
                        f"{nominal_index}/{nominal_segment_count} "
                        f"depth={subdivision_depth}"
                    ),
                    sent_state_tracker=sent_state_tracker,
                    event_capture_timestamp=(
                        event_capture_timestamp
                        if actual_segment == 0
                        else None
                    ),
                )

            except RelativeJointSafetyError as exc:
                midpoint_z = 0.5 * (from_z + float(target_z))

                if not (
                    from_z + 1e-12
                    < midpoint_z
                    < float(target_z) - 1e-12
                ):
                    raise RuntimeError(
                        "adaptive retract cannot subdivide joint-slew "
                        "violation any further"
                    ) from exc

                next_depth = int(subdivision_depth) + 1

                print(
                    f"[{motion_label} SUBDIVIDE] "
                    f"nominal={nominal_index}/{nominal_segment_count} "
                    f"from={from_z:+.3f}mm "
                    f"rejected={float(target_z):+.3f}mm "
                    f"retry={midpoint_z:+.3f}mm "
                    f"depth={next_depth} "
                    f"limit={exc.limit_deg:.2f}deg/send"
                )

                events.append(
                    {
                        "event": f"{event_name}_subdivide",
                        "nominal_segment": int(nominal_index),
                        "nominal_segment_count": int(nominal_segment_count),
                        "subdivision_depth": int(next_depth),
                        "from_z_mm": float(from_z),
                        "rejected_target_z_mm": float(target_z),
                        "retry_target_z_mm": float(midpoint_z),
                        "joint_slew_limit_deg": float(exc.limit_deg),
                        "violations": [
                            {
                                "joint": str(name),
                                "delta_deg": float(delta_deg),
                            }
                            for name, delta_deg in exc.violations
                        ],
                    }
                )

                pending_targets.append(
                    (float(target_z), next_depth)
                )
                pending_targets.append(
                    (float(midpoint_z), next_depth)
                )
                continue

            state = candidate_state
            actual_segment += 1

            events.append(
                {
                    "event": event_name,
                    "segment": int(actual_segment),
                    "nominal_segment": int(nominal_index),
                    "nominal_segment_count": int(nominal_segment_count),
                    "subdivision_depth": int(subdivision_depth),
                    "xyz_mm": [float(v) for v in state.xyz_mm],
                    "predicted_delta_mm": [
                        float(v)
                        for v in plan.predicted_delta_mm
                    ],
                }
            )

    if abs(float(state.z_mm)) > 1e-12:
        raise RuntimeError(
            f"bounded {event_name} did not finish at Z=0"
        )

    return state




@dataclass(frozen=True, slots=True)
class ReleasedCharacterVerification:
    status: object
    total_frames: int
    success_votes: int
    wrong_votes: int
    uncertain_votes: int
    wrong_character: str | None
    winning_votes: int
    vote_fraction: float


def _normalize_released_character(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    if not text.isascii() or not text.isalnum():
        return None
    return text


def _released_character_for_verdict(
    value,
    confidence,
    *,
    target: str,
) -> str | None:
    """Convert OCR output into semantic verdict evidence.

    The raw recognizer output is preserved. This function only decides whether
    that observation is strong enough to support a semantic verdict.

    - expected target observations remain usable even at confidence 0.0;
      accepted hardware evidence showed correct glyphs can have low confidence.
    - repeated expected text such as QQ remains explicit WRONG evidence.
    - a different observation with non-finite or <=0 confidence is too weak
      to prove WRONG and is downgraded to UNCERTAIN.
    """
    observed = _normalize_released_character(value)
    expected = _normalize_released_character(target)

    if expected is None or len(expected) != 1:
        raise ValueError(
            "released-character verdict requires one ASCII alphanumeric target"
        )

    if observed is None:
        return None

    if observed == expected:
        return observed

    if (
        len(observed) > 1
        and all(character == expected for character in observed)
    ):
        return observed

    confidence_value = float(confidence)
    if (
        not math.isfinite(confidence_value)
        or confidence_value <= 0.0
    ):
        return None

    return observed


def _vote_released_characters(
    observed_characters: list[str | None],
    *,
    target: str,
) -> ReleasedCharacterVerification:
    if not observed_characters:
        raise ValueError("at least one released-character observation is required")

    expected = _normalize_released_character(target)
    if expected is None or len(expected) != 1:
        raise ValueError(
            "Phase-8 released-character verification currently requires "
            "one ASCII alphanumeric target"
        )

    normalized = [
        _normalize_released_character(value)
        for value in observed_characters
    ]
    total = len(normalized)
    required_votes = math.ceil(
        total * float(phase5.SCREEN_MIN_VOTE_FRACTION)
    )
    success_votes = sum(value == expected for value in normalized)
    wrong_counter = Counter(
        value
        for value in normalized
        if value is not None and value != expected
    )
    if wrong_counter:
        wrong_character, wrong_votes = wrong_counter.most_common(1)[0]
    else:
        wrong_character = None
        wrong_votes = 0
    uncertain_votes = sum(value is None for value in normalized)

    if success_votes >= required_votes:
        status = phase5.ScreenVerificationStatus.CONFIRMED_SUCCESS
        winning_votes = success_votes
    elif wrong_votes >= required_votes:
        status = phase5.ScreenVerificationStatus.CONFIRMED_WRONG
        winning_votes = wrong_votes
    else:
        status = phase5.ScreenVerificationStatus.UNCERTAIN
        winning_votes = max(success_votes, wrong_votes)

    return ReleasedCharacterVerification(
        status=status,
        total_frames=total,
        success_votes=int(success_votes),
        wrong_votes=int(wrong_votes),
        uncertain_votes=int(uncertain_votes),
        wrong_character=wrong_character,
        winning_votes=int(winning_votes),
        vote_fraction=float(winning_votes / total),
    )


def _capture_dynamic_side_snapshot(
    *,
    side_camera,
    calibration,
    run_dir: Path,
) -> tuple[dict, int]:
    """Prove SIDE/rectification are live without assuming any screen text."""
    frame = phase5.fresh_camera_frame(
        side_camera,
        after_frame_id=-1,
        min_timestamp=None,
    )
    rectified = calibration.rectify(frame.image)
    roi = calibration.crop_text_roi(rectified)
    if roi is None:
        raise RuntimeError("screen text ROI is missing")

    debug_dir = run_dir / "side_dynamic_baseline"
    debug_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(debug_dir / "full_frame.png"), frame.image):
        raise RuntimeError("failed to save dynamic SIDE baseline full frame")
    if not cv2.imwrite(str(debug_dir / "text_roi.png"), roi):
        raise RuntimeError("failed to save dynamic SIDE baseline text ROI")

    record = {
        "status": "DYNAMIC_SNAPSHOT",
        "frame_id": int(frame.frame_id),
        "capture_timestamp": float(frame.capture_timestamp),
        "frame_age_ms": float(frame.frame_age_ms),
        "roi_shape": [int(v) for v in roi.shape],
        "content_role": "uninterpreted_dynamic_baseline",
    }
    print(
        "[SIDE BASELINE] "
        f"frame={frame.frame_id} age={frame.frame_age_ms:.1f}ms "
        "dynamic ROI captured; no fixed text prefix is required"
    )
    return record, int(frame.frame_id)


def _confirm_no_persistent_screen_event(
    *,
    side_camera,
    screen_event_watcher,
    autonomy_stop: Event,
) -> dict:
    """Confirm watcher health and absence of a persistent screen event.

    The FastSidePressEventWatcher remains the authority for detecting new
    foreground.  This helper only proves that fresh SIDE frames continued to
    arrive during a short confirmation window while no event was latched.
    """
    deadline = time.monotonic() + SCREEN_NO_EVENT_CONFIRM_S
    last_frame_id = -1
    fresh_frames = 0

    while time.monotonic() < deadline:
        if autonomy_stop.is_set():
            raise OperatorStop("operator stop during SIDE no-event confirmation")

        # Raises ScreenPressEventDetected immediately if the watcher latched.
        phase5.raise_if_screen_event()

        frame = side_camera.latest(copy_image=False)
        if frame is not None and int(frame.frame_id) > last_frame_id:
            last_frame_id = int(frame.frame_id)
            if frame.frame_age_ms <= screen_event_watcher.config.max_frame_age_ms:
                fresh_frames += 1

        time.sleep(screen_event_watcher.config.poll_interval_s)

    # Close the small race at the end of the observation window.
    phase5.raise_if_screen_event()

    if fresh_frames < SCREEN_NO_EVENT_MIN_FRESH_FRAMES:
        raise RuntimeError(
            "SIDE no-event confirmation did not observe enough fresh frames: "
            f"{fresh_frames} < {SCREEN_NO_EVENT_MIN_FRESH_FRAMES}"
        )

    print(
        "[SIDE NO-CHANGE] "
        f"fresh_frames={fresh_frames} window={SCREEN_NO_EVENT_CONFIRM_S:.2f}s "
        "persistent_event=False"
    )
    return {
        "status": str(phase5.ScreenVerificationStatus.CONFIRMED_NO_CHANGE),
        "fresh_frames": int(fresh_frames),
        "window_s": float(SCREEN_NO_EVENT_CONFIRM_S),
    }


def _capture_released_character_verification(
    *,
    side_camera,
    calibration,
    change_model,
    char_ocr,
    target: str,
    after_frame_id: int,
    min_timestamp: float,
    artifact_dir: Path,
    event_bbox_xywh,
):
    last_frame_id = int(after_frame_id)

    print()
    print("===== SIDE RELEASED-TEXT TEMPORAL VERIFY =====")

    capture_dir = (
        artifact_dir
        / f"released_temporal_after_{last_frame_id:06d}"
    )
    capture_dir.mkdir(parents=True, exist_ok=False)

    time.sleep(RELEASED_TEXT_GUARD_S)
    capture_start = time.monotonic()
    capture_deadline = capture_start + RELEASED_TEXT_CAPTURE_S
    effective_min_timestamp = max(
        float(min_timestamp),
        float(capture_start),
    )

    post_rois: list[np.ndarray] = []
    frame_ids: list[int] = []

    while time.monotonic() < capture_deadline:
        frame = phase5.fresh_camera_frame(
            side_camera,
            after_frame_id=last_frame_id,
            min_timestamp=effective_min_timestamp,
        )
        last_frame_id = int(frame.frame_id)

        rectified = calibration.rectify(frame.image)
        roi = calibration.crop_text_roi(rectified)
        if roi is None:
            raise RuntimeError(
                "screen text ROI is missing during released-text verification"
            )

        post_rois.append(roi.copy())
        frame_ids.append(int(frame.frame_id))

        roi_path = (
            capture_dir
            / f"released_roi_{len(post_rois):03d}_frame_{frame.frame_id}.png"
        )
        if not cv2.imwrite(str(roi_path), roi):
            raise RuntimeError(f"failed to save {roi_path}")

    if len(post_rois) < RELEASED_TEXT_MIN_FRESH_FRAMES:
        raise RuntimeError(
            "released-text temporal verification did not collect enough "
            f"fresh SIDE frames: {len(post_rois)} < "
            f"{RELEASED_TEXT_MIN_FRESH_FRAMES}"
        )

    clean = build_temporal_clean_text(
        change_model,
        post_rois,
        persistence_fraction=RELEASED_TEXT_PERSISTENCE_FRACTION,
        crop_margin_px=RELEASED_TEXT_CROP_MARGIN_PX,
        event_bbox_xywh=event_bbox_xywh,
    )

    frequency_image = np.clip(
        clean.frequency * 255.0,
        0,
        255,
    ).astype(np.uint8)

    debug_images = {
        "temporal_frequency.png": frequency_image,
        "persistent_mask.png": clean.persistent_mask,
        "selected_stable_new_text_mask.png": clean.selected_mask,
        "median_post_roi.png": clean.median_roi,
        "clean_new_text.png": clean.clean_crop,
    }

    for name, image in debug_images.items():
        path = capture_dir / name
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"failed to save {path}")

    raw_character, confidence = char_ocr.recognize_characters(
        clean.clean_crop
    )
    character = _normalize_released_character(raw_character)
    verdict_character = _released_character_for_verdict(
        raw_character,
        confidence,
        target=target,
    )

    if character is not None and verdict_character is None:
        print(
            "[SIDE OCR GUARD] "
            f"observation={character!r} "
            f"conf={confidence:.1f} -> UNCERTAIN; "
            "insufficient evidence for WRONG"
        )

    result = _vote_released_characters(
        [verdict_character],
        target=target,
    )

    records = [
        {
            "evidence_type": "temporal_aggregate",
            "source_frame_count": len(post_rois),
            "first_frame_id": int(frame_ids[0]),
            "last_frame_id": int(frame_ids[-1]),
            "raw_character": raw_character,
            "character": character,
            "verdict_character": verdict_character,
            "confidence": float(confidence),
            "persistence_fraction": float(
                RELEASED_TEXT_PERSISTENCE_FRACTION
            ),
            "stable_bbox_xywh": [
                int(v) for v in clean.stable_bbox_xywh
            ],
            "crop_bbox_xywh": [
                int(v) for v in clean.crop_bbox_xywh
            ],
            "components": [
                dict(component)
                for component in clean.components
            ],
            "recognition_padding_px": int(
                clean.recognition_padding_px
            ),
            "artifact_dir": str(capture_dir),
        }
    ]

    print(
        "[SIDE CLEAN OCR] "
        f"frames={len(post_rois)} "
        f"char={character!r} "
        f"conf={confidence:.1f} "
        f"components={len(clean.components)} "
        f"stable_bbox={clean.stable_bbox_xywh} "
        f"ocr_pad={clean.recognition_padding_px}px"
    )
    print()
    print(
        "[SIDE EVENT RESULT] "
        f"{result.status} "
        f"observation={character!r} "
        f"verdict_observation={verdict_character!r} "
        f"wrong_char={result.wrong_character}"
    )

    return result, last_frame_id, records


def _run_full_deterministic_press(
    *,
    robot,
    wrist_camera,
    side_camera,
    target: str,
    motor_names: list[str],
    recognizer,
    semantic_lock,
    tool_reference,
    screen_calibration,
    screen_char_ocr,
    endpoint_settled_timestamp: float,
    endpoint_wrist_frame_id: int,
    run_dir: Path,
    autonomy_stop: Event,
    phase: dict,
    safety_context: dict,
) -> dict:
    """Run the accepted Phase-5 deterministic primitive from one Goal anchor.

    This is orchestration only.  Perception, visual servoing, Cartesian planning,
    SIDE event detection, OCR, release, and retract all reuse the accepted
    Phase-5 implementation.
    """
    phase5.TARGET = target
    phase5.ARTIFACT_DIR = run_dir / "deterministic"
    phase5.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    phase5.SEMANTIC_REENTRY_GUARDED = True
    phase5.ACTIVE_SCREEN_WATCHER = None

    jacobian = phase5.ImageJacobianCalibration.load(phase5.IMAGE_JACOBIAN)
    if jacobian is None:
        raise RuntimeError("image Jacobian is missing")

    if (
        semantic_lock is None
        or not semantic_lock.locked
    ):
        raise RuntimeError(
            "deterministic ownership requires the "
            "already-established target identity lock"
        )

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
        # No independent Phase-8 35 mm XY radius.  The total 80 mm
        # fixed-anchor XYZ sphere is authoritative.
        max_xy_norm_mm=phase5.MAX_COMMAND_NORM_MM,
        max_xyz_norm_mm=phase5.MAX_COMMAND_NORM_MM,
    )
    sent_state_tracker = phase5.LatestSentXYZCommandState(state)
    controller = phase5.SingleKeyPressController(
        phase5.SingleKeyPressConfig(
            z_levels_mm=None,
            z_step_mm=phase5.Z_STEP_MM,
            max_descent_mm=AUTONOMOUS_MAX_DESCENT_MM,
            max_uncertain_reobservations=1,
        )
    )

    # Expose only the already-created fixed-anchor press context to the outer
    # recovery owner.  If any exception/CTRL+C escapes this helper after a
    # negative-Z command, main() can retract locally to Z=0 before HOME.
    safety_context.clear()
    safety_context.update(
        {
            "planner": planner,
            "sent_state_tracker": sent_state_tracker,
            "motor_names": motor_names,
            "events": None,
        }
    )

    wrist_frame_id = int(endpoint_wrist_frame_id)
    min_wrist_timestamp = (
        float(endpoint_settled_timestamp)
        + float(phase5.POST_MOTION_GUARD_S)
    )
    side_frame_id = -1

    print()
    print("=" * 78)
    print("PHASE 8 DETERMINISTIC OWNERSHIP")
    print("=" * 78)
    print("ACT queue/processors : RESET; no further ACT command is allowed")
    print("command anchor       : existing Goal_Position")
    print("WRIST alignment      : Phase-5 accepted controller")
    print("Z / SIDE / OCR       : Phase-5 accepted single-key primitive")
    print("autonomous Z fuse    :", f"-{AUTONOMOUS_MAX_DESCENT_MM:.1f} commanded-mm")
    print("normal-path arming   : AUTOMATIC; no ENTER/GO prompt")
    print()
    print("[OWNERSHIP TRANSFER] deterministic controller armed automatically")
    if autonomy_stop.is_set():
        raise OperatorStop("operator stop before deterministic press")

    phase["name"] = "PRESS"
    screen_event_watcher = None
    events: list[dict] = []
    screen_records: list[dict] = []
    safety_context["events"] = events

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
                # Never send another downward command after a SIDE event has latched.
                phase5.raise_if_screen_event()
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
                no_change = _confirm_no_persistent_screen_event(
                    side_camera=side_camera,
                    screen_event_watcher=screen_event_watcher,
                    autonomy_stop=autonomy_stop,
                )
                screen_records.append(
                    {
                        "stage": f"z_{state.z_mm:+.1f}",
                        **no_change,
                    }
                )
                directive = controller.on_verification(
                    phase5.ScreenVerificationStatus.CONFIRMED_NO_CHANGE
                )
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

        # A fixed +20 commanded-mm release is not a physical key-release
        # guarantee on SO101 because backlash/compliance can leave the key held.
        # Once SIDE latches a press event, do not pause for OCR at an arbitrary
        # intermediate Z. Escape all the way to the known pre-press fixed-anchor
        # Z=0 first, using the same adaptive joint-slew-safe retreat path.
        release_target_z = 0.0
        release_directive = controller.on_screen_event(
            release_z_target_mm=release_target_z,
        )
        if (
            release_directive.kind
            is not phase5.PressDirectiveKind.RELEASE_TO_Z
        ):
            raise RuntimeError("SIDE event did not enter release state")

        state = _retract_to_z0(
            robot=robot,
            planner=planner,
            state=state,
            motor_names=motor_names,
            sent_state_tracker=sent_state_tracker,
            events=events,
            label_prefix="RELEASE",
            nominal_step_mm=phase5.SCREEN_EVENT_RELEASE_STEP_MM,
            event_capture_timestamp=event_details.get("capture_timestamp"),
        )
        last_release_timestamp = time.monotonic()

        directive = controller.on_release_complete()
        result, side_frame_id, records = (
            _capture_released_character_verification(
                side_camera=side_camera,
                calibration=screen_calibration,
                change_model=screen_event_watcher.model,
                char_ocr=screen_char_ocr,
                target=target,
                after_frame_id=int(
                    event_details.get("frame_id", side_frame_id)
                ),
                min_timestamp=last_release_timestamp,
                artifact_dir=(phase5.ARTIFACT_DIR / "side_event"),
                event_bbox_xywh=event_details.get("bbox_xywh"),
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
                _capture_released_character_verification(
                    side_camera=side_camera,
                    calibration=screen_calibration,
                    change_model=screen_event_watcher.model,
                    char_ocr=screen_char_ocr,
                    target=target,
                    after_frame_id=side_frame_id,
                    min_timestamp=last_release_timestamp,
                    artifact_dir=(phase5.ARTIFACT_DIR / "side_event"),
                    event_bbox_xywh=event_details.get("bbox_xywh"),
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


def _start_runtime_devices(
    *,
    top,
    wrist,
    side,
    robot,
    home_baseline: dict,
    screen_calibration,
    run_dir: Path,
) -> tuple[list[str], list[str], dict]:
    """Start cameras, capture the initial SIDE baseline, and connect the robot.

    This is the accepted Phase-8 startup sequence extracted without changing
    ordering. Session-level resources remain open after this function returns.
    """
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
    print("===== DYNAMIC SIDE BASELINE =====")
    screen_baseline, _ = _capture_dynamic_side_snapshot(
        side_camera=side,
        calibration=screen_calibration,
        run_dir=run_dir,
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

    return motor_names, keys, screen_baseline


def _stop_runtime_devices(
    *,
    top,
    wrist,
    side,
    robot,
    original_raise_if_screen_event,
    original_wait_motion_stable,
    signal_installed: bool,
    previous_sigint,
) -> None:
    """Stop session-level resources and restore global hooks.

    This preserves the accepted Phase-8 teardown ordering exactly.
    """
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


def _restore_canonical_home(
    *,
    robot,
    motor_names: list[str],
    home_baseline: dict,
    home_stop: Event,
) -> dict:
    """Restore the explicit recovery HOME pose.

    This is a mechanical extraction of the accepted Phase-8 final HOME path.
    It intentionally preserves the existing synchronize -> restore -> verify
    ordering and the two operator-stop checks.
    """
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

    return {
        "status": "PASS",
        "present_hold": home_hold,
        "restore": home_restore,
        "final_goal_diff_deg": float(home_diff),
    }


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
        help="ACT execution horizon before policy replans; Phase 7 retained baseline is 20.",
    )
    parser.add_argument("--robot-port", default=ROBOT_PORT)
    parser.add_argument("--recovery-home", type=Path, default=RECOVERY_HOME_CONFIG)
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()

    target = str(args.target).strip().upper()
    if target not in TARGET_VOCAB[:26]:
        raise ValueError("Phase-8 formal single-key runtime currently supports A-Z targets")
    if not np.isfinite(args.duration) or args.duration <= 0.0:
        raise ValueError("--duration must be finite and > 0")
    if not 1 <= args.n_action_steps <= CHUNK_SIZE:
        raise ValueError(f"--n-action-steps must be in [1, {CHUNK_SIZE}]")

    workspace_norm_limit_mm = _validate_autonomous_safety_contract()

    handoff_contract = HandoffEndpointContract.load(
        HANDOFF_ENDPOINT_CONTRACT
    )

    print("=" * 78)
    print("PHASE 8 — SINGLE-KEY RUNTIME HARDENING v4")
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
        "handoff gate        :",
        f"verified endpoint hull + "
        f"{handoff_contract.acceptance_margin_px:.1f}px margin; "
        f"{handoff_contract.required_consecutive_frames} consecutive "
        f"fresh WRIST frames; >= {HANDOFF_CANDIDATE_MIN_KEYCAPS} keycaps",
    )
    print(
        "80 px legacy role   :",
        "diagnostic reference only; NEVER transfers ownership",
    )
    print(
        "settled endpoint    :",
        "fresh WRIST after motion-stable + POST_MOTION_GUARD; "
        "must pass same verified endpoint contract",
    )
    print("deterministic phase  : SAME Goal anchor through align + Z + release + retract")
    print("success authority    : SIDE novel-character OCR == target")
    print("screen baseline      : dynamic current SIDE ROI; no fixed text prefix")
    print("baseline requirement : detectable current text; content arbitrary")
    print("normal-path prompts  : NONE (no ENTER / GO)")
    print("runtime precondition : caret/text prepared before command launch")
    print("Z policy             :", f"iterative -{phase5.Z_STEP_MM:g} mm command-space steps")
    print("autonomous Z fuse    :", f"-{AUTONOMOUS_MAX_DESCENT_MM:.1f} commanded-mm")
    print(
        "workspace envelope   :",
        f"cumulative XYZ norm <= {workspace_norm_limit_mm:.1f} commanded-mm; "
        "no separate Phase-8 35 mm XY budget",
    )
    print("Ctrl+C safety        : local fixed-anchor retract -> HOME; hold if retract fails")
    print("joint safety limit   :", f"{MAX_RELATIVE_TARGET_DEG:.1f} deg/send")
    print("success              : expected character -> retract -> AUTO HOME")
    print("timeout / Ctrl+C     : local fixed-anchor retract -> AUTO HOME")
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
    screen_char_ocr = phase5.TesseractSingleCharacterOCR(
        language="eng",
        scale=2.0,
        psm=13,
        whitelist="abcdefghijklmnopqrstuvwxyz",
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
        if current in {
            "ACT",
            "HANDOFF_SETTLE",
            "DETERMINISTIC",
            "PRESS",
            "STARTUP",
        }:
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

    # Endpoint handoff requires consecutive distinct WRIST frames.
    # The pure state machine is covered by unit tests.
    handoff_confirmation = HandoffFrameConfirmation()
    handoff_endpoint_streak = 0

    settled_endpoint = None
    deterministic_result = None
    deterministic_safety_context: dict = {}
    local_retract_result = {"status": "NOT_NEEDED"}
    screen_baseline = None
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
        print("[RUNTIME PRECONDITION] The caret must already be positioned after")
        print("the current text before this command is launched. Existing text may")
        print("be arbitrary; no fixed literal prefix is required.")
        print(
            "The current SIDE text ROI becomes the dynamic event baseline. "
            "This runtime still requires detectable baseline text so the accepted "
            "Phase-5 watcher can locate the continuation region. Keep key-repeat "
            "delay at the previously accepted Phase-5 setting."
        )
        print("[AUTO START] camera/baseline checks begin without operator input")

        motor_names, keys, screen_baseline = _start_runtime_devices(
            top=top,
            wrist=wrist,
            side=side,
            robot=robot,
            home_baseline=home_baseline,
            screen_calibration=screen_calibration,
            run_dir=run_dir,
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
        _reset_single_key_attempt_state(
            target=target,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
        )

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

        # Establish target identity once from semantics, then carry that same
        # physical key through ACT motion with keyboard geometry. HOG does
        # not need to rediscover the requested glyph on every moving frame.
        moving_target_lock = SemanticTargetLock(target)

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

            keycap_centers = [
                observation.center_px
                for observation in glyph_observations
            ]

            semantic_target_found = (
                target_observation.found
                and target_observation.center_px is not None
            )

            tracked_target_observation = None
            moving_tracking_source = "unlocked"

            if not moving_target_lock.locked:
                if (
                    semantic_target_found
                    and keycap_count
                    >= HANDOFF_CANDIDATE_MIN_KEYCAPS
                ):
                    tracked_target_observation = (
                        moving_target_lock.observe_semantic(
                            target_center_px=(
                                target_observation.center_px
                            ),
                            keycap_centers=keycap_centers,
                        )
                    )

                    moving_tracking_source = "semantic_seed"

                    print(
                        f"\n[MOVING TARGET LOCK] "
                        f"frame={int(wrist_frame.frame_id)} "
                        f"target={target} "
                        f"center="
                        f"{tracked_target_observation.center_px} "
                        f"keycaps={keycap_count}"
                    )

            elif (
                keycap_count
                >= HANDOFF_CANDIDATE_MIN_KEYCAPS
            ):
                # Geometry preserves the identity of the already-locked
                # physical key. A rejected geometry estimate does not mutate
                # the lock.
                tracked_target_observation = (
                    moving_target_lock.propagate_geometry(
                        keycap_centers
                    )
                )

                if (
                    tracked_target_observation.found
                    and tracked_target_observation.center_px
                    is not None
                ):
                    moving_tracking_source = "geometry"

                elif semantic_target_found:
                    # Geometry can temporarily be unavailable because of
                    # partial visibility or a large inter-frame change.
                    # Semantic recognition may then refresh the identity,
                    # but it still names the same requested key.
                    tracked_target_observation = (
                        moving_target_lock.observe_semantic(
                            target_center_px=(
                                target_observation.center_px
                            ),
                            keycap_centers=keycap_centers,
                        )
                    )

                    moving_tracking_source = "semantic_refresh"

            handoff_error_px = None
            handoff_error_norm_px = None

            handoff_endpoint_accepted = False
            handoff_endpoint_signed_distance_px = None
            handoff_endpoint_clearance_px = None
            handoff_candidate_now = False

            current_wrist_frame_id = int(
                wrist_frame.frame_id
            )

            if (
                tracked_target_observation is not None
                and tracked_target_observation.found
                and tracked_target_observation.center_px
                is not None
            ):
                handoff_error_px = np.asarray(
                    tool_reference.error_px(
                        tracked_target_observation.center_px
                    ),
                    dtype=np.float64,
                )
                handoff_error_norm_px = float(
                    np.linalg.norm(
                        handoff_error_px
                    )
                )

                # TARGET_ACQUIRED only means the requested glyph
                # is visible. Visibility alone NEVER transfers
                # controller ownership.
                if (
                    target_trigger is None
                    and semantic_target_found
                ):
                    target_trigger = {
                        "tick": tick,
                        "commands_sent": command_count,
                        "act_elapsed_s": float(
                            time.perf_counter()
                            - rollout_start
                        ),
                        "top_frame_id": int(
                            top_frame.frame_id
                        ),
                        "wrist_frame_id": (
                            current_wrist_frame_id
                        ),
                        "keycap_count": int(
                            keycap_count
                        ),
                        "tool_tip_px": list(
                            tool_reference.center
                        ),
                        "error_px": (
                            handoff_error_px.tolist()
                        ),
                        "error_norm_px": (
                            handoff_error_norm_px
                        ),
                        "observation": (
                            target_observation.to_dict()
                        ),
                    }

                    _append_jsonl(
                        events_path,
                        {
                            "event": "TARGET_ACQUIRED",
                            **target_trigger,
                        },
                    )

                    print(
                        f"\n[TARGET ACQUIRED] "
                        f"t={target_trigger['act_elapsed_s']:.3f}s "
                        f"commands={command_count} "
                        f"center="
                        f"{target_observation.center_px} "
                        f"error="
                        f"{handoff_error_norm_px:.2f}px "
                        f"keycaps={keycap_count}"
                    )

                    print(
                        "[ACT CONTINUES] visibility is "
                        "diagnostic only; verified endpoint "
                        "contract controls handoff."
                    )

                endpoint_evaluation = (
                    handoff_contract.evaluate(
                        handoff_error_px
                    )
                )

                handoff_endpoint_accepted = bool(
                    endpoint_evaluation.accepted
                )

                handoff_endpoint_signed_distance_px = (
                    float(
                        endpoint_evaluation
                        .signed_distance_to_hull_px
                    )
                )

                handoff_endpoint_clearance_px = float(
                    endpoint_evaluation
                    .acceptance_clearance_px
                )

                endpoint_eligible_now = (
                    keycap_count
                    >= HANDOFF_CANDIDATE_MIN_KEYCAPS
                    and handoff_endpoint_accepted
                )

                previous_streak = (
                    handoff_confirmation.streak
                )

                (
                    handoff_confirmation,
                    handoff_frame_is_fresh,
                ) = handoff_confirmation.observe(
                    current_wrist_frame_id,
                    eligible=endpoint_eligible_now,
                )

                handoff_endpoint_streak = (
                    handoff_confirmation.streak
                )

                if handoff_frame_is_fresh:
                    if endpoint_eligible_now:
                        print(
                            f"\n[HANDOFF WINDOW] "
                            f"frame="
                            f"{current_wrist_frame_id} "
                            f"streak="
                            f"{handoff_endpoint_streak}/"
                            f"{handoff_contract.required_consecutive_frames} "
                            f"error=("
                            f"{handoff_error_px[0]:+.2f},"
                            f"{handoff_error_px[1]:+.2f})px "
                            f"norm="
                            f"{handoff_error_norm_px:.2f}px "
                            f"clearance="
                            f"{handoff_endpoint_clearance_px:+.2f}px "
                            f"source={moving_tracking_source}"
                        )

                    elif previous_streak > 0:
                        print(
                            f"\n[HANDOFF RESET] "
                            f"frame="
                            f"{current_wrist_frame_id} "
                            "fresh observation left the "
                            "verified endpoint region."
                        )

                handoff_candidate_now = (
                    handoff_confirmation.confirmed(
                        handoff_contract
                        .required_consecutive_frames
                    )
                )

                if (
                    handoff_frame_is_fresh
                    and handoff_candidate_now
                ):
                    rollout_stop = (
                        time.perf_counter()
                    )

                    # Ownership boundary:
                    # flush every queued ACT command before
                    # deterministic control can ever be allowed.
                    policy.reset()
                    _processor_reset(preprocessor)
                    _processor_reset(postprocessor)

                    handoff_candidate_trigger = {
                        "tick": tick,
                        "commands_sent": command_count,
                        "act_elapsed_s": float(
                            rollout_stop
                            - rollout_start
                        ),
                        "top_frame_id": int(
                            top_frame.frame_id
                        ),
                        "wrist_frame_id": (
                            current_wrist_frame_id
                        ),
                        "keycap_count": int(
                            keycap_count
                        ),
                        "tool_tip_px": list(
                            tool_reference.center
                        ),
                        "error_px": (
                            handoff_error_px.tolist()
                        ),
                        "error_norm_px": (
                            handoff_error_norm_px
                        ),
                        "contract_path": str(
                            HANDOFF_ENDPOINT_CONTRACT
                        ),
                        "contract_margin_px": float(
                            handoff_contract
                            .acceptance_margin_px
                        ),
                        "required_consecutive_frames": (
                            int(
                                handoff_contract
                                .required_consecutive_frames
                            )
                        ),
                        "confirmed_streak": int(
                            handoff_endpoint_streak
                        ),
                        "signed_distance_to_hull_px": (
                            handoff_endpoint_signed_distance_px
                        ),
                        "acceptance_clearance_px": (
                            handoff_endpoint_clearance_px
                        ),
                        "moving_tracking_source": (
                            moving_tracking_source
                        ),
                        "tracked_target_center_px": list(
                            tracked_target_observation.center_px
                        ),
                        "observation": (
                            target_observation.to_dict()
                        ),
                    }

                    _append_jsonl(
                        events_path,
                        {
                            "event": (
                                "HANDOFF_CANDIDATE"
                            ),
                            **handoff_candidate_trigger,
                        },
                    )

                    task_status = (
                        "HANDOFF_CANDIDATE"
                    )

                    print(
                        f"\n[HANDOFF CONFIRMED] "
                        f"t="
                        f"{handoff_candidate_trigger['act_elapsed_s']:.3f}s "
                        f"commands={command_count} "
                        f"streak="
                        f"{handoff_endpoint_streak}/"
                        f"{handoff_contract.required_consecutive_frames} "
                        f"error=("
                        f"{handoff_error_px[0]:+.2f},"
                        f"{handoff_error_px[1]:+.2f})px "
                        f"norm="
                        f"{handoff_error_norm_px:.2f}px "
                        f"clearance="
                        f"{handoff_endpoint_clearance_px:+.2f}px"
                    )

                    print(
                        "[ACT STOP] locked target reached the "
                        "verified demonstration endpoint; "
                        "settled endpoint verification remains mandatory."
                    )

                    break

            else:
                previous_streak = (
                    handoff_confirmation.streak
                )

                (
                    handoff_confirmation,
                    handoff_frame_is_fresh,
                ) = handoff_confirmation.observe(
                    current_wrist_frame_id,
                    eligible=False,
                )

                handoff_endpoint_streak = (
                    handoff_confirmation.streak
                )

                if (
                    handoff_frame_is_fresh
                    and previous_streak > 0
                ):
                    print(
                        f"\n[HANDOFF RESET] "
                        f"frame="
                        f"{current_wrist_frame_id} "
                        "locked target unavailable in "
                        "fresh WRIST frame."
                    )

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
                    "handoff_endpoint_accepted": bool(
                        handoff_endpoint_accepted
                    ),
                    "handoff_endpoint_signed_distance_px": (
                        handoff_endpoint_signed_distance_px
                    ),
                    "handoff_endpoint_clearance_px": (
                        handoff_endpoint_clearance_px
                    ),
                    "handoff_endpoint_streak": int(
                        handoff_endpoint_streak
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

        if task_status == "HANDOFF_CANDIDATE":
            # ACT has stopped, but deterministic ownership has NOT transferred
            # yet.  First prove that the physically settled arm still lies in
            # the verified demonstration endpoint region.
            phase["name"] = "HANDOFF_SETTLE"

            settled_timestamp = phase5.wait_motion_stable(
                robot,
                motor_names,
            )
            if autonomy_stop.is_set():
                raise OperatorStop(
                    "operator stop before settled endpoint measurement"
                )

            print()
            print("=" * 78)
            print("SETTLED ACT ENDPOINT HANDOFF GATE")
            print("=" * 78)

            try:
                settled_endpoint = _capture_settled_endpoint(
                    wrist_camera=wrist,
                    recognizer=recognizer,
                    target=target,
                    tool_reference=tool_reference,
                    settled_timestamp=settled_timestamp,
                    semantic_lock=moving_target_lock,
                    after_frame_id=int(
                        handoff_candidate_trigger[
                            "wrist_frame_id"
                        ]
                    ),
                )
            except OperatorStop:
                raise
            except Exception as exc:
                settled_endpoint = {
                    "status": "UNAVAILABLE",
                    "error": f"{type(exc).__name__}: {exc}",
                    "settled_timestamp": float(
                        settled_timestamp
                    ),
                    "post_motion_guard_s": float(
                        phase5.POST_MOTION_GUARD_S
                    ),
                    "acceptance_role": "handoff_gate",
                    "accepted": False,
                }

                _append_jsonl(
                    events_path,
                    {
                        "event": "SETTLED_ENDPOINT_UNAVAILABLE",
                        **settled_endpoint,
                    },
                )

                raise HandoffRejected(
                    "settled endpoint could not be measured: "
                    f"{settled_endpoint['error']}"
                ) from exc

            candidate_error = np.asarray(
                handoff_candidate_trigger["error_px"],
                dtype=np.float64,
            )
            endpoint_error = np.asarray(
                settled_endpoint["error_px"],
                dtype=np.float64,
            )
            endpoint_drift = (
                endpoint_error - candidate_error
            )

            endpoint_evaluation = handoff_contract.evaluate(
                endpoint_error
            )

            settled_endpoint[
                "moving_candidate_error_px"
            ] = candidate_error.tolist()

            settled_endpoint[
                "moving_candidate_error_norm_px"
            ] = float(
                handoff_candidate_trigger["error_norm_px"]
            )

            settled_endpoint[
                "error_drift_px"
            ] = endpoint_drift.tolist()

            settled_endpoint[
                "error_drift_norm_px"
            ] = float(
                np.linalg.norm(endpoint_drift)
            )

            settled_endpoint[
                "error_norm_change_px"
            ] = float(
                settled_endpoint["error_norm_px"]
                - handoff_candidate_trigger[
                    "error_norm_px"
                ]
            )

            settled_endpoint[
                "signed_distance_to_hull_px"
            ] = float(
                endpoint_evaluation
                .signed_distance_to_hull_px
            )

            settled_endpoint[
                "acceptance_clearance_px"
            ] = float(
                endpoint_evaluation
                .acceptance_clearance_px
            )

            settled_endpoint[
                "contract_margin_px"
            ] = float(
                handoff_contract.acceptance_margin_px
            )

            settled_endpoint[
                "acceptance_role"
            ] = "handoff_gate"

            settled_endpoint[
                "accepted"
            ] = bool(
                endpoint_evaluation.accepted
            )

            _append_jsonl(
                events_path,
                {
                    "event": (
                        "SETTLED_ENDPOINT_ACCEPTED"
                        if endpoint_evaluation.accepted
                        else "SETTLED_ENDPOINT_REJECTED"
                    ),
                    **settled_endpoint,
                },
            )

            print(
                f"[SETTLED ENDPOINT] "
                f"frame={settled_endpoint['frame_id']} "
                f"error=("
                f"{endpoint_error[0]:+.2f},"
                f"{endpoint_error[1]:+.2f})px "
                f"norm="
                f"{settled_endpoint['error_norm_px']:.2f}px "
                f"drift="
                f"{settled_endpoint['error_drift_norm_px']:.2f}px "
                f"clearance="
                f"{endpoint_evaluation.acceptance_clearance_px:+.2f}px"
            )

            if not endpoint_evaluation.accepted:
                print(
                    "[HANDOFF REJECTED] settled arm is outside "
                    "the verified demonstration endpoint region; "
                    "deterministic ownership is NOT transferred."
                )

                raise HandoffRejected(
                    "settled ACT endpoint outside verified "
                    "handoff region: "
                    f"error=({endpoint_error[0]:+.2f},"
                    f"{endpoint_error[1]:+.2f})px "
                    f"clearance="
                    f"{endpoint_evaluation.acceptance_clearance_px:+.2f}px"
                )

            print(
                "[HANDOFF ACCEPTED] settled endpoint passed "
                "the same verified endpoint contract."
            )

            if autonomy_stop.is_set():
                raise OperatorStop(
                    "operator stop before deterministic ownership"
                )

            # Only here does ownership actually move from ACT to
            # the deterministic WRIST/press controller.
            phase["name"] = "DETERMINISTIC"

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
                semantic_lock=moving_target_lock,
                tool_reference=tool_reference,
                screen_calibration=screen_calibration,
                screen_char_ocr=screen_char_ocr,
                endpoint_settled_timestamp=settled_timestamp,
                endpoint_wrist_frame_id=endpoint_wrist_frame_id,
                run_dir=run_dir,
                autonomy_stop=autonomy_stop,
                phase=phase,
                safety_context=deterministic_safety_context,
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

    except HandoffRejected as exc:
        task_status = "HANDOFF_REJECTED"
        task_error = str(exc)
        if rollout_start is not None and rollout_stop is None:
            rollout_stop = time.perf_counter()
        print("\n[HANDOFF REJECTED]", task_error)

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

            local_retract_safe = True
            tracker = deterministic_safety_context.get("sent_state_tracker")
            planner = deterministic_safety_context.get("planner")
            recovery_motor_names = deterministic_safety_context.get("motor_names")
            recovery_events = deterministic_safety_context.get("events")

            if tracker is not None and planner is not None and recovery_motor_names:
                latest_state = tracker.state
                if latest_state.z_mm < -1e-12:
                    print()
                    print("===== LOCAL FIXED-ANCHOR SAFETY RETRACT =====")
                    print(
                        "latest sent XYZ       : "
                        f"({latest_state.x_mm:+.2f},"
                        f"{latest_state.y_mm:+.2f},"
                        f"{latest_state.z_mm:+.2f}) mm"
                    )
                    print("policy                : retract to local Z=0 before canonical HOME")
                    try:
                        phase5.ACTIVE_SCREEN_WATCHER = None
                        recovered_state = _retract_to_z0(
                            robot=robot,
                            planner=planner,
                            state=latest_state,
                            motor_names=recovery_motor_names,
                            sent_state_tracker=tracker,
                            events=(recovery_events if recovery_events is not None else []),
                        )
                        local_retract_result = {
                            "status": "PASS",
                            "start_z_mm": float(latest_state.z_mm),
                            "final_z_mm": float(recovered_state.z_mm),
                        }
                        print("[LOCAL SAFETY RETRACT PASS] local command-space Z=0")
                    except BaseException as exc:
                        local_retract_safe = False
                        local_retract_result = {
                            "status": "FAIL",
                            "start_z_mm": float(latest_state.z_mm),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        print(
                            "[LOCAL SAFETY RETRACT FAILURE]",
                            local_retract_result["error"],
                        )

            if not local_retract_safe:
                home_result = {
                    "status": "SKIPPED_UNSAFE_LOCAL_RETRACT_FAILED",
                    "error": local_retract_result.get("error"),
                }
                print(
                    "[AUTO HOME SKIPPED] local fixed-anchor retract failed; "
                    "holding torque for operator-controlled recovery."
                )

                phase5.raise_if_screen_event = original_raise_if_screen_event
                phase5.wait_motion_stable = original_wait_motion_stable
                if signal_installed:
                    signal.signal(signal.SIGINT, previous_sigint)
                    signal_installed = False
                _emergency_hold_until_operator(robot, home_result["error"])

            else:
                try:
                    home_result = _restore_canonical_home(
                        robot=robot,
                        motor_names=motor_names,
                        home_baseline=home_baseline,
                        home_stop=home_stop,
                    )
                    print(
                        "[HOME RESTORED] "
                        f"final Goal diff="
                        f"{home_result['final_goal_diff_deg']:.3f}deg"
                    )

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
        _stop_runtime_devices(
            top=top,
            wrist=wrist,
            side=side,
            robot=robot,
            original_raise_if_screen_event=original_raise_if_screen_event,
            original_wait_motion_stable=original_wait_motion_stable,
            signal_installed=signal_installed,
            previous_sigint=previous_sigint,
        )

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
            "schema": "phase8.single_key_integration.v2",
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
                "threshold_px": PHASE5_CAPTURE_REFERENCE_THRESHOLD_PX,
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
        print("local safety retract:", local_retract_result["status"])
        print("HOME recovery       :", home_result["status"])
        print("summary             :", summary_path)
        print("=" * 78)


if __name__ == "__main__":
    main()
