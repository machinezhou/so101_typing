from __future__ import annotations

import argparse
import json
import math
import select
import shutil
import struct
import subprocess
import sys
import termios
import time
import tty
import wave
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

from so101_typing.adapters.cameras import CameraSpec, ThreadedOpenCVCamera


ROBOT_PORT = "/dev/ttyACM0"
LEADER_PORT = "/dev/ttyACM1"
ROBOT_ID = "lawson_follower_arm"
LEADER_ID = "lawson_leader_arm"
MAX_RELATIVE_TARGET_DEG = 10.0

TOP_CAMERA_CONFIG = Path("configs/cameras/top.yaml")
WRIST_CAMERA_CONFIG = Path("configs/cameras/wrist.yaml")
ARTIFACT_ROOT = Path("artifacts/phase6_act_dataset")

# Formal multi-target dataset.
#
# All A-Z demonstrations append to ONE LeRobot dataset.  Target identity
# remains encoded only in observation.state/task; collection source zones
# remain external SOP and are never dataset labels.
FORMAL_DATASET_BASE = ARTIFACT_ROOT / "keyboard_v1"
FORMAL_DATASET_ROOT = FORMAL_DATASET_BASE / "lerobot_dataset"
FORMAL_REPO_ID = "local/so101_typing_act_raw_keyboard_v1"

# Formal natural collection performs no online visual analysis.
# WRIST images are stored raw and any perception diagnostics run offline.

TELEOP_HZ = 60.0
DATASET_FPS = 15
MOTION_START_DELTA_DEG = 0.8
DEFAULT_MAX_EPISODE_S = 30.0
DEFAULT_REVIEW_S = 5.0
# When the operator presses S/SPACE after naturally finishing the motion,
# trim stationary reaction tail from the training clip. Raw audit samples
# remain intact for later inspection/re-clipping.
END_REACTION_TRIM_DELTA_DEG = 0.05

# Advisory trajectory diagnostics. They never silently delete an episode.
PAUSE_DELTA_DEG = 0.08
PAUSE_MIN_S = 0.35
REVERSAL_MIN_NORM_DEG = 0.20
REVERSAL_COSINE = -0.50
LONG_EPISODE_S = 8.0

TARGET_VOCAB = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ("SPACE", "BACKSPACE")
TARGET_DIM = len(TARGET_VOCAB)



@dataclass(slots=True)
class BufferedFrame:
    raw_index: int
    timestamp: float
    state: np.ndarray
    action: np.ndarray
    # Keep BGR copies during the real-time loop. BGR->RGB conversion is deferred
    # to the background dataset committer so color conversion cannot consume
    # control-loop time.
    top_bgr: np.ndarray
    wrist_bgr: np.ndarray
    top_frame_id: int
    top_capture_timestamp: float
    wrist_frame_id: int
    wrist_capture_timestamp: float
    top_frame_age_ms: float
    wrist_frame_age_ms: float



class SoundGuide:
    """Dependency-free audio cues used only for workflow state changes."""

    def __init__(self, root: Path, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.root = root
        self.player = shutil.which("paplay") or shutil.which("aplay")
        self.paths: dict[str, Path] = {}
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            self.paths = {
                "countdown": self._ensure_tone("countdown.wav", [(560.0, 0.085)]),
                "armed": self._ensure_tone("armed.wav", [(980.0, 0.13)]),
                "success": self._ensure_tone(
                    "success_v2.wav",
                    [(820.0, 0.12), (0.0, 0.045), (1080.0, 0.13), (0.0, 0.045), (1480.0, 0.30)],
                ),
                "retry": self._ensure_tone(
                    "retry.wav",
                    [(520.0, 0.10), (0.0, 0.040), (330.0, 0.18)],
                ),
            }

    def _ensure_tone(self, filename: str, sequence: list[tuple[float, float]]) -> Path:
        path = self.root / filename
        if path.exists():
            return path
        sample_rate = 44100
        amplitude = 0.24
        frames = bytearray()
        phase = 0.0
        for frequency, duration_s in sequence:
            count = max(1, int(round(sample_rate * duration_s)))
            for _ in range(count):
                if frequency <= 0.0:
                    value = 0
                else:
                    value = int(32767.0 * amplitude * math.sin(phase))
                    phase += 2.0 * math.pi * frequency / sample_rate
                frames.extend(struct.pack("<h", value))
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(bytes(frames))
        return path

    def _play(self, cue: str, *, blocking: bool) -> None:
        if not self.enabled:
            return
        path = self.paths.get(cue)
        if path is None:
            return
        if self.player:
            try:
                if blocking:
                    subprocess.run(
                        [self.player, str(path)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                else:
                    subprocess.Popen(
                        [self.player, str(path)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                return
            except OSError:
                pass
        print("\a", end="", flush=True)

    def play(self, cue: str) -> None:
        self._play(cue, blocking=False)

    def play_blocking(self, cue: str) -> None:
        self._play(cue, blocking=True)

    def test(self) -> None:
        print("sound: COUNTDOWN 3")
        self.play_blocking("countdown")
        time.sleep(0.35)
        print("sound: COUNTDOWN 2")
        self.play_blocking("countdown")
        time.sleep(0.35)
        print("sound: COUNTDOWN 1")
        self.play_blocking("countdown")
        time.sleep(0.35)
        print("sound: ARMED (natural motion may start after this tone ends)")
        self.play_blocking("armed")
        time.sleep(0.45)
        print("sound: SUCCESS (recording ended; review window follows)")
        self.play_blocking("success")
        time.sleep(0.45)
        print("sound: RETRY (attempt did not count; retry same episode)")
        self.play_blocking("retry")


class TerminalKeyPoller:
    """Non-blocking single-key polling; Ctrl+C remains active in cbreak mode."""

    def __init__(self) -> None:
        self.fd: int | None = None
        self._old_attrs = None

    def __enter__(self) -> "TerminalKeyPoller":
        if sys.stdin.isatty():
            self.fd = sys.stdin.fileno()
            self._old_attrs = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def poll(self) -> str | None:
        if self.fd is None:
            return None
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            return None
        return sys.stdin.read(1)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.fd is not None and self._old_attrs is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._old_attrs)


def _review_episode(
    robot: SO101Follower,
    leader: SO101Leader,
    sounds: SoundGuide,
    *,
    review_s: float,
    episode_number: int,
) -> bool:
    """Keep teleop live while giving the operator a no-ENTER discard window."""
    print()
    print("=" * 72)
    print(f"REVIEW EPISODE {episode_number}")
    print("=" * 72)
    print(
        f"SUCCESS already ended recording. For the next {review_s:.1f}s, "
        "press q (no ENTER) to DISCARD and re-record this same episode."
    )
    print("If you do nothing, the episode is accepted automatically.")
    print("Teleoperation stays LIVE; movement during REVIEW is not recorded.")

    deadline = time.monotonic() + review_s
    period_s = 1.0 / TELEOP_HZ
    next_print = 0.0
    with TerminalKeyPoller() as keys:
        while True:
            started = time.monotonic()
            robot.send_action(leader.get_action())
            now = time.monotonic()
            key = keys.poll()
            if key is not None and key.lower() == "q":
                sounds.play_blocking("retry")
                print("\n[DISCARDED] q pressed. This buffered episode was NOT written to LeRobot dataset.")
                print("[RETRY] Reposition and re-record the same episode according to your external collection protocol.")
                return False

            remaining = max(0.0, deadline - now)
            if now >= next_print:
                print(
                    f"\r[REVIEW] press q to discard | auto-accept in {remaining:4.1f}s"
                    + " " * 20,
                    end="",
                    flush=True,
                )
                next_print = now + 0.20
            if remaining <= 0.0:
                print("\n[ACCEPTED] Review window expired; episode will be committed.")
                return True
            time.sleep(max(0.0, period_s - (time.monotonic() - started)))


class DatasetCommitter:
    """Commit exactly one accepted episode in the background during PREPARE teleop."""

    def __init__(self, dataset: LeRobotDataset) -> None:
        self.dataset = dataset
        self._thread: Thread | None = None
        self._done = Event()
        self._error: BaseException | None = None
        self._label = ""

    @property
    def busy(self) -> bool:
        return self._thread is not None and not self._done.is_set()

    def start(self, frames: list[BufferedFrame], *, task: str, label: str) -> None:
        self.wait()
        self._done.clear()
        self._error = None
        self._label = label

        def _run() -> None:
            try:
                for item in frames:
                    self.dataset.add_frame(
                        {
                            "observation.images.top": cv2.cvtColor(item.top_bgr, cv2.COLOR_BGR2RGB),
                            "observation.images.wrist": cv2.cvtColor(item.wrist_bgr, cv2.COLOR_BGR2RGB),
                            "observation.state": item.state,
                            "action": item.action,
                            "task": task,
                        }
                    )
                self.dataset.save_episode()
            except BaseException as exc:  # surface in foreground before next ARM
                self._error = exc
            finally:
                self._done.set()

        self._thread = Thread(target=_run, name=f"phase6-dataset-save-{label}", daemon=False)
        self._thread.start()

    def wait(self) -> None:
        if self._thread is None:
            return
        self._thread.join()
        self._thread = None
        if self._error is not None:
            error = self._error
            self._error = None
            raise RuntimeError(f"Dataset commit failed for {self._label}") from error

    def ready_for_next_episode(self) -> bool:
        if self._thread is None:
            return True
        if not self._done.is_set():
            return False
        self.wait()
        return True


def _numeric_action(action: dict) -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in action.items()
        if isinstance(value, (int, float, np.floating, np.integer))
    }


def _action_key(name: str) -> str:
    return name if name.endswith(".pos") else f"{name}.pos"


def _vector_from_action(action: dict[str, float], motor_names: list[str]) -> np.ndarray:
    values: list[float] = []
    missing: list[str] = []
    for name in motor_names:
        key = _action_key(name)
        if key not in action:
            missing.append(key)
        else:
            values.append(float(action[key]))
    if missing:
        raise RuntimeError(f"Sent action missing motor keys: {missing}")
    return np.asarray(values, dtype=np.float32)


def _present_state(observation: dict, motor_names: list[str]) -> np.ndarray:
    values: list[float] = []
    missing: list[str] = []
    for name in motor_names:
        key = _action_key(name)
        if key not in observation:
            missing.append(key)
        else:
            values.append(float(observation[key]))
    if missing:
        raise RuntimeError(f"Robot observation missing motor keys: {missing}")
    return np.asarray(values, dtype=np.float32)


def _target_one_hot(target: str) -> np.ndarray:
    target = target.strip().upper()
    if target not in TARGET_VOCAB:
        raise ValueError(f"Unsupported target {target!r}; vocabulary={TARGET_VOCAB}")
    vector = np.zeros(TARGET_DIM, dtype=np.float32)
    vector[TARGET_VOCAB.index(target)] = 1.0
    return vector


def _max_action_delta(left: dict, right: dict, motor_names: list[str]) -> float:
    keys = [_action_key(name) for name in motor_names if "gripper" not in name]
    shared = [key for key in keys if key in left and key in right]
    if not shared:
        return float("inf")
    return max(abs(float(left[key]) - float(right[key])) for key in shared)


def _wait_initial_frames(
    top: ThreadedOpenCVCamera,
    wrist: ThreadedOpenCVCamera,
    timeout_s: float = 5.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if top.latest() is not None and wrist.latest() is not None:
            return
        time.sleep(0.02)
    raise RuntimeError("TOP/WRIST did not both produce an initial frame")


def _dataset_features(
    motor_names: list[str],
    top_spec: CameraSpec,
    wrist_spec: CameraSpec,
) -> dict[str, dict]:
    joint_names = [_action_key(name) for name in motor_names]
    target_names = [f"target_{name}" for name in TARGET_VOCAB]
    return {
        "observation.images.top": {
            "dtype": "image",
            "shape": (top_spec.height, top_spec.width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "image",
            "shape": (wrist_spec.height, wrist_spec.width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(joint_names) + TARGET_DIM,),
            "names": joint_names + target_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": joint_names,
        },
    }


def _normalize_feature_spec(features: dict[str, dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for key, value in features.items():
        item = dict(value)
        if "shape" in item:
            item["shape"] = tuple(item["shape"])
        if "names" in item and item["names"] is not None:
            item["names"] = list(item["names"])
        # Ignore encoder/runtime info that does not alter the model-facing contract.
        item.pop("info", None)
        out[key] = item
    return out


def _open_dataset(
    *,
    root: Path,
    repo_id: str,
    features: dict[str, dict],
    fps: int,
) -> LeRobotDataset:
    info = root / "meta" / "info.json"
    if info.exists():
        dataset = LeRobotDataset.resume(
            repo_id=repo_id,
            root=root,
            image_writer_threads=4,
        )
        actual_all = _normalize_feature_spec(dataset.features)
        expected = _normalize_feature_spec(features)

        # LeRobot augments dataset.features with its own bookkeeping columns
        # (e.g. timestamp/frame_index/episode_index/index/task_index). Those are
        # not part of this collector's model-facing contract and must not make
        # a valid dataset look incompatible on resume. Compare only the feature
        # keys this collector explicitly owns.
        missing = [key for key in expected if key not in actual_all]
        mismatched = {
            key: {"expected": expected[key], "actual": actual_all.get(key)}
            for key in expected
            if key in actual_all and actual_all[key] != expected[key]
        }
        if missing or mismatched:
            details = []
            if missing:
                details.append(f"missing={missing}")
            if mismatched:
                details.append(f"mismatched={mismatched}")
            raise RuntimeError(
                "Existing dataset model-facing feature schema does not match this collector: "
                + "; ".join(details)
                + ". Use a new --dataset-root instead of mixing incompatible data."
            )
        if int(dataset.fps) != int(fps):
            raise RuntimeError(
                f"Existing dataset fps={dataset.fps} but collector fps={fps}. "
                "Use a new --dataset-root."
            )
        return dataset

    if root.exists():
        if any(root.iterdir()):
            raise RuntimeError(
                f"Dataset root exists but is not a LeRobot dataset: {root}. "
                "Choose an empty/new --dataset-root."
            )
        # LeRobotDataset.create() deliberately requires the dataset root itself
        # to NOT exist (its metadata constructor calls mkdir(exist_ok=False)).
        # A previous collector launch or our parent-directory setup can leave an
        # empty root behind, so remove only that empty leaf directory and let
        # LeRobot create it atomically. Never delete a non-empty directory.
        root.rmdir()

    root.parent.mkdir(parents=True, exist_ok=True)
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root,
        robot_type="so101",
        use_videos=False,
        image_writer_threads=4,
    )


def _prepare_and_arm(
    robot: SO101Follower,
    leader: SO101Leader,
    sounds: SoundGuide,
    committer: DatasetCommitter,
    *,
    target: str,
    episode_number: int,
    episode_count: int,
    motor_names: list[str],
    arm_countdown_s: float,
) -> tuple[dict[str, float], np.ndarray]:
    print()
    print("=" * 72)
    print(f"PHASE 6 — PREPARE EPISODE {episode_number}/{episode_count}")
    print("=" * 72)
    print(f"Target: {target}")
    print()
    print("Teleoperation is LIVE during PREPARE.")
    print("Position the robot according to your external collection protocol/SOP.")
    print("PREPARE motion is NOT recorded in the ACT demonstration dataset.")
    print(
        f"No ENTER is required. After the previous save is complete, "
        f"ARMED starts automatically after a {arm_countdown_s:.1f}s countdown."
    )
    print("The collector applies no start-class, distance, pose, or visual gate.")

    last_sent: dict[str, float] | None = None
    period_s = 1.0 / TELEOP_HZ
    next_print = 0.0

    while not committer.ready_for_next_episode():
        started = time.monotonic()
        last_sent = _numeric_action(robot.send_action(leader.get_action()))
        now = time.monotonic()
        if now >= next_print:
            print(
                "\r[PREPARE] SAVING PREVIOUS — reposition freely" + " " * 24,
                end="", flush=True,
            )
            next_print = now + 0.50
        time.sleep(max(0.0, period_s - (time.monotonic() - started)))

    if episode_number > 1:
        print("\n[SAVE COMPLETE] Starting automatic arm countdown.")

    countdown_deadline = time.monotonic() + arm_countdown_s
    last_announced_second: int | None = None
    next_print = 0.0
    while True:
        started = time.monotonic()
        last_sent = _numeric_action(robot.send_action(leader.get_action()))
        now = time.monotonic()
        remaining = max(0.0, countdown_deadline - now)
        shown = int(math.ceil(remaining))
        if shown != last_announced_second:
            last_announced_second = shown
            if shown > 0:
                print(f"\n[ARM COUNTDOWN] {shown}...")
                if shown == 1:
                    sounds.play_blocking("countdown")
                else:
                    sounds.play("countdown")
        if now >= next_print:
            print(
                f"\r[PREPARE] ARM IN {remaining:4.1f}s" + " " * 36,
                end="", flush=True,
            )
            next_print = now + 0.20
        if remaining <= 0.0:
            break
        time.sleep(max(0.0, period_s - (time.monotonic() - started)))

    print()
    if last_sent is None:
        raise RuntimeError("No sent action available at ARM time")
    _vector_from_action(last_sent, motor_names)
    print("[COUNTDOWN COMPLETE] Current operator-positioned start accepted; no task-specific gate applied.")
    print("[ARMED SOUND] Wait for this tone to END, then begin the demonstration.")
    sounds.play_blocking("armed")
    arm_present = _present_state(robot.get_observation(), motor_names)
    print("[ARMED] Tone finished — begin moving naturally NOW.")
    return last_sent, arm_present

def _control_trace_endpoint_index(
    trace: list[dict],
    *,
    start_index: int,
    motor_names: list[str],
) -> int:
    """Return the replay endpoint after trimming only stationary keypress reaction tail.

    The replay trace is captured at the 60 Hz teleop/control cadence.  The endpoint
    is one control sample after the last meaningful change in the *actual sent*
    action, so later physical replay preserves the operator's command history while
    excluding the pause required to reach for S/SPACE.  WRIST metrics are not used.
    """
    if not trace:
        return 0
    if len(trace) <= start_index + 1:
        return len(trace) - 1
    last_motion = start_index
    for i in range(max(start_index + 1, 1), len(trace)):
        prev = trace[i - 1]["sent_action"]
        cur = trace[i]["sent_action"]
        delta = _max_action_delta(cur, prev, motor_names)
        if math.isfinite(delta) and delta > END_REACTION_TRIM_DELTA_DEG:
            last_motion = i
    return min(len(trace) - 1, last_motion + 1)


def _frame_window_for_replay_times(
    raw: list[BufferedFrame],
    *,
    start_timestamp: float,
    endpoint_timestamp: float,
) -> tuple[int | None, int | None]:
    """Map the high-rate replay time window onto the 15 Hz training frames.

    Include the final sampled observation at-or-before replay start when available,
    then retain sampled frames through the replay endpoint.  This keeps the ACT
    training clip aligned with the separately stored high-rate command trace.
    """
    if not raw:
        return None, None
    timestamps = [float(item.timestamp) for item in raw]
    before = [i for i, ts in enumerate(timestamps) if ts <= start_timestamp]
    if before:
        start_index = before[-1]
    else:
        start_index = next((i for i, ts in enumerate(timestamps) if ts >= start_timestamp), 0)
    through_end = [i for i, ts in enumerate(timestamps) if ts <= endpoint_timestamp]
    if through_end:
        end_index = through_end[-1]
    else:
        end_index = start_index
    if end_index < start_index:
        end_index = start_index
    return start_index, end_index


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _performance_summary(
    *,
    send_timestamps: list[float],
    loop_durations_s: list[float],
    send_latencies_s: list[float],
    observation_latencies_s: list[float],
    camera_copy_latencies_s: list[float],
    sample_due_count: int,
    sample_recorded_count: int,
) -> dict:
    inter_send_ms = [
        (send_timestamps[i] - send_timestamps[i - 1]) * 1000.0
        for i in range(1, len(send_timestamps))
    ]
    loop_ms = [v * 1000.0 for v in loop_durations_s]
    send_ms = [v * 1000.0 for v in send_latencies_s]
    obs_ms = [v * 1000.0 for v in observation_latencies_s]
    copy_ms = [v * 1000.0 for v in camera_copy_latencies_s]
    period_ms = 1000.0 / TELEOP_HZ

    elapsed_s = (
        float(send_timestamps[-1] - send_timestamps[0])
        if len(send_timestamps) >= 2
        else 0.0
    )
    achieved_hz = (
        float((len(send_timestamps) - 1) / elapsed_s)
        if elapsed_s > 0.0
        else 0.0
    )

    return {
        "target_control_hz": float(TELEOP_HZ),
        "achieved_send_hz": achieved_hz,
        "control_samples": len(send_timestamps),
        "inter_send_ms": {
            "p50": _percentile(inter_send_ms, 50),
            "p95": _percentile(inter_send_ms, 95),
            "p99": _percentile(inter_send_ms, 99),
            "max": max(inter_send_ms) if inter_send_ms else None,
        },
        "loop_ms": {
            "p50": _percentile(loop_ms, 50),
            "p95": _percentile(loop_ms, 95),
            "p99": _percentile(loop_ms, 99),
            "max": max(loop_ms) if loop_ms else None,
            "overrun_count": int(sum(v > period_ms for v in loop_ms)),
        },
        "send_action_ms": {
            "p50": _percentile(send_ms, 50),
            "p95": _percentile(send_ms, 95),
            "p99": _percentile(send_ms, 99),
            "max": max(send_ms) if send_ms else None,
        },
        "sample_observation_ms": {
            "p50": _percentile(obs_ms, 50),
            "p95": _percentile(obs_ms, 95),
            "p99": _percentile(obs_ms, 99),
            "max": max(obs_ms) if obs_ms else None,
        },
        "sample_camera_copy_ms": {
            "p50": _percentile(copy_ms, 50),
            "p95": _percentile(copy_ms, 95),
            "p99": _percentile(copy_ms, 99),
            "max": max(copy_ms) if copy_ms else None,
        },
        "sample_frame_age_ms": {"top_p95": None, "top_max": None, "wrist_p95": None, "wrist_max": None},
        "sample_due_count": int(sample_due_count),
        "sample_recorded_count": int(sample_recorded_count),
        "sample_missed_count": int(max(0, sample_due_count - sample_recorded_count)),
    }


def _collect_buffered_episode(
    *,
    robot: SO101Follower,
    leader: SO101Leader,
    top: ThreadedOpenCVCamera,
    wrist: ThreadedOpenCVCamera,
    sounds: SoundGuide,
    motor_names: list[str],
    target: str,
    episode_number: int,
    episode_count: int,
    baseline_action: dict[str, float],
    arm_present_state: np.ndarray,
    max_episode_s: float,
) -> tuple[list[BufferedFrame], dict]:
    target_one_hot = _target_one_hot(target)
    raw: list[BufferedFrame] = []
    audit: list[dict] = []
    control_trace: list[dict] = []
    motion_started = False
    motion_start_control_index: int | None = None
    motion_start_raw_index: int | None = None
    endpoint_control_index: int | None = None
    endpoint_raw_index: int | None = None
    requested_clamp_max = 0.0
    operator_aborted = False

    # Passive performance instrumentation. These arrays are tiny compared with
    # image buffers and do not trigger any additional hardware or CV work.
    send_timestamps: list[float] = []
    loop_durations_s: list[float] = []
    send_latencies_s: list[float] = []
    observation_latencies_s: list[float] = []
    camera_copy_latencies_s: list[float] = []
    sample_due_count = 0
    sample_recorded_count = 0

    armed_at = time.monotonic()
    next_sample = armed_at
    next_ui = armed_at
    period_s = 1.0 / TELEOP_HZ
    sample_period_s = 1.0 / DATASET_FPS

    print()
    print("===== ARMED =====")
    print(f"Episode {episode_number}/{episode_count} | target={target}")
    print("Perform one natural, continuous expert approach. The collector gives NO visual guidance.")
    print("When YOU consider the demonstration complete, press s or SPACE (no ENTER).")
    print("Press q at any time to discard/retry this attempt.")

    with TerminalKeyPoller() as keys:
        while True:
            loop_started = time.monotonic()
            now = loop_started
            sample_due = now >= next_sample
            observation = top_frame = wrist_frame = None

            # Preserve causal ACT training semantics: on a 15 Hz sample tick,
            # capture observation BEFORE the human action paired with it. Unlike
            # earlier collectors, no recognition or BGR->RGB conversion runs here.
            if sample_due:
                sample_due_count += 1
                obs_started = time.monotonic()
                observation = robot.get_observation()
                observation_latencies_s.append(time.monotonic() - obs_started)

                copy_started = time.monotonic()
                top_frame = top.latest(copy_image=True)
                wrist_frame = wrist.latest(copy_image=True)
                camera_copy_latencies_s.append(time.monotonic() - copy_started)

            requested = _numeric_action(leader.get_action())
            send_started = time.monotonic()
            sent = _numeric_action(robot.send_action(requested))
            send_done = time.monotonic()
            send_latencies_s.append(send_done - send_started)
            send_timestamps.append(send_done)

            clamp_delta = _max_action_delta(requested, sent, motor_names)
            if math.isfinite(clamp_delta):
                requested_clamp_max = max(requested_clamp_max, clamp_delta)

            control_index = len(control_trace)
            control_trace.append({
                "control_index": control_index,
                "timestamp": float(send_done),
                "elapsed_from_arm_s": float(send_done - armed_at),
                "requested_action": dict(requested),
                "sent_action": dict(sent),
                "send_latency_ms": float((send_done - send_started) * 1000.0),
            })

            if not motion_started:
                delta = _max_action_delta(sent, baseline_action, motor_names)
                if delta >= MOTION_START_DELTA_DEG:
                    motion_started = True
                    motion_start_control_index = max(0, control_index - 1)
                    print(
                        f"\n[EPISODE START] real motion detected at control sample "
                        f"{motion_start_control_index}; full-rate replay trace starts there."
                    )

            if sample_due:
                if observation is not None and top_frame is not None and wrist_frame is not None:
                    present = _present_state(observation, motor_names)
                    state = np.concatenate((present, target_one_hot)).astype(np.float32, copy=False)
                    action = _vector_from_action(sent, motor_names)
                    now_for_age = time.monotonic()
                    top_age_ms = max(0.0, (now_for_age - float(top_frame.capture_timestamp)) * 1000.0)
                    wrist_age_ms = max(0.0, (now_for_age - float(wrist_frame.capture_timestamp)) * 1000.0)
                    item = BufferedFrame(
                        raw_index=len(raw),
                        timestamp=float(send_done),
                        state=state,
                        action=action,
                        top_bgr=top_frame.image,
                        wrist_bgr=wrist_frame.image,
                        top_frame_id=int(top_frame.frame_id),
                        top_capture_timestamp=float(top_frame.capture_timestamp),
                        wrist_frame_id=int(wrist_frame.frame_id),
                        wrist_capture_timestamp=float(wrist_frame.capture_timestamp),
                        top_frame_age_ms=float(top_age_ms),
                        wrist_frame_age_ms=float(wrist_age_ms),
                    )
                    raw.append(item)
                    sample_recorded_count += 1
                    raw_index = len(raw) - 1
                    audit.append({
                        "raw_index": raw_index,
                        "timestamp": float(send_done),
                        "top_frame_id": item.top_frame_id,
                        "top_capture_timestamp": item.top_capture_timestamp,
                        "top_frame_age_ms": item.top_frame_age_ms,
                        "wrist_frame_id": item.wrist_frame_id,
                        "wrist_capture_timestamp": item.wrist_capture_timestamp,
                        "wrist_frame_age_ms": item.wrist_frame_age_ms,
                    })
                next_sample += sample_period_s
                if next_sample < send_done - sample_period_s:
                    next_sample = send_done + sample_period_s

            key = keys.poll()
            if key is not None:
                low = key.lower()
                if low == "q":
                    sounds.play_blocking("retry")
                    operator_aborted = True
                    print("\n[DISCARDED DURING RECORDING] q pressed — same episode slot will retry.")
                    break
                if key == " " or low == "s":
                    if not motion_started or motion_start_control_index is None:
                        print("\n[END IGNORED] No real post-ARM motion has been detected yet.")
                    elif raw and control_trace:
                        endpoint_control_index = _control_trace_endpoint_index(
                            control_trace,
                            start_index=motion_start_control_index,
                            motor_names=motor_names,
                        )
                        replay_start_ts = float(control_trace[motion_start_control_index]["timestamp"])
                        replay_end_ts = float(control_trace[endpoint_control_index]["timestamp"])
                        motion_start_raw_index, endpoint_raw_index = _frame_window_for_replay_times(
                            raw,
                            start_timestamp=replay_start_ts,
                            endpoint_timestamp=replay_end_ts,
                        )
                        sounds.play_blocking("success")
                        print(
                            "\n[EPISODE END] Operator marked the natural demonstration complete. "
                            "SUCCESS tone finished."
                        )
                        print(
                            f"[REACTION TRIM] full-rate replay trace ends at control sample "
                            f"{endpoint_control_index}; 15 Hz training clip ends at sample "
                            f"{endpoint_raw_index}."
                        )
                        break

            now = time.monotonic()
            if now >= next_ui:
                if not motion_started:
                    text = "ARMED — begin one natural move"
                else:
                    text = "RECORDING NATURAL DEMO | press s/SPACE when complete | q=discard"
                print("\r[DATASET] " + text + " " * 20, end="", flush=True)
                next_ui = now + 0.50

            if now - armed_at > max_episode_s:
                sounds.play_blocking("retry")
                print(
                    f"\n[TIMEOUT] No operator end signal within {max_episode_s:.0f}s. "
                    "This attempt will NOT be saved; the same episode slot will retry."
                )
                break

            loop_duration = time.monotonic() - loop_started
            loop_durations_s.append(loop_duration)
            time.sleep(max(0.0, period_s - loop_duration))

    print()
    if (
        operator_aborted
        or motion_start_control_index is None
        or endpoint_control_index is None
        or motion_start_raw_index is None
        or endpoint_raw_index is None
        or endpoint_control_index < motion_start_control_index
        or endpoint_raw_index < motion_start_raw_index
    ):
        trimmed: list[BufferedFrame] = []
        replay_trimmed: list[dict] = []
    else:
        trimmed = raw[motion_start_raw_index : endpoint_raw_index + 1]
        replay_trimmed = control_trace[motion_start_control_index : endpoint_control_index + 1]

    quality = _trajectory_quality(trimmed)
    performance = _performance_summary(
        send_timestamps=send_timestamps,
        loop_durations_s=loop_durations_s,
        send_latencies_s=send_latencies_s,
        observation_latencies_s=observation_latencies_s,
        camera_copy_latencies_s=camera_copy_latencies_s,
        sample_due_count=sample_due_count,
        sample_recorded_count=sample_recorded_count,
    )
    # Frame-age statistics are derived from retained raw samples; no extra timing
    # query or visual processing is added to the hot loop.
    if raw:
        performance["sample_frame_age_ms"] = {
            "top_p95": _percentile([x.top_frame_age_ms for x in raw], 95),
            "top_max": max(x.top_frame_age_ms for x in raw),
            "wrist_p95": _percentile([x.wrist_frame_age_ms for x in raw], 95),
            "wrist_max": max(x.wrist_frame_age_ms for x in raw),
        }

    joint_keys = [_action_key(name) for name in motor_names]
    baseline_vector = _vector_from_action(baseline_action, motor_names)
    arm_present_vector = np.asarray(arm_present_state, dtype=np.float32)
    summary = {
        "schema": "phase6.act_raw_demo.audit.v4",
        "episode_number": episode_number,
        "target": target,
        "accepted": bool(trimmed) and bool(replay_trimmed),
        "motion_started": bool(motion_started),
        "motion_start_control_index": motion_start_control_index,
        "motion_start_raw_index": motion_start_raw_index,
        "endpoint_source": "operator_s_or_space_with_full_rate_sent_action_reaction_trim" if endpoint_control_index is not None else None,
        "endpoint_control_index": endpoint_control_index,
        "endpoint_raw_index": endpoint_raw_index,
        "raw_samples_before_trim": len(raw),
        "saved_samples": len(trimmed),
        "control_samples_before_trim": len(control_trace),
        "replay_control_samples": len(replay_trimmed),
        "replay_control_hz_target": TELEOP_HZ,
        "motor_names": joint_keys,
        "arm_baseline_sent_action": {key: float(value) for key, value in zip(joint_keys, baseline_vector)},
        "arm_present_state": {key: float(value) for key, value in zip(joint_keys, arm_present_vector)},
        "first_saved_present_state": None if not trimmed else {
            key: float(value) for key, value in zip(joint_keys, trimmed[0].state[:len(motor_names)])
        },
        "first_saved_action": None if not trimmed else {
            key: float(value) for key, value in zip(joint_keys, trimmed[0].action)
        },
        "final_saved_present_state": None if not trimmed else {
            key: float(value) for key, value in zip(joint_keys, trimmed[-1].state[:len(motor_names)])
        },
        "final_saved_action": None if not trimmed else {
            key: float(value) for key, value in zip(joint_keys, trimmed[-1].action)
        },
        "max_requested_vs_sent_action_delta_deg": float(requested_clamp_max),
        "quality": quality,
        "performance": performance,
        "audit_samples": audit,
        "replay_trace": replay_trimmed,
    }
    return trimmed, summary

def _trajectory_quality(samples: list[BufferedFrame]) -> dict:
    if len(samples) < 2:
        return {
            "sample_count": len(samples),
            "duration_s": 0.0,
            "pause_windows": [],
            "reversal_count": 0,
            "target_lost_samples_after_first_detection": 0,
            "diagnostics": [],
            "warnings": ["too_few_samples"],
        }

    timestamps = np.asarray([item.timestamp for item in samples], dtype=np.float64)
    vectors = np.asarray([item.action for item in samples], dtype=np.float64)
    warnings: list[str] = []
    diagnostics: list[str] = []
    pause_windows: list[dict] = []
    reversal_count = 0

    deltas = np.diff(vectors, axis=0)
    max_abs = np.max(np.abs(deltas), axis=1)
    run_start: int | None = None
    for i, value in enumerate(max_abs):
        if float(value) <= PAUSE_DELTA_DEG:
            if run_start is None:
                run_start = i
        elif run_start is not None:
            duration = float(timestamps[i] - timestamps[run_start])
            if duration >= PAUSE_MIN_S:
                pause_windows.append(
                    {
                        "start_s": float(timestamps[run_start] - timestamps[0]),
                        "end_s": float(timestamps[i] - timestamps[0]),
                        "duration_s": duration,
                    }
                )
            run_start = None
    if run_start is not None:
        duration = float(timestamps[-1] - timestamps[run_start])
        if duration >= PAUSE_MIN_S:
            pause_windows.append(
                {
                    "start_s": float(timestamps[run_start] - timestamps[0]),
                    "end_s": float(timestamps[-1] - timestamps[0]),
                    "duration_s": duration,
                }
            )

    for i in range(1, len(deltas)):
        a = deltas[i - 1]
        b = deltas[i]
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < REVERSAL_MIN_NORM_DEG or nb < REVERSAL_MIN_NORM_DEG:
            continue
        cosine = float(np.dot(a, b) / (na * nb))
        if cosine <= REVERSAL_COSINE:
            reversal_count += 1

    duration_s = float(timestamps[-1] - timestamps[0])
    if pause_windows:
        warnings.append("possible_hesitation_pause")
    if reversal_count >= 3:
        warnings.append("multiple_strong_joint_space_reversals")
    if duration_s > LONG_EPISODE_S:
        warnings.append("episode_longer_than_expected")


    return {
        "sample_count": len(samples),
        "duration_s": duration_s,
        "pause_windows": pause_windows,
        "reversal_count": int(reversal_count),
        "diagnostics": diagnostics,
        "warnings": warnings,
    }


def _write_episode_audit(run_dir: Path, summary: dict) -> Path:
    episode_number = int(summary["episode_number"])
    attempt_number = int(summary.get("attempt_number", episode_number))
    episode_dir = run_dir / f"attempt_{attempt_number:03d}_episode_{episode_number:03d}"
    episode_dir.mkdir(parents=True, exist_ok=False)
    audit_samples = summary.pop("audit_samples")
    replay_trace = summary.pop("replay_trace")
    (episode_dir / "sample_audit.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in audit_samples),
        encoding="utf-8",
    )
    (episode_dir / "replay_trace.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in replay_trace),
        encoding="utf-8",
    )
    summary["audit_jsonl"] = str(episode_dir / "sample_audit.jsonl")
    summary["replay_trace_jsonl"] = str(episode_dir / "replay_trace.jsonl")
    (episode_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return episode_dir / "summary.json"


def _run_recovery(
    robot: SO101Follower,
    leader: SO101Leader,
    committer: DatasetCommitter,
    sounds: SoundGuide,
) -> None:
    print()
    print("===== OPERATOR RECOVERY =====")
    print("Normal teleoperation is live from the leader pose currently in your hand.")
    print("Move follower + leader together to a safe resting pose of your choice.")
    print("The final dataset save may finish in the background while you recover.")
    print("When the robot is safely supported/resting, press Ctrl+C to exit.")
    period_s = 1.0 / TELEOP_HZ
    completion_announced = False
    next_commit_poll = 0.0
    try:
        while True:
            started = time.monotonic()
            robot.send_action(leader.get_action())
            now = time.monotonic()
            if not completion_announced and now >= next_commit_poll:
                if committer.ready_for_next_episode():
                    print("\n[DATASET SAVED] Final episode commit complete.")
                    completion_announced = True
                next_commit_poll = now + 0.25
            time.sleep(max(0.0, period_s - (time.monotonic() - started)))
    except KeyboardInterrupt:
        print("\n[RECOVERY COMPLETE] Waiting for any final dataset commit...")
        committer.wait()
        if not completion_announced:
            print("[DATASET SAVED] Final commit complete.")


def _print_config(
    target: str,
    episodes: int,
    dataset_root: Path,
    repo_id: str,
    sounds: bool,
    arm_countdown_s: float,
    review_s: float,
    max_episode_s: float,
) -> None:
    print("PHASE 6 — GENERIC ACT EPISODE COLLECTOR v4.0")
    print(f"target                    : {target}")
    print(f"episodes                  : {episodes}")
    print(f"dataset fps               : {DATASET_FPS}")
    print(f"dataset root              : {dataset_root}")
    print(f"local repo id             : {repo_id}")
    print("ACT inference              : DISABLED (demonstration collection only)")
    print("SIDE / OCR                 : DISABLED")
    print("PRESS / Z-down             : DISABLED")
    print("autonomous robot motion    : DISABLED")
    print("operator                   : leader teleop only")
    print("episode arm                : automatic countdown; no ENTER")
    print(f"arm countdown              : {arm_countdown_s:.1f}s")
    print(f"max raw demo time          : {max_episode_s:.1f}s")
    print(f"post-success q review      : {review_s:.1f}s")
    print("episode start              : first real post-ARM sent-action motion")
    print("episode end                : operator presses s/SPACE after natural motion is complete")
    print("reaction-tail trim         : 60 Hz actual sent-action trace; no WRIST threshold")
    print("physical replay trace      : full-rate actual sent-action history + original timing")
    print("online WRIST analysis       : DISABLED; perception diagnostics are offline only")
    print("observation.state          : 6 present joints + 28 target one-hot")
    print("action                     : 6 actual sent joint-position goals")
    print("visual observation         : TOP RGB + WRIST RGB")
    print("SIDE pixels                : NOT in ACT dataset")
    print("PREPARE between episodes : teleop LIVE, NOT recorded")
    print(f"sound guide                : {'ON' if sounds else 'OFF'}")
    print("sound COUNTDOWN            : one simple tick per countdown number")
    print("sound ARMED                : start tone; move only after it finishes")
    print("sound SUCCESS              : recording ended after operator s/SPACE")
    print("sound RETRY                : q / timeout; same episode must be retried")
    print("recording itself           : silent; NO Rerun / recognition / proximity guidance")
    print("q discard                  : during recording or REVIEW, q discards/retries")
    print("start distribution         : EXTERNAL SOP; collector has no start-class semantics")

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 6 generic ACT episode collector: TOP+WRIST RGB, 6 joint state + "
            "28 target one-hot, and actual sent 6D action. Start-state coverage is external."
        )
    )
    parser.add_argument("--target", default="G")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--robot-port", default=ROBOT_PORT)
    parser.add_argument("--leader-port", default=LEADER_PORT)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help=(
            "Audit/run-summary root. Defaults to <dataset-root-parent>/runs, "
            "so all targets belonging to one shared dataset have one run history."
        ),
    )
    parser.add_argument("--task", default=None, help="LeRobot task string stored with each episode; defaults to target_key:<TARGET>.")
    parser.add_argument("--no-sound", action="store_true")
    parser.add_argument(
        "--arm-countdown-s",
        "--countdown",
        dest="arm_countdown_s",
        type=float,
        default=5.0,
        help=(
            "Seconds of automatic PREPARE countdown before the start is evaluated and ARMED. "
            "No ENTER is required. Default: 5.0"
        ),
    )
    parser.add_argument(
        "--max-episode-s",
        type=float,
        default=DEFAULT_MAX_EPISODE_S,
        help="Maximum wall time for one raw demonstration before retry. Default: 30s",
    )
    parser.add_argument(
        "--review-s",
        type=float,
        default=DEFAULT_REVIEW_S,
        help=(
            "Seconds after SUCCESS during which q (no ENTER) discards the buffered episode "
            "and retries the same episode slot. Default: 5.0"
        ),
    )
    parser.add_argument("--sound-test", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()

    target = str(args.target).strip().upper()
    if target not in TARGET_VOCAB[:26]:
        raise ValueError(
            "Formal V1 collection currently supports A-Z targets. "
            "SPACE/BACKSPACE remain reserved in the 28D target schema "
            "until deterministic WRIST takeover support is validated."
        )
    if args.episodes < 1:
        raise ValueError("--episodes must be >= 1")
    if not math.isfinite(args.arm_countdown_s) or args.arm_countdown_s <= 0.0:
        raise ValueError("--arm-countdown-s/--countdown must be finite and > 0")
    if not math.isfinite(args.review_s) or args.review_s <= 0.0:
        raise ValueError("--review-s must be finite and > 0")
    if not math.isfinite(args.max_episode_s) or args.max_episode_s <= 0.0:
        raise ValueError("--max-episode-s must be finite and > 0")

    dataset_root = (
        args.dataset_root
        if args.dataset_root is not None
        else FORMAL_DATASET_ROOT
    )

    repo_id = (
        args.repo_id
        or FORMAL_REPO_ID
    )

    runs_root = (
        args.runs_root
        if args.runs_root is not None
        else dataset_root.parent / "runs"
    )

    sounds = SoundGuide(
        ARTIFACT_ROOT / "sounds",
        enabled=not args.no_sound,
    )

    _print_config(
        target,
        args.episodes,
        dataset_root,
        repo_id,
        not args.no_sound,
        args.arm_countdown_s,
        args.review_s,
        args.max_episode_s,
    )
    if args.sound_test:
        sounds.test()
        return
    if args.print_config:
        return

    for path, label in (
        (TOP_CAMERA_CONFIG, "TOP camera config"),
        (WRIST_CAMERA_CONFIG, "WRIST camera config"),
            ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")


    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.robot_port,
            id=ROBOT_ID,
            use_degrees=True,
            max_relative_target=MAX_RELATIVE_TARGET_DEG,
            cameras={},
        )
    )
    leader = SO101Leader(
        SO101LeaderConfig(
            port=args.leader_port,
            id=LEADER_ID,
            use_degrees=True,
        )
    )
    motor_names = list(robot.bus.motors.keys())
    if len(motor_names) != 6:
        raise RuntimeError(f"Expected 6 SO101 motors, got {len(motor_names)}: {motor_names}")

    top_spec = CameraSpec.from_yaml(TOP_CAMERA_CONFIG)
    wrist_spec = CameraSpec.from_yaml(WRIST_CAMERA_CONFIG)
    features = _dataset_features(motor_names, top_spec, wrist_spec)
    dataset = _open_dataset(
        root=dataset_root,
        repo_id=repo_id,
        features=features,
        fps=DATASET_FPS,
    )
    committer = DatasetCommitter(dataset)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    run_dir = (
        runs_root
        / f"run_{timestamp}_{target.lower()}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    top = ThreadedOpenCVCamera(top_spec)
    wrist = ThreadedOpenCVCamera(wrist_spec)

    run_summary = {
        "schema": "phase6.act_raw_demo.run.v4",
        "target": target,
        "task": args.task or f"target_key:{target}",
        "repo_id": repo_id,
        "dataset_root": str(dataset_root),
        "dataset_fps": DATASET_FPS,
        "state_target_vocab": list(TARGET_VOCAB),
        "state_dim": 6 + TARGET_DIM,
        "action_dim": 6,
        "motor_names": [_action_key(name) for name in motor_names],
        "training_fps": DATASET_FPS,
        "replay_control_hz_target": TELEOP_HZ,
        "run_dir": str(run_dir),
        "episodes_requested": args.episodes,
        "dataset_episodes_before_run": int(dataset.meta.total_episodes),
        "episodes": [],
    }

    try:
        top.start()
        wrist.start()
        _wait_initial_frames(top, wrist)

        # Match LeRobot's current practice: connect teleoperator before robot.
        leader.connect()
        robot.connect()
        if not robot.is_connected or not leader.is_connected:
            raise RuntimeError("robot/leader connection did not complete")

        print()
        print("SCOPE:")
        print("  - No ACT inference and no autonomous motion in this collector.")
        print("  - Every accepted episode is human leader teleoperation only.")
        print("  - Normal flow cues: COUNTDOWN -> ARMED -> operator natural motion -> s/SPACE -> SUCCESS.")
        print("  - During movement there is NO proximity sound, READY sound, distance target, or visual coaching.")
        print("  - No online WRIST recognition or Rerun visualization runs during recording; images are analyzed offline if needed.")
        print("  - ACT training samples remain 15 Hz, while a separate ~60 Hz actual sent-action trace is saved for physical whole-episode replay.")
        print("  - Move only AFTER the ARMED tone finishes; press s/SPACE only AFTER your natural motion is complete.")
        print(f"  - After SUCCESS, q is available for {args.review_s:.1f}s to discard and retry the same episode.")
        print("  - RETRY means this attempt did not count (q / timeout).")
        print("  - Only after REVIEW accepts the episode is it saved in background.")
        print("  - Start-state coverage is defined outside this tool; the collector records generic episodes only.")

        accepted_in_run = 0
        attempt_number = 0
        while accepted_in_run < args.episodes:
            attempt_number += 1
            episode_number = accepted_in_run + 1

            baseline_action, arm_present_state = _prepare_and_arm(
                robot,
                leader,
                sounds,
                committer,
                target=target,
                episode_number=episode_number,
                episode_count=args.episodes,
                motor_names=motor_names,
                arm_countdown_s=args.arm_countdown_s,
            )

            frames, summary = _collect_buffered_episode(
                robot=robot,
                leader=leader,
                top=top,
                wrist=wrist,
                sounds=sounds,
                motor_names=motor_names,
                target=target,
                episode_number=episode_number,
                episode_count=args.episodes,
                baseline_action=baseline_action,
                arm_present_state=arm_present_state,
                max_episode_s=args.max_episode_s,
            )
            summary["attempt_number"] = attempt_number

            if not frames:
                summary["accepted"] = False
                summary_path = _write_episode_audit(run_dir, summary)
                run_summary["episodes"].append(summary)
                (run_dir / "run_summary.json").write_text(
                    json.dumps(run_summary, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                print("[RETRY] Attempt not saved to LeRobot dataset. Reposition and retry the same episode slot.")
                print("audit:", summary_path)
                continue

            keep_episode = _review_episode(
                robot,
                leader,
                sounds,
                review_s=args.review_s,
                episode_number=episode_number,
            )
            summary["operator_review_s"] = float(args.review_s)
            summary["operator_discarded"] = not keep_episode

            if not keep_episode:
                summary["accepted"] = False
                summary_path = _write_episode_audit(run_dir, summary)
                run_summary["episodes"].append(summary)
                (run_dir / "run_summary.json").write_text(
                    json.dumps(run_summary, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                print("audit:", summary_path)
                continue

            summary["dataset_episode_index"] = int(dataset.meta.total_episodes)
            summary_path = _write_episode_audit(run_dir, summary)
            task = args.task or f"target_key:{target}"
            committer.start(frames, task=task, label=f"ep{episode_number:03d}")
            accepted_in_run += 1

            run_summary["episodes"].append(summary)
            run_summary["accepted_in_run"] = accepted_in_run
            (run_dir / "run_summary.json").write_text(
                json.dumps(run_summary, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            q = summary["quality"]
            print()
            print("-" * 72)
            print(f"ACCEPTED {episode_number}/{args.episodes}")
            print("-" * 72)
            print("saved samples       :", len(frames))
            print("duration            :", f"{q['duration_s']:.2f}s")
            print("pause windows       :", len(q["pause_windows"]))
            print("reversals           :", q["reversal_count"])
            print("warnings            :", q["warnings"] or "none")
            print("diagnostics         :", q["diagnostics"] or "none")
            print("req→sent max diff   :", f"{summary['max_requested_vs_sent_action_delta_deg']:.3f}deg")
            perf = summary["performance"]
            print("control send Hz     :", f"{perf['achieved_send_hz']:.1f}")
            print("inter-send p99/max :", f"{perf['inter_send_ms']['p99']:.1f}/{perf['inter_send_ms']['max']:.1f}ms")
            print("loop overruns       :", perf["loop_ms"]["overrun_count"])
            print("sample misses       :", perf["sample_missed_count"])
            print("audit summary       :", summary_path)
            if accepted_in_run < args.episodes:
                print("[NEXT] Review accepted. Teleop remains live; prepare the next episode according to your external SOP.")

        print()
        print("=" * 72)
        print("REQUESTED BATCH RECORDED")
        print("=" * 72)
        print(f"accepted this run: {accepted_in_run}/{args.episodes}")
        print("run summary       :", run_dir / "run_summary.json")
        print("dataset root      :", dataset_root)

        _run_recovery(robot, leader, committer, sounds)

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] No autonomous recovery will be attempted.")
        if robot.is_connected and leader.is_connected:
            _run_recovery(robot, leader, committer, sounds)
        else:
            committer.wait()
    finally:
        try:
            committer.wait()
        finally:
            dataset.finalize()
            run_summary["dataset_episodes_after_run"] = int(dataset.meta.total_episodes)
            run_summary["dataset_frames_after_run"] = int(dataset.meta.total_frames)
            (run_dir / "run_summary.json").write_text(
                json.dumps(run_summary, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            wrist.stop()
            top.stop()
            if robot.is_connected:
                robot.disconnect()
            if leader.is_connected:
                leader.disconnect()


if __name__ == "__main__":
    main()
