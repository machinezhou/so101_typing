from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from enum import StrEnum
import json
from pathlib import Path
import signal
import sys
import time
import traceback
from threading import Event
from unittest.mock import patch


TARGET_ALPHABET = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
ARTIFACT_ROOT = Path("artifacts/phase9_multi_key_typing")


class _Tee:
    """Mirror console output to the terminal and a persistent log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(
            bool(getattr(stream, "isatty", lambda: False)())
            for stream in self.streams
        )


class SingleKeyAttemptOutcome(StrEnum):
    SUCCESS = "SUCCESS"
    WRONG = "WRONG"
    UNCERTAIN = "UNCERTAIN"
    HANDOFF_FAILURE = "HANDOFF_FAILURE"
    TIMEOUT = "TIMEOUT"
    ABORTED = "ABORTED"


class MultiKeyTaskState(StrEnum):
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class MultiKeyTaskSnapshot:
    target_text: str
    state: MultiKeyTaskState
    current_index: int
    current_target: str | None
    completed_text: str
    failed_index: int | None
    failed_target: str | None
    failed_outcome: SingleKeyAttemptOutcome | None


def normalize_target_text(text: str) -> str:
    normalized = str(text).strip().upper()
    if not normalized:
        raise ValueError("Phase-9 target text must not be empty")
    if any(ch not in TARGET_ALPHABET for ch in normalized):
        raise ValueError("Phase-9 V0 supports A-Z strings only")
    return normalized


class MultiKeyTypingSupervisor:
    def __init__(self, target_text: str):
        self.target_text = normalize_target_text(target_text)
        self.state = MultiKeyTaskState.RUNNING
        self.current_index = 0
        self.completed_text = ""
        self.failed_index: int | None = None
        self.failed_target: str | None = None
        self.failed_outcome: SingleKeyAttemptOutcome | None = None

    @property
    def current_target(self) -> str | None:
        if self.state is not MultiKeyTaskState.RUNNING:
            return None
        return self.target_text[self.current_index]

    def snapshot(self) -> MultiKeyTaskSnapshot:
        return MultiKeyTaskSnapshot(
            target_text=self.target_text,
            state=self.state,
            current_index=self.current_index,
            current_target=self.current_target,
            completed_text=self.completed_text,
            failed_index=self.failed_index,
            failed_target=self.failed_target,
            failed_outcome=self.failed_outcome,
        )

    def record_attempt(
        self,
        outcome: SingleKeyAttemptOutcome | str,
    ) -> MultiKeyTaskSnapshot:
        if self.state is not MultiKeyTaskState.RUNNING:
            raise RuntimeError("cannot record an attempt after Phase-9 task termination")

        resolved = SingleKeyAttemptOutcome(outcome)
        target = self.target_text[self.current_index]

        if resolved is SingleKeyAttemptOutcome.SUCCESS:
            self.completed_text += target
            self.current_index += 1
            if self.current_index >= len(self.target_text):
                self.state = MultiKeyTaskState.SUCCEEDED
            return self.snapshot()

        self.failed_index = self.current_index
        self.failed_target = target
        self.failed_outcome = resolved
        self.state = MultiKeyTaskState.FAILED
        return self.snapshot()


def _phase9_process_exit_code(
    task_state: MultiKeyTaskState | str,
    *,
    session_contract_ok: bool = True,
) -> int:
    """Return process integrity status, not the physical typing verdict.

    SUCCEEDED and FAILED are both handled Phase-9 task outcomes, so both return
    zero. Non-zero is reserved for a violated runtime/session contract or an
    unhandled software/infrastructure exception.
    """
    MultiKeyTaskState(task_state)
    return 0 if session_contract_ok else 2


def _map_phase8_outcome(
    deterministic_result: dict | None,
    summary: dict | None,
) -> SingleKeyAttemptOutcome:
    if deterministic_result is not None:
        if bool(deterministic_result.get("success")):
            return SingleKeyAttemptOutcome.SUCCESS
        outcome = str(deterministic_result.get("outcome") or "")
        if outcome == "WRONG_KEY":
            return SingleKeyAttemptOutcome.WRONG
        if outcome.startswith("UNCERTAIN"):
            return SingleKeyAttemptOutcome.UNCERTAIN

    task_status = str((summary or {}).get("task_status") or "")
    if "TIMEOUT" in task_status:
        return SingleKeyAttemptOutcome.TIMEOUT
    if "ABORT" in task_status or "OPERATOR" in task_status:
        return SingleKeyAttemptOutcome.ABORTED
    return SingleKeyAttemptOutcome.HANDOFF_FAILURE


class _ReusablePhase8Session:
    """Reuse the accepted Phase-8 main() without reopening its control algorithm.

    Only resource lifetime changes:
      * cameras / robot / ACT policy are created once;
      * every later key gets a fresh SIDE baseline;
      * successful intermediate keys defer canonical HOME and teardown;
      * a failure or the final key performs the original HOME + teardown.
    """

    EXPECTED_KEYS = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]

    def __init__(self, phase8, *, total_attempts: int):
        self.phase8 = phase8
        self.total_attempts = int(total_attempts)
        self.attempt_index = -1
        self.target: str | None = None
        self.current_success: bool | None = None
        self.current_deterministic_result: dict | None = None
        self.current_run_dir: Path | None = None

        self.camera_objects: list[object] = []
        self.camera_factory_calls = 0
        self.robot_object = None
        self.robot_factory_calls = 0

        self.policy_object = None
        self.policy_load_calls = 0
        self.processors = None
        self.processor_build_calls = 0

        self.session_started = False
        self.session_closed = False
        self.session_start_calls = 0
        self.session_reuse_calls = 0
        self.deferred_home_calls = 0
        self.real_home_calls = 0
        self.deferred_teardown_calls = 0
        self.real_teardown_calls = 0

        self.motor_names: list[str] = []
        self.home_baseline: dict | None = None

        self._orig_camera_ctor = phase8.ThreadedOpenCVCamera
        self._orig_robot_ctor = phase8.SO101Follower
        self._orig_policy_loader = phase8.ACTPolicy.from_pretrained
        self._orig_make_processors = phase8.make_pre_post_processors
        self._orig_start = phase8._start_runtime_devices
        self._orig_stop = phase8._stop_runtime_devices
        self._orig_home = phase8._restore_canonical_home
        self._orig_press = phase8._run_full_deterministic_press

    @property
    def is_intermediate_success(self) -> bool:
        return (
            self.current_success is True
            and 0 <= self.attempt_index < self.total_attempts - 1
        )

    def begin_attempt(self, index: int, target: str) -> None:
        self.attempt_index = int(index)
        self.target = str(target)
        self.current_success = None
        self.current_deterministic_result = None
        self.current_run_dir = None

    def _camera_factory(self, spec):
        slot = self.camera_factory_calls % 3
        if self.camera_factory_calls < 3:
            obj = self._orig_camera_ctor(spec)
            self.camera_objects.append(obj)
        else:
            if len(self.camera_objects) != 3:
                raise RuntimeError("Phase-9 camera cache is incomplete")
            obj = self.camera_objects[slot]
        self.camera_factory_calls += 1
        return obj

    def _robot_factory(self, config):
        if self.robot_object is None:
            self.robot_object = self._orig_robot_ctor(config)
        self.robot_factory_calls += 1
        return self.robot_object

    def _policy_loader(self, *args, **kwargs):
        if self.policy_object is None:
            self.policy_object = self._orig_policy_loader(*args, **kwargs)
            self.policy_load_calls += 1
        return self.policy_object

    def _make_processors(self, *args, **kwargs):
        if self.processors is None:
            self.processors = self._orig_make_processors(*args, **kwargs)
            self.processor_build_calls += 1
        return self.processors

    def _validate_connected_session(self, *, robot, home_baseline):
        if not robot.is_connected:
            raise RuntimeError("Phase-9 expected the shared robot session to remain connected")

        motor_names = list(robot.bus.motors.keys())
        if len(motor_names) != 6:
            raise RuntimeError(f"Expected 6 motors, got {motor_names}")

        keys = [self.phase8._action_key(name) for name in motor_names]
        missing_home = [key for key in keys if key not in home_baseline]
        if missing_home:
            raise RuntimeError(f"Recovery HOME missing keys: {missing_home}")
        if keys != self.EXPECTED_KEYS:
            raise RuntimeError(
                "Follower motor order no longer matches the Phase-6 dataset action order: "
                f"{keys} != {self.EXPECTED_KEYS}"
            )
        return motor_names, keys

    def _start_runtime_devices(
        self,
        *,
        top,
        wrist,
        side,
        robot,
        home_baseline: dict,
        screen_calibration,
        run_dir: Path,
    ):
        self.current_run_dir = Path(run_dir)
        self.home_baseline = home_baseline

        if not self.session_started:
            result = self._orig_start(
                top=top,
                wrist=wrist,
                side=side,
                robot=robot,
                home_baseline=home_baseline,
                screen_calibration=screen_calibration,
                run_dir=run_dir,
            )
            self.session_started = True
            self.session_closed = False
            self.session_start_calls += 1
            self.motor_names = list(result[0])
            print("[PHASE9 SESSION] cameras + robot started ONCE")
            return result

        if self.session_closed:
            raise RuntimeError("Phase-9 shared session was already closed")
        if len(self.camera_objects) != 3:
            raise RuntimeError("Phase-9 camera session is not available")
        if top is not self.camera_objects[0]:
            raise RuntimeError("TOP camera object changed between Phase-9 keys")
        if wrist is not self.camera_objects[1]:
            raise RuntimeError("WRIST camera object changed between Phase-9 keys")
        if side is not self.camera_objects[2]:
            raise RuntimeError("SIDE camera object changed between Phase-9 keys")
        if robot is not self.robot_object:
            raise RuntimeError("robot object changed between Phase-9 keys")

        self.phase8._wait_initial_frames(top, wrist)
        side_deadline = time.monotonic() + 5.0
        while time.monotonic() < side_deadline:
            if side.latest() is not None:
                break
            time.sleep(0.02)
        else:
            raise RuntimeError("SIDE stopped producing frames between Phase-9 keys")

        time.sleep(0.5)
        print()
        print("===== PHASE 9 FRESH PER-KEY SIDE BASELINE =====")
        screen_baseline, _ = self.phase8._capture_dynamic_side_snapshot(
            side_camera=side,
            calibration=screen_calibration,
            run_dir=run_dir,
        )

        motor_names, keys = self._validate_connected_session(
            robot=robot,
            home_baseline=home_baseline,
        )
        self.motor_names = list(motor_names)
        self.session_reuse_calls += 1
        print(
            "[PHASE9 SESSION REUSE] "
            f"key={self.target} cameras=KEEP robot=KEEP HOME=NO "
            "SIDE baseline=FRESH"
        )
        return motor_names, keys, screen_baseline

    def _run_full_deterministic_press(self, *args, **kwargs):
        result = self._orig_press(*args, **kwargs)
        self.current_deterministic_result = result
        self.current_success = bool(result.get("success"))
        return result

    def _restore_canonical_home(self, **kwargs):
        if self.is_intermediate_success:
            self.deferred_home_calls += 1
            print()
            print(
                "[PHASE9 INTER-KEY] "
                f"{self.target} SUCCESS -> canonical HOME DEFERRED"
            )
            return {
                "status": "DEFERRED_INTER_KEY_SUCCESS",
                "present_hold": None,
                "restore": None,
                "final_goal_diff_deg": float("nan"),
            }

        self.real_home_calls += 1
        return self._orig_home(**kwargs)

    def _stop_runtime_devices(
        self,
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
        if self.is_intermediate_success:
            self.deferred_teardown_calls += 1
            self.phase8.phase5.ACTIVE_SCREEN_WATCHER = None
            self.phase8.phase5.raise_if_screen_event = original_raise_if_screen_event
            self.phase8.phase5.wait_motion_stable = original_wait_motion_stable
            if signal_installed:
                signal.signal(signal.SIGINT, previous_sigint)

            print(
                "[PHASE9 INTER-KEY] "
                "cameras KEEP / robot KEEP / disconnect NO"
            )
            return

        self.real_teardown_calls += 1
        self._orig_stop(
            top=top,
            wrist=wrist,
            side=side,
            robot=robot,
            original_raise_if_screen_event=original_raise_if_screen_event,
            original_wait_motion_stable=original_wait_motion_stable,
            signal_installed=signal_installed,
            previous_sigint=previous_sigint,
        )
        self.session_closed = True

    def install(self):
        stack = ExitStack()
        stack.enter_context(
            patch.object(
                self.phase8,
                "ThreadedOpenCVCamera",
                side_effect=self._camera_factory,
            )
        )
        stack.enter_context(
            patch.object(
                self.phase8,
                "SO101Follower",
                side_effect=self._robot_factory,
            )
        )
        stack.enter_context(
            patch.object(
                self.phase8.ACTPolicy,
                "from_pretrained",
                side_effect=self._policy_loader,
            )
        )
        stack.enter_context(
            patch.object(
                self.phase8,
                "make_pre_post_processors",
                side_effect=self._make_processors,
            )
        )
        stack.enter_context(
            patch.object(
                self.phase8,
                "_start_runtime_devices",
                side_effect=self._start_runtime_devices,
            )
        )
        stack.enter_context(
            patch.object(
                self.phase8,
                "_run_full_deterministic_press",
                side_effect=self._run_full_deterministic_press,
            )
        )
        stack.enter_context(
            patch.object(
                self.phase8,
                "_restore_canonical_home",
                side_effect=self._restore_canonical_home,
            )
        )
        stack.enter_context(
            patch.object(
                self.phase8,
                "_stop_runtime_devices",
                side_effect=self._stop_runtime_devices,
            )
        )
        return stack

    def read_current_summary(self) -> dict | None:
        if self.current_run_dir is None:
            return None
        path = self.current_run_dir / "summary.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def force_finalize(self) -> None:
        """Best-effort safety finalization if Phase-9 exits between keys."""
        if not self.session_started or self.session_closed:
            return
        robot = self.robot_object
        if robot is None:
            return

        print()
        print("[PHASE9 SAFETY FINALIZE] shared session still open")

        try:
            if (
                robot.is_connected
                and self.home_baseline is not None
                and self.motor_names
            ):
                print("[PHASE9 SAFETY FINALIZE] restoring canonical HOME")
                result = self._orig_home(
                    robot=robot,
                    motor_names=self.motor_names,
                    home_baseline=self.home_baseline,
                    home_stop=Event(),
                )
                self.real_home_calls += 1
                print(
                    "[PHASE9 SAFETY FINALIZE] HOME status=",
                    result.get("status"),
                )
        except BaseException as exc:
            print("[PHASE9 SAFETY FINALIZE] HOME ERROR:", repr(exc))
        finally:
            self.phase8.phase5.ACTIVE_SCREEN_WATCHER = None
            for camera in self.camera_objects:
                try:
                    camera.stop()
                except BaseException as exc:
                    print("[PHASE9 SAFETY FINALIZE] camera stop ERROR:", repr(exc))
            try:
                if robot.is_connected:
                    robot.disconnect()
            except BaseException as exc:
                print("[PHASE9 SAFETY FINALIZE] disconnect ERROR:", repr(exc))
            self.session_closed = True

    def counters(self) -> dict:
        return {
            "camera_factory_calls": int(self.camera_factory_calls),
            "camera_objects_created": int(len(self.camera_objects)),
            "robot_factory_calls": int(self.robot_factory_calls),
            "robot_objects_created": int(self.robot_object is not None),
            "policy_load_calls": int(self.policy_load_calls),
            "processor_build_calls": int(self.processor_build_calls),
            "session_start_calls": int(self.session_start_calls),
            "session_reuse_calls": int(self.session_reuse_calls),
            "deferred_home_calls": int(self.deferred_home_calls),
            "real_home_calls": int(self.real_home_calls),
            "deferred_teardown_calls": int(self.deferred_teardown_calls),
            "real_teardown_calls": int(self.real_teardown_calls),
        }


def _phase8_argv(args, target: str) -> list[str]:
    return [
        "phase8_single_key_integration.py",
        "--target",
        target,
        "--checkpoint",
        str(args.checkpoint),
        "--duration",
        str(args.duration),
        "--n-action-steps",
        str(args.n_action_steps),
        "--robot-port",
        str(args.robot_port),
        "--recovery-home",
        str(args.recovery_home),
    ]


def _assert_phase8_reuse_contract(phase8) -> None:
    required = [
        "_start_runtime_devices",
        "_stop_runtime_devices",
        "_restore_canonical_home",
        "_reset_single_key_attempt_state",
        "_run_full_deterministic_press",
    ]
    missing = [name for name in required if not hasattr(phase8, name)]
    if missing:
        raise RuntimeError(
            "Local Phase-8 reusable-boundary refactor is incomplete; missing: "
            + ", ".join(missing)
        )


def run_hardware(args) -> int:
    import phase8_single_key_integration as phase8

    _assert_phase8_reuse_contract(phase8)

    target_text = normalize_target_text(args.text)
    supervisor = MultiKeyTypingSupervisor(target_text)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    task_dir = ARTIFACT_ROOT / f"run_{timestamp}_{target_text.lower()}"
    task_dir.mkdir(parents=True, exist_ok=False)
    summary_path = task_dir / "summary.json"

    adapter = _ReusablePhase8Session(
        phase8,
        total_attempts=len(target_text),
    )

    print("=" * 78)
    print("PHASE 9 — MULTI-KEY TYPING V0")
    print("=" * 78)
    print("target text          :", target_text)
    print("per-key primitive    : accepted Phase-8 main()")
    print("ACT n_action_steps   :", args.n_action_steps)
    print("inter-key HOME       : DISABLED after SUCCESS")
    print("inter-key disconnect : DISABLED after SUCCESS")
    print("inter-key cameras    : KEEP ALIVE")
    print("per-key SIDE baseline: FRESH")
    print("failure policy       : STOP task -> safety recovery -> ONE final HOME")
    print("final success policy : ONE canonical HOME -> teardown")
    print("auto correction      : DISABLED (Phase 10)")
    print("task artifacts       :", task_dir)
    print("=" * 78)

    attempts: list[dict] = []
    previous_argv = list(sys.argv)
    escaped_error: str | None = None

    try:
        with adapter.install():
            for index, target in enumerate(target_text):
                if supervisor.state is not MultiKeyTaskState.RUNNING:
                    break

                adapter.begin_attempt(index, target)
                print()
                print("#" * 78)
                print(
                    f"PHASE 9 KEY {index + 1}/{len(target_text)} "
                    f"target={target!r} completed={supervisor.completed_text!r}"
                )
                print("#" * 78)

                sys.argv = _phase8_argv(args, target)

                raised = None
                try:
                    phase8.main()
                except KeyboardInterrupt:
                    raised = "KeyboardInterrupt"
                except Exception:
                    # A software/infrastructure exception is not a normal
                    # typing outcome. Let the outer handler record it and
                    # preserve a real traceback/non-zero process status.
                    raise

                phase8_summary = adapter.read_current_summary()
                outcome = _map_phase8_outcome(
                    adapter.current_deterministic_result,
                    phase8_summary,
                )
                if raised is not None:
                    outcome = SingleKeyAttemptOutcome.ABORTED

                snapshot = supervisor.record_attempt(outcome)
                attempts.append(
                    {
                        "index": int(index),
                        "target": target,
                        "outcome": str(outcome),
                        "phase8_run_dir": (
                            None
                            if adapter.current_run_dir is None
                            else str(adapter.current_run_dir)
                        ),
                        "phase8_task_status": (
                            None
                            if phase8_summary is None
                            else phase8_summary.get("task_status")
                        ),
                        "deterministic_result": adapter.current_deterministic_result,
                        "raised": raised,
                        "completed_text_after": snapshot.completed_text,
                    }
                )

                summary_path.write_text(
                    json.dumps(
                        {
                            "schema": "phase9.multi_key_typing.v0",
                            "target_text": target_text,
                            "supervisor": asdict(supervisor.snapshot()),
                            "attempts": attempts,
                            "session": adapter.counters(),
                            "status": "RUNNING",
                        },
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n",
                    encoding="utf-8",
                )

                print()
                print(
                    "[PHASE9 KEY RESULT] "
                    f"target={target} outcome={outcome} "
                    f"completed={supervisor.completed_text!r}"
                )

                if outcome is not SingleKeyAttemptOutcome.SUCCESS:
                    print("[PHASE9 STOP] non-success outcome; next character is NOT attempted")
                    break

    except BaseException as exc:
        escaped_error = repr(exc)
        raise
    finally:
        sys.argv = previous_argv
        adapter.force_finalize()

        final_snapshot = supervisor.snapshot()
        final_payload = {
            "schema": "phase9.multi_key_typing.v0",
            "target_text": target_text,
            "supervisor": asdict(final_snapshot),
            "attempts": attempts,
            "session": adapter.counters(),
            "verified_sequence": supervisor.completed_text,
            "status": str(final_snapshot.state),
            "escaped_error": escaped_error,
        }
        summary_path.write_text(
            json.dumps(
                final_payload,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )

    print()
    print("=" * 78)
    print("PHASE 9 TASK RESULT")
    print("=" * 78)
    print("target text          :", target_text)
    print("verified sequence    :", supervisor.completed_text)
    print("task state           :", supervisor.state)
    print("attempts             :", len(attempts))
    print("session counters     :", adapter.counters())
    print("summary              :", summary_path)

    if supervisor.state is MultiKeyTaskState.SUCCEEDED:
        expected_deferred = max(0, len(target_text) - 1)
        counters = adapter.counters()
        contract_ok = (
            counters["camera_objects_created"] == 3
            and counters["robot_objects_created"] == 1
            and counters["policy_load_calls"] == 1
            and counters["processor_build_calls"] == 1
            and counters["session_start_calls"] == 1
            and counters["session_reuse_calls"] == expected_deferred
            and counters["deferred_home_calls"] == expected_deferred
            and counters["real_home_calls"] == 1
            and counters["deferred_teardown_calls"] == expected_deferred
            and counters["real_teardown_calls"] == 1
        )
        print("session contract     :", "PASS" if contract_ok else "FAIL")
        if not contract_ok:
            print("[PHASE9 FAIL] typing succeeded but shared-session contract was violated")
            return _phase9_process_exit_code(
                supervisor.state,
                session_contract_ok=False,
            )
        print("[PHASE9 PASS] every character succeeded in order; final HOME completed once.")
        return _phase9_process_exit_code(supervisor.state)

    print(
        "[PHASE9 TASK FAILED] "
        f"index={supervisor.failed_index} "
        f"target={supervisor.failed_target} "
        f"outcome={supervisor.failed_outcome}"
    )
    print(
        "[PHASE9 PROCESS] task failure was handled safely; "
        "process exit remains 0. Inspect summary.json for the task verdict."
    )
    return _phase9_process_exit_code(supervisor.state)


def main() -> None:
    import phase8_single_key_integration as phase8

    parser = argparse.ArgumentParser(
        description=(
            "Phase 9 V0 multi-key typing: reuse the accepted Phase-8 single-key "
            "primitive in one shared hardware/model session."
        )
    )
    parser.add_argument("--text", default="CAT")
    parser.add_argument("--checkpoint", type=Path, default=phase8.DEFAULT_CHECKPOINT)
    parser.add_argument("--duration", type=float, default=phase8.DEFAULT_DURATION_S)
    parser.add_argument("--n-action-steps", type=int, default=phase8.DEFAULT_N_ACTION_STEPS)
    parser.add_argument("--robot-port", default=phase8.ROBOT_PORT)
    parser.add_argument("--recovery-home", type=Path, default=phase8.RECOVERY_HOME_CONFIG)
    parser.add_argument("--print-plan", action="store_true")
    args = parser.parse_args()

    target_text = normalize_target_text(args.text)
    if not 1 <= int(args.n_action_steps) <= int(phase8.CHUNK_SIZE):
        raise ValueError(
            f"--n-action-steps must be in [1, {phase8.CHUNK_SIZE}]"
        )

    if args.print_plan:
        print("=" * 78)
        print("PHASE 9 — MULTI-KEY PLAN")
        print("=" * 78)
        print("target               :", target_text)
        print("sequence             :", " -> ".join(target_text))
        print("inter-key HOME       : NO")
        print("inter-key disconnect : NO")
        print("cameras/model        : ONE shared session")
        print("SIDE baseline        : fresh before every key")
        print("failure              : stop immediately + final recovery/HOME")
        print("final                : ONE HOME + teardown")
        print("=" * 78)
        return

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    log_stamp = time.strftime("%Y%m%d_%H%M%S")
    console_log = (
        ARTIFACT_ROOT
        / f"console_{log_stamp}_{target_text.lower()}.log"
    )

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with console_log.open(
        "w",
        encoding="utf-8",
        buffering=1,
    ) as log_stream:
        with redirect_stdout(_Tee(original_stdout, log_stream)):
            with redirect_stderr(_Tee(original_stderr, log_stream)):
                print("[PHASE9 CONSOLE LOG]", console_log)
                try:
                    exit_code = run_hardware(args)
                except BaseException:
                    print()
                    print("[PHASE9 UNHANDLED EXCEPTION]")
                    traceback.print_exc()
                    print("[PHASE9 CONSOLE LOG]", console_log)
                    raise

    print("[PHASE9 CONSOLE LOG]", console_log)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
