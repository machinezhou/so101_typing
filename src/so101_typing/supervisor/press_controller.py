from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math

from so101_typing.supervisor.verification import (
    ScreenVerificationStatus,
)


class PressControllerState(StrEnum):
    ALIGNING = "ALIGNING"
    WAITING_Z_SETTLE = "WAITING_Z_SETTLE"
    CHECKING_ALIGNMENT = "CHECKING_ALIGNMENT"
    REALIGNING = "REALIGNING"
    VERIFYING = "VERIFYING"
    HOLD_REOBSERVE = "HOLD_REOBSERVE"
    RELEASING = "RELEASING"
    RETRACTING = "RETRACTING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class PressDirectiveKind(StrEnum):
    ALIGN_XY = "ALIGN_XY"
    OBSERVE_WRIST = "OBSERVE_WRIST"
    DESCEND_TO_Z = "DESCEND_TO_Z"
    VERIFY_SCREEN = "VERIFY_SCREEN"
    REOBSERVE_SCREEN = "REOBSERVE_SCREEN"
    RELEASE_TO_Z = "RELEASE_TO_Z"
    RETRACT_TO_Z0 = "RETRACT_TO_Z0"
    COMPLETE = "COMPLETE"


class PressOutcome(StrEnum):
    SUCCESS = "SUCCESS"
    WRONG_KEY = "WRONG_KEY"
    DEPTH_EXHAUSTED = "DEPTH_EXHAUSTED"
    UNCERTAIN_EXHAUSTED = "UNCERTAIN_EXHAUSTED"


@dataclass(frozen=True, slots=True)
class PressDirective:
    kind: PressDirectiveKind
    z_target_mm: float | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class SingleKeyPressConfig:
    """Pure press-supervisor policy.

    Two Z modes are supported:

    1. Predefined levels:
       z_levels_mm=(-3.0, -6.0)

    2. Iterative descent:
       z_levels_mm=None
       z_step_mm=2.0

       When max_descent_mm is also None, the controller has no automatic
       cumulative-depth stop. This is intended only for supervised validation
       where an operator provides the absolute safety stop.

    SIDE verification remains the only press-success authority.
    """

    z_levels_mm: tuple[float, ...] | None = (
        -3.0,
        -6.0,
    )

    z_step_mm: float = 2.0

    # Optional software safety fuse for iterative mode.
    # None means no controller-side cumulative Z limit.
    max_descent_mm: float | None = None

    max_uncertain_reobservations: int = 3

    def __post_init__(self) -> None:
        if self.z_levels_mm is not None:
            levels = tuple(
                float(value)
                for value in self.z_levels_mm
            )

            if not levels:
                raise ValueError(
                    "z_levels_mm must not be empty "
                    "when predefined-level mode is used"
                )

            if not all(
                math.isfinite(value)
                and value < 0.0
                for value in levels
            ):
                raise ValueError(
                    "all predefined Z levels must be "
                    "finite and negative"
                )

            for previous, current in zip(
                levels,
                levels[1:],
            ):
                if current >= previous:
                    raise ValueError(
                        "predefined Z levels must become "
                        "strictly more negative"
                    )

            object.__setattr__(
                self,
                "z_levels_mm",
                levels,
            )

        step = float(
            self.z_step_mm
        )

        if (
            not math.isfinite(step)
            or step <= 0.0
        ):
            raise ValueError(
                "z_step_mm must be finite and positive"
            )

        object.__setattr__(
            self,
            "z_step_mm",
            step,
        )

        if self.max_descent_mm is not None:
            maximum = float(
                self.max_descent_mm
            )

            if (
                not math.isfinite(maximum)
                or maximum <= 0.0
            ):
                raise ValueError(
                    "max_descent_mm must be finite and "
                    "positive when provided"
                )

            object.__setattr__(
                self,
                "max_descent_mm",
                maximum,
            )

        if (
            isinstance(
                self.max_uncertain_reobservations,
                bool,
            )
            or not isinstance(
                self.max_uncertain_reobservations,
                int,
            )
            or self.max_uncertain_reobservations < 0
        ):
            raise ValueError(
                "max_uncertain_reobservations "
                "must be an integer >= 0"
            )


class SingleKeyPressController:
    """Pure state machine for one deterministic local key press."""

    def __init__(
        self,
        config: SingleKeyPressConfig | None = None,
    ) -> None:
        self.config = (
            config
            or SingleKeyPressConfig()
        )

        self.state = (
            PressControllerState.ALIGNING
        )

        self.current_z_target_mm = 0.0

        self.pending_outcome: (
            PressOutcome | None
        ) = None

        self._z_level_index = 0
        self._has_descended = False
        self._uncertain_count = 0
        self._screen_event_latched = False

    @property
    def z_level_index(self) -> int:
        """Index of the most recently authorized predefined Z level.

        -1 means no descent has been authorized yet.
        """
        return self._z_level_index - 1

    @property
    def current_z_mm(self) -> float:
        """Backward-compatible current cumulative Z command."""
        return self.current_z_target_mm

    @property
    def screen_event_latched(self) -> bool:
        return self._screen_event_latched

    def _require_state(
        self,
        *allowed: PressControllerState,
    ) -> None:
        if self.state not in allowed:
            raise RuntimeError(
                f"invalid press-controller transition "
                f"from state {self.state}"
            )

    def _begin_retraction(
        self,
        outcome: PressOutcome,
        *,
        reason: str,
    ) -> PressDirective:
        self.pending_outcome = outcome

        self.state = (
            PressControllerState.RETRACTING
        )

        return PressDirective(
            kind=(
                PressDirectiveKind.RETRACT_TO_Z0
            ),
            z_target_mm=0.0,
            reason=reason,
        )

    def _authorize_next_z(
        self,
    ) -> PressDirective:
        levels = (
            self.config.z_levels_mm
        )

        if levels is not None:
            if (
                self._z_level_index
                >= len(levels)
            ):
                return self._begin_retraction(
                    PressOutcome.DEPTH_EXHAUSTED,
                    reason=(
                        "screen still reports no-change "
                        "after final allowed Z level"
                    ),
                )

            next_z = float(
                levels[
                    self._z_level_index
                ]
            )

            self._z_level_index += 1

        else:
            # True iterative local descent.
            #
            # There is deliberately no implicit total-depth limit here.
            # Each authorization advances one small cumulative command-space
            # Z step from the same immutable Goal anchor.
            next_z = (
                float(
                    self.current_z_target_mm
                )
                - self.config.z_step_mm
            )

            maximum = (
                self.config.max_descent_mm
            )

            if (
                maximum is not None
                and abs(next_z)
                > maximum + 1e-12
            ):
                return self._begin_retraction(
                    PressOutcome.DEPTH_EXHAUSTED,
                    reason=(
                        "next iterative Z step would "
                        "exceed configured safety fuse"
                    ),
                )

        self.current_z_target_mm = (
            next_z
        )

        self._has_descended = True

        self.state = (
            PressControllerState.WAITING_Z_SETTLE
        )

        return PressDirective(
            kind=(
                PressDirectiveKind.DESCEND_TO_Z
            ),
            z_target_mm=next_z,
            reason=(
                "alignment authorized next "
                "bounded cumulative Z step"
            ),
        )

    def on_alignment(
        self,
        *,
        aligned: bool,
    ) -> PressDirective:
        self._require_state(
            PressControllerState.ALIGNING,
            PressControllerState.CHECKING_ALIGNMENT,
            PressControllerState.REALIGNING,
        )

        if not aligned:
            self.state = (
                PressControllerState.REALIGNING
            )

            return PressDirective(
                kind=(
                    PressDirectiveKind.ALIGN_XY
                ),
                z_target_mm=(
                    self.current_z_target_mm
                ),
                reason=(
                    "WRIST alignment required "
                    "at current Z"
                ),
            )

        if not self._has_descended:
            return self._authorize_next_z()

        self.state = (
            PressControllerState.VERIFYING
        )

        return PressDirective(
            kind=(
                PressDirectiveKind.VERIFY_SCREEN
            ),
            z_target_mm=(
                self.current_z_target_mm
            ),
            reason=(
                "WRIST is aligned at the "
                "current stopped Z level"
            ),
        )

    def on_z_motion_stable(
        self,
    ) -> PressDirective:
        self._require_state(
            PressControllerState.WAITING_Z_SETTLE
        )

        self.state = (
            PressControllerState.CHECKING_ALIGNMENT
        )

        return PressDirective(
            kind=(
                PressDirectiveKind.OBSERVE_WRIST
            ),
            z_target_mm=(
                self.current_z_target_mm
            ),
            reason=(
                "Z motion stopped; acquire "
                "fresh WRIST observation"
            ),
        )

    def on_screen_event(
        self,
        *,
        release_z_target_mm: float,
    ) -> PressDirective:
        """Latch a high-priority SIDE press event.

        After this transition, this press attempt may never authorize another
        downward command.  The caller must first release the key, then verify
        the released screen state.
        """

        if self.state in {
            PressControllerState.RETRACTING,
            PressControllerState.SUCCEEDED,
            PressControllerState.FAILED,
        }:
            raise RuntimeError(
                f"cannot latch SIDE event from terminal state {self.state}"
            )

        target = float(release_z_target_mm)
        if not math.isfinite(target):
            raise ValueError("release_z_target_mm must be finite")

        if target < self.current_z_target_mm - 1e-12:
            raise ValueError("release target must not move farther downward")

        self._screen_event_latched = True
        self.current_z_target_mm = target
        self.state = PressControllerState.RELEASING

        return PressDirective(
            kind=PressDirectiveKind.RELEASE_TO_Z,
            z_target_mm=target,
            reason=(
                "SIDE detected persistent new screen foreground; "
                "release key before OCR classification"
            ),
        )

    def on_release_complete(self) -> PressDirective:
        self._require_state(PressControllerState.RELEASING)
        if not self._screen_event_latched:
            raise RuntimeError("release completed without a latched SIDE event")

        self.state = PressControllerState.VERIFYING
        return PressDirective(
            kind=PressDirectiveKind.VERIFY_SCREEN,
            z_target_mm=self.current_z_target_mm,
            reason="key released; classify the latched SIDE screen event",
        )

    def on_verification(
        self,
        status: ScreenVerificationStatus,
    ) -> PressDirective:
        self._require_state(
            PressControllerState.VERIFYING,
            PressControllerState.HOLD_REOBSERVE,
        )

        if (
            status
            is ScreenVerificationStatus.CONFIRMED_SUCCESS
        ):
            self._uncertain_count = 0

            return self._begin_retraction(
                PressOutcome.SUCCESS,
                reason=(
                    "SIDE confirmed expected "
                    "screen character"
                ),
            )

        if (
            status
            is ScreenVerificationStatus.CONFIRMED_WRONG
        ):
            self._uncertain_count = 0

            return self._begin_retraction(
                PressOutcome.WRONG_KEY,
                reason=(
                    "SIDE confirmed wrong "
                    "screen continuation"
                ),
            )

        if (
            status
            is ScreenVerificationStatus.UNCERTAIN
        ):
            self._uncertain_count += 1

            if (
                self._uncertain_count
                > self.config.max_uncertain_reobservations
            ):
                return self._begin_retraction(
                    PressOutcome.UNCERTAIN_EXHAUSTED,
                    reason=(
                        "SIDE remained uncertain "
                        "beyond re-observation budget"
                    ),
                )

            self.state = (
                PressControllerState.HOLD_REOBSERVE
            )

            return PressDirective(
                kind=(
                    PressDirectiveKind.REOBSERVE_SCREEN
                ),
                z_target_mm=(
                    self.current_z_target_mm
                ),
                reason=(
                    "SIDE uncertain; hold "
                    "current Z and re-observe"
                ),
            )

        if (
            status
            is ScreenVerificationStatus.CONFIRMED_NO_CHANGE
        ):
            if self._screen_event_latched:
                # Once SIDE has observed a real screen-change event, NO_CHANGE
                # after release is treated as uncertainty.  Never descend again.
                self._uncertain_count += 1

                if (
                    self._uncertain_count
                    > self.config.max_uncertain_reobservations
                ):
                    return self._begin_retraction(
                        PressOutcome.UNCERTAIN_EXHAUSTED,
                        reason=(
                            "SIDE event was latched but released-screen "
                            "classification remained unresolved"
                        ),
                    )

                self.state = PressControllerState.HOLD_REOBSERVE
                return PressDirective(
                    kind=PressDirectiveKind.REOBSERVE_SCREEN,
                    z_target_mm=self.current_z_target_mm,
                    reason=(
                        "SIDE event already latched; hold released Z and "
                        "re-observe, never descend"
                    ),
                )

            self._uncertain_count = 0
            return self._authorize_next_z()

        raise ValueError(
            f"unsupported screen verification status: "
            f"{status}"
        )

    def on_retraction_complete(
        self,
    ) -> PressDirective:
        self._require_state(
            PressControllerState.RETRACTING
        )

        if self.pending_outcome is None:
            raise RuntimeError(
                "retraction completed without "
                "a pending outcome"
            )

        self.current_z_target_mm = 0.0

        if (
            self.pending_outcome
            is PressOutcome.SUCCESS
        ):
            self.state = (
                PressControllerState.SUCCEEDED
            )

        else:
            self.state = (
                PressControllerState.FAILED
            )

        return PressDirective(
            kind=(
                PressDirectiveKind.COMPLETE
            ),
            z_target_mm=0.0,
            reason=str(
                self.pending_outcome
            ),
        )
