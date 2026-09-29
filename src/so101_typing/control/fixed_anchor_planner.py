from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np

from so101_typing.control.cartesian_kinematics import (
    end_effector_xyz_mm,
    make_cartesian_delta_action,
    ordered_joint_observation,
)
from so101_typing.control.fixed_anchor_xyz import (
    FixedAnchorXYZCommandState,
    FixedGoalAnchor,
)


@dataclass(frozen=True, slots=True)
class FixedAnchorPlannerConfig:
    max_command_norm_mm: float = 35.0
    max_model_xy_error_mm: float = 1.0
    max_model_z_error_mm: float = 1.0
    max_relative_target_deg: float = 10.0
    max_zero_joint_shift_deg: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "max_command_norm_mm",
            "max_model_xy_error_mm",
            "max_model_z_error_mm",
            "max_relative_target_deg",
            "max_zero_joint_shift_deg",
        ):
            value = float(
                getattr(self, name)
            )

            if (
                not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(
                    f"{name} must be finite and positive"
                )

            object.__setattr__(
                self,
                name,
                value,
            )


@dataclass(frozen=True, slots=True)
class FixedAnchorPlan:
    joint_action: dict
    requested_xyz_mm: tuple[
        float,
        float,
        float,
    ]
    predicted_xyz_mm: tuple[
        float,
        float,
        float,
    ]
    predicted_delta_mm: tuple[
        float,
        float,
        float,
    ]
    model_error_mm: tuple[
        float,
        float,
        float,
    ]
    model_xy_error_mm: float
    model_z_error_mm: float
    max_relative_joint_delta_deg: float


class FixedAnchorCartesianPlanner:
    """Plan cumulative XYZ against one immutable existing Goal_Position anchor.

    Important runtime contract:

    * the supplied Cartesian pipeline must be fresh for this press attempt;
    * the first pipeline call is zero-delta at the Goal anchor;
    * the underlying EEReferenceAndDelta must use use_latched_reference=True;
    * later Present_Position observations are IK/safety input only;
    * Present_Position never replaces the command anchor.
    """

    def __init__(
        self,
        *,
        pipeline,
        validation_kinematics,
        motor_names: Sequence[str],
        anchor: FixedGoalAnchor,
        anchor_xyz_mm: Sequence[float],
        config: FixedAnchorPlannerConfig | None = None,
    ) -> None:
        names = tuple(
            str(name)
            for name in motor_names
        )

        if not names:
            raise ValueError(
                "motor_names must not be empty"
            )

        if anchor.motor_names != names:
            raise ValueError(
                "anchor motor order must exactly match motor_names"
            )

        anchor_xyz = np.asarray(
            anchor_xyz_mm,
            dtype=np.float64,
        )

        if (
            anchor_xyz.shape != (3,)
            or not np.isfinite(anchor_xyz).all()
        ):
            raise ValueError(
                "anchor_xyz_mm must contain exactly "
                "three finite values"
            )

        self.pipeline = pipeline
        self.validation_kinematics = (
            validation_kinematics
        )
        self.motor_names = names
        self.anchor = anchor
        self.anchor_xyz_mm = (
            anchor_xyz.copy()
        )
        self.config = (
            config
            or FixedAnchorPlannerConfig()
        )

        self._latched = False

    @property
    def latched(self) -> bool:
        return self._latched

    def latch_zero_delta(
        self,
    ) -> float:
        """Latch the fresh Cartesian pipeline at the Goal anchor exactly once."""

        if self._latched:
            raise RuntimeError(
                "fixed Goal anchor is already latched"
            )

        observation = (
            ordered_joint_observation(
                self.anchor.as_observation(),
                self.motor_names,
            )
        )

        zero_action = (
            make_cartesian_delta_action(
                0.0,
                0.0,
                delta_z_mm=0.0,
                max_delta_norm_mm=(
                    self.config.max_command_norm_mm
                ),
            )
        )

        joint_action = self.pipeline(
            (
                zero_action,
                observation,
            )
        )

        shifts: list[float] = []

        violations: list[
            tuple[str, float]
        ] = []

        for name in self.motor_names:
            if name == "gripper":
                continue

            key = f"{name}.pos"

            if key not in joint_action:
                raise ValueError(
                    f"planned action is missing {key}"
                )

            shift = abs(
                float(joint_action[key])
                - float(observation[key])
            )

            shifts.append(
                shift
            )

            if (
                shift
                > self.config.max_zero_joint_shift_deg
            ):
                violations.append(
                    (name, shift)
                )

        max_shift = max(
            shifts,
            default=0.0,
        )

        if violations:
            details = ", ".join(
                f"{name}={shift:.3f}deg"
                for name, shift in violations
            )

            raise RuntimeError(
                "zero-delta Goal-anchor latch changes "
                "arm joints beyond safety limit: "
                + details
            )

        self._latched = True

        return float(
            max_shift
        )

    def plan(
        self,
        state: FixedAnchorXYZCommandState,
        current_observation: Mapping[
            str,
            object,
        ],
        *,
        max_model_xy_error_mm: float | None = None,
        retreat_from_z_mm: float | None = None,
    ) -> FixedAnchorPlan:
        """Plan one anchor-relative cumulative XYZ joint target.

        Normal press/alignment planning uses the configured model-Z
        magnitude gate.

        Safety retreat is different: when ``retreat_from_z_mm`` is supplied,
        the fixed 1 mm model-Z magnitude gate is not used. Instead, both the
        requested cumulative Z target and the FK-predicted target must move
        monotonically upward toward the fixed-anchor Z=0 level.

        This is a planner-model safety check only. It is not a claim that
        commanded millimetres equal physical TCP displacement.
        """

        if not self._latched:
            raise RuntimeError(
                "Goal anchor must be latched before planning"
            )

        # Identity, not merely value equality, is deliberate.  A press attempt
        # owns one anchor object for its entire lifetime.
        if state.anchor is not self.anchor:
            raise RuntimeError(
                "command state belongs to a different Goal anchor"
            )

        ordered = (
            ordered_joint_observation(
                current_observation,
                self.motor_names,
            )
        )

        expected = np.asarray(
            state.xyz_mm,
            dtype=np.float64,
        )

        command_norm = float(
            np.linalg.norm(expected)
        )

        if (
            command_norm
            > self.config.max_command_norm_mm
            + 1e-12
        ):
            raise RuntimeError(
                "fixed-anchor cumulative XYZ command "
                "exceeds planner safety bound: "
                f"{command_norm:.3f} > "
                f"{self.config.max_command_norm_mm:.3f} mm"
            )

        delta_action = (
            make_cartesian_delta_action(
                float(expected[0]),
                float(expected[1]),
                delta_z_mm=float(
                    expected[2]
                ),
                max_delta_norm_mm=(
                    self.config.max_command_norm_mm
                ),
            )
        )

        joint_action = self.pipeline(
            (
                delta_action,
                ordered,
            )
        )

        predicted_xyz = (
            end_effector_xyz_mm(
                self.validation_kinematics,
                joint_action,
                self.motor_names,
            )
        )

        predicted_delta = (
            predicted_xyz
            - self.anchor_xyz_mm
        )

        model_error = (
            predicted_delta
            - expected
        )

        model_xy_error = float(
            np.linalg.norm(
                model_error[:2]
            )
        )

        model_z_error = abs(
            float(model_error[2])
        )

        xy_error_limit = (
            self.config.max_model_xy_error_mm
            if max_model_xy_error_mm is None
            else float(max_model_xy_error_mm)
        )

        if (
            not math.isfinite(xy_error_limit)
            or xy_error_limit <= 0.0
        ):
            raise ValueError(
                "max_model_xy_error_mm override must be finite and positive"
            )

        retreat_from_z = None

        if retreat_from_z_mm is not None:
            retreat_from_z = float(
                retreat_from_z_mm
            )

            if not math.isfinite(
                retreat_from_z
            ):
                raise ValueError(
                    "retreat_from_z_mm must be finite"
                )

            # Keyboard safety retreat is defined only from a non-positive
            # fixed-anchor Z state back toward Z=0.
            if retreat_from_z > 1e-12:
                raise ValueError(
                    "safety retreat must start at or below "
                    "fixed-anchor Z=0"
                )

            # The command-space target itself must advance upward toward zero.
            if expected[2] <= retreat_from_z + 1e-12:
                raise RuntimeError(
                    "safety retreat command does not move cumulative "
                    "Z upward toward zero"
                )

            if expected[2] > 1e-12:
                raise RuntimeError(
                    "safety retreat command must not request "
                    "cumulative Z above zero"
                )

        if (
            retreat_from_z is None
            and model_xy_error
            > xy_error_limit
        ):
            raise RuntimeError(
                "planned joint target fails XY FK "
                "sanity check: "
                f"{model_xy_error:.3f} mm > "
                f"{xy_error_limit:.3f} mm"
            )

        if retreat_from_z is not None:
            # RELEASE / RETRACT are escape motions, not precision
            # Cartesian positioning tasks. Model XY residual magnitude
            # is therefore diagnostic only here.
            #
            # Safety is still fail-closed: the PLANNED FK target must
            # remain inside the command state's fixed-anchor XY envelope.
            predicted_xy_norm = float(
                np.linalg.norm(
                    predicted_delta[:2]
                )
            )

            if (
                predicted_xy_norm
                > float(state.max_xy_norm_mm)
                + 1e-9
            ):
                raise RuntimeError(
                    "safety retreat planned FK XY target exceeds "
                    "fixed-anchor XY envelope: "
                    f"{predicted_xy_norm:.3f} mm > "
                    f"{float(state.max_xy_norm_mm):.3f} mm"
                )

        if retreat_from_z is None:
            # Normal ALIGN / XY / downward PRESS planning keeps the original
            # strict model-consistency gate.
            if (
                model_z_error
                > self.config.max_model_z_error_mm
            ):
                raise RuntimeError(
                    "planned joint target fails Z FK "
                    "sanity check: "
                    f"{model_z_error:.3f} mm > "
                    f"{self.config.max_model_z_error_mm:.3f} mm"
                )

            # Direction sanity only. This is not a measured-motion
            # requirement.
            if (
                expected[2] < 0.0
                and predicted_delta[2] >= -1e-6
            ):
                raise RuntimeError(
                    "downward cumulative Z command does "
                    "not produce a downward planned FK target"
                )

        else:
            # RELEASE / RETRACT are safety escape motions, not precision
            # Cartesian targeting. Do not reject them because an arbitrary
            # model-error magnitude crossed 1 mm.
            #
            # Still fail closed if the planner's own FK says the new target
            # would not move upward relative to the latest actually-sent
            # command-space Z state.
            if (
                predicted_delta[2]
                <= retreat_from_z + 1e-6
            ):
                raise RuntimeError(
                    "safety retreat planned FK target does not "
                    "move upward toward fixed-anchor Z=0: "
                    f"from={retreat_from_z:+.3f} mm, "
                    f"predicted={predicted_delta[2]:+.3f} mm"
                )

        relative_deltas: list[
            tuple[str, float]
        ] = []

        for name in self.motor_names:
            if name == "gripper":
                continue

            key = f"{name}.pos"

            if key not in joint_action:
                raise ValueError(
                    f"planned action is missing {key}"
                )

            delta_deg = abs(
                float(joint_action[key])
                - float(ordered[key])
            )

            relative_deltas.append(
                (name, delta_deg)
            )

        violations = [
            (name, delta)
            for name, delta
            in relative_deltas
            if (
                delta
                > self.config.max_relative_target_deg
            )
        ]

        if violations:
            detail = ", ".join(
                f"{name}={delta:.2f}deg"
                for name, delta
                in violations
            )

            raise RuntimeError(
                "planned joint target exceeds "
                "relative safety gate: "
                + detail
            )

        max_relative = max(
            (
                delta
                for _, delta
                in relative_deltas
            ),
            default=0.0,
        )

        return FixedAnchorPlan(
            joint_action=dict(
                joint_action
            ),
            requested_xyz_mm=(
                float(expected[0]),
                float(expected[1]),
                float(expected[2]),
            ),
            predicted_xyz_mm=(
                float(predicted_xyz[0]),
                float(predicted_xyz[1]),
                float(predicted_xyz[2]),
            ),
            predicted_delta_mm=(
                float(predicted_delta[0]),
                float(predicted_delta[1]),
                float(predicted_delta[2]),
            ),
            model_error_mm=(
                float(model_error[0]),
                float(model_error[1]),
                float(model_error[2]),
            ),
            model_xy_error_mm=(
                model_xy_error
            ),
            model_z_error_mm=(
                model_z_error
            ),
            max_relative_joint_delta_deg=float(
                max_relative
            ),
        )

    @staticmethod
    def validate_sent_action(
        plan: FixedAnchorPlan,
        sent_action: Mapping[str, object],
    ) -> None:
        """Fail if send_action clipped an arm joint target."""

        clipped: list[
            tuple[
                str,
                float,
                float,
            ]
        ] = []

        for key, requested in (
            plan.joint_action.items()
        ):
            if (
                not key.endswith(".pos")
                or key == "gripper.pos"
                or key not in sent_action
            ):
                continue

            sent = float(
                sent_action[key]
            )

            requested_value = float(
                requested
            )

            if (
                abs(
                    sent
                    - requested_value
                )
                > 1e-6
            ):
                clipped.append(
                    (
                        key,
                        requested_value,
                        sent,
                    )
                )

        if clipped:
            details = ", ".join(
                (
                    f"{key}: "
                    f"requested={requested:.3f}, "
                    f"sent={sent:.3f}"
                )
                for key, requested, sent
                in clipped
            )

            raise RuntimeError(
                "joint target was clipped after "
                "send_action: "
                + details
            )
