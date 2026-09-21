from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math


def _finite_float(value: object, *, name: str) -> float:
    result = float(value)

    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")

    return result


def stepped_z_targets_to_zero(
    current_z_mm: float,
    *,
    max_step_mm: float,
) -> tuple[float, ...]:
    """Return bounded cumulative-Z targets leading exactly to zero.

    This only plans command-space Z targets. XY and the immutable Goal
    anchor remain the caller's responsibility.
    """

    current = _finite_float(
        current_z_mm,
        name="current_z_mm",
    )

    step = _finite_float(
        max_step_mm,
        name="max_step_mm",
    )

    if step <= 0.0:
        raise ValueError(
            "max_step_mm must be positive"
        )

    if abs(current) <= 1e-12:
        return ()

    targets: list[float] = []

    while abs(current) > 1e-12:
        if current < 0.0:
            next_z = min(
                0.0,
                current + step,
            )
        else:
            next_z = max(
                0.0,
                current - step,
            )

        targets.append(
            float(next_z)
        )

        current = float(
            next_z
        )

    return tuple(
        targets
    )


@dataclass(frozen=True, slots=True)
class FixedGoalAnchor:
    """Immutable command-space anchor captured from existing Goal_Position.

    This object represents servo command preload at deterministic handoff.
    Present_Position must never be used to mutate or replace this anchor during
    one local press attempt.
    """

    motor_names: tuple[str, ...]
    joint_positions_deg: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.motor_names:
            raise ValueError("motor_names must not be empty")

        if len(set(self.motor_names)) != len(self.motor_names):
            raise ValueError("motor_names must be unique")

        if len(self.motor_names) != len(self.joint_positions_deg):
            raise ValueError(
                "motor_names and joint_positions_deg must have equal length"
            )

        if not all(
            math.isfinite(float(value))
            for value in self.joint_positions_deg
        ):
            raise ValueError(
                "anchor joint positions must all be finite"
            )

    @classmethod
    def from_goal_positions(
        cls,
        goal_positions: Mapping[str, object],
        motor_names: Sequence[str],
    ) -> FixedGoalAnchor:
        names = tuple(str(name) for name in motor_names)

        if not names:
            raise ValueError("motor_names must not be empty")

        values: list[float] = []

        for name in names:
            raw_key = name
            observation_key = f"{name}.pos"

            if raw_key in goal_positions:
                raw_value = goal_positions[raw_key]
            elif observation_key in goal_positions:
                raw_value = goal_positions[observation_key]
            else:
                raise ValueError(
                    f"Goal_Position mapping is missing motor {name!r}"
                )

            values.append(
                _finite_float(
                    raw_value,
                    name=f"Goal_Position[{name}]",
                )
            )

        return cls(
            motor_names=names,
            joint_positions_deg=tuple(values),
        )

    def as_observation(self) -> dict[str, float]:
        """Return the fixed anchor in the joint-observation contract."""

        return {
            f"{name}.pos": float(value)
            for name, value in zip(
                self.motor_names,
                self.joint_positions_deg,
                strict=True,
            )
        }

    def position_deg(self, motor_name: str) -> float:
        try:
            index = self.motor_names.index(motor_name)
        except ValueError as exc:
            raise KeyError(motor_name) from exc

        return float(
            self.joint_positions_deg[index]
        )


@dataclass(frozen=True, slots=True)
class FixedAnchorXYZCommandState:
    """Cumulative XYZ command expressed against one immutable Goal anchor."""

    anchor: FixedGoalAnchor

    x_mm: float = 0.0
    y_mm: float = 0.0
    z_mm: float = 0.0

    max_xy_norm_mm: float = 35.0
    max_xyz_norm_mm: float = 35.0

    def __post_init__(self) -> None:
        values = (
            self.x_mm,
            self.y_mm,
            self.z_mm,
        )

        if not all(
            math.isfinite(float(value))
            for value in values
        ):
            raise ValueError(
                "cumulative XYZ values must be finite"
            )

        max_xy = float(
            self.max_xy_norm_mm
        )
        max_xyz = float(
            self.max_xyz_norm_mm
        )

        if (
            not math.isfinite(max_xy)
            or max_xy <= 0.0
        ):
            raise ValueError(
                "max_xy_norm_mm must be finite and positive"
            )

        if (
            not math.isfinite(max_xyz)
            or max_xyz <= 0.0
        ):
            raise ValueError(
                "max_xyz_norm_mm must be finite and positive"
            )

        xy_norm = math.hypot(
            float(self.x_mm),
            float(self.y_mm),
        )

        xyz_norm = math.sqrt(
            float(self.x_mm) ** 2
            + float(self.y_mm) ** 2
            + float(self.z_mm) ** 2
        )

        if xy_norm > max_xy + 1e-12:
            raise ValueError(
                "cumulative XY command exceeds budget: "
                f"{xy_norm:.6g} > {max_xy:.6g} mm"
            )

        if xyz_norm > max_xyz + 1e-12:
            raise ValueError(
                "cumulative XYZ command exceeds budget: "
                f"{xyz_norm:.6g} > {max_xyz:.6g} mm"
            )

    @classmethod
    def at_anchor(
        cls,
        anchor: FixedGoalAnchor,
        *,
        max_xy_norm_mm: float = 35.0,
        max_xyz_norm_mm: float = 35.0,
    ) -> FixedAnchorXYZCommandState:
        return cls(
            anchor=anchor,
            max_xy_norm_mm=max_xy_norm_mm,
            max_xyz_norm_mm=max_xyz_norm_mm,
        )

    @property
    def xyz_mm(self) -> tuple[float, float, float]:
        return (
            float(self.x_mm),
            float(self.y_mm),
            float(self.z_mm),
        )

    @property
    def xy_mm(self) -> tuple[float, float]:
        return (
            float(self.x_mm),
            float(self.y_mm),
        )

    def with_xy_increment(
        self,
        delta_x_mm: float,
        delta_y_mm: float,
    ) -> FixedAnchorXYZCommandState:
        """Accumulate XY while preserving both Goal anchor and current Z."""

        return FixedAnchorXYZCommandState(
            anchor=self.anchor,
            x_mm=(
                float(self.x_mm)
                + _finite_float(
                    delta_x_mm,
                    name="delta_x_mm",
                )
            ),
            y_mm=(
                float(self.y_mm)
                + _finite_float(
                    delta_y_mm,
                    name="delta_y_mm",
                )
            ),
            z_mm=float(self.z_mm),
            max_xy_norm_mm=self.max_xy_norm_mm,
            max_xyz_norm_mm=self.max_xyz_norm_mm,
        )

    def with_xy_target(
        self,
        x_mm: float,
        y_mm: float,
    ) -> FixedAnchorXYZCommandState:
        """Replace cumulative XY target while preserving the same anchor/Z."""

        return FixedAnchorXYZCommandState(
            anchor=self.anchor,
            x_mm=_finite_float(
                x_mm,
                name="x_mm",
            ),
            y_mm=_finite_float(
                y_mm,
                name="y_mm",
            ),
            z_mm=float(self.z_mm),
            max_xy_norm_mm=self.max_xy_norm_mm,
            max_xyz_norm_mm=self.max_xyz_norm_mm,
        )

    def with_z_level(
        self,
        z_mm: float,
    ) -> FixedAnchorXYZCommandState:
        """Change cumulative Z while holding current cumulative XY fixed."""

        return FixedAnchorXYZCommandState(
            anchor=self.anchor,
            x_mm=float(self.x_mm),
            y_mm=float(self.y_mm),
            z_mm=_finite_float(
                z_mm,
                name="z_mm",
            ),
            max_xy_norm_mm=self.max_xy_norm_mm,
            max_xyz_norm_mm=self.max_xyz_norm_mm,
        )

    def retracted(self) -> FixedAnchorXYZCommandState:
        """Return to the handoff Z level while preserving cumulative XY."""

        return self.with_z_level(0.0)

@dataclass(slots=True)
class LatestSentXYZCommandState:
    """Runtime record of the latest command state actually sent to the robot.

    This is intentionally mutable while FixedAnchorXYZCommandState remains
    immutable. It exists so asynchronous SIDE events can recover the latest
    command-space XYZ even when an interrupt escapes from inside a nested WRIST
    alignment call before that call returns its local state to the outer loop.
    """

    state: FixedAnchorXYZCommandState

    def record_sent(
        self,
        state: FixedAnchorXYZCommandState,
    ) -> None:
        if state.anchor is not self.state.anchor:
            raise ValueError(
                "latest-sent command state must preserve the same FixedGoalAnchor"
            )

        self.state = state

