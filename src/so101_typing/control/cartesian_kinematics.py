from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


SO101_ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)

MOTION_FRAME = "base_link_xy"
MOTION_UNIT = "mm"


class KinematicsBackend(Protocol):
    def forward_kinematics(
        self,
        joint_pos_deg: np.ndarray,
    ) -> np.ndarray: ...

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
    ) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class CartesianIKSafetyConfig:
    max_request_norm_mm: float = 1.0
    max_position_error_mm: float = 0.05
    max_unintended_z_mm: float = 0.05
    max_joint_step_deg: float = 2.0
    min_joint_margin_deg: float = 10.0
    max_orientation_change_deg: float = 0.10
    position_weight: float = 1.0
    orientation_weight: float = 0.01

    def __post_init__(self) -> None:
        positive = (
            "max_request_norm_mm",
            "max_position_error_mm",
            "max_unintended_z_mm",
            "max_joint_step_deg",
            "min_joint_margin_deg",
            "max_orientation_change_deg",
            "position_weight",
        )

        for name in positive:
            value = float(getattr(self, name))

            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{name} must be finite and positive"
                )

            object.__setattr__(self, name, value)

        orientation_weight = float(
            self.orientation_weight
        )

        if (
            not math.isfinite(orientation_weight)
            or orientation_weight < 0.0
        ):
            raise ValueError(
                "orientation_weight must be finite and non-negative"
            )

        object.__setattr__(
            self,
            "orientation_weight",
            orientation_weight,
        )


@dataclass(frozen=True, slots=True)
class CartesianIKCandidate:
    accepted: bool
    rejection_reasons: tuple[str, ...]

    motion_frame: str
    motion_unit: str

    requested_xy_mm: tuple[float, float]

    achieved_xyz_mm: tuple[float, float, float]
    position_error_xyz_mm: tuple[float, float, float]
    position_error_norm_mm: float

    orientation_change_deg: float

    current_joint_pos_deg: tuple[float, ...]
    candidate_joint_pos_deg: tuple[float, ...]
    joint_delta_deg: tuple[float, ...]

    max_joint_step_deg: float

    current_min_joint_margin_deg: float
    candidate_min_joint_margin_deg: float


def load_revolute_joint_limits_deg(
    urdf_path: str | Path,
    joint_names: Sequence[str],
) -> dict[str, tuple[float, float]]:

    root = ET.parse(urdf_path).getroot()

    wanted = tuple(joint_names)

    limits: dict[str, tuple[float, float]] = {}

    for joint in root.findall("joint"):
        name = joint.attrib.get("name")

        if name not in wanted:
            continue

        if joint.attrib.get("type") not in {
            "revolute",
            "continuous",
        }:
            raise ValueError(
                f"{name} is not a revolute/continuous joint"
            )

        limit = joint.find("limit")

        if (
            limit is None
            or "lower" not in limit.attrib
            or "upper" not in limit.attrib
        ):
            raise ValueError(
                f"{name} is missing finite lower/upper limits"
            )

        limits[name] = (
            math.degrees(float(limit.attrib["lower"])),
            math.degrees(float(limit.attrib["upper"])),
        )

    missing = [
        name
        for name in wanted
        if name not in limits
    ]

    if missing:
        raise ValueError(
            f"URDF is missing joint limits for: {missing}"
        )

    return limits


def _as_finite_vector(
    values: Sequence[float] | np.ndarray,
    *,
    length: int,
    name: str,
) -> np.ndarray:

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.shape != (length,):
        raise ValueError(
            f"{name} must have shape ({length},)"
        )

    if not np.isfinite(array).all():
        raise ValueError(
            f"{name} must contain only finite values"
        )

    return array


def _joint_margin_deg(
    q_deg: np.ndarray,
    joint_names: Sequence[str],
    limits_deg: dict[str, tuple[float, float]],
) -> tuple[float, bool]:

    margins: list[float] = []
    inside = True

    for name, q in zip(
        joint_names,
        q_deg,
        strict=True,
    ):
        lower, upper = limits_deg[name]

        inside &= (
            lower
            <= float(q)
            <= upper
        )

        margins.append(
            min(
                float(q) - lower,
                upper - float(q),
            )
        )

    return min(margins), inside


def _rotation_change_deg(
    reference: np.ndarray,
    actual: np.ndarray,
) -> float:

    delta = reference.T @ actual

    cosine = np.clip(
        (np.trace(delta) - 1.0) / 2.0,
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(cosine)
        )
    )


class SafeCartesianIK:
    """Generate and audit an XY IK candidate without commanding hardware."""

    def __init__(
        self,
        kinematics: KinematicsBackend,
        *,
        joint_names: Sequence[str],
        joint_limits_deg: dict[
            str,
            tuple[float, float],
        ],
        config: CartesianIKSafetyConfig | None = None,
    ) -> None:

        self.kinematics = kinematics

        self.joint_names = tuple(
            joint_names
        )

        if not self.joint_names:
            raise ValueError(
                "joint_names must not be empty"
            )

        self.joint_limits_deg = dict(
            joint_limits_deg
        )

        missing = [
            name
            for name in self.joint_names
            if name not in self.joint_limits_deg
        ]

        if missing:
            raise ValueError(
                f"joint_limits_deg is missing: {missing}"
            )

        self.config = (
            config
            or CartesianIKSafetyConfig()
        )

    @classmethod
    def from_urdf(
        cls,
        urdf_path: str | Path,
        *,
        target_frame_name: str = "gripper_frame_link",
        joint_names: Sequence[str] = SO101_ARM_JOINTS,
        config: CartesianIKSafetyConfig | None = None,
    ) -> SafeCartesianIK:

        # Lazy import:
        # ordinary project imports remain independent
        # of the optional Placo runtime.
        from lerobot.model.kinematics import RobotKinematics

        names = tuple(
            joint_names
        )

        limits = load_revolute_joint_limits_deg(
            urdf_path,
            names,
        )

        kinematics = RobotKinematics(
            urdf_path=str(urdf_path),
            target_frame_name=target_frame_name,
            joint_names=list(names),
        )

        return cls(
            kinematics,
            joint_names=names,
            joint_limits_deg=limits,
            config=config,
        )

    def candidate_xy(
        self,
        current_joint_pos_deg:
            Sequence[float] | np.ndarray,
        delta_xy_mm:
            Sequence[float] | np.ndarray,
    ) -> CartesianIKCandidate:

        q0 = _as_finite_vector(
            current_joint_pos_deg,
            length=len(self.joint_names),
            name="current_joint_pos_deg",
        )

        delta_xy = _as_finite_vector(
            delta_xy_mm,
            length=2,
            name="delta_xy_mm",
        )

        request_norm = float(
            np.linalg.norm(delta_xy)
        )

        if (
            request_norm
            > self.config.max_request_norm_mm
            + 1e-12
        ):
            raise ValueError(
                "requested XY correction exceeds "
                "max_request_norm_mm: "
                f"{request_norm:.6g} > "
                f"{self.config.max_request_norm_mm:.6g}"
            )

        current_margin, current_inside = (
            _joint_margin_deg(
                q0,
                self.joint_names,
                self.joint_limits_deg,
            )
        )

        # Fail fast before ANY FK/IK work.
        # An invalid or insufficient-margin current pose
        # must never be handed to the IK solver for repair.
        if not current_inside:
            raise ValueError(
                "current_joint_pos_deg is outside URDF limits"
            )

        if (
            current_margin
            < self.config.min_joint_margin_deg
        ):
            raise ValueError(
                "current joint margin is below "
                "min_joint_margin_deg: "
                f"{current_margin:.6g} < "
                f"{self.config.min_joint_margin_deg:.6g}"
            )

        # CRITICAL:
        # LeRobot 0.6.1 inverse_kinematics()
        # does not update robot kinematics after
        # setting current_joint_pos and before solve().
        #
        # FK here is therefore mandatory cold-start
        # synchronization before every IK solve.
        T0 = np.asarray(
            self.kinematics.forward_kinematics(q0),
            dtype=np.float64,
        )

        if (
            T0.shape != (4, 4)
            or not np.isfinite(T0).all()
        ):
            raise ValueError(
                "forward_kinematics returned "
                "an invalid 4x4 pose"
            )

        target = T0.copy()

        target[0, 3] += (
            delta_xy[0] / 1000.0
        )

        target[1, 3] += (
            delta_xy[1] / 1000.0
        )

        # target Z and target orientation are unchanged.

        q1 = np.asarray(
            self.kinematics.inverse_kinematics(
                current_joint_pos=q0,
                desired_ee_pose=target,
                position_weight=(
                    self.config.position_weight
                ),
                orientation_weight=(
                    self.config.orientation_weight
                ),
            ),
            dtype=np.float64,
        )

        if (
            q1.shape != q0.shape
            or not np.isfinite(q1).all()
        ):
            raise ValueError(
                "inverse_kinematics returned "
                "an invalid joint vector"
            )

        # Mandatory FK back-projection.
        # Never trust the requested Cartesian
        # displacement without measuring what
        # the IK candidate actually achieves.
        T1 = np.asarray(
            self.kinematics.forward_kinematics(q1),
            dtype=np.float64,
        )

        if (
            T1.shape != (4, 4)
            or not np.isfinite(T1).all()
        ):
            raise ValueError(
                "candidate forward_kinematics "
                "returned an invalid 4x4 pose"
            )

        achieved_xyz_mm = (
            T1[:3, 3]
            - T0[:3, 3]
        ) * 1000.0

        desired_xyz_mm = np.array(
            [
                delta_xy[0],
                delta_xy[1],
                0.0,
            ],
            dtype=np.float64,
        )

        error_xyz_mm = (
            achieved_xyz_mm
            - desired_xyz_mm
        )

        error_norm_mm = float(
            np.linalg.norm(
                error_xyz_mm
            )
        )

        orientation_change_deg = (
            _rotation_change_deg(
                T0[:3, :3],
                T1[:3, :3],
            )
        )

        dq = q1 - q0

        max_joint_step_deg = float(
            np.max(
                np.abs(dq)
            )
        )

        candidate_margin, candidate_inside = (
            _joint_margin_deg(
                q1,
                self.joint_names,
                self.joint_limits_deg,
            )
        )

        reasons: list[str] = []

        if not candidate_inside:
            reasons.append(
                "candidate_joint_outside_urdf"
            )

        elif (
            candidate_margin
            < self.config.min_joint_margin_deg
        ):
            reasons.append(
                "candidate_joint_margin"
            )

        if (
            max_joint_step_deg
            > self.config.max_joint_step_deg
        ):
            reasons.append(
                "joint_step"
            )

        if (
            error_norm_mm
            > self.config.max_position_error_mm
        ):
            reasons.append(
                "position_error"
            )

        if (
            abs(
                float(
                    achieved_xyz_mm[2]
                )
            )
            > self.config.max_unintended_z_mm
        ):
            reasons.append(
                "unintended_z"
            )

        if (
            orientation_change_deg
            > self.config.max_orientation_change_deg
        ):
            reasons.append(
                "orientation_change"
            )

        return CartesianIKCandidate(
            accepted=not reasons,
            rejection_reasons=tuple(reasons),

            motion_frame=MOTION_FRAME,
            motion_unit=MOTION_UNIT,

            requested_xy_mm=(
                float(delta_xy[0]),
                float(delta_xy[1]),
            ),

            achieved_xyz_mm=tuple(
                float(value)
                for value
                in achieved_xyz_mm
            ),

            position_error_xyz_mm=tuple(
                float(value)
                for value
                in error_xyz_mm
            ),

            position_error_norm_mm=(
                error_norm_mm
            ),

            orientation_change_deg=(
                orientation_change_deg
            ),

            current_joint_pos_deg=tuple(
                float(value)
                for value
                in q0
            ),

            candidate_joint_pos_deg=tuple(
                float(value)
                for value
                in q1
            ),

            joint_delta_deg=tuple(
                float(value)
                for value
                in dq
            ),

            max_joint_step_deg=(
                max_joint_step_deg
            ),

            current_min_joint_margin_deg=(
                float(current_margin)
            ),

            candidate_min_joint_margin_deg=(
                float(candidate_margin)
            ),
        )
