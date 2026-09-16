from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)


SO101_ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)

SO101_MOTOR_NAMES = (
    *SO101_ARM_JOINTS,
    "gripper",
)

MOTION_FRAME = "base_link_xy"
MOTION_UNIT = "mm"

# Project-facing Cartesian corrections use millimetres.
# LeRobot kinematics use metres.
MM_TO_M = 0.001

# Task-level first-line safety bound.
# LeRobot EEBoundsAndSafety rate-limits against its previous command,
# so its very first command has no previous EE target to compare with.
DEFAULT_MAX_DELTA_NORM_MM = 20.0


def make_cartesian_delta_action(
    delta_x_mm: float,
    delta_y_mm: float,
    *,
    delta_z_mm: float = 0.0,
    enabled: bool = True,
    max_delta_norm_mm: float = DEFAULT_MAX_DELTA_NORM_MM,
) -> RobotAction:
    """Build the delta-action contract consumed by EEReferenceAndDelta.

    Translation is expressed in millimetres in the SO-101 base frame.
    Orientation delta is zero; LeRobot's SO-101 IK treats orientation as
    a soft constraint while prioritising Cartesian position.
    """

    values = (
        float(delta_x_mm),
        float(delta_y_mm),
        float(delta_z_mm),
    )

    if not all(math.isfinite(v) for v in values):
        raise ValueError("Cartesian delta must contain only finite values")

    max_delta_norm_mm = float(max_delta_norm_mm)

    if (
        not math.isfinite(max_delta_norm_mm)
        or max_delta_norm_mm <= 0.0
    ):
        raise ValueError(
            "max_delta_norm_mm must be finite and positive"
        )

    delta_norm_mm = math.sqrt(
        sum(v * v for v in values)
    )

    if delta_norm_mm > max_delta_norm_mm + 1e-12:
        raise ValueError(
            "Cartesian delta exceeds max_delta_norm_mm: "
            f"{delta_norm_mm:.6g} > {max_delta_norm_mm:.6g}"
        )

    return {
        "enabled": bool(enabled),
        "target_x": values[0],
        "target_y": values[1],
        "target_z": values[2],
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        # Visual servo does not operate the gripper.
        # Zero velocity preserves the measured gripper position.
        "gripper_vel": 0.0,
    }


def build_official_cartesian_pipeline(
    urdf_path: str | Path,
    *,
    end_effector_bounds: dict,
    motor_names: Sequence[str] = SO101_MOTOR_NAMES,
    max_ee_step_m: float = 0.02,
    orientation_weight: float = 0.01,
    raise_on_jump: bool = True,
) -> RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation],
    RobotAction,
]:
    """Build the LeRobot-native Cartesian-delta -> joint-target pipeline.

    so101_typing owns:
      - millimetre-facing task semantics
      - visual-servo correction
      - task-level safety parameters

    LeRobot owns:
      - FK / reference pose
      - Cartesian workspace safety
      - IK
      - joint-target generation
    """

    urdf_path = Path(urdf_path)

    if not urdf_path.is_file():
        raise FileNotFoundError(
            f"SO-101 URDF not found: {urdf_path}"
        )

    max_ee_step_m = float(max_ee_step_m)
    orientation_weight = float(orientation_weight)

    if (
        not math.isfinite(max_ee_step_m)
        or max_ee_step_m <= 0.0
    ):
        raise ValueError(
            "max_ee_step_m must be finite and positive"
        )

    if (
        not math.isfinite(orientation_weight)
        or orientation_weight < 0.0
    ):
        raise ValueError(
            "orientation_weight must be finite and non-negative"
        )

    names = list(motor_names)

    kinematics = RobotKinematics(
        urdf_path=str(urdf_path),
        target_frame_name="gripper_frame_link",
        joint_names=names,
    )

    return RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation],
        RobotAction,
    ](
        steps=[
            # Each visual-servo correction is relative to the CURRENT
            # measured robot pose. Do not accumulate an idealised pose.
            EEReferenceAndDelta(
                kinematics=kinematics,
                end_effector_step_sizes={
                    "x": MM_TO_M,
                    "y": MM_TO_M,
                    "z": MM_TO_M,
                },
                motor_names=names,
                use_latched_reference=False,
            ),

            # This is a safety/rate limit, not a positioning-accuracy claim.
            EEBoundsAndSafety(
                end_effector_bounds=end_effector_bounds,
                max_ee_step_m=max_ee_step_m,
                raise_on_jump=raise_on_jump,
            ),

            # gripper_vel=0 preserves the current measured gripper position.
            GripperVelocityToJoint(
                speed_factor=1.0,
            ),

            # Official SO-101 IK.
            # For closed-loop visual servo, seed IK from measured joints.
            InverseKinematicsEEToJoints(
                kinematics=kinematics,
                motor_names=names,
                initial_guess_current_joints=True,
                orientation_weight=orientation_weight,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
