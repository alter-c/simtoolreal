from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import yourdfpy
from scipy.spatial.transform import Rotation as R


def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0)[..., None]
    b = np.cross(q_vec, v, axis=-1) * q_w[..., None] * 2.0
    c = (
        q_vec
        * (q_vec.reshape(shape[0], 1, 3) @ v.reshape(shape[0], 3, 1))[..., 0]
        * 2.0
    )
    return a + b + c


def tensor_clamp(t, min_t, max_t):
    return np.maximum(np.minimum(t, max_t), min_t)


NUM_ARM_DOFS = 7
NUM_HAND_DOFS = 11
NUM_HAND_ARM_DOFS = NUM_ARM_DOFS + NUM_HAND_DOFS  # 18
NUM_FINGERTIPS = 5

JOINT_NAMES_ISAACGYM = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "lh_thumb_cmc_yaw",
    "lh_thumb_cmc_pitch",
    "lh_thumb_ip",
    "lh_index_mcp_pitch",
    "lh_index_dip",
    "lh_middle_mcp_pitch",
    "lh_middle_dip",
    "lh_ring_mcp_pitch",
    "lh_ring_dip",
    "lh_pinky_mcp_pitch",
    "lh_pinky_dip",
]

PALM_LINK = "left_wrist_yaw_link"

FINGERTIP_LINKS = [
    "lh_index_distal",
    "lh_middle_distal",
    "lh_ring_distal",
    "lh_thumb_distal",
    "lh_pinky_distal",
]

# NOTE: IsaacGym sorts joints alphabetically at the same depth.
# The actual joint order must be validated at runtime after asset loading.
# The order above is our best guess based on alphabetical sorting of revolute
# joints in the URDF. If IsaacGym reports a different order, this list will
# be overwritten via the validate_and_update_joint_names() function.
# DO NOT assume this list is correct without runtime verification.

assert len(JOINT_NAMES_ISAACGYM) == NUM_HAND_ARM_DOFS, (
    f"len(JOINT_NAMES_ISAACGYM): {len(JOINT_NAMES_ISAACGYM)}, "
    f"expected: {NUM_HAND_ARM_DOFS}"
)

DES_LEFT_ARM_POS = np.array(
    [0.0, 0.5, 0.0, -1.0, 0.0, 0.0, 0.0],
    dtype=np.float32,
)

DES_LEFT_HAND_POS = np.zeros(NUM_HAND_DOFS, dtype=np.float32)

Q_LOWER_LIMITS_np = np.array(
    [
        -3.0892,
        -1.5882,
        -2.618,
        -1.0472,
        -1.972222054,
        -1.614429558,
        -1.614429558,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)

Q_UPPER_LIMITS_np = np.array(
    [
        2.6704,
        2.2515,
        2.618,
        2.0944,
        1.972222054,
        1.614429558,
        1.614429558,
        1.3,
        0.58,
        1.3282,
        1.6,
        1.424,
        1.6,
        1.424,
        1.6,
        1.424,
        1.6,
        1.424,
    ],
    dtype=np.float32,
)

assert Q_LOWER_LIMITS_np.shape == (NUM_HAND_ARM_DOFS,), (
    f"Q_LOWER_LIMITS_np.shape: {Q_LOWER_LIMITS_np.shape}, "
    f"expected: ({NUM_HAND_ARM_DOFS},)"
)
assert Q_UPPER_LIMITS_np.shape == (NUM_HAND_ARM_DOFS,), (
    f"Q_UPPER_LIMITS_np.shape: {Q_UPPER_LIMITS_np.shape}, "
    f"expected: ({NUM_HAND_ARM_DOFS},)"
)

Q_LOWER_LIMITS_restricted_np = Q_LOWER_LIMITS_np.copy()
Q_LOWER_LIMITS_restricted_np[:NUM_ARM_DOFS] += np.deg2rad(10.0)

Q_UPPER_LIMITS_restricted_np = Q_UPPER_LIMITS_np.copy()
Q_UPPER_LIMITS_restricted_np[:NUM_ARM_DOFS] -= np.deg2rad(10.0)
assert Q_LOWER_LIMITS_restricted_np.shape == (NUM_HAND_ARM_DOFS,), (
    f"Q_LOWER_LIMITS_restricted_np.shape: {Q_LOWER_LIMITS_restricted_np.shape}, "
    f"expected: ({NUM_HAND_ARM_DOFS},)"
)
assert Q_UPPER_LIMITS_restricted_np.shape == (NUM_HAND_ARM_DOFS,), (
    f"Q_UPPER_LIMITS_restricted_np.shape: {Q_UPPER_LIMITS_restricted_np.shape}, "
    f"expected: ({NUM_HAND_ARM_DOFS},)"
)

OBS_NAME_TO_NAMES = {
    "joint_pos": [f"{name}_q" for name in JOINT_NAMES_ISAACGYM],
    "joint_vel": [f"{name}_qd" for name in JOINT_NAMES_ISAACGYM],
    "prev_action_targets": [
        f"{name}_prev_action_target" for name in JOINT_NAMES_ISAACGYM
    ],
    "palm_pos": [f"palm_center_pos_{x}" for x in "xyz"],
    "palm_rot": [f"palm_rot_{x}" for x in "xyzw"],
    "object_rot": [f"object_rot_{x}" for x in "xyzw"],
    "keypoints_rel_palm": [
        f"keypoints_rel_palm_{i}_{x}" for i in range(4) for x in "xyz"
    ],
    "keypoints_rel_goal": [
        f"keypoints_rel_goal_{i}_{x}" for i in range(4) for x in "xyz"
    ],
    "fingertip_pos_rel_palm": [
        f"fingertip_rel_pos_{finger}_{x}"
        for finger in ["index", "middle", "ring", "thumb", "pinky"]
        for x in "xyz"
    ],
    "object_scales": [f"object_scales_{x}" for x in "xyz"],
}

OBS_NAMES = sum(OBS_NAME_TO_NAMES.values(), [])

N_OBS = (
    NUM_HAND_ARM_DOFS  # joint_pos
    + NUM_HAND_ARM_DOFS  # joint_vel
    + NUM_HAND_ARM_DOFS  # prev_action_targets
    + 3  # palm_pos
    + 4  # palm_rot
    + 4  # object_rot
    + 3 * NUM_FINGERTIPS  # fingertip_pos_rel_palm
    + 3 * 4  # keypoints_rel_palm
    + 3 * 4  # keypoints_rel_goal
    + 3  # object_scales
)
assert len(OBS_NAMES) == N_OBS, f"len(OBS_NAMES): {len(OBS_NAMES)}, expected: {N_OBS}"

T_W_R_np = np.eye(4)
T_W_R_np[:3, 3] = np.array([0.0, 0.8, 0.0])

PALM_OFFSET_np = np.array([0.0, 0.0, 0.13], dtype=np.float32)

FINGERTIP_OFFSETS_np = np.array(
    [
        [0.02, 0.002, 0],
        [0.02, 0.002, 0],
        [0.02, 0.002, 0],
        [0.02, 0.002, 0],
        [0.02, 0.002, 0],
    ],
    dtype=np.float32,
)

OBJECT_KEYPOINT_OFFSETS_np = np.array(
    [[1, 1, 1], [1, 1, -1], [-1, -1, 1], [-1, -1, -1]]
)


def validate_urdf(urdf: yourdfpy.URDF) -> None:
    runtime_joint_names = list(urdf.actuated_joint_names)
    num_actuated = len(runtime_joint_names)

    if num_actuated != NUM_HAND_ARM_DOFS:
        print(
            f"[URDF VALIDATION ERROR] DOF count: {num_actuated}, "
            f"expected: {NUM_HAND_ARM_DOFS}",
            file=sys.stderr,
        )

    expected_set = set(JOINT_NAMES_ISAACGYM)
    runtime_set = set(runtime_joint_names)

    missing = expected_set - runtime_set
    extra = runtime_set - expected_set

    if missing:
        print(
            f"[URDF VALIDATION WARNING] Joints in expected list but missing from "
            f"URDF actuated joints: {sorted(missing)}",
            file=sys.stderr,
        )
    if extra:
        print(
            f"[URDF VALIDATION WARNING] Extra actuated joints in URDF not in "
            f"expected list: {sorted(extra)}",
            file=sys.stderr,
        )

    if runtime_joint_names != JOINT_NAMES_ISAACGYM:
        print(
            f"[URDF VALIDATION WARNING] Joint name order differs from expected.\n"
            f"  Expected: {JOINT_NAMES_ISAACGYM}\n"
            f"  Runtime:  {runtime_joint_names}\n"
            f"  This may indicate IsaacGym sorted joints differently than anticipated.",
            file=sys.stderr,
        )

    for link_name in [PALM_LINK] + FINGERTIP_LINKS:
        if link_name not in urdf.link_map:
            print(
                f"[URDF VALIDATION ERROR] Required link '{link_name}' not found in "
                f"URDF. Available links: {sorted(urdf.link_map.keys())}",
                file=sys.stderr,
            )

    print(
        f"[URDF VALIDATION] DOF count: {num_actuated} (expected: {NUM_HAND_ARM_DOFS})"
    )
    print(f"[URDF VALIDATION] Palm link '{PALM_LINK}': ", end="")
    if PALM_LINK in urdf.link_map:
        print("found ✓")
    else:
        print("NOT FOUND ✗")
    print(f"[URDF VALIDATION] Fingertip links ({len(FINGERTIP_LINKS)}/5): ", end="")
    found_count = sum(1 for fl in FINGERTIP_LINKS if fl in urdf.link_map)
    print(f"{found_count}/5 found {'✓' if found_count == 5 else '✗'}")


def create_urdf_object(
    robot_name: Literal["g1_29dof_with_left_linkerhand"],
) -> yourdfpy.URDF:
    asset_root = Path(__file__).parent / "../../assets"
    assert asset_root.exists(), f"Asset root {asset_root} does not exist"
    if robot_name == "g1_29dof_with_left_linkerhand":
        urdf_path = (
            asset_root
            / "urdf/g1_description/g1_29dof_with_left_linkerhand_actuated.urdf"
        )
    else:
        raise ValueError(f"Invalid robot name: {robot_name}")
    assert urdf_path.exists(), f"URDF file {urdf_path} does not exist"
    urdf = yourdfpy.URDF.load(urdf_path)
    validate_urdf(urdf)
    return urdf


def compute_fk_dict(
    urdf: yourdfpy.URDF, q: np.ndarray, link_names: list[str]
) -> dict[str, np.ndarray]:
    N = q.shape[0]
    assert q.shape == (N, NUM_HAND_ARM_DOFS), (
        f"q.shape: {q.shape}, expected: (N, {NUM_HAND_ARM_DOFS})"
    )
    fk_dict = defaultdict(list)
    for i in range(N):
        urdf.update_cfg(q[i])
        for link_name in link_names:
            fk_dict[link_name].append(urdf.get_transform(frame_to=link_name))
    for link_name in link_names:
        fk_dict[link_name] = np.stack(fk_dict[link_name], axis=0)
        assert fk_dict[link_name].shape == (N, 4, 4), (
            f"fk_dict[link_name].shape: {fk_dict[link_name].shape}, expected: (N, 4, 4)"
        )
    return fk_dict


def compute_observation(
    q: np.ndarray,
    qd: np.ndarray,
    prev_action_targets: np.ndarray,
    object_pose: np.ndarray,
    goal_object_pose: np.ndarray,
    object_scales: np.ndarray,
    urdf: yourdfpy.URDF,
    obs_list: list[str],
) -> np.ndarray:
    N = q.shape[0]
    J = NUM_HAND_ARM_DOFS
    assert q.shape == (N, J), f"q.shape: {q.shape}, expected: (N, {J})"
    assert qd.shape == (N, J), f"qd.shape: {qd.shape}, expected: (N, {J})"
    assert prev_action_targets.shape == (N, J), (
        f"prev_action_targets.shape: {prev_action_targets.shape}, expected: (N, {J})"
    )
    q_lower_limits = Q_LOWER_LIMITS_np
    q_upper_limits = Q_UPPER_LIMITS_np
    assert q_lower_limits.shape == (J,), (
        f"q_lower_limits.shape: {q_lower_limits.shape}, expected: ({J},)"
    )
    assert q_upper_limits.shape == (J,), (
        f"q_upper_limits.shape: {q_upper_limits.shape}, expected: ({J},)"
    )
    assert object_pose.shape == (N, 7), (
        f"object_pose.shape: {object_pose.shape}, expected: (N, 7)"
    )
    assert goal_object_pose.shape == (N, 7), (
        f"goal_object_pose.shape: {goal_object_pose.shape}, expected: (N, 7)"
    )
    assert object_scales.shape == (N, 3), (
        f"object_scales.shape: {object_scales.shape}, expected: (N, 3)"
    )
    assert set(obs_list).issubset(set(OBS_NAME_TO_NAMES.keys())), (
        f"obs_list: {obs_list} is not a subset of OBS_NAME_TO_NAMES.keys(): "
        f"{OBS_NAME_TO_NAMES.keys()}"
    )

    q_unscaled = unscale(
        x=q,
        lower=q_lower_limits,
        upper=q_upper_limits,
    )

    actual_joint_names = list(urdf.actuated_joint_names)
    assert actual_joint_names == JOINT_NAMES_ISAACGYM, (
        f"Joint names from URDF ({actual_joint_names}) do not match "
        f"JOINT_NAMES_ISAACGYM ({JOINT_NAMES_ISAACGYM}). "
        f"This will cause FK to produce incorrect results."
    )

    LINK_NAMES = [PALM_LINK] + FINGERTIP_LINKS
    fk_dict = compute_fk_dict(urdf=urdf, q=q, link_names=LINK_NAMES)
    palm_center_pos, palm_rot = _compute_palm_center_pos_and_rot(fk_dict=fk_dict)
    fingertip_positions_with_offsets = _compute_fingertip_positions_with_offsets(
        fk_dict=fk_dict
    )
    fingertip_rel_pos = fingertip_positions_with_offsets - palm_center_pos[:, None]

    assert palm_center_pos.shape == (N, 3), (
        f"palm_center_pos.shape: {palm_center_pos.shape}, expected: (N, 3)"
    )
    assert fingertip_rel_pos.shape == (N, NUM_FINGERTIPS, 3), (
        f"fingertip_rel_pos.shape: {fingertip_rel_pos.shape}, "
        f"expected: (N, {NUM_FINGERTIPS}, 3)"
    )

    N_KEYPOINTS = 4
    object_keypoint_positions = _compute_keypoint_positions(
        pose=object_pose, scales=object_scales
    )
    goal_keypoint_positions = _compute_keypoint_positions(
        pose=goal_object_pose, scales=object_scales
    )
    keypoints_rel_palm = object_keypoint_positions - palm_center_pos[:, None]
    keypoints_rel_goal = object_keypoint_positions - goal_keypoint_positions

    assert keypoints_rel_palm.shape == (N, N_KEYPOINTS, 3), (
        f"keypoints_rel_palm.shape: {keypoints_rel_palm.shape}, "
        f"expected: (N, {N_KEYPOINTS}, 3)"
    )
    assert keypoints_rel_goal.shape == (N, N_KEYPOINTS, 3), (
        f"keypoints_rel_goal.shape: {keypoints_rel_goal.shape}, "
        f"expected: (N, {N_KEYPOINTS}, 3)"
    )

    object_rot = object_pose[:, 3:7]
    assert object_rot.shape == (N, 4), (
        f"object_rot.shape: {object_rot.shape}, expected: (N, 4)"
    )

    obs_dict = {
        "joint_pos": q_unscaled,
        "joint_vel": qd,
        "prev_action_targets": prev_action_targets,
        "palm_pos": palm_center_pos,
        "palm_rot": palm_rot,
        "object_rot": object_rot,
        "keypoints_rel_palm": keypoints_rel_palm.reshape(N, -1),
        "keypoints_rel_goal": keypoints_rel_goal.reshape(N, -1),
        "fingertip_pos_rel_palm": fingertip_rel_pos.reshape(N, -1),
        "object_scales": object_scales,
    }
    for k, v in obs_dict.items():
        assert v.ndim == 2, f"v.ndim: {v.ndim}, expected: 2 for key: {k}: {v.shape}"
        assert v.shape[0] == N, f"v.shape[0]: {v.shape[0]}, expected: {N} for key: {k}"
    for name, names in OBS_NAME_TO_NAMES.items():
        assert name in obs_dict, f"name: {name} not in obs_dict"
        assert obs_dict[name].shape[1] == len(names), (
            f"obs_dict[name].shape[1]: {obs_dict[name].shape[1]}, "
            f"expected: {len(names)} for name: {name}"
        )

    obs = np.concatenate(
        [obs_dict[key] for key in obs_list],
        axis=-1,
    )

    assert obs.shape == (N, N_OBS), f"obs.shape: {obs.shape}, expected: (N, {N_OBS})"
    return obs


def compute_joint_pos_targets(
    actions: np.ndarray,
    prev_targets: np.ndarray,
    hand_moving_average: float,
    arm_moving_average: float,
    hand_dof_speed_scale: float,
    dt: float,
) -> np.ndarray:
    N = actions.shape[0]
    J = NUM_HAND_ARM_DOFS
    assert actions.shape == (N, J), (
        f"actions.shape: {actions.shape}, expected: (N, {J})"
    )
    assert prev_targets.shape == (N, J), (
        f"prev_targets.shape: {prev_targets.shape}, expected: (N, {J})"
    )
    q_lower_limits = Q_LOWER_LIMITS_np
    q_upper_limits = Q_UPPER_LIMITS_np
    assert q_lower_limits.shape == (J,), (
        f"q_lower_limits.shape: {q_lower_limits.shape}, expected: ({J},)"
    )
    assert q_upper_limits.shape == (J,), (
        f"q_upper_limits.shape: {q_upper_limits.shape}, expected: ({J},)"
    )
    assert 0.0 <= hand_moving_average <= 1.0, (
        f"hand_moving_average: {hand_moving_average}, expected: (0.0, 1.0)"
    )
    assert 0.0 <= arm_moving_average <= 1.0, (
        f"arm_moving_average: {arm_moving_average}, expected: (0.0, 1.0)"
    )

    cur_targets = prev_targets.copy()

    # Hand joints: scale action to joint limits, then EMA
    cur_targets[:, NUM_ARM_DOFS:] = scale(
        actions[:, NUM_ARM_DOFS:],
        q_lower_limits[NUM_ARM_DOFS:],
        q_upper_limits[NUM_ARM_DOFS:],
    )
    cur_targets[:, NUM_ARM_DOFS:] = (
        hand_moving_average * cur_targets[:, NUM_ARM_DOFS:]
        + (1.0 - hand_moving_average) * prev_targets[:, NUM_ARM_DOFS:]
    )
    cur_targets[:, NUM_ARM_DOFS:] = tensor_clamp(
        cur_targets[:, NUM_ARM_DOFS:],
        q_lower_limits[NUM_ARM_DOFS:],
        q_upper_limits[NUM_ARM_DOFS:],
    )

    # Arm joints: delta position control, then EMA
    cur_targets[:, :NUM_ARM_DOFS] = (
        prev_targets[:, :NUM_ARM_DOFS]
        + hand_dof_speed_scale * dt * actions[:, :NUM_ARM_DOFS]
    )
    cur_targets[:, :NUM_ARM_DOFS] = tensor_clamp(
        cur_targets[:, :NUM_ARM_DOFS],
        q_lower_limits[:NUM_ARM_DOFS],
        q_upper_limits[:NUM_ARM_DOFS],
    )
    cur_targets[:, :NUM_ARM_DOFS] = (
        arm_moving_average * cur_targets[:, :NUM_ARM_DOFS]
        + (1.0 - arm_moving_average) * prev_targets[:, :NUM_ARM_DOFS]
    )
    return cur_targets


def _compute_palm_center_pos_and_rot(
    fk_dict: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    T_R_Ps = fk_dict[PALM_LINK]
    N = T_R_Ps.shape[0]
    T_W_Rs = T_W_R_np[None]

    T_W_Ps = T_W_Rs @ T_R_Ps

    palm_offset = PALM_OFFSET_np[None].repeat(N, axis=0)
    palm_pos = T_W_Ps[:, :3, 3]
    palm_rot = T_W_Ps[:, :3, :3]
    palm_quat_xyzw = matrix_to_quaternion_xyzw_scipy(palm_rot)

    palm_center_pos = palm_pos + quat_rotate(palm_quat_xyzw, palm_offset)
    assert palm_center_pos.shape == (N, 3), (
        f"palm_center_pos.shape: {palm_center_pos.shape}, expected: (N, 3)"
    )
    return palm_center_pos, palm_quat_xyzw


def _compute_fingertip_positions_with_offsets(
    fk_dict: dict[str, np.ndarray],
) -> np.ndarray:
    T_R_F_list = [fk_dict[name] for name in FINGERTIP_LINKS]
    T_R_Fs = np.stack(T_R_F_list, axis=1)
    N = T_R_Fs.shape[0]
    assert T_R_Fs.shape == (N, NUM_FINGERTIPS, 4, 4), (
        f"T_R_Fs.shape: {T_R_Fs.shape}, expected: (N, {NUM_FINGERTIPS}, 4, 4)"
    )

    T_W_Rs = T_W_R_np[None, None]

    T_W_Fs = T_W_Rs @ T_R_Fs
    fingertip_positions = T_W_Fs[:, :, :3, 3]
    fingertip_rots = T_W_Fs[:, :, :3, :3]
    fingertip_quat_xyzw = matrix_to_quaternion_xyzw_scipy(
        fingertip_rots.reshape(-1, 3, 3)
    ).reshape(N, NUM_FINGERTIPS, 4)

    fingertip_offsets = FINGERTIP_OFFSETS_np[None].repeat(N, axis=0)
    assert fingertip_offsets.shape == (N, NUM_FINGERTIPS, 3), (
        f"fingertip_offsets.shape: {fingertip_offsets.shape}, "
        f"expected: (N, {NUM_FINGERTIPS}, 3)"
    )
    fingertip_positions_with_offsets = np.zeros(
        (N, NUM_FINGERTIPS, 3), dtype=np.float32
    )
    for i in range(NUM_FINGERTIPS):
        fingertip_positions_with_offsets[:, i] = fingertip_positions[
            :, i
        ] + quat_rotate(fingertip_quat_xyzw[:, i], fingertip_offsets[:, i])
    return fingertip_positions_with_offsets


def _compute_keypoint_positions(
    pose: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    N = pose.shape[0]
    assert pose.shape == (N, 7), f"pose.shape: {pose.shape}, expected: (N, 7)"
    assert scales.shape == (N, 3), f"scales.shape: {scales.shape}, expected: (N, 3)"

    OBJECT_BASE_SIZE = 0.04
    KEYPOINT_SCALE = 1.5
    object_keypoint_offsets = (
        OBJECT_KEYPOINT_OFFSETS_np[None]
        * OBJECT_BASE_SIZE
        * KEYPOINT_SCALE
        / 2
        * scales[:, None]
    )
    N_KEYPOINTS = 4
    assert object_keypoint_offsets.shape == (N, N_KEYPOINTS, 3), (
        f"object_keypoint_offsets.shape: {object_keypoint_offsets.shape}, "
        f"expected: (N, {N_KEYPOINTS}, 3)"
    )

    pos = pose[:, :3]
    quat_xyzw = pose[:, 3:7]

    keypoint_positions = np.zeros((N, N_KEYPOINTS, 3), dtype=np.float32)
    for i in range(N_KEYPOINTS):
        keypoint_positions[:, i] = pos + quat_rotate(
            quat_xyzw, object_keypoint_offsets[:, i]
        )
    return keypoint_positions


def matrix_to_quaternion_xyzw_scipy(matrix: np.ndarray) -> np.ndarray:
    return R.from_matrix(matrix).as_quat()


"""
Frames and Transforms
====================
W = world, R = robot base, P = palm (left_wrist_yaw_link + offset),
    O = object, G = goal, F = fingertip

W != R because the robot base is offset from the world origin:
  T_W_R = eye(4) with translation (x=0, y=0.8, z=0)

Robot: G1 left arm (7 DOF) + LinkerHand O6 left (11 DOF) = 18 DOF total

Observation (N_OBS-dim):
  joint_pos (18), joint_vel (18), prev_action_targets (18),
  palm_pos (3), palm_rot (4), object_rot (4),
  keypoints_rel_palm (4x3=12), keypoints_rel_goal (4x3=12),
  fingertip_pos_rel_palm (5x3=15), object_scales (3)
  Total = 18+18+18+3+4+4+12+12+15+3 = 107

Action -> Joint Position Targets:
  Hand (joints 7:18): scale action to joint limits, then EMA with hand_moving_average
  Arm  (joints 0:7):  prev_targets + hand_dof_speed_scale * dt * action, then EMA with arm_moving_average
"""
