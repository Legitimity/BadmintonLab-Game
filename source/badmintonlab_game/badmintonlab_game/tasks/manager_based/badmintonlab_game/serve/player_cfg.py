"""Actor observations and initial state for one player (design doc P2).

This file is the single source used by both training and the submission. It may
import only Isaac Lab and ``badmintonlab_game.assets.court``, which both branches
provide. Set ``SIDE`` to 0 or 1 to generate the two submission files; they differ
only in that constant.

The reset pose values come from ``scripts/tools/ready_pose.py`` and are unverified
in Isaac until ``scripts/tools/racket_impact_test.py --mode robot`` has been run.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp import (
    base_ang_vel,
    base_lin_vel,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
    projected_gravity,
    reset_root_state_uniform,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

SIDE = 0
"""0: negative-x half. 1: positive-x half (the same pose rotated 180 degrees about z)."""

OBS_DIM = 95

# Player-0 trunk pose. Offsets are added to the default root state (0, 0, 0.70).
READY_ROOT = dict(x=-3.251, y=0.605, yaw=-0.043)
# Joint targets at reset; joints not listed keep their default value.
READY_JOINTS = {
    "Waist": -0.657,
    "Right_Shoulder_Pitch": -1.896,
    "Right_Shoulder_Roll": 1.024,
    "Right_Elbow_Pitch": -0.085,
    "Right_Elbow_Yaw": -1.198,
}


def canonical_robot_heading_court(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, side: int) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    forward_b = torch.zeros_like(robot.data.root_pos_w)
    forward_b[:, 0] = 1.0
    forward_w = quat_apply(robot.data.root_quat_w, forward_b)
    heading_xy = forward_w[:, :2]
    heading = heading_xy / torch.clamp(torch.linalg.norm(heading_xy, dim=-1, keepdim=True), min=1e-6)
    return heading if side == 0 else -heading


def canonical_robot_pos_court(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, side: int) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    root_local = robot.data.root_pos_w - env.scene.env_origins
    if side == 0:
        return root_local
    return torch.cat((-root_local[:, :2], root_local[:, 2:3]), dim=-1)


def shuttle_pos_b(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, shuttle_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    shuttle: RigidObject = env.scene[shuttle_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w, shuttle.data.root_pos_w - robot.data.root_pos_w)


def shuttle_vel_b(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, shuttle_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    shuttle: RigidObject = env.scene[shuttle_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w, shuttle.data.root_lin_vel_w - robot.data.root_lin_vel_w)


def racket_pos_b(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Sweet-spot position in the trunk frame. ``asset_cfg.body_names`` selects the sweet spot."""
    robot: Articulation = env.scene[asset_cfg.name]
    sweet_w = robot.data.body_pos_w[:, asset_cfg.body_ids[0]]
    return quat_apply_inverse(robot.data.root_quat_w, sweet_w - robot.data.root_pos_w)


def shuttle_from_racket_b(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, shuttle_cfg: SceneEntityCfg) -> torch.Tensor:
    """Shuttle position relative to the sweet spot, in the trunk frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    shuttle: RigidObject = env.scene[shuttle_cfg.name]
    sweet_w = robot.data.body_pos_w[:, asset_cfg.body_ids[0]]
    return quat_apply_inverse(robot.data.root_quat_w, shuttle.data.root_pos_w - sweet_w)


def reset_joints_to_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    joint_pos: dict[str, float],
) -> None:
    """Write a fixed pose. ``reset_joints_by_scale`` is multiplicative, so it cannot set zero-default joints."""
    robot: Articulation = env.scene[asset_cfg.name]
    joint_ids, _ = robot.find_joints(list(joint_pos), preserve_order=True)
    positions = robot.data.default_joint_pos[env_ids].clone()
    positions[:, joint_ids] = torch.tensor(list(joint_pos.values()), device=positions.device, dtype=positions.dtype)
    limits = robot.data.soft_joint_pos_limits[env_ids]
    positions = positions.clamp(limits[..., 0], limits[..., 1])
    velocities = torch.zeros_like(positions)
    robot.write_joint_state_to_sim(positions, velocities, env_ids=env_ids)


def make_obs_cfg(side: int) -> type:
    me, opponent = (f"robot_{side}", f"robot_{1 - side}")
    del opponent  # All opponent information is left out on purpose (design doc 6.3).
    action_name = f"player_{side}"

    @configclass
    class ObsCfg(ObsGroup):
        robot_heading_court = ObsTerm(func=canonical_robot_heading_court, params={"asset_cfg": SceneEntityCfg(me), "side": side})
        robot_pos_court = ObsTerm(func=canonical_robot_pos_court, params={"asset_cfg": SceneEntityCfg(me), "side": side})
        shuttle_pos_b = ObsTerm(func=shuttle_pos_b, params={"asset_cfg": SceneEntityCfg(me), "shuttle_cfg": SceneEntityCfg("shuttle")})
        shuttle_vel_b = ObsTerm(func=shuttle_vel_b, params={"asset_cfg": SceneEntityCfg(me), "shuttle_cfg": SceneEntityCfg("shuttle")})
        racket_pos_b = ObsTerm(func=racket_pos_b, params={"asset_cfg": SceneEntityCfg(me, body_names="racket_sweet_spot")})
        shuttle_from_racket_b = ObsTerm(
            func=shuttle_from_racket_b,
            params={"asset_cfg": SceneEntityCfg(me, body_names="racket_sweet_spot"), "shuttle_cfg": SceneEntityCfg("shuttle")},
        )
        base_lin_vel = ObsTerm(func=base_lin_vel, params={"asset_cfg": SceneEntityCfg(me)})
        base_ang_vel = ObsTerm(func=base_ang_vel, params={"asset_cfg": SceneEntityCfg(me)})
        projected_gravity = ObsTerm(func=projected_gravity, params={"asset_cfg": SceneEntityCfg(me)})
        joint_pos = ObsTerm(func=joint_pos_rel, params={"asset_cfg": SceneEntityCfg(me)})
        joint_vel = ObsTerm(func=joint_vel_rel, params={"asset_cfg": SceneEntityCfg(me)})
        last_action = ObsTerm(func=last_action, params={"action_name": action_name})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    return ObsCfg


def make_events_cfg(side: int) -> type:
    sign = 1.0 if side == 0 else -1.0
    yaw = READY_ROOT["yaw"] + (0.0 if side == 0 else math.pi)
    x, y = sign * READY_ROOT["x"], sign * READY_ROOT["y"]
    me = f"robot_{side}"

    # Attribute names must differ per side: play merges the two classes by inheritance.
    attrs = {
        f"reset_player_{side}_root": EventTerm(
            func=reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(me),
                "pose_range": {"x": (x, x), "y": (y, y), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (yaw, yaw)},
                "velocity_range": {axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")},
            },
        ),
        f"reset_player_{side}_joints": EventTerm(
            func=reset_joints_to_pose,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg(me, joint_names=[".*"]), "joint_pos": dict(READY_JOINTS)},
        ),
    }
    annotations = {name: EventTerm for name in attrs}
    cls = type(f"EventsCfg{side}", (), {"__annotations__": annotations, **attrs})
    return configclass(cls)


ObsCfg = make_obs_cfg(SIDE)
EventsCfg = make_events_cfg(SIDE)
