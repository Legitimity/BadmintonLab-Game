"""Receive-expert actor observations (P5 V3.1): the 95-D serve observation plus a 5-D intercept prediction.

Generated from serve/player_cfg.py (same ready pose and events); self-contained for submission.
Original header follows.

Actor observations and initial state for one player (design doc P2).

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

OBS_DIM = 100
BASE_OBS_DIM = 95

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


def make_base_obs_cfg(side: int) -> type:
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




# ------------------------------------------------------------------ intercept prediction (P5 V3.1 4.1; V4 11-09)
# V4: free hitting height. Among the descending crossings of _HIT_ZS inside the band, pick the one with the
# lowest score d - _V_REF t + _Z_PREF |z - 1|, d = distance from the ready root to the station of that point.
# The score depends only on the trajectory and is invariant to a common time shift, so the target is stable
# during the flight. Must stay identical to receive/core.predict_intercept_multi (tested).
_DRAG_OVER_MASS = 0.00110 / 0.00519
_HIT_ZS = (0.8, 0.9, 1.0, 1.1, 1.2)
_BAND_X = (-5.0, -0.5)
_BAND_Y = 2.4
_V_REF = 1.5
_Z_PREF = 0.3
_READY_XY = (-3.251, 0.605)
_READY_YAW = -0.043
_O_READY = (0.825, -0.521)


def _predict_intercept(pos: torch.Tensor, vel: torch.Tensor, dt: float = 0.03, steps: int = 70):
    """Hitting point with a free height, canonical frame (own half x<0). Returns (point, time, valid)."""
    p = pos.clone()
    v = vel.clone()
    n = p.shape[0]
    zs = torch.tensor(_HIT_ZS, device=p.device, dtype=p.dtype)
    k_n = zs.numel()
    gv = torch.tensor((0.0, 0.0, -9.81), device=p.device, dtype=p.dtype)
    found = torch.zeros(n, k_n, dtype=torch.bool, device=p.device)
    alive = p[:, 2] > 0.03
    pt = torch.zeros(n, k_n, 3, device=p.device, dtype=p.dtype)
    t_hit = torch.zeros(n, k_n, device=p.device, dtype=p.dtype)
    for k in range(steps):
        speed = torch.linalg.norm(v, dim=-1, keepdim=True)
        vm = v + 0.5 * dt * (gv - _DRAG_OVER_MASS * speed * v)
        speed_m = torch.linalg.norm(vm, dim=-1, keepdim=True)
        vn = v + dt * (gv - _DRAG_OVER_MASS * speed_m * vm)
        pn = p + dt * vm
        z0, z1 = p[:, 2:3], pn[:, 2:3]
        cross = alive[:, None] & ~found & (z0 >= zs) & (z1 < zs)
        a = (z0 - zs) / (z0 - z1).clamp(min=1e-9)
        q = p[:, None, :] + a[..., None] * (pn - p)[:, None, :]
        pt = torch.where(cross[..., None], q, pt)
        t_hit = torch.where(cross, (k + a) * dt, t_hit)
        found = found | cross
        p, v = pn, vn
        alive = alive & (p[:, 2] > 0.03)
    ok = found & (pt[..., 0] >= _BAND_X[0]) & (pt[..., 0] <= _BAND_X[1]) & (pt[..., 1].abs() <= _BAND_Y)
    c, s = math.cos(_READY_YAW), math.sin(_READY_YAW)
    ox = c * _O_READY[0] - s * _O_READY[1]
    oy = s * _O_READY[0] + c * _O_READY[1]
    sx = pt[..., 0] - ox - _READY_XY[0]
    sy = pt[..., 1] - oy - _READY_XY[1]
    d = torch.sqrt(sx * sx + sy * sy)
    score = d - _V_REF * t_hit + _Z_PREF * (zs - 1.0).abs()
    score = torch.where(ok, score, torch.full_like(score, 1e9))
    best = score.argmin(dim=1)
    idx = torch.arange(n, device=p.device)
    return pt[idx, best], t_hit[idx, best], ok.any(dim=1)


def intercept_obs(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, shuttle_cfg: SceneEntityCfg, side: int) -> torch.Tensor:
    """[intercept relative to the root in the trunk frame (3), time to it (1), valid flag (1)].

    Uses only the current shuttle state. Zero when the shuttle is not flying towards this player
    or will not enter the hitting band.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    shuttle: RigidObject = env.scene[shuttle_cfg.name]
    local = shuttle.data.root_pos_w - env.scene.env_origins
    vel = shuttle.data.root_lin_vel_w
    sign = 1.0 if side == 0 else -1.0
    flip = torch.tensor((sign, sign, 1.0), device=local.device, dtype=local.dtype)
    pt_c, t_hit, valid = _predict_intercept(local * flip, vel * flip)
    valid = valid & (vel[:, 0] * sign < -0.05)
    pt_w = pt_c * flip + env.scene.env_origins
    rel_b = quat_apply_inverse(robot.data.root_quat_w, pt_w - robot.data.root_pos_w)
    out = torch.cat((rel_b, t_hit.clamp(0.0, 1.5).unsqueeze(-1), valid.float().unsqueeze(-1)), dim=-1)
    return out * valid.float().unsqueeze(-1)


def make_obs_cfg(side: int) -> type:
    base = make_base_obs_cfg(side)
    me = f"robot_{side}"

    @configclass
    class ObsCfg(base):
        intercept = ObsTerm(
            func=intercept_obs,
            params={"asset_cfg": SceneEntityCfg(me), "shuttle_cfg": SceneEntityCfg("shuttle"), "side": side},
        )

    return ObsCfg


ObsCfg = make_obs_cfg(SIDE)
EventsCfg = make_events_cfg(SIDE)
