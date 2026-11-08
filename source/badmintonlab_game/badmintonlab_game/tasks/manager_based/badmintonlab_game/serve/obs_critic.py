"""Critic-only observation terms. They read environment internals, so they must stay out of the actor."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse


def is_server(env, side: int) -> torch.Tensor:
    # The observation manager probes term shapes inside ManagerBasedEnv.__init__,
    # before DualAgentBadmintonEnv creates its rally state.
    if not hasattr(env, "initial_hitter"):
        return torch.zeros(env.num_envs, 1, device=env.device)
    return (env.initial_hitter == side).float().unsqueeze(-1)


def rally_phase(env, side: int) -> torch.Tensor:
    """[own hit made, opponent hit made, episode time in seconds]."""
    if not hasattr(env, "hit_count"):
        return torch.zeros(env.num_envs, 3, device=env.device)
    own = (env.hit_count[:, side] > 0).float()
    other = (env.hit_count[:, 1 - side] > 0).float()
    return torch.stack((own, other, env.episode_length_buf.float() * env.step_dt), dim=-1)


def actuator_delay(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Per-episode delay of the arm and waist groups, in units of 5 ms (the P3 physics step, so the critic input
    keeps its range at any physics_dt: 2-8). Hidden from the actor."""
    robot: Articulation = env.scene[asset_cfg.name]
    lags = [robot.actuators[name].positions_delay_buffer.time_lags.float() for name in ("arms", "waist")]
    return torch.stack(lags, dim=-1) * (float(env.physics_dt) / 0.005)


def racket_vel_b(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w, robot.data.body_lin_vel_w[:, asset_cfg.body_ids[0]])


def opponent_root_court(env, asset_cfg: SceneEntityCfg, side: int) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    local = robot.data.root_pos_w - env.scene.env_origins
    return local if side == 0 else torch.cat((-local[:, :2], local[:, 2:3]), dim=-1)
