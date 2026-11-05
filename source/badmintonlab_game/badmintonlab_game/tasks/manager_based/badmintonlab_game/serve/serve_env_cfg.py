"""Task configuration: the parent's rewards and scene, ``play`` rules, the shared player config."""

from __future__ import annotations

import dataclasses

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .. import mdp
from ..badminton_scene_cfg import _COURT_CFG
from ..dual_agent_env_cfg import DualAgentBadmintonEnvCfg, Player0RewardsCfg, Player1RewardsCfg
from . import player_cfg


@sim_utils.clone
def spawn_net_collider_filtered(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Copy of the ``play`` spawner: the filter must sit on the collider, not on the Xform above it."""
    from isaaclab.sim.spawners.shapes.shapes import spawn_cuboid
    from pxr import UsdPhysics

    prim = spawn_cuboid(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    shuttle_path = prim.GetPath().GetParentPath().AppendChild("Shuttle")
    pending = [prim]
    while pending:
        child = pending.pop()
        if child.HasAPI(UsdPhysics.CollisionAPI):
            filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(child)
            filtered_pairs.CreateFilteredPairsRel().AddTarget(shuttle_path)
        pending.extend(child.GetChildren())
    return prim


def _critic_terms(side: int) -> dict:
    """Training-only inputs (design doc 6.4). These never enter the submitted model."""
    from .obs_critic import actuator_delay, is_server, racket_vel_b, rally_phase, opponent_root_court

    me = SceneEntityCfg(f"robot_{side}")
    return {
        "is_server": ObsTerm(func=is_server, params={"side": side}),
        "rally_phase": ObsTerm(func=rally_phase, params={"side": side}),
        "actuator_delay": ObsTerm(func=actuator_delay, params={"asset_cfg": me}),
        "racket_vel_b": ObsTerm(func=racket_vel_b, params={"asset_cfg": SceneEntityCfg(f"robot_{side}", body_names="racket_sweet_spot")}),
        "opponent_root": ObsTerm(func=opponent_root_court, params={"asset_cfg": SceneEntityCfg(f"robot_{1 - side}"), "side": side}),
    }


def _make_critic_cfg(side: int) -> type:
    terms = _critic_terms(side)
    cls = type(f"Critic{side}Cfg", (ObsGroup,), {"__annotations__": {k: ObsTerm for k in terms}, **terms, "__post_init__": _critic_post_init})
    return configclass(cls)


def _critic_post_init(self):
    self.enable_corruption = False
    self.concatenate_terms = True


_Player0Obs = player_cfg.make_obs_cfg(0)
_Player1Obs = player_cfg.make_obs_cfg(1)
_Critic0 = _make_critic_cfg(0)
_Critic1 = _make_critic_cfg(1)


@configclass
class ServeObservationsCfg:
    player_0: _Player0Obs = _Player0Obs()
    player_1: _Player1Obs = _Player1Obs()
    critic_0: _Critic0 = _Critic0()
    critic_1: _Critic1 = _Critic1()


@configclass
class ServeEventsCfg(player_cfg.make_events_cfg(1), player_cfg.make_events_cfg(0)):
    """Same inheritance order as ``play``: player 1, player 0, then the shuttle."""

    reset_shuttle_pose = EventTerm(
        func=mdp.reset_shuttle_drop_state,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("shuttle"),
            "canonical_position": (-0.5 * _COURT_CFG.half_length + 0.8, 0.0, 2.2),
        },
    )


@configclass
class ServeEnvCfg(DualAgentBadmintonEnvCfg):
    observations: ServeObservationsCfg = ServeObservationsCfg()
    events: ServeEventsCfg = ServeEventsCfg()
    rewards: Player0RewardsCfg = Player0RewardsCfg()
    player_1_rewards: Player1RewardsCfg = Player1RewardsCfg()
    max_serve_height = 1.15

    def __post_init__(self):
        super().__post_init__()
        collider = self.scene.net_collider
        collider.spawn = dataclasses.replace(collider.spawn, func=spawn_net_collider_filtered)
        # 4096/8192 courts overflow the default 5*2^15 contact patches (PhysX asks for ~176k / ~352k).
        # Training only; play keeps its own setting. Costs a few tens of MB of GPU memory.
        self.sim.physx.gpu_max_rigid_patch_count = 2**19
