"""Receive task configuration (P5 V3.1 S1). Learner: robot 0. Robot 1 is parked off court."""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.envs.mdp import reset_root_state_uniform
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from .. import mdp
from ..dual_agent_env_cfg import DualTerminationsCfg
from ..serve import player_cfg as serve_player_cfg
from ..serve.obs_critic import actuator_delay, racket_vel_b
from ..serve.serve_env_cfg import ServeEnvCfg, ServeEventsCfg
from . import player_cfg as recv_player_cfg
from .receive_env import TERMS, critic_feet, critic_targets, recv_balls_done, recv_fall_done, recv_term

_Player0Obs = recv_player_cfg.make_obs_cfg(0)
_Player1Obs = serve_player_cfg.make_obs_cfg(1)


@configclass
class ReceiveCriticCfg(ObsGroup):
    actuator_delay = ObsTerm(func=actuator_delay, params={"asset_cfg": SceneEntityCfg("robot_0")})
    racket_vel_b = ObsTerm(func=racket_vel_b, params={"asset_cfg": SceneEntityCfg("robot_0", body_names="racket_sweet_spot")})
    targets = ObsTerm(func=critic_targets)
    feet = ObsTerm(func=critic_feet)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ReceiveObservationsCfg:
    player_0: _Player0Obs = _Player0Obs()
    player_1: _Player1Obs = _Player1Obs()
    critic_0: ReceiveCriticCfg = ReceiveCriticCfg()


@configclass
class ReceiveEventsCfg(ServeEventsCfg):
    # Applied after the player-1 ready pose: move robot 1 off court (out of the doubles lines).
    park_player_1 = EventTerm(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_1"),
            "pose_range": {"x": (4.0, 4.0), "y": (5.0, 5.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (3.14159, 3.14159)},
            "velocity_range": {axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")},
        },
    )


def _recv_rewards() -> type:
    terms = {f"recv_{t}": RewTerm(func=recv_term, weight=1.0, params={"term": t}) for t in TERMS}
    terms["joint_limit"] = RewTerm(func=mdp.joint_pos_limits, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot_0", joint_names=[".*"])})
    terms["racket_ground"] = RewTerm(
        func=mdp.racket_ground_support_penalty,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("racket_ground_contact_0", body_names=["racket_link"]), "force_threshold": 3.0, "force_scale": 10.0, "max_penalty": 1.0},
    )
    terms["action_rate"] = RewTerm(func=mdp.agent_action_rate_l2, weight=-0.03, params={"action_name": "player_0"})
    body = {"__annotations__": {k: RewTerm for k in terms}, **terms}
    return configclass(type("ReceiveRewardsCfg", (), body))


@configclass
class _Player1Rewards:
    joint_limit = RewTerm(func=mdp.joint_pos_limits, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot_1", joint_names=[".*"])})


ReceiveRewardsCfg = _recv_rewards()


@configclass
class ReceiveTerminationsCfg(DualTerminationsCfg):
    rally_resolved: DoneTerm = None
    robot_fell = DoneTerm(func=recv_fall_done)
    balls_done = DoneTerm(func=recv_balls_done)


@configclass
class ReceiveEnvCfg(ServeEnvCfg):
    observations: ReceiveObservationsCfg = ReceiveObservationsCfg()
    events: ReceiveEventsCfg = ReceiveEventsCfg()
    rewards: ReceiveRewardsCfg = ReceiveRewardsCfg()
    player_1_rewards: _Player1Rewards = _Player1Rewards()
    terminations: ReceiveTerminationsCfg = ReceiveTerminationsCfg()
    feeder_rows: int = 200_000
    feeder_seed: int = 0
    balls_per_episode: int = 6
    ball_gap_s: float = 0.5
    ball_timeout_s: float = 2.5
    z_ready: float = 0.68
    fall_height: float = 0.45
    feeder_v4: bool = False  # 11-09 S1 V4: wide table, free hitting height, speed levels, return-home gaps
    swing_v_cap: float = 8.0  # S2 r_v input cap, m/s (V3.1 had 3.5 as a 5 ms tunnelling guard)

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        self.scene.feet_contact_0 = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot0/.*_foot_link", history_length=3, track_air_time=True, force_threshold=1.0
        )
