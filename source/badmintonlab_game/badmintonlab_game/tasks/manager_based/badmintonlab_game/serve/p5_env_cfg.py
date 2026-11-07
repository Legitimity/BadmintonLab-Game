"""P5 receive task configuration: the serve task plus the receive rewards, the P5 critic group and the feeder."""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .receive import RECV_TERMS, ReceiveParams, recv_fall_done, recv_term
from .serve_env_cfg import (
    ServeEnvCfg,
    ServeObservationsCfg,
    ServeRewards0Cfg,
    ServeRewards1Cfg,
    ServeTerminationsCfg,
)


def p5_critic(env, side: int):
    """Critic-only P5 inputs (P5_RECEIVE_DESIGN 4.1), appended after the serve critic group."""
    import torch

    if not hasattr(env, "recv_reward"):
        return torch.zeros(env.num_envs, 11, device=env.device)
    return env.p5_critic_obs(side)


def _p5_group(side: int) -> type:
    def post_init(self):
        self.enable_corruption = False
        self.concatenate_terms = True

    term = ObsTerm(func=p5_critic, params={"side": side})
    cls = type(f"P5Critic{side}Cfg", (ObsGroup,), {"__annotations__": {"p5": ObsTerm}, "p5": term, "__post_init__": post_init})
    return configclass(cls)


_P5C0 = _p5_group(0)
_P5C1 = _p5_group(1)


@configclass
class ReceiveObservationsCfg(ServeObservationsCfg):
    p5c_0: _P5C0 = _P5C0()
    p5c_1: _P5C1 = _P5C1()


def _recv_terms(base: type, side: int) -> type:
    terms = {t: RewTerm(func=recv_term, weight=1.0, params={"term": t, "side": side}) for t in RECV_TERMS}
    return configclass(type(f"ReceiveRewards{side}Cfg", (base,), {"__annotations__": {k: RewTerm for k in terms}, **terms}))


ReceiveRewards0Cfg = _recv_terms(ServeRewards0Cfg, 0)
ReceiveRewards1Cfg = _recv_terms(ServeRewards1Cfg, 1)


@configclass
class ReceiveTerminationsCfg(ServeTerminationsCfg):
    receiver_fell = DoneTerm(func=recv_fall_done)


@configclass
class ReceiveParamsCfg(ReceiveParams):
    """Hydra-overridable, e.g. ``env.recv_reward.approach_hi=0.7``."""


@configclass
class P5Cfg:
    natural_share: float = 0.5  # initial; the runner sets it from the curriculum
    level: int = 0  # initial feeder level; the runner sets it from the curriculum
    feeder_x: float = 5.8  # feeder root, |x| in its own half (canonical x = -feeder_x)
    feeder_y: float = 1.9  # random sign
    teach_after_hit: float = 0.3  # s of teacher samples after the server's first hit
    natural_scan_dt: float = 0.02
    natural_scan_s: float = 2.5
    verify_mode: bool = False  # receiver off court, levels drawn uniformly: landing check of the table (2.2)
    verify_recv_x: float = 3.25
    verify_recv_y: float = 4.0
    verify_tol: float = 0.10


@configclass
class ReceiveEnvCfg(ServeEnvCfg):
    observations: ReceiveObservationsCfg = ReceiveObservationsCfg()
    rewards: ReceiveRewards0Cfg = ReceiveRewards0Cfg()
    player_1_rewards: ReceiveRewards1Cfg = ReceiveRewards1Cfg()
    terminations: ReceiveTerminationsCfg = ReceiveTerminationsCfg()
    recv_reward: ReceiveParamsCfg = ReceiveParamsCfg()
    p5: P5Cfg = P5Cfg()

    def __post_init__(self):
        # P5: the rally goes on after the return; the receiver may step (no hold_ready); C serve weights.
        self.serve_reward.end_on_return = False
        self.serve_reward.receiver_hold = False
        self.serve_reward.initial_stage = "C"
        # P3 final (p3_c7) was trained in C with --stage_param C.v_close_max=7.0: same serve reward.
        self.serve_reward.v_close_max = 7.0
        super().__post_init__()
        if not self.recv_reward.terminate_on_fall:
            self.terminations.receiver_fell = None
