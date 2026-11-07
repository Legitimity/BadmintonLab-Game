"""PPO configuration. Only the observation groups, noise and save interval differ from the baseline."""

from isaaclab.utils import configclass

from ..agents.rsl_rl_ppo_cfg import PPORunnerCfg


@configclass
class ServePPORunnerCfg(PPORunnerCfg):
    experiment_name = "badmintonlab_serve"
    save_interval = 250
    obs_groups = {"policy": ["policy"], "critic": ["policy", "critic"]}

    def __post_init__(self):
        # The default of 1.0 means about +/-1 rad of noise on the arm: it destroys balance and the ready pose.
        self.policy.init_noise_std = 0.3


@configclass
class ReceivePPORunnerCfg(ServePPORunnerCfg):
    """P5: the P5 critic group is appended after the serve critic (warm start pads the first critic layer).
    ``aux`` (sample weights) is stored by the rollout storage but read only by ``PPOExt``."""

    experiment_name = "badmintonlab_receive"
    save_interval = 250
    obs_groups = {"policy": ["policy"], "critic": ["policy", "critic", "p5c"]}
