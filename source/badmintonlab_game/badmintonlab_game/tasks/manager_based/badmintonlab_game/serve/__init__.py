"""Registers the rule-aligned serve task. Environment modules are imported lazily by Gym."""

import gymnasium as gym

gym.register(
    id="BadmintonLab-Game-Serve",
    entry_point=f"{__name__}.serve_env:ServeBadmintonEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.serve_env_cfg:ServeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:ServePPORunnerCfg",
    },
)

gym.register(
    id="BadmintonLab-Game-Receive",
    entry_point=f"{__name__}.p5_env:ReceiveBadmintonEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.p5_env_cfg:ReceiveEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:ReceivePPORunnerCfg",
    },
)
