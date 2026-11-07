"""Registers the receive-expert task (P5 V3.1). Environment modules are imported lazily by Gym."""

import gymnasium as gym

gym.register(
    id="BadmintonLab-Game-Receive",
    entry_point=f"{__name__}.receive_env:ReceiveBadmintonEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.receive_env_cfg:ReceiveEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:ReceivePPORunnerCfg",
    },
)
