# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gymnasium registration for two-player badminton self-play."""

import gymnasium as gym

gym.register(
    id="BadmintonLab-Game",
    entry_point=f"{__name__}.dual_agent_env:DualAgentBadmintonEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dual_agent_env_cfg:DualAgentBadmintonEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

from . import serve  # noqa: E402,F401  registers BadmintonLab-Game-Serve
