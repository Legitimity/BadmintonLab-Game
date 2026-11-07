"""Shared-policy wrapper with a critic observation group, plus the actor pose initialisation."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from ..shared_policy_wrapper import SharedPolicyVecEnvWrapper


class ServePolicyVecEnvWrapper(SharedPolicyVecEnvWrapper):
    def _agent_obs(self, obs_dict: dict[str, torch.Tensor]) -> TensorDict:
        policy = torch.cat((obs_dict["player_0"], obs_dict["player_1"]), dim=0)
        critic = torch.cat((obs_dict["critic_0"], obs_dict["critic_1"]), dim=0)
        return TensorDict({"policy": policy, "critic": critic}, batch_size=[self.num_envs])


class ReceivePolicyVecEnvWrapper(ServePolicyVecEnvWrapper):
    """P5: adds the P5 critic group ``p5c``, ``aux`` = [learn_w, teach_w, m_serve, m_recv, m_body] and
    ``extras["group_rewards"]`` (2N, 3), same sample order as the rewards (side 0 courts, then side 1)."""

    def _agent_obs(self, obs_dict: dict[str, torch.Tensor]) -> TensorDict:
        td = super()._agent_obs(obs_dict)
        td["p5c"] = torch.cat((obs_dict["p5c_0"], obs_dict["p5c_1"]), dim=0)
        aux = self.unwrapped.p5_aux()
        td["aux"] = torch.cat((aux[:, 0], aux[:, 1]), dim=0)
        return td

    def step(self, actions: torch.Tensor):
        obs, rewards, dones, extras = super().step(actions)
        g = self.unwrapped.group_reward_buf
        extras["group_rewards"] = torch.cat((g[:, 0], g[:, 1]), dim=0)
        return obs, rewards, dones, extras


def hold_action(q_ready: torch.Tensor, offset: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Raw action whose target is ``q_ready``: the inverse of ``raw * scale + offset``."""
    return (q_ready - offset) / scale


@torch.no_grad()
def init_actor_output(policy, action_bias: torch.Tensor, weight_scale: float = 0.01) -> None:
    """Start with the actor near ``action_bias``.

    The action offset is fixed to the course default pose, so a zero output would leave the ready pose.
    Works on an RSL-RL ``ActorCritic``, whose ``actor`` ends in the output ``Linear``.
    """
    last = policy.actor[-1]
    if not isinstance(last, torch.nn.Linear) or last.out_features != action_bias.numel():
        raise ValueError(f"Unexpected actor output layer: {last}")
    last.weight.mul_(weight_scale)
    last.bias.copy_(action_bias.to(last.bias.device, last.bias.dtype))
