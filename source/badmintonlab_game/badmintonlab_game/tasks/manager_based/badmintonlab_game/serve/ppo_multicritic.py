"""Multi-critic PPO for P5 V2 (P5_RECEIVE_DESIGN_V2 2). No Isaac import.

* ``MultiCriticActorCritic``: RSL-RL ``ActorCritic`` whose critic ends in ``128 -> num_values`` (one value head per
  reward group on a shared trunk) and whose actor normaliser can be frozen (V2 5).
* ``MultiCriticStorage``: rewards / values / returns ``(T, N, G)``, ``group_mask (T, N, G)``; per-group GAE with
  the same ``gamma``, ``lam`` and ``dones``; per-group normalisation of the advantages over the whole rollout on the
  samples with ``group_mask * learn_w = 1`` (the others get 0, a group without samples gets 0, no NaN; a group
  whose advantage std is below ``adv_std_min`` = 1e-4, e.g. G_fall in a rollout without a fall, gets 0 instead
  of amplified numerical noise); the combined advantage ``A = sum_g w_g * A_g`` is not normalised again.
  The number of groups G is ``len(group_names)`` (default ``receive.REWARD_GROUPS``: serve, recv, fall, body).
* ``MultiCriticPPO``: ``PPOExt`` (sample mask, teacher, std clamp, critic warm-up) with the grouped storage, time-out
  bootstrap per group (``r_g += gamma * V_g``), value loss ``sum_g mean_{mask_g}(clipped loss_g)``. ``w_g`` can
  change between rollouts (``set_group_weights``: V2 2.5 stage weights, set by the runner from the curriculum).
  Logs ``Rollout/events_<group>`` (non-zero group rewards on learned samples; ``events_fall`` = fall events of the
  rollout) in both modes.

Group rewards come from ``extras["group_rewards"]`` ``(N, G_env)``; the group mask of a sample is read from its
stored observation ``aux[:, 2:2+G_env]`` (state of the action, i.e. the rally the reward belongs to).
``group_mode="single"`` is the control S: the groups are summed into one reward, one value head, the mask is
``learn_w`` only; with one group and ``w = 1`` this is ``PPOExt`` (tested element-wise against RSL-RL PPO).
"""

from __future__ import annotations

from collections.abc import Generator

import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

from .ppo_ext import PPOExt
from .receive import REWARD_GROUPS

GROUPS = REWARD_GROUPS
DEFAULT_GROUP_WEIGHTS = (1.0, 1.0, 0.8, 0.2)  # V2 2.5, R0
ADV_STD_MIN = 1e-4


class MultiCriticActorCritic(ActorCritic):
    def __init__(self, obs, obs_groups, num_actions, num_values: int = 1, freeze_actor_normalizer: bool = False,
                 **kwargs):
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        self.num_values = int(num_values)
        last = self.critic[-1]
        if not isinstance(last, nn.Linear) or last.out_features != 1:
            raise ValueError(f"Unexpected critic output layer: {last}")
        if self.num_values != 1:
            self.critic[-1] = nn.Linear(last.in_features, self.num_values)
        self.freeze_actor_normalizer = bool(freeze_actor_normalizer)

    def update_normalization(self, obs) -> None:
        if self.actor_obs_normalization and not self.freeze_actor_normalizer:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))


class MultiCriticStorage(RolloutStorage):
    def __init__(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape, device="cpu",
                 num_groups: int = 1):
        super().__init__(training_type, num_envs, num_transitions_per_env, obs, actions_shape, device)
        g, t, n = int(num_groups), num_transitions_per_env, num_envs
        self.num_groups = g
        z = lambda: torch.zeros(t, n, g, device=self.device)  # noqa: E731
        self.rewards, self.values, self.returns, self.group_mask, self.group_adv = z(), z(), z(), z(), z()
        self.advantages = torch.zeros(t, n, 1, device=self.device)
        self.stats: dict[str, torch.Tensor] = {}

    def add_transitions(self, transition) -> None:
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")
        g = self.num_groups
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, g))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values.view(-1, g))
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self.group_mask[self.step].copy_(transition.group_mask.view(-1, g))
        self.step += 1

    @torch.no_grad()
    def compute_group_returns(self, last_values, gamma: float, lam: float, weights, learn_w,
                              adv_std_min: float = 0.0) -> None:
        """``last_values`` (N, G); ``weights`` (G,); ``learn_w`` (T, N, 1). Groups with fewer than 2 samples or an
        advantage std below ``adv_std_min`` get a zero normalised advantage."""
        adv = torch.zeros_like(last_values)
        for step in reversed(range(self.num_transitions_per_env)):
            next_values = last_values if step == self.num_transitions_per_env - 1 else self.values[step + 1]
            not_done = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + not_done * gamma * next_values - self.values[step]
            adv = delta + not_done * gamma * lam * adv
            self.returns[step] = adv + self.values[step]
        raw = self.returns - self.values
        m = self.group_mask * learn_w
        cnt = m.sum((0, 1))
        mean = (raw * m).sum((0, 1)) / cnt.clamp(min=1.0)
        var = ((raw - mean) ** 2 * m).sum((0, 1)) / (cnt - 1.0).clamp(min=1.0)  # unbiased, as torch.std in RSL-RL
        std = var.sqrt()
        ok = ((cnt > 1.0) & (std >= adv_std_min)).float()
        self.group_adv = (raw - mean) / std.clamp(min=1e-8) * m * ok
        w = torch.as_tensor(weights, dtype=raw.dtype, device=raw.device).view(1, 1, -1)
        self.advantages = (self.group_adv * w).sum(-1, keepdim=True)
        # Explained variance of the values on the group's samples.
        ret_mean = (self.returns * m).sum((0, 1)) / cnt.clamp(min=1.0)
        var_ret = ((self.returns - ret_mean) ** 2 * m).sum((0, 1)) / cnt.clamp(min=1.0)
        res = self.returns - self.values
        res_mean = (res * m).sum((0, 1)) / cnt.clamp(min=1.0)
        var_res = ((res - res_mean) ** 2 * m).sum((0, 1)) / cnt.clamp(min=1.0)
        ev = torch.where(var_ret > 1e-12, 1.0 - var_res / var_ret.clamp(min=1e-12), torch.zeros_like(var_ret))
        self.stats = dict(adv_std=std * (cnt > 1.0).float(), adv_mean=mean, explained_var=ev,
                          share=cnt / float(m[..., 0].numel()), adv_zeroed=1.0 - ok,
                          reward_mean=(self.rewards * m).sum((0, 1)) / cnt.clamp(min=1.0))

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator:
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        group_mask = self.group_mask.flatten(0, 1)
        for _epoch in range(num_epochs):
            for i in range(num_mini_batches):
                idx = indices[i * mini_batch_size:(i + 1) * mini_batch_size]
                yield (observations[idx], actions[idx], values[idx], advantages[idx], returns[idx],
                       old_actions_log_prob[idx], old_mu[idx], old_sigma[idx], (None, None), group_mask[idx])


class MultiCriticPPO(PPOExt):
    def __init__(self, policy, group_mode: str = "multi", group_weights=DEFAULT_GROUP_WEIGHTS, group_names=GROUPS,
                 adv_std_min: float = ADV_STD_MIN, **kwargs):
        super().__init__(policy, **kwargs)
        self.adv_std_min = float(adv_std_min)
        self.env_group_names = tuple(group_names)
        self._events = None
        if self.normalize_advantage_per_mini_batch:
            raise ValueError("MultiCriticPPO normalises the advantages per group over the rollout")
        if group_mode not in ("multi", "single"):
            raise ValueError(f"group_mode {group_mode!r}")
        self.group_mode = group_mode
        self.env_groups = len(group_names)
        self.num_values = getattr(policy, "num_values", 1)
        if group_mode == "multi":
            self.group_names = tuple(group_names)
            self.group_weights = tuple(float(w) for w in group_weights)
            if len(self.group_weights) != len(self.group_names):
                raise ValueError("one weight per group")
        else:
            self.group_names, self.group_weights = ("sum",), (1.0,)
        if self.num_values != len(self.group_names):
            raise ValueError(f"policy has {self.num_values} value heads, {group_mode} mode needs {len(self.group_names)}")

    def set_group_weights(self, weights) -> None:
        """V2 2.5: ``w_g`` for the next rollouts (multi mode; control S ignores them, its groups are summed)."""
        if self.group_mode != "multi":
            return
        w = tuple(float(x) for x in weights)
        if len(w) != len(self.group_names):
            raise ValueError(f"{len(w)} weights for groups {self.group_names}")
        self.group_weights = w

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape) -> None:
        self.storage = MultiCriticStorage(training_type, num_envs, num_transitions_per_env, obs, actions_shape,
                                          self.device, num_groups=self.num_values)

    def _aux(self, obs, n: int) -> torch.Tensor:
        if obs is not None and self.aux_group in obs.keys():
            return obs[self.aux_group].float()
        return torch.ones(n, 2 + self.env_groups, device=self.device)

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        self.policy.update_normalization(obs)
        n = rewards.shape[0]
        gr = extras.get("group_rewards")
        gr = rewards.view(-1, 1).float() if gr is None else gr.to(self.device).float()
        if gr.shape[-1] == self.env_groups:
            learn = self._aux(self.transition.observations, n)[:, 0:1]
            ev = ((gr != 0).float() * learn).sum(0)
            self._events = ev if self._events is None else self._events + ev
        if self.group_mode == "single":
            r = gr.sum(-1, keepdim=True)
            m = torch.ones(n, 1, device=self.device)
        else:
            if gr.shape[-1] != self.num_values:
                raise ValueError(f"{gr.shape[-1]} group rewards for {self.num_values} value heads")
            aux = self._aux(self.transition.observations, n)
            m = aux[:, 2:2 + self.num_values] if aux.shape[-1] >= 2 + self.num_values else torch.ones_like(gr)
            r = gr
        self.transition.rewards = r.clone()
        self.transition.dones = dones
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * self.transition.values * extras["time_outs"].to(self.device).view(-1, 1)
        self.transition.group_mask = m
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    @torch.no_grad()
    def compute_returns(self, obs) -> None:
        last_values = self.policy.evaluate(obs).detach()
        st = self.storage
        if self.aux_group in st.observations.keys():
            learn = st.observations[self.aux_group][..., 0:1].float()
        else:
            learn = torch.ones(st.num_transitions_per_env, st.num_envs, 1, device=self.device)
        st.compute_group_returns(last_values, self.gamma, self.lam, self.group_weights, learn, self.adv_std_min)
        stats = {}
        for key, v in st.stats.items():
            for i, name in enumerate(self.group_names):
                stats[f"Critic/{key}_{name}"] = float(v[i])
        for i, name in enumerate(self.group_names if self.group_mode == "multi" else ()):
            stats[f"Critic/w_{name}"] = self.group_weights[i]
        if self._events is not None:
            for i, name in enumerate(self.env_group_names):
                stats[f"Rollout/events_{name}"] = float(self._events[i])
            self._events = None
        stats.update(self.normalizer_range(st))
        self._rollout_stats = stats

    @torch.no_grad()
    def normalizer_range(self, st, limit: float = 5.0) -> dict:
        """V2 risk 4: share of the actor inputs beyond +-``limit`` after the (frozen) actor normaliser, on the
        serve-role and the receive-role samples; and the max |z| of each."""
        pol = self.policy
        if not getattr(pol, "actor_obs_normalization", False) or self.aux_group not in st.observations.keys():
            return {}
        x = pol.actor_obs_normalizer(pol.get_actor_obs(st.observations.flatten(0, 1)))
        aux = st.observations[self.aux_group].flatten(0, 1).float()
        out = {}
        if aux.shape[-1] < 4:
            return out
        cols = {g: 2 + i for i, g in enumerate(self.env_group_names) if g in ("serve", "recv")}
        for name, col in cols.items():
            sel = (aux[:, col] * aux[:, 0]) > 0
            if bool(sel.any()):
                z = x[sel].abs()
                out[f"Norm/over{limit:g}_{name}"] = float((z > limit).float().mean())
                out[f"Norm/zmax_{name}"] = float(z.max())
        return out

    def _renormalize_advantages(self) -> None:
        return  # done per group in compute_returns

    def value_loss(self, value_batch, target_values_batch, returns_batch, learn_w, group_mask=None):
        g = value_batch.shape[-1]
        m = (torch.ones_like(value_batch) if group_mask is None else group_mask) * learn_w.view(-1, 1)
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
            per = torch.max((value_batch - returns_batch).pow(2), (value_clipped - returns_batch).pow(2))
        else:
            per = (returns_batch - value_batch).pow(2)
        per_g = (per * m).sum(0) / m.sum(0).clamp(min=1e-8)
        for i in range(g):
            self._batch_stats[f"Critic/value_loss_{self.group_names[i]}"] = float(per_g[i].detach())
        return per_g.sum()


def pad_value_head(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """P3's ``1 -> G`` value head: row 0 (``G_serve``) copied, the other heads zero (V2 2.2)."""
    out = torch.zeros_like(dst)
    out[: src.shape[0]] = src.to(dst.device, dst.dtype)
    return out
