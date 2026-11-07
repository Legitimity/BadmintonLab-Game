"""PPO for P5 (P5_RECEIVE_DESIGN 4.3): sample mask, teacher term (Kickstarting), noise range, critic warm-up.

RSL-RL 3.1.2 ``PPO.update`` with four changes, nothing else (no RND, symmetry or recurrent support):

1. **Sample mask.** The observation group ``aux`` (stored by the rollout storage, never fed to the actor or the
   critic because it is not in ``obs_groups``) carries ``learn_w`` (0 for the feeder's samples) and ``teach_w``.
   Surrogate, value and entropy losses and the KL used for the adaptive learning rate are ``learn_w``-weighted
   means.
2. **Teacher.** ``beta * mean_teach_w( sum_a (mu_s - mu_t)^2 / sigma_t^2 )``: ``mu_t`` from the frozen P3 actor
   with its own normaliser (a TorchScript export, raw actor observations in), ``sigma_t = 0.3``. ``beta`` falls
   linearly from ``beta_start`` to ``beta_end`` over ``beta_iters`` updates and then stays at ``beta_end``.
3. **Noise range.** ``std`` is clamped to ``[std_min, std_max]`` after every optimiser step (and once at start).
4. **Critic warm-up.** For the first ``critic_only_iters`` updates the actor and the noise get no gradient and
   the learning rate is not adapted (the actor KL is 0 then; the adaptive rule would only raise it).

``updates`` counts the calls of ``update`` and is saved with the checkpoint (``state_dict_extra``).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO


class PPOExt(PPO):
    def __init__(self, policy, teacher=None, teacher_path: str | None = None, teacher_sigma: float = 0.3,
                 beta_start: float = 1.0, beta_end: float = 0.1, beta_iters: int = 2000, std_min: float = 0.15,
                 std_max: float = 0.40, critic_only_iters: int = 30, aux_group: str = "aux", **kwargs):
        super().__init__(policy, **kwargs)
        if teacher is None and teacher_path:
            teacher = torch.jit.load(teacher_path, map_location=self.device).eval()
        self.teacher = teacher
        if isinstance(self.teacher, nn.Module):
            for prm in self.teacher.parameters():
                prm.requires_grad_(False)
        self.teacher_sigma = float(teacher_sigma)
        self.beta_start, self.beta_end, self.beta_iters = float(beta_start), float(beta_end), int(beta_iters)
        self.std_min, self.std_max = float(std_min), float(std_max)
        self.critic_only_iters = int(critic_only_iters)
        self.aux_group = aux_group
        self.updates = 0
        self.last_stats: dict[str, float] = {}
        self.clamp_std()

    # ------------------------------------------------------------------ helpers
    def beta(self) -> float:
        if self.teacher is None:
            return 0.0
        if self.beta_iters <= 0:
            return self.beta_end
        f = min(self.updates / self.beta_iters, 1.0)
        return self.beta_start + (self.beta_end - self.beta_start) * f

    def _std_param(self):
        pol = self.policy
        if getattr(pol, "noise_std_type", "scalar") == "log" and hasattr(pol, "log_std"):
            return pol.log_std, True
        return getattr(pol, "std", None), False

    @torch.no_grad()
    def clamp_std(self) -> None:
        prm, is_log = self._std_param()
        if prm is None:
            return
        if is_log:
            prm.clamp_(math.log(self.std_min), math.log(self.std_max))
        else:
            prm.clamp_(self.std_min, self.std_max)

    @torch.no_grad()
    def reset_std(self, value: float) -> None:
        prm, is_log = self._std_param()
        if prm is not None:
            prm.fill_(math.log(value) if is_log else value)
        self.clamp_std()

    def std_mean(self) -> float:
        prm, is_log = self._std_param()
        if prm is None:
            return float("nan")
        return float((prm.exp() if is_log else prm).mean())

    def actor_parameters(self):
        prm, _ = self._std_param()
        out = list(self.policy.actor.parameters())
        return out + ([prm] if prm is not None else [])

    def _weights(self, obs_batch, n: int):
        if self.aux_group in obs_batch.keys():
            aux = obs_batch[self.aux_group]
            return aux[:, 0].float(), aux[:, 1].float()
        one = torch.ones(n, device=self.device)
        return one, torch.zeros(n, device=self.device)

    @staticmethod
    def _wmean(x, w):
        return (x * w).sum() / w.sum().clamp(min=1e-8)

    def state_dict_extra(self) -> dict:
        return {"updates": self.updates}

    def load_state_dict_extra(self, d: dict | None) -> None:
        if d:
            self.updates = int(d.get("updates", 0))

    # ------------------------------------------------------------------ losses of one mini-batch
    def batch_loss(self, obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch,
                   old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, adapt_lr: bool):
        n = obs_batch.batch_size[0]
        learn_w, teach_w = self._weights(obs_batch, n)
        if self.normalize_advantage_per_mini_batch:
            with torch.no_grad():
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
        self.policy.act(obs_batch)
        actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
        value_batch = self.policy.evaluate(obs_batch)
        mu_batch = self.policy.action_mean
        sigma_batch = self.policy.action_std
        entropy_batch = self.policy.entropy

        if adapt_lr and self.desired_kl is not None and self.schedule == "adaptive":
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = self._wmean(kl, learn_w)
                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate

        adv = torch.squeeze(advantages_batch, -1) if advantages_batch.dim() > 1 else advantages_batch
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch, -1))
        surrogate = -adv * ratio
        surrogate_clipped = -adv * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        surrogate_loss = self._wmean(torch.max(surrogate, surrogate_clipped), learn_w)

        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = self._wmean(torch.max(value_losses, value_losses_clipped).squeeze(-1), learn_w)
        else:
            value_loss = self._wmean((returns_batch - value_batch).pow(2).squeeze(-1), learn_w)
        entropy = self._wmean(entropy_batch, learn_w)
        loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

        beta = self.beta()
        teach_loss = torch.zeros((), device=self.device)
        teach_mse = float("nan")
        n_teach = float(teach_w.sum())
        if self.teacher is not None and n_teach > 0:
            with torch.no_grad():
                mu_t = self.teacher(obs_batch["policy"]).to(mu_batch.dtype)
            d2 = ((mu_batch - mu_t) / self.teacher_sigma).pow(2)
            per = d2.sum(-1)
            teach_loss = self._wmean(per, teach_w)
            teach_mse = float(self._wmean(d2.mean(-1), teach_w).detach())
            if beta > 0:
                loss = loss + beta * teach_loss
        parts = dict(value_function=value_loss, surrogate=surrogate_loss, entropy=entropy, teach=teach_loss)
        return loss, parts, teach_mse, n_teach

    # ------------------------------------------------------------------ update
    def update(self) -> dict[str, float]:
        if self.rnd or self.symmetry or self.policy.is_recurrent:
            raise NotImplementedError("PPOExt supports plain feed-forward PPO only")
        critic_only = self.updates < self.critic_only_iters
        actor_params = self.actor_parameters() if critic_only else []
        sums = dict(value_function=0.0, surrogate=0.0, entropy=0.0, teach=0.0)
        mse_sum, mse_n, n_upd, n_samples, learn_sum = 0.0, 0.0, 0, 0, 0.0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch,
             old_mu_batch, old_sigma_batch, _hid, _masks) in generator:
            loss, parts, teach_mse, n_teach = self.batch_loss(
                obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch,
                old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, adapt_lr=not critic_only)
            self.optimizer.zero_grad()
            loss.backward()
            for prm in actor_params:
                prm.grad = None if prm.grad is None else torch.zeros_like(prm.grad)
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.clamp_std()
            for k, v in parts.items():
                sums[k] += float(v.detach())
            if n_teach > 0 and not math.isnan(teach_mse):
                mse_sum += teach_mse * n_teach
                mse_n += n_teach
            n_upd += 1
            n_samples += obs_batch.batch_size[0]
            learn_sum += float(self._weights(obs_batch, obs_batch.batch_size[0])[0].sum())
        self.storage.clear()
        self.last_stats = {
            "Teach/mse": mse_sum / mse_n if mse_n > 0 else float("nan"),
            "Teach/beta": self.beta(),
            "Teach/samples_share": mse_n / max(n_samples, 1),
            "Policy/learn_share": learn_sum / max(n_samples, 1),
            "Policy/std_mean": self.std_mean(),
            "Policy/critic_only": float(critic_only),
        }
        self.updates += 1
        return {k: v / max(n_upd, 1) for k, v in sums.items()}
