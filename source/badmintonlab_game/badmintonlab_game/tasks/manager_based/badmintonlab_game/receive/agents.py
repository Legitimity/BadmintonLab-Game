"""Single-learner wrapper (robot 0), PPO config, and the receive runner."""

from __future__ import annotations

import math
import os

import torch
from isaaclab.utils import configclass
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from ..agents.rsl_rl_ppo_cfg import PPORunnerCfg
from ..serve.runner import ServeRunner, StopTraining
from . import core


class ReceiveVecEnvWrapper(VecEnv):
    """Exposes robot 0 of every court; robot 1 (parked) receives a fixed hold action."""

    def __init__(self, env, hold_action_1: torch.Tensor, clip_actions: float | None = None):
        self.env = env
        self.clip_actions = clip_actions
        self.device = self.unwrapped.device
        self.num_envs = self.unwrapped.num_envs
        self.num_actions = self.unwrapped.single_agent_action_dim
        self.max_episode_length = self.unwrapped.max_episode_length
        self.hold_1 = hold_action_1.to(self.device).expand(self.num_envs, -1).contiguous()
        self.env.reset()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def cfg(self):
        return self.unwrapped.cfg

    @property
    def episode_length_buf(self):
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.unwrapped.episode_length_buf = value

    def seed(self, seed: int = -1):
        return self.unwrapped.seed(seed)

    def _obs(self, obs: dict) -> TensorDict:
        return TensorDict({"policy": obs["player_0"], "critic": obs["critic_0"]}, batch_size=[self.num_envs])

    def get_observations(self):
        return self._obs(self.unwrapped.observation_manager.compute())

    def reset(self):
        obs, extras = self.env.reset()
        return self._obs(obs), extras

    def step(self, actions: torch.Tensor):
        if self.clip_actions is not None:
            actions = actions.clamp(-self.clip_actions, self.clip_actions)
        obs, _, terminated, truncated, extras = self.env.step(torch.cat((actions, self.hold_1), dim=-1))
        rewards = self.unwrapped.agent_reward_buf[:, 0]
        dones = (terminated | truncated).long()
        extras["time_outs"] = truncated
        return self._obs(obs), rewards, dones, extras

    def close(self):
        return self.env.close()


@configclass
class ReceivePPORunnerCfg(PPORunnerCfg):
    experiment_name = "badmintonlab_receive"
    save_interval = 250
    obs_groups = {"policy": ["policy"], "critic": ["policy", "critic"]}

    def __post_init__(self):
        self.policy.init_noise_std = 0.3
        self.algorithm.entropy_coef = 0.005
        self.algorithm.num_mini_batches = 8
        self.algorithm.learning_rate = 1.0e-4


# Normaliser statistics for the 5 intercept inputs (frozen): rel pos (3), time, valid.
EXTRA_MEAN = torch.tensor((0.6, -0.4, 0.3, 0.6, 0.6))
EXTRA_STD = torch.tensor((0.8, 0.8, 0.3, 0.5, 0.5))


class ReceiveRunner(ServeRunner):
    """Adds the S1 metrics, the feeder-level curriculum, per-group std clamping and a critic warm-up."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu", time_budget_hours=None, curriculum=None,
                 leg_mask=None, critic_warmup=50):
        super().__init__(env, train_cfg, log_dir=log_dir, device=device, time_budget_hours=time_budget_hours)
        self.recv_curriculum = curriculum
        self.leg_mask = leg_mask.to(self.device)
        self.critic_warmup = critic_warmup
        self._actor_snapshot = None

    def snapshot_actor(self) -> None:
        """Critic-only warm-up for ``critic_warmup`` iterations counted from now (also after loading S1)."""
        self._warmup_from = int(self.current_learning_iteration)
        pol = self.alg.policy
        self._actor_snapshot = (
            {k: v.detach().clone() for k, v in pol.actor.state_dict().items()},
            pol.log_std.detach().clone(),
        )

    @torch.no_grad()
    def _restore_actor(self) -> None:
        pol = self.alg.policy
        pol.actor.load_state_dict(self._actor_snapshot[0])
        pol.log_std.copy_(self._actor_snapshot[1])

    def log(self, locs, *args, **kwargs):
        it = locs["it"]
        pol = self.alg.policy
        if self._actor_snapshot is not None and self.critic_warmup > 0 and it < getattr(self, "_warmup_from", 0) + self.critic_warmup:
            self._restore_actor()  # critic-only warm-up: the actor (and its noise) stay at p3_final
        else:
            core.clamp_log_std(pol.log_std.data, self.leg_mask)
        unwrapped = self.env.unwrapped
        rates = unwrapped.pop_stats()
        cur = self.recv_curriculum
        if cur is not None:
            cur.update(it, rates)
            if cur.level != unwrapped.level:
                unwrapped.set_level(cur.level)
            self.apply_recv_params()
            for event in cur.events:
                print(f"[Curriculum] {event} at iteration {it}", flush=True)
                if self.log_dir is not None:
                    self.save(os.path.join(self.log_dir, f"model_{event}_{it}.pt"))
        std = pol.log_std.exp()
        values = {f"Recv/{k}": v for k, v in rates.items()}
        values["Recv/std_leg"] = float(std[self.leg_mask].mean())
        values["Recv/std_other"] = float(std[~self.leg_mask].mean())
        if cur is not None:
            values.update(cur.log_values())
        if self.writer is not None:
            for k, v in values.items():
                if isinstance(v, float) and math.isnan(v):
                    continue
                self.writer.add_scalar(k, v, it)
        if getattr(cur, "stage", "S1") == "S2":
            p = cur.params()
            print(
                f"[S2] it={cur.iters_in_stage} sp={p['sigma_p']:.3f} sr={p['sigma_r']:.2f} "
                + " ".join(f"{k}={rates[k]:.3f}" for k in ("hit", "ret_in", "ret_net", "ret_out", "epos_p50", "eori_deg", "vn_mean", "swing_mean"))
                + " | " + " ".join(f"L{i}={rates[f'L{i}_hit']:.2f}" for i in range(4)),
                flush=True,
            )
        src = [k[4:-4] for k in rates if k.startswith("src_") and k.endswith("_hit")]
        if len(src) > 1:
            print("[SRC] " + " ".join(f"{s}:hit={rates[f'src_{s}_hit']:.2f}/st={rates[f'src_{s}_station_ok']:.2f}/n={rates[f'src_{s}_n']:.0f}" for s in src), flush=True)
        print(
            "[Recv] lvl={} ".format(unwrapped.level)
            + " ".join(f"{k}={rates[k]:.3f}" for k in ("station_ok", "face_ok", "d_mean", "d_p50", "head_deg", "v_to_station", "fall", "touch"))
            + f" eps={rates['episodes']:.0f} low={rates['fall_low']:.0f} tilt={rates['fall_tilt']:.0f} fall_t={rates['fall_t']:.2f} to={rates['timeouts']:.0f} balls={rates['balls']:.0f} std_leg={values['Recv/std_leg']:.3f}",
            flush=True,
        )
        super().log(locs, *args, **kwargs)
        if cur is not None and cur.passed and getattr(self, "stop_on_pass", True):
            self.stop_reason = f"passed_{getattr(cur, 'stage', 'S1')}"
            raise StopTraining
        if cur is not None and cur.stop_reason:
            self.stop_reason = cur.stop_reason
            raise StopTraining

    def apply_recv_params(self) -> None:
        """S2: push the blended weights and annealed tolerances into the env (S1 uses the env defaults)."""
        cur = self.recv_curriculum
        if cur is None or getattr(cur, "stage", "S1") != "S2":
            return
        p = cur.params()
        u = self.env.unwrapped
        u.set_stage("S2")
        u.set_stage_params(p["weights"], p["sigma_p"], p["sigma_r"])

    def _infos(self) -> dict:
        out = super()._infos()
        out["recv_curriculum"] = None if self.recv_curriculum is None else self.recv_curriculum.state_dict()
        return out

    def load(self, path, load_optimizer=True, map_location=None):
        infos = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
        cur = self.recv_curriculum
        state = (infos or {}).get("recv_curriculum")
        if state and cur is not None:
            # Starting S2 from an S1 checkpoint: keep the fresh S2 curriculum (and its uniform feeds).
            if state.get("stage", "S1") == getattr(cur, "stage", "S1"):
                cur.load_state_dict(state)
            else:
                print(f"[INFO] checkpoint curriculum stage {state.get('stage', 'S1')} != {cur.stage}: starting {cur.stage} fresh", flush=True)
        if cur is not None:
            self.env.unwrapped.set_level(cur.level)
            self.apply_recv_params()
        return infos
