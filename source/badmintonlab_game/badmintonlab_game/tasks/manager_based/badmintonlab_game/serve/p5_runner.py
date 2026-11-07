"""P5 runner: ``ServeRunner`` with ``PPOExt``, the P3 warm start, the P5 curriculum and the receive metrics.

No Isaac dependency (the environment is duck-typed: ``recv_reward``, ``set_level``, ``set_natural_share``,
``pop_verify``).
"""

from __future__ import annotations

import json
import os

import rsl_rl.runners.on_policy_runner as _opr
import torch

from .p5_curriculum import P5Curriculum
from .ppo_ext import PPOExt
from .runner import ServeRunner

_opr.PPOExt = PPOExt  # OnPolicyRunner resolves the algorithm class by name in its own module


def pad_state_dict(src: dict, dst: dict) -> tuple[dict, dict]:
    """P3 state into the P5 ``ActorCritic``: equal shapes copied; the first critic layer and the critic
    normaliser get the old input columns, new columns are zero weights and mean 0 / var 1 / std 1."""
    out, report = {}, {"copied": 0, "padded": [], "missing": [], "unused": sorted(set(src) - set(dst))}
    for k, v in dst.items():
        s = src.get(k)
        if s is None:
            out[k] = v
            report["missing"].append(k)
        elif s.shape == v.shape:
            out[k] = s.to(v.device, v.dtype)
            report["copied"] += 1
        elif k == "critic.0.weight" and s.shape[0] == v.shape[0] and s.shape[1] < v.shape[1]:
            t = torch.zeros_like(v)
            t[:, : s.shape[1]] = s
            out[k] = t
            report["padded"].append(f"{k} {tuple(s.shape)}->{tuple(v.shape)}")
        elif k.startswith("critic_obs_normalizer._") and s.dim() == 2 and s.shape[1] < v.shape[1]:
            t = v.clone()  # fresh buffers: mean 0, var 1, std 1
            t[:, : s.shape[1]] = s
            out[k] = t
            report["padded"].append(f"{k} {tuple(s.shape)}->{tuple(v.shape)}")
        else:
            raise ValueError(f"cannot map {k}: {tuple(s.shape)} -> {tuple(v.shape)}")
    return out, report


class P5Runner(ServeRunner):
    CONTROL_INTERVAL = 10

    def __init__(self, env, train_cfg, log_dir=None, device="cpu", time_budget_hours=None, ppo_ext: dict | None = None,
                 p5_curriculum: P5Curriculum | None = None, control_file: str | None = None):
        alg = train_cfg["algorithm"]
        alg["class_name"] = "PPOExt"
        alg.update(ppo_ext or {})
        self.p5_curriculum = p5_curriculum
        self.control_file = control_file
        super().__init__(env, train_cfg, log_dir=log_dir, device=device, time_budget_hours=time_budget_hours)
        self.apply_p5()

    # ------------------------------------------------------------------ curriculum -> environment
    def apply_p5(self) -> None:
        u, cur = self.env.unwrapped, self.p5_curriculum
        if cur is None:
            return
        u.set_level(cur.level)
        u.set_natural_share(cur.natural_share)
        u.recv_reward.set_weights(cur.weights())

    def _read_control(self, it: int) -> None:
        if not self.control_file or self.p5_curriculum is None or it % self.CONTROL_INTERVAL:
            return
        try:
            with open(self.control_file) as f:
                ctl = json.load(f)
        except (OSError, ValueError):
            return
        share = ctl.get("natural_share")
        if share is not None and self.p5_curriculum.set_natural_share(float(share)):
            print(f"[P5] natural share -> {self.p5_curriculum.natural_share:.2f} at iteration {it} ({self.control_file})",
                  flush=True)

    # ------------------------------------------------------------------ warm start
    def warm_start(self, path: str, std: float | None = 0.3, load_optimizer: bool = True, lr: float | None = None) -> dict:
        """P3 -> P5. With ``load_optimizer`` the Adam moments and the adapted learning rate of P3 are kept (a fresh
        Adam at the default 1e-3 makes sign-sized steps on every weight and wrecked the warm-started actor in the
        first pilot: receiver falls 0.02 -> 0.99 in 5 iterations). ``lr`` overrides the learning rate."""
        ck = torch.load(path, map_location=self.device, weights_only=False)
        state, report = pad_state_dict(ck["model_state_dict"], self.alg.policy.state_dict())
        self.alg.policy.load_state_dict(state)
        if load_optimizer and "optimizer_state_dict" in ck:
            report["optimizer"] = self._load_padded_optimizer(ck["optimizer_state_dict"])
        if lr is not None:
            self.alg.learning_rate = float(lr)
            for g in self.alg.optimizer.param_groups:
                g["lr"] = float(lr)
        if std is not None:
            self.alg.reset_std(std)
        report.update(source=path, source_iter=ck.get("iter"), std=self.alg.std_mean(), lr=self.alg.learning_rate)
        print(f"[P5] warm start from {path}: {report['copied']} tensors copied, padded {report['padded']}, "
              f"missing {report['missing']}, optimizer {report.get('optimizer')}, lr {report['lr']:.2e}, "
              f"std {report['std']:.3f}", flush=True)
        return report

    def _load_padded_optimizer(self, src: dict) -> str:
        opt = self.alg.optimizer
        params = [p for g in opt.param_groups for p in g["params"]]
        if len(src["param_groups"]) != len(opt.param_groups) or len(params) != sum(len(g["params"]) for g in src["param_groups"]):
            return "skipped (parameter count differs)"
        padded = 0
        for i, prm in enumerate(params):
            st = src["state"].get(i)
            if st is None:
                continue
            new = {}
            for k, v in st.items():
                if torch.is_tensor(v) and v.dim() > 0 and v.shape != prm.shape:
                    if v.dim() != prm.dim() or any(a > b for a, b in zip(v.shape, prm.shape)):
                        return f"skipped (state shape {tuple(v.shape)} vs {tuple(prm.shape)})"
                    t = torch.zeros_like(prm)
                    t[tuple(slice(0, s) for s in v.shape)] = v
                    new[k] = t
                    padded += 1
                else:
                    new[k] = v.to(self.device) if torch.is_tensor(v) else v
            src["state"][i] = new
        opt.load_state_dict(src)
        self.alg.learning_rate = float(opt.param_groups[0]["lr"])
        return f"loaded ({padded} moment tensors padded)"

    # ------------------------------------------------------------------ metrics
    def _serve_metrics(self, it: int) -> None:
        super()._serve_metrics(it)
        u = self.env.unwrapped
        rates = u.recv_reward.pop_iteration()
        values = {f"Recv/{k}": v for k, v in rates.items()}
        values.update(getattr(self.alg, "last_stats", {}))
        cur = self.p5_curriculum
        if cur is not None:
            self._read_control(it)
            u.recv_reward.set_weights(cur.update(it, rates))
            u.set_level(cur.level)
            u.set_natural_share(cur.natural_share)
            values.update(cur.log_values())
            for event in cur.events:
                print(f"[P5Curriculum] {event} at iteration {it}: level={cur.level} ema="
                      f"{ {k: round(v, 3) for k, v in cur.ema.items() if k.startswith('f_')} }", flush=True)
                if self.log_dir is not None and event.startswith("passed_"):
                    self.save(os.path.join(self.log_dir, f"model_{event}_{it}.pt"))
        values.update({f"Recv/weight_{k}": v for k, v in u.recv_reward.weights.items()})
        ver = u.pop_verify()
        for i in range(len(ver) // 3):
            n = ver[f"L{i}_n"]
            if n > 0:
                values[f"Feeder/L{i}_n"] = n
                values[f"Feeder/L{i}_ok"] = ver[f"L{i}_ok"] / n
                values[f"Feeder/L{i}_err_mean"] = ver[f"L{i}_err_sum"] / n
        self._verify_acc = getattr(self, "_verify_acc", {})
        for k, v in ver.items():
            self._verify_acc[k] = self._verify_acc.get(k, 0.0) + v
        if self.writer is not None:
            for key, value in values.items():
                self.writer.add_scalar(key, value, it)
        keys = ("f_rallies", "f_touch", "f_crossed", "f_in", "f_fell", "f_tunnel", "f_need_mean", "n_rallies", "n_touch",
                "n_crossed", "n_in", "vc_mean", "approach_mean", "timeout")
        teach = self.alg.last_stats if hasattr(self.alg, "last_stats") else {}
        print("[Recv] " + " ".join(f"{k}={rates[k]:.3f}" for k in keys if k in rates)
              + f" | L={u.feeder_level} nat={u.natural_share:.2f}"
              + f" | mse={teach.get('Teach/mse', float('nan')):.2e} beta={teach.get('Teach/beta', 0):.3f}"
              + f" std={teach.get('Policy/std_mean', float('nan')):.3f} co={int(teach.get('Policy/critic_only', 0))}"
              + f" learn={teach.get('Policy/learn_share', float('nan')):.3f}"
              + f" teach={teach.get('Teach/samples_share', float('nan')):.3f}",
              flush=True)

    def verify_summary(self) -> dict:
        acc = getattr(self, "_verify_acc", {})
        out = {}
        for i in range(4):
            n = acc.get(f"L{i}_n", 0.0)
            out[f"L{i}"] = dict(n=n, ok=acc.get(f"L{i}_ok", 0.0) / n if n else None,
                                err_mean=acc.get(f"L{i}_err_sum", 0.0) / n if n else None)
        return out

    # ------------------------------------------------------------------ checkpoint
    def _infos(self) -> dict:
        infos = super()._infos()
        infos["p5"] = {"curriculum": None if self.p5_curriculum is None else self.p5_curriculum.state_dict(),
                       "ppo": self.alg.state_dict_extra()}
        return infos

    def load(self, path, load_optimizer=True, map_location=None):
        infos = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
        p5 = (infos or {}).get("p5") or {}
        self.alg.load_state_dict_extra(p5.get("ppo"))
        if self.p5_curriculum is not None and p5.get("curriculum"):
            self.p5_curriculum.load_state_dict(p5["curriculum"])
        self.apply_p5()
        return infos
