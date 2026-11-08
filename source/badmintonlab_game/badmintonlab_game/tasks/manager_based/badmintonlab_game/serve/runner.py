"""OnPolicyRunner with atomic checkpoints, a wall-clock budget and graceful stop. No Isaac dependency."""

from __future__ import annotations

import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner


class StopTraining(Exception):
    """Raised from the logging hook after an iteration has completed."""


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


class ServeRunner(OnPolicyRunner):
    LATEST_INTERVAL = 50
    FINITE_CHECK_INTERVAL = 50

    def __init__(self, env, train_cfg, log_dir=None, device="cpu", time_budget_hours=None, curriculum=None,
                 stop_after_stage=None, std_range=None):
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self.std_range = std_range
        self.time_budget_s = None if time_budget_hours is None else time_budget_hours * 3600.0
        self.curriculum = curriculum
        self.stop_after_stage = stop_after_stage
        self.start_wall = time.time()
        self.iter_times: list[float] = []
        self.stop_reason: str | None = None
        self.samples = 0
        self.nan_rollbacks = 0
        self._stop_signal = False
        self._last_iter_end = None
        computer = getattr(env.unwrapped, "serve_reward", None)
        if curriculum is not None and computer is not None:
            names = {k for overrides in curriculum.stage_params.values() for k in overrides}
            curriculum.base_params = {k: getattr(computer.p, k) for k in names}
        self.apply_stage_params()

    def apply_stage_params(self) -> None:
        """Puts the curriculum's per-stage reward parameters into the reward computer."""
        computer = getattr(self.env.unwrapped, "serve_reward", None)
        if self.curriculum is not None and computer is not None:
            for name, value in self.curriculum.params().items():
                setattr(computer.p, name, value)

    def install_signal_handlers(self) -> None:
        def handler(signum, _frame):
            # Only set a flag: the checkpoint is written after the current iteration.
            print(f"[ServeRunner] signal {signum}: stopping after this iteration", flush=True)
            self._stop_signal = True

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _serve_metrics(self, it: int) -> None:
        """Serve rates of this iteration, the curriculum step and the weight update."""
        unwrapped = self.env.unwrapped
        computer = getattr(unwrapped, "serve_reward", None)
        if computer is None:
            return
        rates = computer.pop_iteration()
        values = {f"Serve/{k}": v for k, v in rates.items()}
        if self.curriculum is not None:
            computer.set_weights(self.curriculum.update(it, rates))
            self.apply_stage_params()
            values.update(self.curriculum.log_values())
            for event in self.curriculum.events:
                print(f"[Curriculum] {event} at iteration {it}: {self.curriculum.history[-1] if self.curriculum.history else ''}", flush=True)
                if self.log_dir is not None:
                    self.save(os.path.join(self.log_dir, f"model_{event}_{it}.pt"))
        values.update({f"Serve/weight_{k}": v for k, v in computer.weights.items()})
        param_names = ["v_close_max"]
        if self.curriculum is not None:
            param_names += [k for k in self.curriculum.base_params if k != "v_close_max"]
        params = {f"param_{k}": float(getattr(computer.p, k)) for k in param_names}
        values.update({f"Serve/{k}": v for k, v in params.items()})
        if self.writer is not None:
            for key, value in values.items():
                self.writer.add_scalar(key, value, it)
        shown = {k: rates[k] for k in ("legal", "crossed", "in", "fault", "no_hit", "tunnel", "fall", "fall_after_hit", "vc_mean", "el_mean", "launch_el_mean", "launch_v_mean", "pred_in", "land_x_mean", "land_ay_mean", "tof_mean") if k in rates}
        print("[Serve] " + " ".join(f"{k}={v:.3f}" for k, v in {**shown, **params}.items()), flush=True)

    def clamp_std(self) -> None:
        """Keeps the action noise std in ``std_range`` (scalar or log parameterisation)."""
        if self.std_range is None:
            return
        lo, hi = self.std_range
        policy = self.alg.policy
        with torch.no_grad():
            if getattr(policy, "std", None) is not None and isinstance(policy.std, torch.nn.Parameter):
                policy.std.clamp_(lo, hi)
            elif getattr(policy, "log_std", None) is not None:
                policy.log_std.clamp_(math.log(lo), math.log(hi))

    def log(self, locs, *args, **kwargs):
        # Called after the update of this iteration and before the next rollout.
        self.clamp_std()
        super().log(locs, *args, **kwargs)
        it = locs["it"]
        self._serve_metrics(it)
        self.samples += self.num_steps_per_env * self.env.num_envs
        now = time.time()
        if self._last_iter_end is not None:
            self.iter_times.append(now - self._last_iter_end)
        self._last_iter_end = now
        if (it + 1) % self.FINITE_CHECK_INTERVAL == 0 and not self._finite(locs):
            self.stop_reason = "nonfinite"
            raise StopTraining
        if (it + 1) % self.LATEST_INTERVAL == 0:
            self.save_latest()
        if self._stop_signal:
            self.stop_reason = "signal"
            raise StopTraining
        if self.curriculum is not None and self.curriculum.stop_reason:
            # A stage hit its maximum without passing its gate: stop and investigate (V3 7.3).
            self.stop_reason = self.curriculum.stop_reason
            raise StopTraining
        if self.curriculum is not None and self.stop_after_stage and any(
                h["stage"] == self.stop_after_stage for h in self.curriculum.history):
            # model_passed_<stage>_<it>.pt was written above and holds the next stage's curriculum state.
            self.stop_reason = f"stop_after_{self.stop_after_stage}"
            raise StopTraining
        if self.time_budget_s is not None and self._over_budget(now):
            self.stop_reason = "time_budget"
            raise StopTraining

    def _over_budget(self, now: float) -> bool:
        recent = self.iter_times[-20:]
        mean = sum(recent) / len(recent) if recent else 0.0
        return now - self.start_wall + 1.2 * mean + 5.0 > self.time_budget_s

    def _finite(self, locs) -> bool:
        ok = all(torch.isfinite(p).all() for p in self.alg.policy.parameters())
        for value in locs.get("loss_dict", {}).values():
            ok = ok and np.isfinite(value)
        return bool(ok)

    def _infos(self) -> dict:
        return {
            "samples": self.samples,
            "learning_rate": self.alg.learning_rate,
            "curriculum": None if self.curriculum is None else self.curriculum.state_dict(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }

    def save(self, path, infos=None):
        """Atomic: a crash during the write cannot damage an existing checkpoint."""
        tmp = f"{path}.tmp"
        super().save(tmp, infos=self._infos() if infos is None else infos)
        os.replace(tmp, path)

    def save_latest(self) -> None:
        self.save(os.path.join(self.log_dir, "latest.pt"))

    def load(self, path, load_optimizer=True, map_location=None):
        infos = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
        if infos:
            self.samples = infos.get("samples", 0)
            if "learning_rate" in infos:
                self.alg.learning_rate = infos["learning_rate"]
                for group in self.alg.optimizer.param_groups:
                    group["lr"] = infos["learning_rate"]
            if self.curriculum is not None and infos.get("curriculum") is not None:
                self.curriculum.load_state_dict(infos["curriculum"])
                computer = getattr(self.env.unwrapped, "serve_reward", None)
                if computer is not None:
                    computer.set_weights(self.curriculum.weights())
                self.apply_stage_params()
            rng = infos.get("rng")
            if rng:
                random.setstate(rng["python"])
                np.random.set_state(rng["numpy"])
                torch.set_rng_state(rng["torch"].cpu())
                if rng.get("cuda") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
        return infos

    def learn_until_stop(self, num_iterations: int, init_at_random_ep_len: bool = False) -> str:
        """Runs ``learn``, then always leaves a final checkpoint. Returns the stop reason."""
        try:
            self.learn(num_iterations, init_at_random_ep_len=init_at_random_ep_len)
            self.stop_reason = self.stop_reason or "max_iterations"
        except StopTraining:
            pass
        if self.log_dir is not None and self.stop_reason != "nonfinite":
            self.save_latest()
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
        return self.stop_reason


def write_run_meta(log_dir: str, repo: Path, params: dict) -> Path:
    """Records what is needed to reproduce a run, including the uncommitted diff."""
    out = Path(log_dir)
    out.mkdir(parents=True, exist_ok=True)
    diff = _git(repo, "diff", "HEAD")
    (out / "git.diff").write_text(diff, encoding="utf-8")
    meta = {
        "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "commit": _git(repo, "rev-parse", "HEAD"),
        "dirty": bool(diff),
        "command": sys.argv,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": params,
    }
    path = out / "run_meta.json"
    path.write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    return path


def finish_run_meta(path: Path, runner: ServeRunner) -> None:
    meta = json.loads(path.read_text(encoding="utf-8"))
    times = runner.iter_times[20:] or runner.iter_times
    meta.update(
        stop_reason=runner.stop_reason,
        iterations=runner.current_learning_iteration + 1,
        player_samples=runner.samples,
        elapsed_seconds=time.time() - runner.start_wall,
        steady_iter_seconds=(sum(times) / len(times)) if times else None,
        peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
    )
    path.write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")


def find_latest(log_root: str, run_name: str) -> str | None:
    """Most recent ``latest.pt`` among the runs with this name."""
    runs = sorted(Path(log_root).glob(f"*_{run_name}"))
    for run in reversed(runs):
        latest = run / "latest.pt"
        if latest.exists():
            return str(latest)
    return None
