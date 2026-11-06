"""Time breakdown of one control step of the Serve task (rollout side of a PPO iteration).

Wraps the env's step components with CUDA-synchronised timers. Two passes over the same number
of steps: an uninstrumented pass for the true wall time per step, then the instrumented pass for
the distribution. Actions come from an exported policy so rallies behave like in training.

    python scripts/tools/profile_serve_step.py --headless --num_envs 8192 --policy exports/p3_s1_108/policy.pt
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=8192)
parser.add_argument("--steps", type=int, default=96, help="Control steps per pass (24 = one PPO iteration).")
parser.add_argument("--warmup", type=int, default=48)
parser.add_argument("--policy", default=None, help="TorchScript policy (95-dim obs). Default: zero actions.")
parser.add_argument("--task", default="BadmintonLab-Game-Serve")
parser.add_argument("--noise_std", type=float, default=0.0, help="Gaussian action noise, like the stochastic training policy.")
parser.add_argument("--out", default=None, help="JSON output path.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
args_cli.headless = True
app = AppLauncher(args_cli).app

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import badmintonlab_game.tasks  # noqa: E402,F401
from badmintonlab_game.tasks.manager_based.badmintonlab_game.serve.wrapper import ServePolicyVecEnvWrapper  # noqa: E402

TIMES = defaultdict(float)
CALLS = defaultdict(int)
ENABLED = [False]


def timed(name, fn):
    def wrapper(*a, **kw):
        if not ENABLED[0]:
            return fn(*a, **kw)
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn(*a, **kw)
        torch.cuda.synchronize()
        TIMES[name] += time.perf_counter() - t
        CALLS[name] += 1
        return out

    return wrapper


def patch(obj, attr, name):
    setattr(obj, attr, timed(name, getattr(obj, attr)))


def main():
    cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env = ServePolicyVecEnvWrapper(gym.make(args_cli.task, cfg=cfg), clip_actions=None)
    u = env.unwrapped
    policy = torch.jit.load(args_cli.policy, map_location=u.device).eval() if args_cli.policy else None

    def act(obs):
        with torch.inference_mode():
            if policy is None:
                return torch.zeros(env.num_envs, env.num_actions, device=u.device)
            a = policy(obs["policy"])
            return a + args_cli.noise_std * torch.randn_like(a) if args_cli.noise_std > 0 else a

    # Top level (inside env.step)
    patch(u.action_manager, "process_action", "1 process_action")
    patch(u.rules, "begin_control_step", "1 rules.begin_control_step")
    patch(u.action_manager, "apply_action", "2 substep/apply_action")
    patch(u, "_apply_shuttle_forces", "2 substep/shuttle_forces")
    patch(u.scene, "write_data_to_sim", "2 substep/write_data_to_sim (+reset)")
    patch(u.sim, "step", "2 substep/sim.step (PhysX)")
    for name, asset in list(u.scene.articulations.items()) + list(u.scene.rigid_objects.items()):
        patch(asset, "write_data_to_sim", f"6 write/{name}")
    for name, asset in u.scene.articulations.items():
        patch(asset, "_apply_actuator_model", f"6 write/{name}/_apply_actuator_model")
    patch(u.scene, "update", "2 substep/scene.update")
    # Line-level copy of ServeBadmintonEnv._physics_substep (keep in sync with serve_env.py).
    def section(name):
        class _S:
            def __enter__(self):
                if ENABLED[0]:
                    torch.cuda.synchronize()
                self.t = time.perf_counter()

            def __exit__(self, *exc):
                if ENABLED[0]:
                    torch.cuda.synchronize()
                    TIMES[name] += time.perf_counter() - self.t
                    CALLS[name] += 1

        return _S()

    def physics_substep_lines():
        self = u
        with section("3 phys/a shuttle pos + env_origins"):
            pos_w = self._shuttle.data.root_pos_w
            local_z = pos_w[:, 2] - self.scene.env_origins[:, 2]
        self.rules.detect_landing(local_z, self.cfg.shuttle_landing_detection_height)
        with section("3 phys/b contact force_matrix_w -> contacts"):
            contacts = torch.stack(
                [torch.linalg.norm(c.data.force_matrix_w, dim=-1).flatten(1).amax(dim=1) > self.cfg.hit_contact_threshold
                 for c in self._contacts], dim=1)
        before = self.rules.just_hit.clone()
        self.rules.update_hits(contacts, local_z)
        sid = self._sweet_spot_ids
        with section("3 phys/c body_link_quat_w (racket)"):
            new_hits = self.rules.just_hit & ~before
            quat = torch.stack([r.data.body_link_quat_w[:, i] for r, i in zip(self._robots, sid)], dim=1)
        with section("3 phys/d racket velocity (cached COM)"):
            vel = self._racket_link_lin_vel_w(quat)
        self.serve_reward.capture_contact(new_hits, self.initial_hitter, quat, vel)
        with section("3 phys/e body_link_pos_w (racket)"):
            racket_pos = torch.stack([r.data.body_link_pos_w[:, i] for r, i in zip(self._robots, sid)], dim=1)
        self.serve_reward.capture_tunnel(self.initial_hitter, racket_pos, quat, pos_w, self._prev_phys_shuttle, contacts)
        self._prev_phys_shuttle.copy_(pos_w)
        self.rules.update_rally_result(pos_w, self.scene.env_origins)

    u._physics_substep = physics_substep_lines
    patch(u, "_physics_substep", "2 substep/_physics_substep (rules+reward)")
    # Inside _physics_substep
    patch(u.rules, "detect_landing", "3 phys/rules.detect_landing")
    patch(u.rules, "update_hits", "3 phys/rules.update_hits")
    patch(u.serve_reward, "capture_contact", "3 phys/serve_reward.capture_contact")
    patch(u.serve_reward, "capture_tunnel", "3 phys/serve_reward.capture_tunnel")
    patch(u.rules, "update_rally_result", "3 phys/rules.update_rally_result")
    # After the substeps
    patch(u, "_finish_control_step", "4 post/_finish_control_step")
    patch(u, "_update_return_history", "4 post/_update_return_history")
    patch(u, "_step_inputs", "4 post/_step_inputs")
    patch(u.serve_reward, "end_step", "4 post/serve_reward.end_step")
    from badmintonlab_game.tasks.manager_based.badmintonlab_game.serve import serve_reward as _sr

    patch(_sr, "predict_flight", "5 end_step/predict_flight")
    patch(u.termination_manager, "compute", "4 post/termination_manager")
    patch(u.serve_reward, "count_timeouts", "4 post/serve_reward.count_timeouts")
    patch(u.reward_manager, "compute", "4 post/reward_manager p0")
    patch(u.player_1_reward_manager, "compute", "4 post/reward_manager p1")
    patch(u, "_reset_idx", "4 post/_reset_idx")
    patch(u.sim, "forward", "4 post/sim.forward (after reset)")
    patch(u.observation_manager, "compute", "4 post/observation_manager")

    obs, _ = env.reset()
    for _ in range(args_cli.warmup):
        obs, *_ = env.step(act(obs))

    # Pass 1: uninstrumented wall time.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args_cli.steps):
        obs, *_ = env.step(act(obs))
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / args_cli.steps

    # Pass 2: instrumented.
    ENABLED[0] = True
    t_pol = t_step = 0.0
    for _ in range(args_cli.steps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        a = act(obs)
        torch.cuda.synchronize()
        t_pol += time.perf_counter() - t
        t = time.perf_counter()
        obs, *_ = env.step(a)
        torch.cuda.synchronize()
        t_step += time.perf_counter() - t
    ENABLED[0] = False
    n = args_cli.steps
    rows = {k: (v / n * 1000.0, CALLS[k] / n) for k, v in TIMES.items()}
    rows["0 policy inference"] = (t_pol / n * 1000.0, 1.0)
    step_ms = t_step / n * 1000.0
    # Nested timers: "write_data_to_sim" is also called once after a reset; sim.forward only after a reset.
    accounted = sum(v for k, (v, _) in rows.items() if k[0] in "124" and k != "0 policy inference")
    rows["9 env.step other (wrapper, extras, buffers)"] = (step_ms - accounted, 1.0)

    print(f"\n== {args_cli.task} num_envs={args_cli.num_envs} policy={'zero' if policy is None else args_cli.policy} noise={args_cli.noise_std}")
    print(f"uninstrumented: {wall * 1000:.1f} ms/step ({wall * 24:.2f} s per 24-step rollout)")
    print(f"instrumented:   {(t_pol + t_step) / n * 1000:.1f} ms/step (policy {t_pol / n * 1000:.1f} + env.step {step_ms:.1f})")
    print(f"{'section':58s} {'ms/step':>9s} {'calls':>6s} {'% env.step':>10s}")
    for k in sorted(rows):
        ms, c = rows[k]
        nested = k.startswith(("3 ", "5 ", "6 "))
        print(f"{('    ' if nested else '') + k:58s} {ms:9.2f} {c:6.2f} {100 * ms / step_ms:9.1f}%")
    if args_cli.out:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
        json.dump(dict(task=args_cli.task, num_envs=args_cli.num_envs, wall_ms=wall * 1000, step_ms=step_ms,
                       policy_ms=t_pol / n * 1000, rows=rows), open(args_cli.out, "w"), indent=1)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)
