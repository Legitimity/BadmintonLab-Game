"""GPU check: cached-COM racket velocity == Isaac Lab body_link_lin_vel_w at every physics step.

    python scripts/tools/check_racket_vel.py --headless --num_envs 1024 --policy exports/p3_s1_108/policy.pt
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--policy", default=None)
parser.add_argument("--noise_std", type=float, default=0.28)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
args_cli.headless = True
app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import badmintonlab_game.tasks  # noqa: E402,F401
from badmintonlab_game.tasks.manager_based.badmintonlab_game.serve.wrapper import ServePolicyVecEnvWrapper  # noqa: E402


def main():
    task = "BadmintonLab-Game-Serve"
    env = ServePolicyVecEnvWrapper(gym.make(task, cfg=parse_env_cfg(task, num_envs=args_cli.num_envs)), clip_actions=None)
    u = env.unwrapped
    policy = torch.jit.load(args_cli.policy, map_location=u.device).eval() if args_cli.policy else None
    stats = {"max_abs": 0.0, "max_speed": 0.0, "n": 0}
    original = u._physics_substep

    def checked():
        original()
        sid = u._sweet_spot_ids
        quat = torch.stack([r.data.body_link_quat_w[:, i] for r, i in zip(u._robots, sid)], dim=1)
        fast = u._racket_link_lin_vel_w(quat)
        ref = torch.stack([r.data.body_link_lin_vel_w[:, i] for r, i in zip(u._robots, sid)], dim=1)
        stats["max_abs"] = max(stats["max_abs"], float((fast - ref).abs().max()))
        stats["max_speed"] = max(stats["max_speed"], float(torch.linalg.norm(ref, dim=-1).max()))
        stats["n"] += 1

    u._physics_substep = checked
    obs, _ = env.reset()
    for _ in range(args_cli.steps):
        with torch.inference_mode():
            a = policy(obs["policy"]) if policy is not None else torch.zeros(env.num_envs, env.num_actions, device=u.device)
            a = a + args_cli.noise_std * torch.randn_like(a)
        obs, *_ = env.step(a)
    print(f"[CHECK] physics steps={stats['n']} max |fast-ref|={stats['max_abs']:.3e} m/s, max racket speed={stats['max_speed']:.2f} m/s,"
          f" com offset={[c[0].tolist() for c in u._sweet_com_b]}", flush=True)
    ok = stats["max_abs"] < 1e-4
    print("[CHECK] PASS" if ok else "[CHECK] FAIL", flush=True)
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
