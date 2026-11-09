"""Diagnose why the receive expert never touches the shuttle in play (11-09).

Runs inside the play package (same setup as _eval_serve_inner.py): side 0 = our receive model, side 1 = a
serve model. For every rally where side 1 serves, logs per control step (first ``--steps`` steps):
  - side-0 observation (100-D): intercept block (last 5), base pos/heading,
  - shuttle position / velocity (env-local), side-1 just_hit, last_hitter, next_hitter,
  - side-0 root xy, sweet-spot position, distance shuttle <-> sweet spot,
  - action norm of side 0.
Writes a torch file and prints a short summary.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--play_root", required=True)
parser.add_argument("--model0", required=True)
parser.add_argument("--model1", required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import badmintonlab_game  # noqa: E402
import badmintonlab_game.tasks  # noqa: E402,F401

assert args.play_root.replace("\\", "/").lower() in badmintonlab_game.__file__.replace("\\", "/").lower()


def main():
    cfg = parse_env_cfg("BadmintonLab-Game-Play", num_envs=args.num_envs)
    cfg.seed = args.seed
    env = gym.make("BadmintonLab-Game-Play", cfg=cfg)
    obs, _ = env.reset()
    u = env.unwrapped
    dev = u.device
    p0 = torch.jit.load(args.model0, map_location=dev).eval()
    p1 = torch.jit.load(args.model1, map_location=dev).eval()
    r0 = u.scene["robot_0"]
    sh = u.scene["shuttle"]
    ss_id = r0.find_bodies("racket_sweet_spot")[0][0]
    org = u.scene.env_origins
    print("[DIAG] obs dims", obs["player_0"].shape, obs["player_1"].shape, flush=True)
    print("[DIAG] obs terms p0", u.observation_manager.active_terms["player_0"], flush=True)
    print("[DIAG] obs term dims p0", u.observation_manager.group_obs_term_dim["player_0"], flush=True)
    rec = {k: [] for k in ("obs0", "sh_pos", "sh_vel", "root0", "ss0", "hit1", "hit0", "last_hitter", "next_hitter", "resolved", "a0")}
    with torch.inference_mode():
        for k in range(args.steps):
            a0 = p0(obs["player_0"]).float()
            a1 = p1(obs["player_1"]).float()
            rec["obs0"].append(obs["player_0"].clone().cpu())
            obs, info = env.step(torch.cat((a0, a1), dim=-1))
            rec["sh_pos"].append((sh.data.root_pos_w - org).cpu())
            rec["sh_vel"].append(sh.data.root_lin_vel_w.clone().cpu())
            rec["root0"].append((r0.data.root_pos_w - org).cpu())
            rec["ss0"].append((r0.data.body_pos_w[:, ss_id] - org).cpu())
            rec["hit1"].append(u.just_hit[:, 1].clone().cpu())
            rec["hit0"].append(u.just_hit[:, 0].clone().cpu())
            rec["last_hitter"].append(u.last_hitter.clone().cpu())
            rec["next_hitter"].append(u.next_hitter.clone().cpu())
            rec["resolved"].append(info["rally_resolved"].clone().cpu())
            rec["a0"].append(a0.cpu())
    out = {k: torch.stack(v) for k, v in rec.items()}  # (T, N, ...)
    torch.save(out, args.out)
    o = out["obs0"]
    inter = o[..., -5:]
    valid = inter[..., 4]
    print("[DIAG] intercept valid fraction per step (first 80):", [round(float(x), 2) for x in valid.mean(1)[:80]], flush=True)
    print("[DIAG] next_hitter at t0:", out["next_hitter"][0].tolist(), flush=True)
    print("[DIAG] hit1 any per env:", out["hit1"].any(0).tolist(), flush=True)
    print("[DIAG] resolved first step per env:", [int(x) if x >= 0 else -1 for x in torch.where(out["resolved"].any(0), out["resolved"].float().argmax(0), torch.tensor(-1.0))], flush=True)
    d = (out["sh_pos"] - out["ss0"]).norm(dim=-1)
    print("[DIAG] min shuttle-sweetspot distance per env:", [round(float(x), 2) for x in d.min(0).values], flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import os
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    finally:
        app.close()
