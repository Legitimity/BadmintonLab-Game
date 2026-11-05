"""Inner process for ``eval_serve.py``: runs inside the play package, reads only play's info and state."""

import argparse
import json
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--play_root", required=True)
parser.add_argument("--model0", required=True)
parser.add_argument("--model1", required=True)
parser.add_argument("--ours", type=int, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--rallies", type=int, default=200)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--watchdog_s", type=float, default=60.0)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import badmintonlab_game  # noqa: E402
import badmintonlab_game.tasks  # noqa: E402,F401

assert args.play_root.replace("\\", "/").lower() in badmintonlab_game.__file__.replace("\\", "/").lower(), (
    f"not the play package: {badmintonlab_game.__file__}"
)


def _load(path, device):
    if path == "zero":
        return lambda obs: torch.zeros(obs.shape[0], 23, device=device)
    return torch.jit.load(path, map_location=device).eval()


def main():
    task = "BadmintonLab-Game-Play"
    cfg = parse_env_cfg(task, num_envs=args.num_envs)
    cfg.seed = args.seed
    env = gym.make(task, cfg=cfg)
    obs, _ = env.reset()
    u = env.unwrapped
    device = u.device
    pols = [_load(args.model0, device), _load(args.model1, device)]
    me, other = args.ours, 1 - args.ours
    n = u.num_envs
    keys = ("rallies", "legal", "served_in", "height", "double", "no_hit", "net", "out", "fell", "watchdog", "hits_ours")
    stats = {"serve": {k: 0 for k in keys}, "receive": {k: 0 for k in keys}}
    dims = [int(obs["player_0"].shape[-1]), int(obs["player_1"].shape[-1])]
    server = u.next_hitter.clone()
    ours_hit = torch.zeros(n, dtype=torch.bool, device=device)
    ours_hits = torch.zeros(n, dtype=torch.long, device=device)
    fell = torch.zeros(n, dtype=torch.bool, device=device)
    started = torch.zeros(n, device=device)
    sim_t = 0.0
    total = 0
    robot = u.scene[f"robot_{me}"]
    wall = time.time()
    while total < args.rallies * n:
        with torch.inference_mode():
            a0 = pols[0](obs["player_0"]).to(device=device, dtype=torch.float32)
            a1 = pols[1](obs["player_1"]).to(device=device, dtype=torch.float32)
            obs, info = env.step(torch.cat((a0, a1), dim=-1))
        sim_t += u.step_dt
        new_hit = u.just_hit[:, me]
        ours_hit |= new_hit
        ours_hits += new_hit.long()
        fell |= robot.data.root_pos_w[:, 2] < 0.45
        stuck = (sim_t - started) > args.watchdog_s
        done = info["rally_resolved"] | stuck
        if done.any():
            for i in done.nonzero().flatten().tolist():
                role = "serve" if int(server[i]) == me else "receive"
                s = stats[role]
                s["rallies"] += 1
                total += 1
                if stuck[i] and not info["rally_resolved"][i]:
                    s["watchdog"] += 1
                    continue
                won = int(info["rally_winner"][i]) == me
                s["height"] += int(info["serve_height_fault"][i] and role == "serve" and not won)
                s["double"] += int(info["double_hit_fault"][i] and not won)
                s["legal"] += int(ours_hit[i] and not (info["serve_height_fault"][i] and not won))
                s["no_hit"] += int(not ours_hit[i])
                s["served_in"] += int(won and bool(ours_hit[i]))
                s["fell"] += int(fell[i])
                s["hits_ours"] += int(ours_hits[i])
            # Rally state of the resolved courts was reset by play; capture the new server.
            server[done] = u.next_hitter[done]
            ours_hit[done] = False
            ours_hits[done] = 0
            fell[done] = False
            started[done] = sim_t
    for role, s in stats.items():
        r = max(s["rallies"], 1)
        s["rates"] = {k: s[k] / r for k in keys if k != "rallies"}
    out = dict(obs_dims=dims, num_envs=n, seed=args.seed, wall_s=time.time() - wall, **stats)
    print("[SERVE_EVAL]" + json.dumps(out), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
