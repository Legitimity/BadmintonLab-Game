"""Record real serves in the ``play`` package (P5 V3.1 7.1). Read-only: play's rules are untouched.

Side 1 serves with ``--server_model``; side 0 holds the ready pose (zero action). For every legal serve
hit by side 1 we store the shuttle position/velocity one control step after the hit, in the receiver's
(side 0) env-local frame, which is the frame of the receive feeder table. Rallies where side 0 serves
are skipped (side 0 does nothing; the rally ends by itself).

Run inside play (``eval_serve.py``-style cfg install is done by the caller script)::

    python scripts/tools/record_serves.py --headless --play_root D:/rl-badminton-eval/BadmintonLab-Game-Play \
        --server_model D:/.../policy.pt --num_envs 256 --serves 2000 --out D:/.../serves_s0.pt
"""

import argparse
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--play_root", required=True)
parser.add_argument("--server_model", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--serves", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_sim_s", type=float, default=600.0)
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

assert args.play_root.replace("\\", "/").lower() in badmintonlab_game.__file__.replace("\\", "/").lower(), badmintonlab_game.__file__


def main():
    task = "BadmintonLab-Game-Play"
    cfg = parse_env_cfg(task, num_envs=args.num_envs)
    cfg.seed = args.seed
    env = gym.make(task, cfg=cfg)
    obs, _ = env.reset()
    u = env.unwrapped
    dev = u.device
    n = u.num_envs
    pol = torch.jit.load(args.server_model, map_location=dev).eval()
    sh = u.scene["shuttle"]
    origins = u.scene.env_origins
    server = u.next_hitter.clone()
    pending = torch.zeros(n, dtype=torch.bool, device=dev)  # side 1 hit last step: record this step
    done_rec = torch.zeros(n, dtype=torch.bool, device=dev)  # already recorded this rally
    pre_hit1 = torch.zeros(n, dtype=torch.bool, device=dev)
    orig_reset = u._reset_idx

    def reset_snap(env_ids):
        ids = torch.as_tensor(env_ids, device=dev, dtype=torch.long)
        pre_hit1[ids] = u.just_hit[ids, 1]
        return orig_reset(env_ids)

    u._reset_idx = reset_snap
    pos_l, vel_l, hit_z = [], [], []
    outcome = {"recorded": 0, "rallies_s1_serve": 0, "s1_no_hit": 0, "s1_fault": 0, "lost_before_record": 0}
    sim_t, wall = 0.0, time.time()
    zero0 = torch.zeros(n, 23, device=dev)
    while outcome["recorded"] < args.serves and sim_t < args.max_sim_s:
        with torch.inference_mode():
            a1 = pol(obs["player_1"]).to(device=dev, dtype=torch.float32)
            pre_hit1.zero_()
            obs, info = env.step(torch.cat((zero0, a1), dim=-1))
        sim_t += u.step_dt
        resolved = info["rally_resolved"]
        # Record shuttle state one control step after side 1's serve hit (courts still live).
        rec = pending & ~resolved & ~done_rec
        if rec.any():
            ids = rec.nonzero().flatten()
            p = sh.data.root_pos_w[ids] - origins[ids]
            v = sh.data.root_lin_vel_w[ids]
            pos_l.append(p.cpu())
            vel_l.append(v.cpu())
            outcome["recorded"] += int(ids.numel())
            done_rec[ids] = True
        outcome["lost_before_record"] += int((pending & resolved & ~done_rec).sum())
        hit1 = torch.where(resolved, pre_hit1, u.just_hit[:, 1]) & (server == 1)
        pending = hit1 & ~done_rec & ~resolved
        if resolved.any():
            ids = resolved.nonzero().flatten()
            s1 = server[ids] == 1
            outcome["rallies_s1_serve"] += int(s1.sum())
            outcome["s1_fault"] += int((s1 & (info["rally_winner"][ids] == 0) & ~done_rec[ids]).sum())
            server[ids] = u.next_hitter[ids]
            done_rec[ids] = False
            pending[ids] = False
    pos = torch.cat(pos_l)[: args.serves]
    vel = torch.cat(vel_l)[: args.serves]
    torch.save({"pos": pos, "vel": vel, "frame": "side0 env-local (receiver), net x=0", "server_model": args.server_model,
                "physics_dt": u.physics_dt, "decimation": cfg.decimation, "stats": outcome}, args.out)  # fmt: skip
    print(f"[RECORD] {outcome} sim_s={sim_t:.1f} wall_s={time.time() - wall:.1f} -> {args.out}", flush=True)
    print(f"[RECORD] pos mean {pos.mean(0).tolist()} vel mean {vel.mean(0).tolist()}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
