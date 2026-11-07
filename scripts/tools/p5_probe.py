"""P5 probe: run the receive env with a TorchScript policy (or hold actions) and print feeder rallies.

    python scripts/tools/p5_probe.py --headless --num_envs 8 --steps 300 [--policy exports/p3_final/policy.pt] \
        [env.p5.verify_mode=true]

Per court and control step (``--trace`` courts): shuttle position (feeder frame), launch flag, receiver root
height / projected gravity, racket-ball distance, fall and rule state. At the end: feeder landing check per level.
"""

import argparse
import json
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--policy", default="exports/p3_final/policy.pt", help="TorchScript, or 'hold' (ready-pose action).")
parser.add_argument("--trace", type=int, default=2, help="Courts to trace.")
parser.add_argument("--every", type=int, default=5)
parser.add_argument("--natural_share", type=float, default=0.0)
parser.add_argument("--level", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import badmintonlab_game.tasks  # noqa: E402,F401
from badmintonlab_game.tasks.manager_based.badmintonlab_game.serve.serve_reward import to_canonical  # noqa: E402


@hydra_task_config("BadmintonLab-Game-Receive", "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.p5.natural_share = args.natural_share
    env_cfg.p5.level = args.level
    env = gym.make("BadmintonLab-Game-Receive", cfg=env_cfg)
    u = env.unwrapped
    obs, _ = env.reset()
    dev = u.device
    n = u.num_envs
    if args.policy == "hold":
        pol = None
    else:
        pol = torch.jit.load(args.policy, map_location=dev).eval()
    adim = u.single_agent_action_dim
    for t in range(args.steps):
        with torch.inference_mode():
            if pol is None:
                act = torch.zeros(n, 2 * adim, device=dev)
            else:
                act = torch.cat((pol(obs["player_0"]), pol(obs["player_1"])), -1)
            pre_nat = u.natural.clone()
            obs, _, term, trunc, info = env.step(act)
        if t % args.every == 0 or bool((term | trunc)[: args.trace].any()):
            for i in range(min(args.trace, n)):
                s = int(u.initial_hitter[i])
                r = 1 - s
                sh = to_canonical((u._shuttle.data.root_pos_w[i] - u.scene.env_origins[i])[None], torch.tensor([s], device=dev))[0]
                root = u._robots[r].data.root_pos_w[i] - u.scene.env_origins[i]
                feeder_root = u._robots[s].data.root_pos_w[i] - u.scene.env_origins[i]
                gz = float(u._robots[r].data.projected_gravity_b[i, 2])
                gzf = float(u._robots[s].data.projected_gravity_b[i, 2])
                done = bool(term[i] | trunc[i])
                terms = {k: bool(u.termination_manager.get_term(k)[i]) for k in u.termination_manager.active_terms}
                print(f"t={t:3d} c{i} nat={int(pre_nat[i])} srv={s} L={int(u.feeder_row_level[i])} launched={int(u.launched[i])} "
                      f"shuttle_c=({sh[0]:+.2f},{sh[1]:+.2f},{sh[2]:.2f}) recv_root=({root[0]:+.2f},{root[1]:+.2f},{root[2]:.2f}) "
                      f"gz={gz:+.2f} feeder_root_z={feeder_root[2]:.2f} gzf={gzf:+.2f} active={int(u.recv_reward.active[i])} "
                      f"hit={int(u.recv_reward.hit[i])} fall={u.serve_reward.fall_given[i].tolist()} "
                      f"done={int(done)} {[k for k, v in terms.items() if v] if done else ''}", flush=True)
    ver = u.pop_verify()
    print("[PROBE_VERIFY]" + json.dumps(ver), flush=True)
    print("[PROBE_RECV]" + json.dumps(u.recv_reward.pop_iteration()), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
