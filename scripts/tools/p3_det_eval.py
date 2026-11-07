"""Training-env evaluation without updates (P3 F2 wrap-up): serve rates of a checkpoint with the deterministic
actor mean (``act_inference``, what play runs) and with the sampled training actions (``act``), same env.

    python scripts/tools/p3_det_eval.py --headless --num_envs 2048 --checkpoint logs/.../model_3152.pt \
        --iters 20 --out logs/v3l/p3/final_eval/det_eval.json

Each mode resets the env, discards ``--warmup`` iterations (all courts start their first rally together),
then pools ``--iters`` iterations of 24 control steps. Rates come from the reward computer's statistics,
the same as ``Serve/*`` in training. Stage C weights, default reward parameters (v_close_max 8.0).
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rsl_rl"))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="BadmintonLab-Game-Serve")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--iters", type=int, default=20)
parser.add_argument("--warmup", type=int, default=2)
parser.add_argument("--modes", default="det,stoch")
parser.add_argument("--out", required=True)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import badmintonlab_game.tasks  # noqa: E402,F401
from badmintonlab_game.tasks.manager_based.badmintonlab_game.serve.serve_reward import STAGE_WEIGHTS  # noqa: E402
from badmintonlab_game.tasks.manager_based.badmintonlab_game.serve.wrapper import ServePolicyVecEnvWrapper  # noqa: E402
from common import load_weights, prepare_agent_cfg  # noqa: E402

KEYS = ("legal", "crossed", "in", "fault", "no_hit", "tunnel", "net", "out", "fall", "fall_after_hit", "timeout",
        "vc_mean", "el_mean", "launch_el_mean", "outcome_mean", "rallies")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = prepare_agent_cfg(cli_args.update_rsl_rl_cfg(agent_cfg, args_cli))
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.serve_reward.initial_stage = "C"
    env = ServePolicyVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        load_weights(runner, args_cli.checkpoint)
        policy = runner.alg.policy
        policy.eval()
        computer = env.unwrapped.serve_reward
        computer.set_weights(STAGE_WEIGHTS["C"])
        std = getattr(policy, "std", None)
        std = torch.exp(policy.log_std) if std is None and hasattr(policy, "log_std") else std
        noise = None if std is None else float(std.detach().mean())
        result = dict(checkpoint=args_cli.checkpoint, courts=env.court_count, iters=args_cli.iters,
                      warmup=args_cli.warmup, steps_per_iter=agent_cfg.num_steps_per_env, mean_noise_std=noise,
                      v_close_max=computer.p.v_close_max, modes={})
        for mode in args_cli.modes.split(","):
            obs, _ = env.reset()
            for it in range(args_cli.warmup + args_cli.iters):
                if it == args_cli.warmup:
                    computer.pop_iteration()
                with torch.no_grad():
                    for _ in range(agent_cfg.num_steps_per_env):
                        actions = policy.act_inference(obs) if mode == "det" else policy.act(obs)
                        obs, _, _, _ = env.step(actions)
            # Pooled over the measured iterations: the counters accumulate until popped.
            rates = computer.pop_iteration()
            r = {k: round(float(rates[k]), 4) for k in KEYS if k in rates}
            result["modes"][mode] = r
            print(f"[DET_EVAL] {mode}: " + " ".join(f"{k}={v}" for k, v in r.items()), flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
        with open(args_cli.out, "w") as f:
            json.dump(result, f, indent=1)
        print(json.dumps(result, indent=1))
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
