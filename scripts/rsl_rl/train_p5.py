"""Train the P5 receive task (P5_RECEIVE_DESIGN). Warm start from P3 final, teacher = P3 final export.

Smoke::

    python scripts/rsl_rl/train_p5.py --headless --num_envs 4 --max_iterations 2 --run_name p5_smoke --swanlab_mode disabled

Pilot part 1 (feeder only, L0-L1)::

    python scripts/rsl_rl/train_p5.py --headless --num_envs 2048 --run_name p5_pilot_a --natural_share 0 \
        --natural_min 0 --max_level 1 --time_budget_hours 1.5

Feeder landing check (2.2): ``env.p5.verify_mode=True`` (receiver off court, levels drawn uniformly).
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args

REPO = Path(__file__).resolve().parents[2]
P3_CKPT = REPO / "logs/rsl_rl/badmintonlab_serve/2026-11-06_22-49-04_p3_c7/model_3000.pt"
P3_EXPORT = REPO / "exports/p3_final/policy.pt"

_ORIGINAL_ARGV = list(sys.argv)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="BadmintonLab-Game-Receive")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--time_budget_hours", type=float, default=None)
parser.add_argument("--auto_resume", action="store_true", help="Resume from the newest latest.pt of this run_name.")
parser.add_argument("--resume_path", default=None,
                    help="Resume this checkpoint (optimizer, curriculum, PPOExt counters) as a new run; the CLI natural "
                         "share / min / max and --max_level then replace the stored ones (pilot part 2).")
parser.add_argument("--init_checkpoint", default=str(P3_CKPT), help="Warm start (actor, normalisers, padded critic).")
parser.add_argument("--teacher", default=str(P3_EXPORT), help="TorchScript teacher; 'none' disables the teacher term.")
parser.add_argument("--teacher_sigma", type=float, default=0.3)
parser.add_argument("--beta_start", type=float, default=1.0)
parser.add_argument("--beta_end", type=float, default=0.1)
parser.add_argument("--beta_iters", type=int, default=2000)
parser.add_argument("--std_init", type=float, default=0.3)
parser.add_argument("--init_lr", type=float, default=None, help="Learning rate after the warm start (default: P3's).")
parser.add_argument("--fresh_optimizer", action="store_true", help="Do not load P3's Adam state at the warm start.")
parser.add_argument("--std_min", type=float, default=0.15)
parser.add_argument("--std_max", type=float, default=0.40)
parser.add_argument("--critic_only_iters", type=int, default=30)
parser.add_argument("--natural_share", type=float, default=0.5)
parser.add_argument("--natural_min", type=float, default=0.3, help="0 for the feeder-only pilot.")
parser.add_argument("--natural_max", type=float, default=0.7)
parser.add_argument("--level", type=int, default=0, help="Start level (not when resuming).")
parser.add_argument("--max_level", type=int, default=3)
parser.add_argument("--r1_iters", type=int, default=1500)
parser.add_argument("--reward_blend_iters", type=int, default=300)
parser.add_argument("--level_iters", default=None, help="'min=300,max=3000,ema=200'")
parser.add_argument("--recv_weights", default=None, help="e.g. 'R1.recv_approach=2.0,R2.recv_approach=1.0'")
parser.add_argument("--mini_batches", type=int, default=None)
parser.add_argument("--swanlab_mode", default="cloud", choices=("cloud", "local", "offline", "disabled"))
parser.add_argument("--swanlab_project", default="BadmintonLab-Game")
parser.add_argument("--swanlab_workspace", default="Legitimity")
parser.add_argument("--swanlab_group", default="P5")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import dataclasses  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import badmintonlab_game.tasks  # noqa: E402,F401
from badmintonlab_game.tasks.manager_based.badmintonlab_game.serve.p5_curriculum import P5Curriculum  # noqa: E402
from badmintonlab_game.tasks.manager_based.badmintonlab_game.serve.p5_runner import P5Runner  # noqa: E402
from badmintonlab_game.tasks.manager_based.badmintonlab_game.serve.receive import RECV_STAGE_WEIGHTS  # noqa: E402
from badmintonlab_game.tasks.manager_based.badmintonlab_game.serve.runner import (  # noqa: E402
    find_latest,
    finish_run_meta,
    write_run_meta,
)
from badmintonlab_game.tasks.manager_based.badmintonlab_game.serve.wrapper import ReceivePolicyVecEnvWrapper  # noqa: E402
from common import prepare_agent_cfg  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _git_short() -> str:
    try:
        sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip())
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def _start_swanlab(agent_cfg, env_cfg, log_dir: str):
    if args_cli.swanlab_mode == "disabled":
        return None
    python = os.environ.get("SWANLAB_PYTHON", "/root/autodl-tmp/env_swanlab/bin/python")
    if not os.path.isfile(python):
        print(f"[WARN] SwanLab interpreter not found at {python}; TensorBoard only.")
        return None
    name = f"{agent_cfg.run_name or 'p5'}-s{agent_cfg.seed}-{_git_short()}"
    config_path = os.path.join(log_dir, "params", "swanlab_config.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump({"task": args_cli.task, "num_envs": env_cfg.scene.num_envs, "seed": agent_cfg.seed, "log_dir": log_dir,
                   "command": " ".join(_ORIGINAL_ARGV), "agent": agent_cfg.to_dict()}, f, indent=1, default=str)
    cmd = [python, str(REPO / "scripts/tools/swanlab_bridge.py"), "--logdir", log_dir, "--project", args_cli.swanlab_project,
           "--name", name, "--config", config_path, "--mode", args_cli.swanlab_mode, "--parent_pid", str(os.getpid()),
           "--workspace", args_cli.swanlab_workspace, "--group", args_cli.swanlab_group]
    log_file = open(os.path.join(log_dir, "swanlab_bridge.log"), "w")
    return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)


def _stop_swanlab(proc, log_dir: str) -> None:
    if proc is None:
        return
    open(os.path.join(log_dir, ".swanlab_done"), "w").close()
    try:
        proc.wait(timeout=180)
    except subprocess.TimeoutExpired:
        proc.terminate()


def _curriculum() -> P5Curriculum:
    cur = P5Curriculum(level=args_cli.level, max_level=args_cli.max_level, r1_iters=args_cli.r1_iters,
                       blend_iters=args_cli.reward_blend_iters, natural_min=args_cli.natural_min,
                       natural_max=args_cli.natural_max, natural_share=args_cli.natural_share)
    if args_cli.level_iters:
        kv = {k.strip(): int(v) for k, v in (x.split("=") for x in args_cli.level_iters.split(","))}
        if "ema" in kv:
            cur.ema_iters = kv["ema"]
        cur.levels = tuple(dataclasses.replace(s, min_iters=kv.get("min", s.min_iters), max_iters=kv.get("max", s.max_iters))
                           for s in cur.levels)
    return cur


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = prepare_agent_cfg(cli_args.update_rsl_rl_cfg(agent_cfg, args_cli))
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    if args_cli.mini_batches is not None:
        agent_cfg.algorithm.num_mini_batches = args_cli.mini_batches
    env_cfg.seed = agent_cfg.seed
    env_cfg.p5.natural_share = args_cli.natural_share
    env_cfg.p5.level = args_cli.level
    if args_cli.recv_weights:
        for item in args_cli.recv_weights.split(","):
            key, value = (s.strip() for s in item.split("="))
            stage, term = key.split(".")
            if term not in RECV_STAGE_WEIGHTS[stage]:
                raise KeyError(f"--recv_weights: unknown term {term!r}")
            RECV_STAGE_WEIGHTS[stage][term] = float(value)
        print(f"[INFO] receive weights overridden: {args_cli.recv_weights}")

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = find_latest(log_root, agent_cfg.run_name) if (args_cli.auto_resume and agent_cfg.run_name) else None
    if resume_path is None and args_cli.resume_path:
        resume_path = os.path.abspath(args_cli.resume_path)
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + (f"_{agent_cfg.run_name}" if agent_cfg.run_name else "")
    log_dir = os.path.join(log_root, run_name)
    env_cfg.log_dir = log_dir
    teacher = None if args_cli.teacher.lower() == "none" else args_cli.teacher
    ppo_ext = dict(teacher_path=teacher, teacher_sigma=args_cli.teacher_sigma, beta_start=args_cli.beta_start,
                   beta_end=args_cli.beta_end, beta_iters=args_cli.beta_iters, std_min=args_cli.std_min,
                   std_max=args_cli.std_max, critic_only_iters=args_cli.critic_only_iters)
    meta_path = write_run_meta(log_dir, REPO, dict(
        task=args_cli.task, num_envs=env_cfg.scene.num_envs, seed=agent_cfg.seed, max_iterations=agent_cfg.max_iterations,
        time_budget_hours=args_cli.time_budget_hours, resume=resume_path, init_checkpoint=args_cli.init_checkpoint,
        ppo_ext=ppo_ext, std_init=args_cli.std_init, natural_share=args_cli.natural_share,
        natural_min=args_cli.natural_min, level=args_cli.level, max_level=args_cli.max_level, r1_iters=args_cli.r1_iters,
        level_iters=args_cli.level_iters, recv_weights=args_cli.recv_weights,
        recv_stage_weights=RECV_STAGE_WEIGHTS,
        serve_reward={k: v for k, v in vars(env_cfg.serve_reward).items() if isinstance(v, (int, float, str, bool))},
        recv_reward={k: v for k, v in vars(env_cfg.recv_reward).items() if isinstance(v, (int, float, str, bool))},
        p5={k: v for k, v in vars(env_cfg.p5).items() if isinstance(v, (int, float, str, bool))},
        num_mini_batches=agent_cfg.algorithm.num_mini_batches,
        samples_per_iteration=2 * env_cfg.scene.num_envs * agent_cfg.num_steps_per_env,
    ))

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env = ReceivePolicyVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        print(f"[INFO] {env.court_count} courts, {env.num_envs} player samples per step, log {log_dir}")
        curriculum = _curriculum()
        runner = P5Runner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device,
                          time_budget_hours=args_cli.time_budget_hours, ppo_ext=ppo_ext, p5_curriculum=curriculum,
                          control_file=os.path.join(log_dir, "p5_control.json"))
        runner.install_signal_handlers()
        if resume_path:
            print(f"[INFO] Resuming from: {resume_path}")
            runner.load(resume_path)
            if args_cli.resume_path and os.path.abspath(args_cli.resume_path) == resume_path:
                curriculum.natural_min, curriculum.natural_max = args_cli.natural_min, args_cli.natural_max
                curriculum.max_level = args_cli.max_level
                curriculum.level = min(curriculum.level, curriculum.max_level)
                curriculum.natural_share = curriculum.clamp_share(args_cli.natural_share)
                runner.apply_p5()
                print(f"[INFO] resumed curriculum overridden: level {curriculum.level} (max {curriculum.max_level}), natural "
                      f"share {curriculum.natural_share:.2f} in [{curriculum.natural_min}, {curriculum.natural_max}]")
        else:
            report = runner.warm_start(args_cli.init_checkpoint, std=args_cli.std_init,
                                       load_optimizer=not args_cli.fresh_optimizer, lr=args_cli.init_lr)
            with open(os.path.join(log_dir, "warm_start.json"), "w") as f:
                json.dump(report, f, indent=1, default=str)
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        swan = _start_swanlab(agent_cfg, env_cfg, log_dir)
        try:
            reason = runner.learn_until_stop(agent_cfg.max_iterations, init_at_random_ep_len=True)
            print(f"[INFO] stopped: {reason}")
            ver = runner.verify_summary()
            print("[P5_VERIFY]" + json.dumps(ver), flush=True)
            with open(os.path.join(log_dir, "feeder_verify.json"), "w") as f:
                json.dump(ver, f, indent=1)
        finally:
            finish_run_meta(meta_path, runner)
            _stop_swanlab(swan, log_dir)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
