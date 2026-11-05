# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train one PPO policy controlling both players on each badminton court."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

import cli_args

_ORIGINAL_ARGV = list(sys.argv)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="BadmintonLab-Game", help="Registered badminton task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of two-player courts.")
parser.add_argument("--seed", type=int, default=None, help="Random seed (-1 samples a seed).")
parser.add_argument("--max_iterations", type=int, default=None, help="Number of PPO iterations.")
parser.add_argument("--distributed", action="store_true", help="Train using multiple GPUs.")
parser.add_argument("--video", action="store_true", help="Record training videos.")
parser.add_argument("--video_length", type=int, default=200, help="Video length in control steps.")
parser.add_argument("--video_interval", type=int, default=2000, help="Control steps between videos.")
parser.add_argument(
    "--swanlab_mode",
    default="cloud",
    choices=("cloud", "local", "offline", "disabled"),
    help="Mirror TensorBoard scalars to SwanLab through scripts/tools/swanlab_bridge.py "
    "(credentials come from `swanlab login`, never from the repo).",
)
parser.add_argument("--swanlab_project", default="BadmintonLab-Game", help="SwanLab project.")
parser.add_argument("--swanlab_workspace", default="Legitimity", help="SwanLab workspace (user or organisation).")
parser.add_argument("--swanlab_group", default=None, help="SwanLab experiment group, e.g. p0-bench.")
parser.add_argument(
    "--swanlab_name", default=None, help="SwanLab experiment name; defaults to <run_name|task>-s<seed>-<shortsha>."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import badmintonlab_game.tasks  # noqa: F401
from badmintonlab_game.tasks.manager_based.badmintonlab_game.shared_policy_wrapper import SharedPolicyVecEnvWrapper

from common import prepare_agent_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _git_state() -> dict:
    """Short commit hash, branch and dirty flag of the training source tree."""
    try:
        sha = subprocess.check_output(["git", "-C", _REPO_ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", _REPO_ROOT, "status", "--porcelain"], text=True).strip())
        branch = subprocess.check_output(
            ["git", "-C", _REPO_ROOT, "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "nogit", "dirty": False, "branch": ""}
    return {"commit": sha, "dirty": dirty, "branch": branch}


def _start_swanlab_bridge(agent_cfg, env_cfg, log_dir: str):
    """Launch the SwanLab sidecar that mirrors this run's TensorBoard scalars.

    The project needs SwanLab >= 0.9, which cannot be installed next to Isaac Sim
    (wrapt pin), so the upload runs in an isolated interpreter
    (``$SWANLAB_PYTHON``, default ``/root/autodl-tmp/env_swanlab/bin/python``).
    Returns the process handle, or ``None`` when disabled/unavailable.
    """
    if args_cli.swanlab_mode == "disabled":
        return None
    python = os.environ.get("SWANLAB_PYTHON", "/root/autodl-tmp/env_swanlab/bin/python")
    if not os.path.isfile(python):
        print(f"[WARN] SwanLab interpreter not found at {python}; continuing with TensorBoard only.")
        return None
    git = _git_state()
    prefix = agent_cfg.run_name or args_cli.task
    name = args_cli.swanlab_name or f"{prefix}-s{agent_cfg.seed}-{git['commit']}{'-dirty' if git['dirty'] else ''}"
    config = {
        "task": args_cli.task,
        "num_envs": env_cfg.scene.num_envs,
        "player_samples_per_step": 2 * env_cfg.scene.num_envs,
        "seed": agent_cfg.seed,
        "log_dir": log_dir,
        "git": git,
        "command": " ".join(_ORIGINAL_ARGV),
        "agent": agent_cfg.to_dict(),
    }
    config_path = os.path.join(log_dir, "params", "swanlab_config.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=1, default=str)
    cmd = [
        python,
        os.path.join(_REPO_ROOT, "scripts", "tools", "swanlab_bridge.py"),
        "--logdir", log_dir,
        "--project", args_cli.swanlab_project,
        "--name", name,
        "--config", config_path,
        "--mode", args_cli.swanlab_mode,
        "--parent_pid", str(os.getpid()),
    ]  # fmt: skip
    if args_cli.swanlab_workspace:
        cmd += ["--workspace", args_cli.swanlab_workspace]
    if args_cli.swanlab_group:
        cmd += ["--group", args_cli.swanlab_group]
    log_file = open(os.path.join(log_dir, "swanlab_bridge.log"), "w")
    print(f"[INFO] SwanLab bridge: {args_cli.swanlab_project}/{name} ({args_cli.swanlab_mode})")
    return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)


def _stop_swanlab_bridge(proc, log_dir: str) -> None:
    """Ask the sidecar to drain the remaining events and finish the SwanLab run."""
    if proc is None:
        return
    open(os.path.join(log_dir, ".swanlab_done"), "w").close()
    try:
        proc.wait(timeout=180)
    except subprocess.TimeoutExpired:
        proc.terminate()


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Collect both players' experience and update their shared policy."""
    agent_cfg = prepare_agent_cfg(cli_args.update_rsl_rl_cfg(agent_cfg, args_cli))
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    if args_cli.distributed:
        if "cpu" in env_cfg.sim.device:
            raise ValueError("Distributed training requires a CUDA device.")
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = env_cfg.sim.device
        agent_cfg.seed += app_launcher.local_rank
    env_cfg.seed = agent_cfg.seed

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = None
    if agent_cfg.resume:
        resume_path = (
            retrieve_file_path(args_cli.checkpoint)
            if args_cli.checkpoint
            else get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
        )
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        run_name += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root, run_name)
    env_cfg.log_dir = log_dir
    print(f"[INFO] Logging experiment in: {log_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    swanlab_bridge = None
    try:
        if args_cli.video:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=os.path.join(log_dir, "videos", "train"),
                step_trigger=lambda step: step % args_cli.video_interval == 0,
                video_length=args_cli.video_length,
                disable_logger=True,
            )
        env = SharedPolicyVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        print(f"[INFO] Self-play: {env.court_count} courts, {env.num_envs} player samples per step.")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        if os.path.isdir(os.path.join(os.path.dirname(__file__), "..", "..", ".git")):
            runner.add_git_repo_to_log(__file__)
        if resume_path:
            print(f"[INFO] Resuming from: {resume_path}")
            runner.load(resume_path)
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        swanlab_bridge = _start_swanlab_bridge(agent_cfg, env_cfg, log_dir)
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    finally:
        _stop_swanlab_bridge(swanlab_bridge, log_dir)
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
