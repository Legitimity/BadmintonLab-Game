"""Train the serve task (design doc P3). Same launcher as ``train.py`` plus the P0-P2 infrastructure.

Example::

    python scripts/rsl_rl/train_serve.py --headless --num_envs 2048 --seed 1 \
        --run_name serve_s1 --time_budget_hours 20 --auto_resume

Use ``--max_iterations 2 --num_envs 4`` as the smoke test.
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

_ORIGINAL_ARGV = list(sys.argv)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="BadmintonLab-Game-Serve")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--time_budget_hours", type=float, default=None, help="Stop cleanly before this wall-clock budget.")
parser.add_argument("--auto_resume", action="store_true", help="Resume from the newest latest.pt of this run_name.")
parser.add_argument("--no_pose_init", action="store_true", help="Ablation AB3: skip the actor output initialisation.")
parser.add_argument(
    "--swanlab_mode",
    default="cloud",
    choices=("cloud", "local", "offline", "disabled"),
    help="Mirror TensorBoard scalars to SwanLab through scripts/tools/swanlab_bridge.py (separate interpreter).",
)
parser.add_argument("--swanlab_project", default="BadmintonLab-Game")
parser.add_argument("--swanlab_workspace", default="Legitimity")
parser.add_argument("--swanlab_group", default=None)
parser.add_argument("--swanlab_name", default=None)
parser.add_argument("--stage", choices=("A", "B", "C"), default="A", help="Start stage when not resuming.")
parser.add_argument("--no_curriculum", action="store_true", help="Fixed weights of --stage, no promotion or stop.")
parser.add_argument(
    "--mini_batches",
    type=int,
    default=None,
    help="PPO mini-batches per epoch. Default 16 (2048 courts). At 8192 courts use 64 to keep the mini-batch size.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import badmintonlab_game.tasks  # noqa: E402,F401
from badmintonlab_game.tasks.manager_based.badmintonlab_game.serve import player_cfg  # noqa: E402
from badmintonlab_game.tasks.manager_based.badmintonlab_game.serve.curriculum import ServeCurriculum  # noqa: E402
from badmintonlab_game.tasks.manager_based.badmintonlab_game.serve.runner import (  # noqa: E402
    ServeRunner,
    find_latest,
    finish_run_meta,
    write_run_meta,
)
from badmintonlab_game.tasks.manager_based.badmintonlab_game.serve.wrapper import (  # noqa: E402
    ServePolicyVecEnvWrapper,
    hold_action,
    init_actor_output,
)
from common import prepare_agent_cfg  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
REPO = Path(__file__).resolve().parents[2]


def _git_state() -> dict:
    try:
        sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "nogit", "dirty": False}
    return {"commit": sha, "dirty": dirty}


def _start_swanlab_bridge(agent_cfg, env_cfg, log_dir: str):
    """SwanLab >= 0.9 cannot live next to Isaac Sim (wrapt pin), so a sidecar in
    ``$SWANLAB_PYTHON`` tails the TensorBoard events. Returns the process or ``None``."""
    if args_cli.swanlab_mode == "disabled":
        return None
    python = os.environ.get("SWANLAB_PYTHON", "/root/autodl-tmp/env_swanlab/bin/python")
    if not os.path.isfile(python):
        print(f"[WARN] SwanLab interpreter not found at {python}; TensorBoard only.")
        return None
    git = _git_state()
    prefix = agent_cfg.run_name or args_cli.task
    name = args_cli.swanlab_name or f"{prefix}-s{agent_cfg.seed}-{git['commit']}{'-dirty' if git['dirty'] else ''}"
    config = {
        "task": args_cli.task,
        "num_envs": env_cfg.scene.num_envs,
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
        python, str(REPO / "scripts" / "tools" / "swanlab_bridge.py"),
        "--logdir", log_dir, "--project", args_cli.swanlab_project, "--name", name,
        "--config", config_path, "--mode", args_cli.swanlab_mode, "--parent_pid", str(os.getpid()),
    ]  # fmt: skip
    if args_cli.swanlab_workspace:
        cmd += ["--workspace", args_cli.swanlab_workspace]
    if args_cli.swanlab_group:
        cmd += ["--group", args_cli.swanlab_group]
    log_file = open(os.path.join(log_dir, "swanlab_bridge.log"), "w")
    print(f"[INFO] SwanLab bridge: {args_cli.swanlab_project}/{name} ({args_cli.swanlab_mode})")
    return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)


def _stop_swanlab_bridge(proc, log_dir: str) -> None:
    if proc is None:
        return
    open(os.path.join(log_dir, ".swanlab_done"), "w").close()
    try:
        proc.wait(timeout=180)
    except subprocess.TimeoutExpired:
        proc.terminate()


def _ready_action(env) -> torch.Tensor:
    """Raw action that holds the reset pose, for the player-0 robot (both robots share it)."""
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot_0"]
    term = unwrapped.action_manager.get_term("player_0")
    q_ready = robot.data.default_joint_pos[0].clone()
    ids, _ = robot.find_joints(list(player_cfg.READY_JOINTS), preserve_order=True)
    q_ready[ids] = torch.tensor(list(player_cfg.READY_JOINTS.values()), device=q_ready.device)
    q_ready = q_ready[term._joint_ids] if not isinstance(term._joint_ids, slice) else q_ready
    scale = term._scale[0] if isinstance(term._scale, torch.Tensor) else torch.full_like(q_ready, term._scale)
    offset = term._offset[0] if isinstance(term._offset, torch.Tensor) else torch.full_like(q_ready, term._offset)
    return hold_action(q_ready, offset, scale)


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
    env_cfg.seed = agent_cfg.seed
    env_cfg.serve_reward.initial_stage = args_cli.stage
    if args_cli.mini_batches is not None:
        agent_cfg.algorithm.num_mini_batches = args_cli.mini_batches

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = None
    if args_cli.auto_resume and agent_cfg.run_name:
        resume_path = find_latest(log_root, agent_cfg.run_name)
    if resume_path is None and agent_cfg.resume:
        resume_path = (
            retrieve_file_path(args_cli.checkpoint)
            if args_cli.checkpoint
            else get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
        )
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + (f"_{agent_cfg.run_name}" if agent_cfg.run_name else "")
    log_dir = os.path.join(log_root, run_name)
    env_cfg.log_dir = log_dir
    meta_path = write_run_meta(
        log_dir,
        REPO,
        dict(
            task=args_cli.task,
            num_envs=env_cfg.scene.num_envs,
            seed=agent_cfg.seed,
            max_iterations=agent_cfg.max_iterations,
            time_budget_hours=args_cli.time_budget_hours,
            resume=resume_path,
            pose_init=not args_cli.no_pose_init,
            stage=args_cli.stage,
            curriculum=not args_cli.no_curriculum,
            num_mini_batches=agent_cfg.algorithm.num_mini_batches,
            samples_per_iteration=2 * env_cfg.scene.num_envs * agent_cfg.num_steps_per_env,
        ),
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env = ServePolicyVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        print(f"[INFO] {env.court_count} courts, {env.num_envs} player samples per step, log {log_dir}")
        curriculum = None if args_cli.no_curriculum else ServeCurriculum(stage=args_cli.stage)
        runner = ServeRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=log_dir,
            device=agent_cfg.device,
            time_budget_hours=args_cli.time_budget_hours,
            curriculum=curriculum,
        )
        runner.install_signal_handlers()
        if resume_path:
            print(f"[INFO] Resuming from: {resume_path}")
            runner.load(resume_path)
        elif not args_cli.no_pose_init:
            init_actor_output(runner.alg.policy, _ready_action(env))
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        swan = _start_swanlab_bridge(agent_cfg, env_cfg, log_dir)
        try:
            reason = runner.learn_until_stop(agent_cfg.max_iterations, init_at_random_ep_len=True)
            print(f"[INFO] stopped: {reason}")
        finally:
            finish_run_meta(meta_path, runner)
            _stop_swanlab_bridge(swan, log_dir)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
