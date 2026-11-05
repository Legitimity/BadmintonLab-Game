"""Train the serve task (design doc P3). Same launcher as ``train.py`` plus the P0-P2 infrastructure.

Example::

    python scripts/rsl_rl/train_serve.py --headless --num_envs 2048 --seed 1 \
        --run_name serve_s1 --time_budget_hours 20 --auto_resume

Use ``--max_iterations 2 --num_envs 4`` as the smoke test.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="BadmintonLab-Game-Serve")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--time_budget_hours", type=float, default=None, help="Stop cleanly before this wall-clock budget.")
parser.add_argument("--auto_resume", action="store_true", help="Resume from the newest latest.pt of this run_name.")
parser.add_argument("--no_pose_init", action="store_true", help="Ablation AB3: skip the actor output initialisation.")
parser.add_argument("--swanlab", action="store_true", help="Mirror TensorBoard scalars to SwanLab (offline by default).")
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
            samples_per_iteration=2 * env_cfg.scene.num_envs * agent_cfg.num_steps_per_env,
        ),
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env = ServePolicyVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        print(f"[INFO] {env.court_count} courts, {env.num_envs} player samples per step, log {log_dir}")
        runner = ServeRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=log_dir,
            device=agent_cfg.device,
            time_budget_hours=args_cli.time_budget_hours,
        )
        runner.install_signal_handlers()
        if resume_path:
            print(f"[INFO] Resuming from: {resume_path}")
            runner.load(resume_path)
        elif not args_cli.no_pose_init:
            init_actor_output(runner.alg.policy, _ready_action(env))
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        swan = None
        if args_cli.swanlab:
            try:
                import swanlab

                swan = swanlab
                swanlab.init(
                    project=os.environ.get("SWANLAB_PROJECT", "BadmintonLab-Game"),
                    name=run_name,
                    mode=os.environ.get("SWANLAB_MODE", "offline"),
                    log_dir=str(REPO / "swanlog"),
                )
                swanlab.sync_tensorboard_torch()
            except Exception as exc:  # Tracking must never stop a run.
                print(f"[SwanLab] disabled: {exc}")
                swan = None
        try:
            reason = runner.learn_until_stop(agent_cfg.max_iterations, init_at_random_ep_len=True)
            print(f"[INFO] stopped: {reason}")
        finally:
            finish_run_meta(meta_path, runner)
            if swan is not None:
                swan.finish()
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
