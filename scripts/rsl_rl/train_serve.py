"""Train the serve task (design doc P3). Same launcher as ``train.py`` plus the P0-P2 infrastructure.

Example::

    python scripts/rsl_rl/train_serve.py --headless --num_envs 2048 --seed 1 \
        --run_name serve_s1 --time_budget_hours 20 --auto_resume

Use ``--max_iterations 2 --num_envs 4`` as the smoke test.
"""

import argparse
import dataclasses
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
    "--stage_max_iters",
    default=None,
    help="Override curriculum max iterations per stage, e.g. 'A=4500,B=12000,C=12000'.",
)
parser.add_argument(
    "--curriculum_iters",
    default=None,
    help="Override curriculum iteration counts, e.g. 'ema=100,blend=150,min=150' (min: every stage's minimum).",
)
parser.add_argument(
    "--stage_gates",
    default=None,
    help="Override gate thresholds (not their direction), e.g. 'C.in=0.75,B.crossed=0.65'.",
)
parser.add_argument(
    "--stage_param",
    default=None,
    help="Reward parameter while in a stage, e.g. 'C.v_close_max=7.0'; float fields of ServeRewardParams only. "
    "Blended like the weights after a promotion. Needs the curriculum.",
)
parser.add_argument(
    "--stop_after_stage",
    choices=("A", "B", "C"),
    default=None,
    help="Stop once this stage's gate passed (model_passed_<stage>_<it>.pt holds the next stage's state).",
)
parser.add_argument(
    "--stage_weights",
    default=None,
    help="Override stage reward weights, e.g. 'B.launch_dir=3,C.launch_dir=1'. Reward parameters go through "
    "hydra, e.g. env.serve_reward.reach_lambda=0.5.",
)
parser.add_argument(
    "--mini_batches",
    type=int,
    default=None,
    help="PPO mini-batches per epoch. Default 16 (2048 courts). At 8192 courts use 64 to keep the mini-batch size.",
)
parser.add_argument(
    "--physics_dt",
    type=float,
    default=0.001,
    choices=(0.001, 0.002, 0.005),
    help="Physics step (play a5e0780: 0.001). decimation = 0.02 / dt; actuator delays stay 10-40 ms.",
)
parser.add_argument("--entropy_coef", type=float, default=None, help="Override algorithm.entropy_coef.")
parser.add_argument("--save_interval_override", type=int, default=None, help="Checkpoint every N iterations.")
parser.add_argument("--style", default=None, help="Serve style preset (serve/styles.py), applied before hydra overrides "
                    "of env.serve_reward and before --stage_weights.")
parser.add_argument(
    "--std_range",
    default=None,
    help="Clamp the action noise std into 'lo,hi' after loading and after every update, e.g. '0.15,0.35'.",
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
from badmintonlab_game.tasks.manager_based.badmintonlab_game.dual_agent_env_cfg import set_physics_dt  # noqa: E402
from badmintonlab_game.tasks.manager_based.badmintonlab_game.serve import player_cfg  # noqa: E402
from badmintonlab_game.tasks.manager_based.badmintonlab_game.serve.styles import style_config  # noqa: E402
from badmintonlab_game.tasks.manager_based.badmintonlab_game.serve.curriculum import (  # noqa: E402
    ServeCurriculum,
    apply_stage_gates,
    parse_stage_params,
)
from badmintonlab_game.tasks.manager_based.badmintonlab_game.serve.serve_reward import STAGE_WEIGHTS  # noqa: E402
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


def _install_profiler(runner, env) -> None:
    """SERVE_PROFILE=1: CUDA-synchronised time per rollout component, printed every iteration."""
    import time
    from collections import defaultdict

    import torch

    acc = defaultdict(float)

    def wrap(obj, attr, name):
        fn = getattr(obj, attr)

        def timed(*a, **kw):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = fn(*a, **kw)
            torch.cuda.synchronize()
            acc[name] += time.perf_counter() - t
            return out

        setattr(obj, attr, timed)

    wrap(runner.alg, "act", "alg.act")
    wrap(env, "step", "env.step")
    wrap(runner.alg, "process_env_step", "alg.process_env_step")
    wrap(runner.alg, "compute_returns", "alg.compute_returns")
    wrap(runner.alg, "update", "alg.update")
    wrap(runner, "log", "runner.log (+serve metrics, curriculum)")
    original_log = runner.log

    def log_and_print(locs, *a, **kw):
        out = original_log(locs, *a, **kw)
        total = locs["collection_time"] + locs["learn_time"]
        parts = " ".join(f"{k}={v:.2f}s" for k, v in acc.items())
        print(f"[PROFILE] it={locs['it']} collection={locs['collection_time']:.2f}s learn={locs['learn_time']:.2f}s "
              f"total={total:.2f}s | {parts}", flush=True)
        acc.clear()
        return out

    runner.log = log_and_print


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
    set_physics_dt(env_cfg, args_cli.physics_dt)
    if args_cli.entropy_coef is not None:
        agent_cfg.algorithm.entropy_coef = args_cli.entropy_coef
    if args_cli.save_interval_override is not None:
        agent_cfg.save_interval = args_cli.save_interval_override
    std_range = None
    if args_cli.std_range:
        std_range = tuple(float(v) for v in args_cli.std_range.split(","))
    env_cfg.serve_reward.initial_stage = args_cli.stage
    style = None
    if args_cli.style:
        # Hydra overrides were applied to env_cfg already: a style value only replaces a field still at its default.
        sp, sw = style_config(args_cli.style)
        defaults = type(env_cfg.serve_reward)()
        for k, v in sp.items():
            if not hasattr(env_cfg.serve_reward, k):
                raise KeyError(f"--style {args_cli.style}: unknown reward parameter {k!r}")
            if getattr(env_cfg.serve_reward, k) == getattr(defaults, k):
                setattr(env_cfg.serve_reward, k, v)
        for stage in STAGE_WEIGHTS:
            STAGE_WEIGHTS[stage].update(sw)
        style = dict(name=args_cli.style, params=sp, weights=sw)
        print(f"[INFO] serve style {args_cli.style}: params {sp}, weights {sw}")
    if args_cli.stage_weights:
        # Before the env is built: the reward computer and the curriculum both read STAGE_WEIGHTS.
        for item in args_cli.stage_weights.split(","):
            key, value = (s.strip() for s in item.split("="))
            stage, term = key.split(".")
            if term not in STAGE_WEIGHTS[stage]:
                raise KeyError(f"--stage_weights: unknown term {term!r}")
            STAGE_WEIGHTS[stage][term] = float(value)
        print(f"[INFO] stage weights overridden: {args_cli.stage_weights}")
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
            curriculum_iters=args_cli.curriculum_iters,
            stage_weights=args_cli.stage_weights,
            stage_max_iters=args_cli.stage_max_iters,
            stage_gates=args_cli.stage_gates,
            stage_param=args_cli.stage_param,
            stop_after_stage=args_cli.stop_after_stage,
            physics_dt=env_cfg.sim.dt,
            decimation=env_cfg.decimation,
            entropy_coef=agent_cfg.algorithm.entropy_coef,
            std_range=std_range,
            style=style,
            stage_weights_effective={k: dict(v) for k, v in STAGE_WEIGHTS.items()},
            serve_reward={k: v for k, v in vars(env_cfg.serve_reward).items() if isinstance(v, (int, float, str, bool))},
            num_mini_batches=agent_cfg.algorithm.num_mini_batches,
            samples_per_iteration=2 * env_cfg.scene.num_envs * agent_cfg.num_steps_per_env,
        ),
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env = ServePolicyVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        print(f"[INFO] {env.court_count} courts, {env.num_envs} player samples per step, log {log_dir}")
        curriculum = None if args_cli.no_curriculum else ServeCurriculum(stage=args_cli.stage)
        if curriculum is not None and args_cli.stage_max_iters:
            for item in args_cli.stage_max_iters.split(","):
                key, value = item.split("=")
                spec = curriculum.stages[key.strip()]
                curriculum.stages[key.strip()] = dataclasses.replace(spec, max_iters=int(value))
            print("[INFO] stage max iters: " + ", ".join(f"{k}={v.max_iters}" for k, v in curriculum.stages.items()))
        if curriculum is not None and args_cli.curriculum_iters:
            for item in args_cli.curriculum_iters.split(","):
                key, value = (s.strip() for s in item.split("="))
                if key == "ema":
                    curriculum.ema_iters = int(value)
                elif key == "blend":
                    curriculum.blend_iters = int(value)
                elif key == "min":
                    for k, spec in curriculum.stages.items():
                        curriculum.stages[k] = dataclasses.replace(spec, min_iters=int(value))
                else:
                    raise ValueError(f"--curriculum_iters: unknown key {key!r}")
            print(f"[INFO] curriculum iters: ema={curriculum.ema_iters} blend={curriculum.blend_iters} "
                  + "min=" + ",".join(f"{k}={v.min_iters}" for k, v in curriculum.stages.items()))
        if curriculum is not None and args_cli.stage_gates:
            curriculum.stages = apply_stage_gates(curriculum.stages, args_cli.stage_gates)
        if curriculum is not None:
            print("[INFO] stage gates: " + "; ".join(
                f"{k}: " + ",".join(f"{m}{op}{thr}" for m, (op, thr) in v.gate.items()) for k, v in curriculum.stages.items()))
        if args_cli.stage_param:
            if curriculum is None:
                raise ValueError("--stage_param needs the curriculum (drop --no_curriculum)")
            curriculum.stage_params = parse_stage_params(args_cli.stage_param)
            print(f"[INFO] stage params: {curriculum.stage_params}")
        if args_cli.stop_after_stage and curriculum is None:
            raise ValueError("--stop_after_stage needs the curriculum (drop --no_curriculum)")
        runner = ServeRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=log_dir,
            device=agent_cfg.device,
            time_budget_hours=args_cli.time_budget_hours,
            curriculum=curriculum,
            stop_after_stage=args_cli.stop_after_stage,
            std_range=std_range,
        )
        runner.install_signal_handlers()
        if resume_path:
            print(f"[INFO] Resuming from: {resume_path}")
            runner.load(resume_path)
            runner.clamp_std()
        elif not args_cli.no_pose_init:
            init_actor_output(runner.alg.policy, _ready_action(env))
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        if os.environ.get("SERVE_PROFILE"):
            _install_profiler(runner, env)
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
