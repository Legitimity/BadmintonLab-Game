"""Train the receive expert (P5 V3.1): stage S1 footwork, stage S2 precision swing. Local runs (RTX 5070 8 GB).

Example::

    python scripts/rsl_rl/train_receive.py --headless --num_envs 2048 --run_name recv_s1 \
        --p3_policy D:/rl-badminton-dev/p3_sync_v4/p3_final/policy.pt --time_budget_hours 8 --auto_resume

    python scripts/rsl_rl/train_receive.py --headless --num_envs 2048 --run_name recv_s2 --stage S2 \
        --init_from D:/.../model_passed_S1_2514.pt --dt_ms 1 --time_budget_hours 8 --auto_resume

``--dt_ms`` keeps the 20 ms control step and the 10-40 ms actuator delay (physics.apply_physics); 1 = play.

Smoke: ``--num_envs 64 --max_iterations 2``.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Windows: tensordict's native module crashes (access violation) when first loaded after Kit starts.
import torch  # noqa: E402,F401
import tensordict  # noqa: F401
from rsl_rl.runners import OnPolicyRunner  # noqa: F401

from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="BadmintonLab-Game-Receive")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--max_iterations", type=int, default=3000)
parser.add_argument("--time_budget_hours", type=float, default=None)
parser.add_argument("--auto_resume", action="store_true")
parser.add_argument("--p3_policy", default="D:/rl-badminton-dev/p3_sync_v4/p3_final/policy.pt")
parser.add_argument("--mini_batches", type=int, default=None)
parser.add_argument("--critic_warmup", type=int, default=50)
parser.add_argument("--leg_std", type=float, default=0.5)
parser.add_argument("--other_std", type=float, default=0.3)
parser.add_argument("--no_curriculum", action="store_true")
parser.add_argument("--level", type=int, default=0, help="Start feeder level (fixed if --no_curriculum).")
parser.add_argument("--stage", default="S1", choices=("S1", "S2", "S3"))
parser.add_argument("--s3_timeout_s", type=float, default=4.0, help="S3: per-ball timeout (the return must land)")
parser.add_argument("--init_from", default=None, help="S2: checkpoint of the passed S1 (weights, optimizer, lr).")
parser.add_argument("--dt_ms", type=int, default=1, help="physics step in ms (1 = play a5e0780)")
parser.add_argument("--s2_anneal", type=int, default=1000, help="earliest S2 gate iteration")
parser.add_argument("--s2_tau", type=float, default=220.0, help="sigma annealing time constant")
parser.add_argument("--no_stop_on_pass", action="store_true", help="keep training after the stage gate passes")
# 11-08 V3.4: real serves + footwork. The fall weight is NOT raised by default (a larger fall penalty
# pushes the policy towards not moving, the opposite of what V3.4 needs).
parser.add_argument("--real_serves", nargs="*", default=[], help="serve files from record_serves.py")
parser.add_argument("--real_frac", type=float, default=0.0, help="fraction of balls from the real serves")
parser.add_argument("--w", nargs="*", default=[], help="S2 weight overrides, e.g. track=3 approach=2")
parser.add_argument("--approach_cap", type=float, default=0.8, help="approach reward speed cap, m/s")
# 11-09 S1 V4 (from p3_final): wide feeder with a free hitting height, required-speed levels, return-home gaps.
parser.add_argument("--v4", action="store_true", help="S1 V4 feeder / curriculum (see core.SpeedCurriculum)")
parser.add_argument("--v4_real_from", type=int, default=3, help="V4: mix real serves from this speed level on")
parser.add_argument("--face_relax", action="store_true", help="V4: wider heading tolerance while travelling")
parser.add_argument("--reset_leg_std", type=float, default=None, help="after resume: set the leg action std once")
parser.add_argument("--ball_gap_s", type=float, default=None, help="gap between balls (V4 default 1.0 s)")
parser.add_argument("--s2_other_std", type=float, default=0.3, help="S2 start: raise non-leg std to at least this.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import badmintonlab_game.tasks  # noqa: E402,F401
from badmintonlab_game.tasks.manager_based.badmintonlab_game import receive as _receive  # noqa: E402,F401
from badmintonlab_game.tasks.manager_based.badmintonlab_game.receive import core  # noqa: E402
from badmintonlab_game.tasks.manager_based.badmintonlab_game.receive.receive_env import S1_WEIGHTS, S2_WEIGHTS, S3_WEIGHTS  # noqa: E402
from badmintonlab_game.tasks.manager_based.badmintonlab_game.physics import apply_physics  # noqa: E402
from badmintonlab_game.tasks.manager_based.badmintonlab_game.receive.agents import (  # noqa: E402
    EXTRA_MEAN,
    EXTRA_STD,
    ReceiveRunner,
    ReceiveVecEnvWrapper,
)
from badmintonlab_game.tasks.manager_based.badmintonlab_game.serve import player_cfg  # noqa: E402
from badmintonlab_game.tasks.manager_based.badmintonlab_game.serve.runner import (  # noqa: E402
    find_latest,
    finish_run_meta,
    write_run_meta,
)
from badmintonlab_game.tasks.manager_based.badmintonlab_game.serve.wrapper import hold_action  # noqa: E402
from common import prepare_agent_cfg  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
LEG = ("Hip", "Knee", "Ankle")


def _ready_action(unwrapped, side: int) -> torch.Tensor:
    robot = unwrapped.scene[f"robot_{side}"]
    term = unwrapped.action_manager.get_term(f"player_{side}")
    q = robot.data.default_joint_pos[0].clone()
    ids, _ = robot.find_joints(list(player_cfg.READY_JOINTS), preserve_order=True)
    q[ids] = torch.tensor(list(player_cfg.READY_JOINTS.values()), device=q.device)
    q = q[term._joint_ids] if not isinstance(term._joint_ids, slice) else q
    scale = term._scale[0] if isinstance(term._scale, torch.Tensor) else torch.full_like(q, term._scale)
    offset = term._offset[0] if isinstance(term._offset, torch.Tensor) else torch.full_like(q, term._offset)
    return hold_action(q, offset, scale)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = prepare_agent_cfg(cli_args.update_rsl_rl_cfg(agent_cfg, args_cli))
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.seed = args_cli.seed
    env_cfg.seed = agent_cfg.seed
    if args_cli.mini_batches is not None:
        agent_cfg.algorithm.num_mini_batches = args_cli.mini_batches
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    apply_physics(env_cfg, args_cli.dt_ms)
    if args_cli.v4:
        env_cfg.feeder_v4 = True
        env_cfg.ball_gap_s = 1.0
    if args_cli.stage == "S3":
        env_cfg.ball_timeout_s = args_cli.s3_timeout_s
        env_cfg.episode_length_s = 30.0
    if args_cli.ball_gap_s is not None:
        env_cfg.ball_gap_s = args_cli.ball_gap_s

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = find_latest(log_root, agent_cfg.run_name) if (args_cli.auto_resume and agent_cfg.run_name) else None
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + (f"_{agent_cfg.run_name}" if agent_cfg.run_name else "")
    log_dir = os.path.join(log_root, run_name)
    env_cfg.log_dir = log_dir
    meta = write_run_meta(log_dir, REPO, dict(num_envs=env_cfg.scene.num_envs, resume=resume_path, p3=args_cli.p3_policy,
                                              mini_batches=agent_cfg.algorithm.num_mini_batches, leg_std=args_cli.leg_std,
                                              stage=args_cli.stage, init_from=args_cli.init_from, dt_ms=args_cli.dt_ms,
                                              real_serves=args_cli.real_serves, real_frac=args_cli.real_frac, w=args_cli.w,
                                              approach_cap=args_cli.approach_cap, v4=args_cli.v4, face_relax=args_cli.face_relax,
                                              reset_leg_std=args_cli.reset_leg_std,
                                              v4_real_from=args_cli.v4_real_from, ball_gap_s=env_cfg.ball_gap_s))  # fmt: skip

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        unwrapped = env.unwrapped
        wrapped = ReceiveVecEnvWrapper(env, _ready_action(unwrapped, 1), clip_actions=agent_cfg.clip_actions)
        term = unwrapped.action_manager.get_term("player_0")
        names = term._joint_names
        leg_mask = torch.tensor([any(k in n for k in LEG) for n in names])
        print(f"[INFO] {wrapped.num_envs} learners; {int(leg_mask.sum())} leg joints; feeder rows {len(unwrapped.table)}; log {log_dir}")
        print(f"[INFO] stage {args_cli.stage}; physics dt {unwrapped.physics_dt} s, decimation {env_cfg.decimation}", flush=True)
        if args_cli.stage == "S3":
            s3w = dict(S3_WEIGHTS)
            for kv in args_cli.w:
                k, v = kv.split("=")
                if k not in s3w:
                    raise SystemExit(f"unknown weight {k}")
                s3w[k] = float(v)
            s2w = dict(S2_WEIGHTS, track=4.0, approach=2.0)  # the S2 V4 run used --w track=4 approach=2
            cur = core.S3Curriculum(s2_weights=s2w, s3_weights=s3w)
            unwrapped.set_level(cur.level)
            unwrapped.set_s3(True, args_cli.s3_timeout_s)
            unwrapped.approach_cap = args_cli.approach_cap
            if args_cli.real_serves:
                for info in unwrapped.set_real_serves(args_cli.real_serves, args_cli.real_frac):
                    print(f"[INFO] real serves {info}", flush=True)
            print(f"[INFO] S3 weights {s3w}; approach_cap {args_cli.approach_cap}; real_frac {args_cli.real_frac}; "
                  f"ball timeout {args_cli.s3_timeout_s} s", flush=True)
        elif args_cli.stage == "S2":
            s2w = dict(S2_WEIGHTS)
            for kv in args_cli.w:
                k, v = kv.split("=")
                if k not in s2w:
                    raise SystemExit(f"unknown weight {k}")
                s2w[k] = float(v)
            cur = core.S2Curriculum(s1_weights=dict(S1_WEIGHTS), s2_weights=s2w, anneal_iters=args_cli.s2_anneal, anneal_tau=args_cli.s2_tau)
            if args_cli.v4:
                # S2 V4: speed-level mix over the V4 table; gate on the first ball of an episode (= play).
                cur.level = core.LEVEL_MIX_V4
                cur.gate_hit_key = "first_all_hit"
                print(f"[INFO] S2 V4: mix {core.S2_MIX_V4}; gate on first_all_hit", flush=True)
            unwrapped.set_level(cur.level)
            unwrapped.approach_cap = args_cli.approach_cap
            if args_cli.real_serves:
                for info in unwrapped.set_real_serves(args_cli.real_serves, args_cli.real_frac):
                    print(f"[INFO] real serves {info}", flush=True)
            print(f"[INFO] S2 weights {s2w}; approach_cap {args_cli.approach_cap}; real_frac {args_cli.real_frac}", flush=True)
        elif args_cli.v4:
            cur = core.SpeedCurriculum(level=args_cli.level)
            unwrapped.set_level(cur.level)
            if args_cli.real_serves:
                for info in unwrapped.set_real_serves(args_cli.real_serves, 0.0):
                    print(f"[INFO] real serves {info}", flush=True)
            print(f"[INFO] S1 V4: table rows {len(unwrapped.table)}; speed edges {core.SPEED_EDGES}; caps {core.APPROACH_CAPS}; "
                  f"real serves {args_cli.real_frac} from level {args_cli.v4_real_from}; gap {env_cfg.ball_gap_s} s", flush=True)
        else:
            cur = None if args_cli.no_curriculum else core.ReceiveCurriculum(level=args_cli.level)
            unwrapped.set_level(args_cli.level)
        runner = ReceiveRunner(wrapped, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device,
                               time_budget_hours=args_cli.time_budget_hours, curriculum=cur, leg_mask=leg_mask,
                               critic_warmup=0 if resume_path else args_cli.critic_warmup)  # fmt: skip
        runner.stop_on_pass = not args_cli.no_stop_on_pass
        runner.v4_real_frac = args_cli.real_frac
        runner.v4_real_from = args_cli.v4_real_from
        runner.apply_v4_params()
        runner.install_signal_handlers()
        unwrapped.face_relax = bool(args_cli.face_relax)
        if resume_path:
            print(f"[INFO] Resuming from {resume_path}")
            runner.load(resume_path)
            if args_cli.reset_leg_std is not None:
                ls = runner.alg.policy.log_std.data
                lm = leg_mask.to(ls.device)
                import math as _m

                ls.copy_(torch.where(lm, torch.full_like(ls, _m.log(args_cli.reset_leg_std)), ls))
                print(f"[INFO] leg std reset to {args_cli.reset_leg_std}", flush=True)
        elif args_cli.stage in ("S2", "S3"):
            if not args_cli.init_from:
                raise SystemExit(f"--stage {args_cli.stage} needs --init_from <checkpoint>")
            # Weights, optimizer state and learning rate continue from S1 (spec 2.1: never a fresh Adam).
            runner.load(args_cli.init_from)
            ls = runner.alg.policy.log_std.data
            lm = leg_mask.to(ls.device)
            import math as _m

            floor = torch.full_like(ls, _m.log(args_cli.s2_other_std))
            ls.copy_(torch.where(~lm, torch.maximum(ls, floor), ls))
            runner.snapshot_actor()  # critic-only warm-up: the critic has never seen the swing reward
            print(f"[INFO] S2 from {args_cli.init_from}; std leg {ls[lm].exp().mean():.3f} other {ls[~lm].exp().mean():.3f}; "
                  f"critic-only for {args_cli.critic_warmup} iterations", flush=True)  # fmt: skip
        else:
            ts = core.load_p3_torchscript(args_cli.p3_policy)
            core.warm_start_actor(runner.alg.policy, ts, EXTRA_MEAN, EXTRA_STD)
            core.set_log_std(runner.alg.policy.log_std.data, leg_mask.to(runner.alg.policy.log_std.device), args_cli.leg_std, args_cli.other_std)
            runner.snapshot_actor()
            print(f"[INFO] warm start from {args_cli.p3_policy}; critic-only for {args_cli.critic_warmup} iterations")
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        try:
            reason = runner.learn_until_stop(agent_cfg.max_iterations, init_at_random_ep_len=False)
            print(f"[INFO] stopped: {reason}")
        finally:
            finish_run_meta(meta, runner)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
