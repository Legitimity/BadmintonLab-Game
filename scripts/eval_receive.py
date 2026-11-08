"""Evaluate a receive-expert checkpoint in the training env (no updates), per distance band.

Example::

    python scripts/eval_receive.py --headless --num_envs 512 --dt_ms 1 --steps 1500 \
        --checkpoint D:/.../model_passed_S1_2514.pt --level 4 --out D:/rl-badminton-dev/recv_logs/eval_s1_dt1.json

``--level 4`` = uniform over the filtered feeder table; 3 = the S1 gate mix (80% >= 1.4 m).
``--stochastic`` samples actions (as in training); default is the deterministic mean.
"""

import argparse
import json
import sys
import time

import torch  # noqa: F401
import tensordict  # noqa: F401
from rsl_rl.runners import OnPolicyRunner  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="BadmintonLab-Game-Receive")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--dt_ms", type=int, default=1)
parser.add_argument("--steps", type=int, default=1500, help="control steps (20 ms each)")
parser.add_argument("--level", type=int, default=4)
parser.add_argument("--stage", default="S1")
parser.add_argument("--stochastic", action="store_true")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import badmintonlab_game.tasks  # noqa: E402,F401
from badmintonlab_game.tasks.manager_based.badmintonlab_game import receive as _receive  # noqa: E402,F401
from badmintonlab_game.tasks.manager_based.badmintonlab_game.physics import apply_physics  # noqa: E402
from badmintonlab_game.tasks.manager_based.badmintonlab_game.receive.agents import ReceiveRunner, ReceiveVecEnvWrapper  # noqa: E402

from badmintonlab_game.tasks.manager_based.badmintonlab_game.serve import player_cfg  # noqa: E402
from badmintonlab_game.tasks.manager_based.badmintonlab_game.serve.wrapper import hold_action  # noqa: E402

LEG = ("Hip", "Knee", "Ankle")


def _ready_action(unwrapped, side: int) -> torch.Tensor:
    """Same as train_receive._ready_action (that module launches the app at import)."""
    robot = unwrapped.scene[f"robot_{side}"]
    term = unwrapped.action_manager.get_term(f"player_{side}")
    q = robot.data.default_joint_pos[0].clone()
    ids, _ = robot.find_joints(list(player_cfg.READY_JOINTS), preserve_order=True)
    q[ids] = torch.tensor(list(player_cfg.READY_JOINTS.values()), device=q.device)
    q = q[term._joint_ids] if not isinstance(term._joint_ids, slice) else q
    scale = term._scale[0] if isinstance(term._scale, torch.Tensor) else torch.full_like(q, term._scale)
    offset = term._offset[0] if isinstance(term._offset, torch.Tensor) else torch.full_like(q, term._offset)
    return hold_action(q, offset, scale)


def main():
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    apply_physics(env_cfg, args_cli.dt_ms)
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    print(f"[EVAL] dt={u.physics_dt} decimation={env_cfg.decimation} step_dt={u.step_dt}", flush=True)
    wrapped = ReceiveVecEnvWrapper(env, _ready_action(u, 1), clip_actions=agent_cfg.clip_actions)
    names = u.action_manager.get_term("player_0")._joint_names
    leg_mask = torch.tensor([any(k in n for k in LEG) for n in names])
    runner = ReceiveRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device, leg_mask=leg_mask, critic_warmup=0)
    runner.load(args_cli.checkpoint, load_optimizer=False)
    u.set_level(args_cli.level)
    u.set_stage(args_cli.stage)
    policy = runner.alg.policy
    policy.eval()
    obs = wrapped.get_observations().to(runner.device)
    u.pop_stats()
    t0 = time.time()
    with torch.inference_mode():
        for k in range(args_cli.steps):
            a = policy.act(obs) if args_cli.stochastic else policy.act_inference(obs)
            obs, _, _, _ = wrapped.step(a)
            if k == 49:
                u.pop_stats()  # drop the first second (all courts start in sync)
                t1 = time.time()
    wall = time.time() - t1
    r = u.pop_stats()
    r.update(dt_ms=args_cli.dt_ms, level=args_cli.level, steps=args_cli.steps, num_envs=args_cli.num_envs,
             stochastic=args_cli.stochastic, checkpoint=args_cli.checkpoint, sec_per_ctrl_step=wall / (args_cli.steps - 50),
             wall_s=time.time() - t0)  # fmt: skip
    line = json.dumps(r)
    print("[RECV_EVAL]" + line, flush=True)
    if args_cli.out:
        with open(args_cli.out, "w", encoding="utf-8") as f:
            f.write(line + "\n")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
