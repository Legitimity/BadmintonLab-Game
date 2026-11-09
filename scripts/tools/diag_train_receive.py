"""Training-env counterpart of diag_play_receive.py: V3.4 code (git worktree _v34wt), only real s0 serves.

Logs per step the 100-D actor observation, shuttle state, root, sweet spot, hits, ball activity, for
comparison with the play rollout. Deterministic actions.
"""

import argparse
import sys

import torch  # noqa: F401
import tensordict  # noqa: F401
from rsl_rl.runners import OnPolicyRunner  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--serves", required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--dt_ms", type=int, default=1)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import badmintonlab_game.tasks  # noqa: E402,F401
from badmintonlab_game.tasks.manager_based.badmintonlab_game import receive as _r  # noqa: E402,F401
from badmintonlab_game.tasks.manager_based.badmintonlab_game.physics import apply_physics  # noqa: E402
from badmintonlab_game.tasks.manager_based.badmintonlab_game.receive.agents import ReceiveRunner, ReceiveVecEnvWrapper  # noqa: E402
from badmintonlab_game.tasks.manager_based.badmintonlab_game.serve import player_cfg  # noqa: E402
from badmintonlab_game.tasks.manager_based.badmintonlab_game.serve.wrapper import hold_action  # noqa: E402

print("[DIAGT] package", badmintonlab_game.tasks.__file__, flush=True)


def _ready_action(u, side):
    robot = u.scene[f"robot_{side}"]
    term = u.action_manager.get_term(f"player_{side}")
    q = robot.data.default_joint_pos[0].clone()
    ids, _ = robot.find_joints(list(player_cfg.READY_JOINTS), preserve_order=True)
    q[ids] = torch.tensor(list(player_cfg.READY_JOINTS.values()), device=q.device)
    q = q[term._joint_ids] if not isinstance(term._joint_ids, slice) else q
    scale = term._scale[0] if isinstance(term._scale, torch.Tensor) else torch.full_like(q, term._scale)
    offset = term._offset[0] if isinstance(term._offset, torch.Tensor) else torch.full_like(q, term._offset)
    return hold_action(q, offset, scale)


def main():
    task = "BadmintonLab-Game-Receive"
    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = 7
    apply_physics(env_cfg, args.dt_ms)
    env = gym.make(task, cfg=env_cfg)
    u = env.unwrapped
    w = ReceiveVecEnvWrapper(env, _ready_action(u, 1), clip_actions=agent_cfg.clip_actions)
    names = u.action_manager.get_term("player_0")._joint_names
    leg = torch.tensor([any(k in n for k in ("Hip", "Knee", "Ankle")) for n in names])
    runner = ReceiveRunner(w, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device, leg_mask=leg, critic_warmup=0)
    runner.recv_curriculum = None
    runner.load(args.checkpoint, load_optimizer=False)
    u.set_level(4)
    u.set_stage("S2")
    u.set_real_serves([args.serves], 1.0)
    pol = runner.alg.policy
    pol.eval()
    obs = w.get_observations().to(runner.device)
    # Re-inject so every court starts with a real serve.
    u._reset_idx(torch.arange(u.num_envs, device=u.device))
    obs = w.get_observations().to(runner.device)
    r0 = u._robots[0]
    sh = u._shuttle
    org = u.scene.env_origins
    ss = u._sweet_spot_ids[0]
    rec = {k: [] for k in ("obs0", "sh_pos", "sh_vel", "root0", "ss0", "hit0", "active", "a0")}
    with torch.inference_mode():
        for k in range(args.steps):
            a = pol.act_inference(obs)
            rec["obs0"].append(obs["policy"].clone().cpu())
            obs, _, _, _ = w.step(a)
            rec["sh_pos"].append((sh.data.root_pos_w - org).cpu())
            rec["sh_vel"].append(sh.data.root_lin_vel_w.clone().cpu())
            rec["root0"].append((r0.data.root_pos_w - org).cpu())
            rec["ss0"].append((r0.data.body_pos_w[:, ss] - org).cpu())
            rec["hit0"].append(u.rules.just_hit[:, 0].clone().cpu())
            rec["active"].append(u.ball_active.clone().cpu())
            rec["a0"].append(a.cpu())
    out = {k: torch.stack(v) for k, v in rec.items()}
    torch.save(out, args.out)
    print("[DIAGT] touched first ball:", u.stats.tolist()[:3], flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import os
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    finally:
        app.close()
