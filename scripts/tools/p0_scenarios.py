"""P0 scenario tests in Isaac (design doc 4.3). Needs the GPU. Run once on the train task and once on play.

Each scenario writes the shuttle state directly and, where a racket is needed, holds
the racket of player 0 at a fixed pose. The outcome is read from the environment's
``rally_resolved`` / ``rally_winner`` / fault flags.

    python scripts/tools/p0_scenarios.py --headless --task BadmintonLab-Game-Serve
    # in a play checkout, with PYTHONPATH pointing at play:
    python scripts/tools/p0_scenarios.py --headless --task BadmintonLab-Game-Play

Both runs write ``docs/P0_<task>.json``; ``--compare`` diffs them.
"""

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="BadmintonLab-Game-Serve")
parser.add_argument("--repeats", type=int, default=8)
parser.add_argument("--out_dir", default=str(Path(__file__).resolve().parents[2] / "docs"))
parser.add_argument("--compare", nargs=2, default=None, help="Two result files; no simulation.")
args_cli, hydra_args = parser.parse_known_args()
if args_cli.compare:
    a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in args_cli.compare)
    bad = {k: (a[k], b.get(k)) for k in a if a[k] != b.get(k)}
    print(json.dumps(bad or "identical", indent=1))
    raise SystemExit(1 if bad else 0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
sys.argv = [sys.argv[0]] + hydra_args
args_cli.headless = True
app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import badmintonlab_game.tasks  # noqa: E402,F401

R = args_cli.repeats
SCENARIOS = {
    # name: (shuttle pos, shuttle vel, opener, racket centre or None, expected winner, expected flag)
    "1_static_racket_high": ((-2.55, 0.0, 1.45), (0, 0, -0.5), 0, (-2.55, 0.0, 1.30), 1, "serve_height_fault"),
    "2_static_racket_low": ((-2.55, 0.0, 1.10), (0, 0, -3.0), 0, (-2.55, 0.0, 0.95), None, "legal_hit"),
    "4_no_hit_lands_own_half": ((-2.55, 0.0, 0.30), (0, 0, -4.0), 0, None, 1, None),
    "4b_no_hit_deflected_to_other_half": ((0.6, 0.0, 0.20), (0.5, 0, -4.0), 0, None, 0, None),
    "5_net_fault": ((-1.0, 0.0, 1.00), (6.0, 0, 0.5), 0, None, None, "net"),
    "6_out_in_the_air": ((-1.0, 2.40, 2.50), (0.0, 3.0, 0.0), 0, None, None, "out"),
    "8_through_the_net_plane": ((-0.3, 0.0, 1.20), (3.0, 0, 0.0), 0, None, None, "no_bounce"),
}


def _env(task):
    cfg = parse_env_cfg(task, num_envs=R)
    if hasattr(cfg, "episode_length_s"):
        cfg.episode_length_s = 5.0
    env = gym.make(task, cfg=cfg).unwrapped
    env.reset()
    return env


def _hold(env):
    """Zero action keeps both robots at their reset pose (as close as PD allows)."""
    return torch.zeros(env.num_envs, 46, device=env.device)


def _place_racket(env, centre):
    """Teleport player 0's root so its sweet spot sits at ``centre``; keeps orientation."""
    robot = env.scene["robot_0"]
    sweet = robot.data.body_pos_w[:, robot.find_bodies("racket_sweet_spot")[0][0]]
    root = robot.data.root_pos_w.clone()
    target = torch.tensor(centre, device=env.device) + env.scene.env_origins
    pose = torch.cat((root + (target - sweet), robot.data.root_quat_w), dim=-1)
    robot.write_root_pose_to_sim(pose)


def run_scenario(env, spec):
    pos, vel, opener, racket, _, _ = spec
    env.reset()
    sh = env.scene["shuttle"]
    if racket is not None:
        _place_racket(env, racket)
    state = torch.zeros(env.num_envs, 13, device=env.device)
    state[:, :3] = torch.tensor(pos, device=env.device) + env.scene.env_origins
    state[:, 3] = 1.0
    state[:, 7:10] = torch.tensor(vel, device=env.device)
    sh.write_root_state_to_sim(state)
    env.next_hitter[:] = opener
    if hasattr(env, "rules"):
        env.rules.reset(torch.arange(env.num_envs, device=env.device), torch.full((env.num_envs,), opener, device=env.device), state[:, :3])
    else:
        env.prev_shuttle_pos_w[:] = state[:, :3]
        env.curr_shuttle_pos_w[:] = state[:, :3]
    env.scene.write_data_to_sim()
    env.sim.forward()
    result = dict(resolved=[False] * env.num_envs, winner=[-1] * env.num_envs, flags=[[] for _ in range(env.num_envs)])
    x_prev = None
    for _ in range(150):
        out = env.step(_hold(env))
        info = out[-1]
        x = sh.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
        if x_prev is not None and spec[5] == "no_bounce":
            for i in torch.nonzero((x_prev < 0) & (x >= 0)).flatten().tolist():
                result["flags"][i].append("crossed_net_plane")
        x_prev = x
        for i in torch.nonzero(info["rally_resolved"]).flatten().tolist():
            if not result["resolved"][i]:
                result["resolved"][i] = True
                result["winner"][i] = int(info["rally_winner"][i])
                for k in ("serve_height_fault", "double_hit_fault"):
                    if bool(info[k][i]):
                        result["flags"][i].append(k)
        if all(result["resolved"]):
            break
    return result


def main():
    env = _env(args_cli.task)
    if "rally_winner" not in env.extras and not hasattr(env, "rally_winner"):
        raise SystemExit("this task does not expose rally_winner")
    out = {}
    for name, spec in SCENARIOS.items():
        r = run_scenario(env, spec)
        winners = sorted(set(r["winner"]))
        flags = sorted({f for fl in r["flags"] for f in fl})
        out[name] = dict(winners=winners, flags=flags, all_resolved=all(r["resolved"]))
        exp_w, exp_f = spec[4], spec[5]
        ok = (exp_w is None or winners == [exp_w]) and (exp_f in (None, "legal_hit", "net", "out", "no_bounce") or exp_f in flags)
        out[name]["meets_expectation"] = bool(ok)
        print(name, out[name], flush=True)
    path = Path(args_cli.out_dir) / f"P0_{args_cli.task}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {path}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
