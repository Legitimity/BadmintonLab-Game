"""P1: racket impact and swing measurements in Isaac (design doc 5). Not run yet: needs the GPU.

Two modes, both on the ``BadmintonLab-Game-Serve`` task:

``plate``  The robot is kinematically held; every physics step the racket link is
           teleported along a straight line at a commanded normal speed through the
           drop path. This isolates the contact model: hit rate, tunnelling,
           restitution, double contacts and a shuttle resting on the racket.
``robot``  The real robot runs a scripted swing (``docs/swing_search.json``) with the
           real actuators and delays, from the ``player_cfg`` reset pose. This is the
           true serve test, scored with the play rules.

    python scripts/tools/racket_impact_test.py --headless --mode plate --num_envs 1024
    python scripts/tools/racket_impact_test.py --headless --mode robot --num_envs 1024 --candidate 0

Output: ``docs/IMPACT_TABLE.md`` and a JSON next to it.
"""

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("plate", "robot"), required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--speeds", default="2,3,4,5,6,7,8,9,10")
parser.add_argument("--elevations", default="45,55,65")
parser.add_argument("--candidate", type=int, default=0)
parser.add_argument("--rallies", type=int, default=2)
parser.add_argument("--out", default="docs/IMPACT_TABLE")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_from_matrix  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import badmintonlab_game.tasks  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[2]
TASK = "BadmintonLab-Game-Serve"


def _env(num_envs):
    cfg = parse_env_cfg(TASK, num_envs=num_envs)
    cfg.episode_length_s = 4.0
    env = gym.make(TASK, cfg=cfg).unwrapped
    env.reset()
    return env


def plate_mode(env, speeds, elevations):
    """Teleport the racket along its normal through the drop path.

    The racket is a kinematic replay: its pose and velocity are written every
    physics step, so the shuttle sees a moving rigid plate with the real collider
    and material. The robot is held still far from the shuttle otherwise.
    """
    n = env.num_envs
    robot = env.scene["robot_0"]
    racket_id = robot.find_bodies("racket_link")[0][0]
    sweet_offset = 0.52  # sweet spot along the racket z axis
    shuttle = env.scene["shuttle"]
    origins = env.scene.env_origins
    rows = []
    for el in elevations:
        for v in speeds:
            env.reset()
            # Contact point: the drop path at z=0.95. Spread the phase over one physics step.
            phase = torch.rand(n, device=env.device)
            a = math.radians(el)
            normal = torch.tensor([math.cos(a), 0.0, math.sin(a)], device=env.device)
            # Racket frame: x = face normal, z = handle direction (in the face plane, pointing up the slope).
            z_axis = torch.tensor([-math.sin(a), 0.0, math.cos(a)], device=env.device)
            y_axis = torch.linalg.cross(z_axis, normal)
            rot = torch.stack((normal, y_axis, z_axis), dim=1)
            quat = quat_from_matrix(rot).expand(n, 4)
            contact = torch.tensor([-2.55, 0.0, 0.95], device=env.device) + origins
            t_hit = 0.53 + phase * env.physics_dt  # seconds after the drop, ~z=0.95 [computed]
            stats = dict(hit=0, double=0, tunnel=0, rest=0, out_speed=[], restitution=[])
            prev_side = None
            touched = torch.zeros(n, dtype=torch.bool, device=env.device)
            hits = torch.zeros(n, dtype=torch.long, device=env.device)
            for k in range(int(1.0 / env.physics_dt)):
                t = k * env.physics_dt
                centre = contact + (t - t_hit).unsqueeze(-1) * v * normal
                link = centre - sweet_offset * z_axis
                pose = torch.cat((link, quat), dim=-1)
                robot.write_root_pose_to_sim(pose)  # robot root follows; arm is the rigid body chain
                vel = torch.zeros(n, 6, device=env.device)
                vel[:, :3] = v * normal
                robot.write_root_velocity_to_sim(vel)
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(dt=env.physics_dt)
                force = torch.linalg.norm(env.scene.sensors["racket_contact_0"].data.force_matrix_w, dim=-1).flatten(1).amax(1)
                now = force > env.cfg.hit_contact_threshold
                hits += (now & ~touched).long()
                touched = now
                rel = shuttle.data.root_pos_w - centre
                side = torch.sign(rel @ normal)
                if prev_side is not None:
                    crossed = (side != prev_side) & (hits == 0)
                    stats["tunnel"] += int(crossed.sum())
                prev_side = side
            vout = shuttle.data.root_lin_vel_w
            stats["hit"] = int((hits > 0).sum())
            stats["double"] = int((hits > 1).sum())
            z = shuttle.data.root_pos_w[:, 2] - origins[:, 2]
            stats["rest"] = int(((z > 0.03) & (torch.linalg.norm(vout, dim=-1) < 0.05)).sum())
            vn = vout @ normal
            stats["out_speed"] = float(torch.linalg.norm(vout, dim=-1)[hits > 0].mean()) if stats["hit"] else 0.0
            # Effective restitution against the racket normal speed v and incoming normal speed.
            v_in = -4.4 * math.sin(a)
            stats["restitution"] = float(((vn - v) / (v - v_in))[hits > 0].mean()) if stats["hit"] else 0.0
            rows.append(dict(elevation=el, speed=v, n=n, **stats))
            print(rows[-1], flush=True)
    return rows


def robot_mode(env, candidate, rallies):
    """Scripted serve with the real robot. Joint targets switch at the searched control step."""
    data = json.loads((ROOT / "docs/swing_search.json").read_text(encoding="utf-8"))
    cand = data["results"][candidate]
    joints = data["joints"]
    robot = env.scene["robot_0"]
    term = env.action_manager.get_term("player_0")
    ids, _ = robot.find_joints(joints, preserve_order=True)
    default = robot.data.default_joint_pos[0]
    scale = term._scale[0]

    def raw(q):
        a = torch.zeros(23, device=env.device)
        a[ids] = (torch.tensor(q, device=env.device) - default[ids]) / scale[ids]
        return a

    hold, swing = raw(cand["q0"]), raw(cand["qc"])
    trig = int(cand["pl"]["trig"] // 4)
    counts = dict(rallies=0, legal=0, served_in=0, height=0, double=0, no_hit=0, net=0, out=0, stuck=0)
    step = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    target = rallies * env.num_envs
    while counts["rallies"] < target:
        server0 = env.initial_hitter == 0
        a0 = torch.where((step >= trig).unsqueeze(-1), swing, hold)
        a0 = torch.where(server0.unsqueeze(-1), a0, hold)
        action = torch.cat((a0, hold.expand(env.num_envs, 23)), dim=-1)
        _, _, term_, trunc, extras = env.step(action)
        step += 1
        done = extras["rally_resolved"] & server0
        counts["stuck"] += int((trunc & ~extras["rally_resolved"]).sum())
        if done.any():
            winner = extras["rally_winner"][done]
            counts["rallies"] += int(done.sum())
            counts["height"] += int(extras["serve_height_fault"][done].sum())
            counts["double"] += int(extras["double_hit_fault"][done].sum())
            counts["served_in"] += int(((winner == 0) & ~extras["double_hit_fault"][done]).sum())
        step[(term_ | trunc)] = 0
    return dict(candidate=candidate, **counts)


def main():
    env = _env(args_cli.num_envs)
    if args_cli.mode == "plate":
        rows = plate_mode(env, [float(s) for s in args_cli.speeds.split(",")], [float(e) for e in args_cli.elevations.split(",")])
        md = ["| elevation | speed | hit | double | tunnel | rest | out speed | restitution |", "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            md.append(
                f"| {r['elevation']:.0f} | {r['speed']:.0f} | {r['hit'] / r['n']:.3f} | {r['double'] / r['n']:.3f} | "
                f"{r['tunnel'] / r['n']:.3f} | {r['rest'] / r['n']:.3f} | {r['out_speed']:.2f} | {r['restitution']:.2f} |"
            )
        result = rows
    else:
        result = robot_mode(env, args_cli.candidate, args_cli.rallies)
        md = [f"robot mode: {json.dumps(result)}"]
    out = ROOT / args_cli.out
    out.parent.mkdir(exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    with open(out.with_suffix(".md"), "a", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n\n")
    print("\n".join(md))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
