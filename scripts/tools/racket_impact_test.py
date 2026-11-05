"""P1: racket impact and swing measurements in Isaac (design doc 5).

Two modes, both on the ``BadmintonLab-Game-Serve`` task:

``plate``  Player 0 is lifted off the ground and kinematically replayed every physics
           step: joints held at the ready pose, the whole body rigidly rotated so the
           racket face has the commanded elevation and translated so the sweet spot
           moves along the face normal at the commanded speed. The shuttle comes
           straight down at ``--v_in`` m/s onto the sweet spot. This isolates the
           contact model: hit rate, tunnelling, restitution, double contacts and a
           shuttle resting on the racket.
``robot``  The real robot runs a scripted swing (``docs/swing_search.json``) with the
           real actuators and delays, from the ``player_cfg`` reset pose. Whichever
           player serves runs the swing (side 1 is a rotation of side 0, so the same
           joint commands apply). Scored with the play rules.

    python scripts/tools/racket_impact_test.py --headless --mode plate --num_envs 1024
    python scripts/tools/racket_impact_test.py --headless --mode robot --num_envs 1024 --candidate 0

Output: ``<out>.json`` and an appended ``<out>.md`` table.
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
parser.add_argument("--v_in", type=float, default=4.4, help="Plate mode: incoming vertical shuttle speed.")
parser.add_argument("--candidate", type=int, default=0)
parser.add_argument("--rallies", type=int, default=2, help="Robot mode: rallies per court.")
parser.add_argument("--max_ctrl_steps", type=int, default=3000, help="Robot mode: hard stop.")
parser.add_argument("--out", default=None, help="Output path without suffix (default docs/IMPACT_<mode>).")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import faulthandler  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402

if os.environ.get("TRACE_EVERY"):
    faulthandler.dump_traceback_later(float(os.environ["TRACE_EVERY"]), repeat=True)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_inv, quat_mul  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import badmintonlab_game.tasks  # noqa: E402,F401
from badmintonlab_game.tasks.manager_based.badmintonlab_game.serve import player_cfg  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TASK = "BadmintonLab-Game-Serve"


def _env(num_envs, candidate=None):
    cfg = parse_env_cfg(TASK, num_envs=num_envs)
    cfg.episode_length_s = 4.0
    if candidate is not None:
        # Each swing candidate comes with its own trunk pose; override the player_cfg reset for both sides.
        data = json.loads((ROOT / "docs/swing_search.json").read_text(encoding="utf-8"))
        cand = data["results"][candidate]
        trunk = cand["trunk"]
        for side in range(2):
            sign = 1.0 if side == 0 else -1.0
            yaw = trunk["yaw"] + (0.0 if side == 0 else math.pi)
            root = getattr(cfg.events, f"reset_player_{side}_root")
            root.params["pose_range"].update(
                x=(sign * trunk["x"],) * 2, y=(sign * trunk["y"],) * 2, yaw=(yaw, yaw)
            )
            joints = getattr(cfg.events, f"reset_player_{side}_joints")
            joints.params["joint_pos"] = dict(zip(data["joints"], cand["q0"]))
        print(f"[robot] reset trunk {trunk}", flush=True)
    env = gym.make(TASK, cfg=cfg).unwrapped
    env.reset()
    return env


def _term_vec(value, like):
    if isinstance(value, torch.Tensor):
        return value[0]
    return torch.full_like(like, float(value))


def _ready_targets(env, side: int, joint_values: dict | None = None):
    """Joint targets (action order) and raw actions holding ``joint_values`` (default: READY_JOINTS)."""
    robot = env.scene[f"robot_{side}"]
    term = env.action_manager.get_term(f"player_{side}")
    q = robot.data.default_joint_pos[0].clone()
    values = joint_values if joint_values is not None else player_cfg.READY_JOINTS
    ids, _ = robot.find_joints(list(values), preserve_order=True)
    q[ids] = torch.tensor(list(values.values()), device=env.device, dtype=q.dtype)
    q_act = q[term._joint_ids]
    raw = (q_act - _term_vec(term._offset, q_act)) / _term_vec(term._scale, q_act)
    return q, raw


# ----------------------------------------------------------------------------- plate


def plate_mode(env, speeds, elevations, v_in):
    n = env.num_envs
    dev = env.device
    dt = env.physics_dt
    robot = env.scene["robot_0"]
    shuttle = env.scene["shuttle"]
    sensor = env.scene.sensors["racket_contact_0"]
    origins = env.scene.env_origins
    sweet_id = robot.find_bodies("racket_sweet_spot")[0][0]
    q_ready, _ = _ready_targets(env, 0)
    q_ready = q_ready.expand(n, -1).clone()
    zeros_j = torch.zeros_like(q_ready)

    # Root -> sweet-spot transform at the ready pose (depends only on the joint angles).
    robot.write_joint_state_to_sim(q_ready, zeros_j)
    env.sim.forward()
    p_r, q_r = robot.data.root_pos_w.clone(), robot.data.root_quat_w.clone()
    p_s, q_s = robot.data.body_pos_w[:, sweet_id].clone(), robot.data.body_quat_w[:, sweet_id].clone()
    p_rel_r = quat_apply(quat_inv(q_r), p_s - p_r)
    n0 = quat_apply(q_s, torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3))
    n0 = torch.where(n0[:, 2:3] < 0.0, -n0, n0)
    natural_el = float(torch.rad2deg(torch.asin(n0[0, 2].clamp(-1, 1))))
    print(f"[plate] natural face elevation at READY pose: {natural_el:.1f} deg", flush=True)

    # Contact point in the air above the serve position: the floating robot cannot touch the ground.
    contact = torch.tensor([-2.55, 0.0, 2.2], device=dev) + origins
    t_lead = 0.06
    steps = int(0.45 / dt)
    rows = []
    for el in elevations:
        a = math.radians(el)
        # Keep the natural heading of the face; only set its elevation.
        horiz = torch.nn.functional.normalize(torch.cat((n0[:, :2], torch.zeros(n, 1, device=dev)), dim=-1), dim=-1)
        normal = horiz * math.cos(a) + torch.tensor([0.0, 0.0, math.sin(a)], device=dev)
        axis = torch.linalg.cross(n0, normal)
        sin_t = torch.linalg.norm(axis, dim=-1)
        angle = torch.atan2(sin_t, (n0 * normal).sum(-1))
        axis = torch.where(sin_t.unsqueeze(-1) > 1e-6, axis / sin_t.clamp_min(1e-9).unsqueeze(-1), torch.tensor([0.0, 0.0, 1.0], device=dev))
        q_root = quat_mul(quat_from_angle_axis(angle, axis), q_r)
        offset = quat_apply(q_root, p_rel_r)
        for v in speeds:
            env.reset()
            t_hit = t_lead + torch.rand(n, device=dev) * dt  # spread the contact phase over one physics step
            # Shuttle straight above the contact point, falling at v_in, arriving at t_hit.
            sh_state = torch.zeros(n, 13, device=dev)
            sh_state[:, :3] = contact + torch.stack((torch.zeros_like(t_hit), torch.zeros_like(t_hit), v_in * t_hit), -1)
            sh_state[:, 3] = 1.0
            sh_state[:, 9] = -v_in
            shuttle.write_root_state_to_sim(sh_state)
            hits = torch.zeros(n, dtype=torch.long, device=dev)
            touching = torch.zeros(n, dtype=torch.bool, device=dev)
            contact_steps = torch.zeros(n, dtype=torch.long, device=dev)
            max_run = torch.zeros(n, dtype=torch.long, device=dev)
            run = torch.zeros(n, dtype=torch.long, device=dev)
            sep_vel = torch.full((n, 3), float("nan"), device=dev)
            separated = torch.zeros(n, dtype=torch.bool, device=dev)
            tunnel = torch.zeros(n, dtype=torch.bool, device=dev)
            prev_side = None
            for k in range(steps):
                t = k * dt
                centre = contact + (t - t_hit).unsqueeze(-1) * v * normal
                root_vel = torch.zeros(n, 6, device=dev)
                root_vel[:, :3] = v * normal
                robot.write_root_pose_to_sim(torch.cat((centre - offset, q_root), dim=-1))
                robot.write_root_velocity_to_sim(root_vel)
                robot.write_joint_state_to_sim(q_ready, zeros_j)
                robot.set_joint_position_target(q_ready)
                env._apply_shuttle_forces()
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(dt=dt)
                force = torch.linalg.norm(sensor.data.force_matrix_w, dim=-1).flatten(1).amax(1)
                now = force > env.cfg.hit_contact_threshold
                hits += (now & ~touching).long()
                ended = touching & ~now & ~separated
                sep_vel[ended] = shuttle.data.root_lin_vel_w[ended]
                separated |= ended
                touching = now
                contact_steps += now.long()
                run = torch.where(now, run + 1, torch.zeros_like(run))
                max_run = torch.maximum(max_run, run)
                # Tunnelling: the shuttle passes to the back of the face, near the sweet spot, without a contact.
                plate_now = contact + (t + dt - t_hit).unsqueeze(-1) * v * normal
                rel = shuttle.data.root_pos_w - plate_now
                side = (rel * normal).sum(-1)
                lateral = torch.linalg.norm(rel - side.unsqueeze(-1) * normal, dim=-1)
                if prev_side is not None:
                    tunnel |= (prev_side > 0) & (side <= 0) & (hits == 0) & (lateral < 0.08)
                prev_side = side
            hit = hits > 0
            nh = int(hit.sum())
            vn_out = (sep_vel * normal).sum(-1)
            v_in_n = -v_in * math.sin(a)
            e = (vn_out - v) / (v - v_in_n)
            good = hit & separated
            row = dict(
                elevation=el,
                speed=v,
                n=n,
                hit=nh / n,
                double=int((hits > 1).sum()) / n,
                tunnel=int(tunnel.sum()) / n,
                rest=int((max_run >= 10).sum()) / n,  # >= 50 ms of continuous contact
                never_separated=int((hit & ~separated).sum()) / n,
                mean_contact_steps=float(contact_steps[hit].float().mean()) if nh else 0.0,
                out_speed=float(torch.linalg.norm(sep_vel[good], dim=-1).mean()) if good.any() else 0.0,
                out_elev_deg=float(torch.rad2deg(torch.asin((sep_vel[good, 2] / torch.linalg.norm(sep_vel[good], dim=-1)).clamp(-1, 1))).mean()) if good.any() else 0.0,
                restitution=float(e[good].mean()) if good.any() else 0.0,
                restitution_std=float(e[good].std()) if int(good.sum()) > 1 else 0.0,
            )
            rows.append(row)
            print(row, flush=True)
    return dict(natural_elevation_deg=natural_el, v_in=v_in, rows=rows)


# ----------------------------------------------------------------------------- robot


def robot_mode(env, candidate, rallies, max_ctrl_steps):
    data = json.loads((ROOT / "docs/swing_search.json").read_text(encoding="utf-8"))
    cand = data["results"][candidate]
    joints = data["joints"]
    n = env.num_envs
    dev = env.device
    hold = []
    swing = []
    for side in range(2):
        _, h = _ready_targets(env, side, dict(zip(joints, cand["q0"])))
        _, s = _ready_targets(env, side, dict(zip(joints, cand["qc"])))
        hold.append(h)
        swing.append(s)
    trig = int(cand["pl"]["trig"]) // int(env.cfg.decimation)
    print(f"[robot] candidate {candidate}: swing at control step {trig}; q0={cand['q0']} qc={cand['qc']}", flush=True)

    c = dict(rallies=0, legal=0, served_in=0, height=0, double=0, no_hit=0, net=0, out=0, receiver_touch=0, timeouts=0)
    per_side = {0: dict(rallies=0, served_in=0), 1: dict(rallies=0, served_in=0)}
    landing_x, hit_z, hit_speed = [], [], []
    step = torch.zeros(n, dtype=torch.long, device=dev)
    target = rallies * n
    env.reset()
    shuttle = env.scene["shuttle"]

    # Per-physics-step diagnosis of serves without a registered hit: did the shuttle
    # pass through the face near the sweet spot (tunnelling) or simply miss?
    robots = (env.scene["robot_0"], env.scene["robot_1"])
    sweet_ids = tuple(r.find_bodies("racket_sweet_spot")[0][0] for r in robots)
    ex_axis = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3)
    diag = dict(
        prev=torch.zeros(n, device=dev),
        valid=torch.zeros(n, dtype=torch.bool, device=dev),
        tunnel=torch.zeros(n, dtype=torch.bool, device=dev),
        min_dist=torch.full((n,), float("inf"), device=dev),
    )
    original_substep = env._physics_substep

    def substep_with_diag():
        original_substep()
        srv0 = (env.initial_hitter == 0).unsqueeze(-1)
        p = torch.where(srv0, robots[0].data.body_pos_w[:, sweet_ids[0]], robots[1].data.body_pos_w[:, sweet_ids[1]])
        q = torch.where(srv0, robots[0].data.body_quat_w[:, sweet_ids[0]], robots[1].data.body_quat_w[:, sweet_ids[1]])
        nrm = quat_apply(q, ex_axis)
        rel = shuttle.data.root_pos_w - p
        side = (rel * nrm).sum(-1)
        lateral = torch.linalg.norm(rel - side.unsqueeze(-1) * nrm, dim=-1)
        no_hit_yet = env.rules.last_hitter < 0
        diag["tunnel"] |= diag["valid"] & (diag["prev"] * side < 0) & (lateral < 0.08) & no_hit_yet
        diag["min_dist"] = torch.where(
            no_hit_yet, torch.minimum(diag["min_dist"], torch.linalg.norm(rel, dim=-1)), diag["min_dist"]
        )
        diag["prev"] = side
        diag["valid"] = torch.ones_like(diag["valid"])

    env._physics_substep = substep_with_diag
    c.update(no_hit_tunnel=0)
    miss_dist = []
    import time

    t_wall = time.time()
    for it in range(max_ctrl_steps):
        if it % 100 == 0:
            print(f"[robot] ctrl step {it}: rallies {c['rallies']}/{target}, wall {time.time() - t_wall:.0f}s", flush=True)
        server = env.initial_hitter.clone()
        swinging = (step >= trig).unsqueeze(-1)
        actions = []
        for side in range(2):
            a = torch.where(swinging & (server == side).unsqueeze(-1), swing[side], hold[side])
            actions.append(a)
        _, _, term_, trunc, ex = env.step(torch.cat(actions, dim=-1))
        step += 1
        # Valid hits this step (env not reset in the same step): racket height and shuttle speed after contact.
        jh = env.just_hit.gather(1, env.initial_hitter.unsqueeze(1)).squeeze(1) & (env.hit_count.sum(1) == 1)
        if jh.any():
            hit_z += env.rules.hit_height.gather(1, env.initial_hitter.unsqueeze(1)).squeeze(1)[jh].tolist()
            hit_speed += torch.linalg.norm(shuttle.data.root_lin_vel_w[jh], dim=-1).tolist()
        done = ex["rally_resolved"]
        c["timeouts"] += int((trunc & ~done).sum())
        if done.any():
            srv = server[done]
            sign = torch.where(srv == 0, 1.0, -1.0)
            hc = ex["hit_count"][done]
            served = hc.gather(1, srv.unsqueeze(1)).squeeze(1) > 0
            recv = hc.gather(1, (1 - srv).unsqueeze(1)).squeeze(1) > 0
            height = ex["serve_height_fault"][done]
            double = ex["double_hit_fault"][done]
            winner = ex["rally_winner"][done]
            net = ex["rally_net_fault"][done]
            out = ex["rally_out_fault"][done]
            legal = served & ~height
            ok = legal & ~double & ~net & ~out & ~recv & (winner == srv)
            c["rallies"] += int(done.sum())
            c["legal"] += int(legal.sum())
            c["served_in"] += int(ok.sum())
            c["height"] += int(height.sum())
            c["double"] += int(double.sum())
            c["no_hit"] += int((~served & ~height).sum())
            c["net"] += int((net & legal).sum())
            c["out"] += int((out & legal).sum())
            c["receiver_touch"] += int((recv & legal).sum())
            for side in range(2):
                m = srv == side
                per_side[side]["rallies"] += int(m.sum())
                per_side[side]["served_in"] += int((ok & m).sum())
            pos = ex["shuttle_local_pos"][done]
            landing_x += (pos[ok, 0] * sign[ok]).tolist()
            nohit = ~served & ~height
            tun = diag["tunnel"][done]
            c["no_hit_tunnel"] += int((nohit & tun).sum())
            miss_dist += diag["min_dist"][done][nohit & ~tun].tolist()
        reset = term_ | trunc
        diag["valid"] &= ~reset
        diag["tunnel"] &= ~reset
        diag["min_dist"][reset] = float("inf")
        step[term_ | trunc] = 0
        if c["rallies"] >= target:
            break
    r = max(c["rallies"], 1)
    rates = {f"{k}_rate": v / r for k, v in c.items() if k not in ("rallies", "timeouts")}
    lx = torch.tensor(landing_x) if landing_x else torch.zeros(0)
    summary = dict(
        candidate=candidate,
        trigger_ctrl_step=trig,
        counts=c,
        rates=rates,
        per_side=per_side,
        landing_x=dict(mean=float(lx.mean()) if len(lx) else None, min=float(lx.min()) if len(lx) else None, max=float(lx.max()) if len(lx) else None),
        hit_height=dict(mean=float(torch.tensor(hit_z).mean()) if hit_z else None, n=len(hit_z)),
        post_hit_speed=dict(mean=float(torch.tensor(hit_speed).mean()) if hit_speed else None),
        no_hit_without_tunnel_min_dist=dict(
            n=len(miss_dist),
            quantiles=torch.tensor(miss_dist).quantile(torch.tensor([0.1, 0.5, 0.9])).tolist() if miss_dist else None,
        ),
    )
    env._physics_substep = original_substep
    return summary


def main():
    env = _env(args_cli.num_envs, args_cli.candidate if args_cli.mode == "robot" else None)
    out = ROOT / (args_cli.out or f"docs/IMPACT_{args_cli.mode}")
    if args_cli.mode == "plate":
        result = plate_mode(
            env,
            [float(s) for s in args_cli.speeds.split(",")],
            [float(e) for e in args_cli.elevations.split(",")],
            args_cli.v_in,
        )
        md = [
            f"plate mode, natural elevation {result['natural_elevation_deg']:.1f} deg, v_in {result['v_in']} m/s",
            "",
            "| elevation | speed | hit | double | tunnel | rest | never sep | contact steps | out speed | out elev | restitution |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in result["rows"]:
            md.append(
                f"| {r['elevation']:.0f} | {r['speed']:.0f} | {r['hit']:.3f} | {r['double']:.3f} | {r['tunnel']:.3f} | "
                f"{r['rest']:.3f} | {r['never_separated']:.3f} | {r['mean_contact_steps']:.1f} | {r['out_speed']:.2f} | "
                f"{r['out_elev_deg']:.0f} | {r['restitution']:.2f}±{r['restitution_std']:.2f} |"
            )
    else:
        result = robot_mode(env, args_cli.candidate, args_cli.rallies, args_cli.max_ctrl_steps)
        md = [f"robot mode: {json.dumps(result)}"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    with open(out.with_suffix(".md"), "a", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n\n")
    print("\n".join(md), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Print before close(): Kit can hang in close() after an exception and swallow it.
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        simulation_app.close()
