"""P0 rule-consistency scenarios (TRAINING_DESIGN_V3 section 4.3).

Runs the same constructed scenarios on any badminton task -- the training task
(``BadmintonLab-Game-Serve``), the default task (``BadmintonLab-Game``) or the
evaluator (``BadmintonLab-Game-Play`` from the ``play`` source tree) -- and
reports the judged rally result of each.  One court per scenario, all in
parallel.

Robots are "held": before every control step their root pose, joint state and
velocities are rewritten, and the actions are zero (default joint targets), so
the racket is quasi-static.  Rally state (``next_hitter``/``last_hitter``) and
the shuttle state are injected directly where V3 allows it (scenarios 5-8).

Usage (training env)::

    python scripts/tools/p0_rule_scenarios.py --task BadmintonLab-Game-Serve

Usage (play env; the play source must shadow the editable install)::

    PYTHONPATH=/root/autodl-tmp/BadmintonLab-Game-Play/source/badmintonlab_game \\
    python scripts/tools/p0_rule_scenarios.py --task BadmintonLab-Game-Play \\
        --expect_source /root/autodl-tmp/BadmintonLab-Game-Play
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="BadmintonLab-Game-Serve")
parser.add_argument("--max_steps", type=int, default=250, help="Control steps after scenario start.")
parser.add_argument("--json_out", default=None)
parser.add_argument("--expect_source", default=None, help="Assert badmintonlab_game is imported from this tree.")
parser.add_argument("--informational", action="store_true", help="Always exit 0 (e.g. for the default task).")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import badmintonlab_game  # noqa: E402
import badmintonlab_game.tasks  # noqa: E402, F401

PACKAGE_FILE = os.path.abspath(badmintonlab_game.__file__)
if args_cli.expect_source and not PACKAGE_FILE.startswith(os.path.abspath(args_cli.expect_source)):
    raise RuntimeError(f"badmintonlab_game imported from {PACKAGE_FILE}, expected under {args_cli.expect_source}")

FAR_0 = (-6.0, -2.2)
FAR_1 = (6.0, -2.2)
SERVE_XY = (-3.35, 0.0)
WARMUP_STEPS = 3
SETTLE_STEPS = 3
T0 = WARMUP_STEPS + SETTLE_STEPS
THROW_OFFSET = 0.07
THROW_SPEED = 2.5
RETHROW_DELAY = 2

# kind: racket (shuttle thrown at a held racket), drop (reset-style drop), inject (state written directly)
SCENARIOS = [
    dict(name="1_racket_1.30m_serve", kind="racket", server=0, racket_z=1.30, expect=dict(winner=1, shf=True, dhf=False)),
    dict(name="2_racket_0.95m_serve", kind="racket", server=0, racket_z=0.95, clear_after_hit=True, expect=dict(hit_by=0, shf=False, dhf=False)),
    dict(name="3_double_hit", kind="racket", server=0, racket_z=0.95, rethrow=True, expect=dict(winner=1, dhf=True)),
    dict(name="4a_no_hit_server0", kind="drop", server=0, pos=(-2.55, 0.0, 2.2), vel=(0, 0, 0), expect=dict(winner=1)),
    dict(name="4b_no_hit_server1", kind="drop", server=1, pos=(2.55, 0.0, 2.2), vel=(0, 0, 0), expect=dict(winner=0)),
    dict(name="5a_net_by_p0", kind="inject", last=0, pos=(-1.0, 0.0, 1.0), vel=(6.0, 0, 1.0), expect=dict(winner=1)),
    dict(name="5b_net_by_p1", kind="inject", last=1, pos=(1.0, 0.0, 1.0), vel=(-6.0, 0, 1.0), expect=dict(winner=0)),
    dict(name="6a_side_out_in_air_p0", kind="inject", last=0, pos=(2.0, 2.3, 2.0), vel=(0, 3.0, 1.0), expect=dict(winner=1)),
    dict(name="6b_long_out_in_air_p0", kind="inject", last=0, pos=(4.5, 0.0, 2.5), vel=(8.0, 0, 2.0), expect=dict(winner=1)),
    dict(name="7_legal_shot_in_p0", kind="inject", last=0, pos=(-1.0, 0.0, 2.0), vel=(5.0, 0, 2.0), expect=dict(winner=0)),
]
RACKET_PARK = (-1.0, 0.0, 3.0)


def main():
    n = len(SCENARIOS)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device or "cuda:0", num_envs=n)
    env_cfg.seed = 0
    if hasattr(env_cfg, "episode_length_s"):
        env_cfg.episode_length_s = 60.0
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    dev = base.device
    origins = base.scene.env_origins
    robots = (base.scene["robot_0"], base.scene["robot_1"])
    shuttle = base.scene["shuttle"]
    sweet_id = robots[0].find_bodies("racket_sweet_spot")[0][0]
    all_ids = torch.arange(n, device=dev)
    racket_ids = torch.tensor([i for i, s in enumerate(SCENARIOS) if s["kind"] == "racket"], device=dev)

    results: list[dict | None] = [None] * n
    hit_info: list[dict | None] = [None] * n
    step_counter = {"k": 0}
    original_reset = base._reset_idx

    def snapshot_reset(env_ids):
        ids = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
        if step_counter["k"] >= T0:
            for i in ids:
                if results[i] is not None:
                    continue
                if hasattr(base, "rally_winner"):
                    winner = int(base.rally_winner[i])
                else:
                    loser = int(base.rally_loser[i])
                    winner = 1 - loser if loser >= 0 else -1
                results[i] = dict(
                    step=step_counter["k"] - T0,
                    resolved=bool(base.rally_resolved[i]),
                    winner=winner,
                    shf=bool(base.serve_height_fault[i]) if hasattr(base, "serve_height_fault") else None,
                    dhf=bool(base.double_hit_fault[i]) if hasattr(base, "double_hit_fault") else None,
                    last_hitter=int(base.last_hitter[i]),
                    shuttle_local=(shuttle.data.root_pos_w[i] - origins[i]).tolist(),
                )
        original_reset(env_ids)

    base._reset_idx = snapshot_reset

    root_z0 = robots[0].data.default_root_state[0, 2].item()
    quat = (
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev),
        torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev),
    )
    hold_pos = torch.zeros((2, n, 3), device=dev)
    for i, s in enumerate(SCENARIOS):
        p0 = (*SERVE_XY, root_z0) if s["kind"] == "racket" else (*FAR_0, root_z0)
        hold_pos[0, i] = torch.tensor(p0, device=dev)
        hold_pos[1, i] = torch.tensor((*FAR_1, root_z0), device=dev)

    park = torch.zeros((n, 3), device=dev)
    for i, s in enumerate(SCENARIOS):
        park[i] = torch.tensor(RACKET_PARK if s["kind"] == "racket" else s["pos"], device=dev)

    def hold_robots(ids):
        for side, robot in enumerate(robots):
            pose = torch.cat((hold_pos[side, ids] + origins[ids], quat[side].expand(len(ids), 4)), dim=-1)
            robot.write_root_pose_to_sim(pose, env_ids=ids)
            robot.write_root_velocity_to_sim(torch.zeros((len(ids), 6), device=dev), env_ids=ids)
            robot.write_joint_state_to_sim(
                robot.data.default_joint_pos[ids], torch.zeros_like(robot.data.default_joint_vel[ids]), env_ids=ids
            )

    def set_shuttle(ids, pos_local, vel):
        state = torch.zeros((len(ids), 13), device=dev)
        state[:, :3] = pos_local + origins[ids]
        state[:, 3] = 1.0
        state[:, 7:10] = vel
        shuttle.write_root_state_to_sim(state, env_ids=ids)

    def throw_at_racket(ids):
        sweet = robots[0].data.body_pos_w[ids, sweet_id]
        sweet_q = robots[0].data.body_quat_w[ids, sweet_id]
        normal = quat_apply(sweet_q, torch.tensor([1.0, 0.0, 0.0], device=dev).expand(len(ids), 3))
        normal = torch.where(normal[:, 2:3] < 0.0, -normal, normal)
        set_shuttle(ids, sweet - origins[ids] + THROW_OFFSET * normal, -THROW_SPEED * normal)
        return (sweet - origins[ids]).tolist(), normal.tolist()

    def set_rally_state(i, next_hitter, last_hitter, server):
        base.next_hitter[i] = next_hitter
        base.last_hitter[i] = last_hitter
        if hasattr(base, "initial_hitter"):
            base.initial_hitter[i] = server
        if hasattr(base, "hit_count"):
            base.hit_count[i] = 0
            if last_hitter >= 0:
                base.hit_count[i, last_hitter] = 1

    actions = torch.zeros((n, base.action_manager.total_action_dim), device=dev)
    env.reset()
    throw_log = {}
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    for k in range(T0 + args_cli.max_steps):
        step_counter["k"] = k
        hold_robots(all_ids)
        if k == WARMUP_STEPS:
            # Lift the held robot so the sweet spot sits at the requested height.
            sweet_z = robots[0].data.body_pos_w[racket_ids, sweet_id, 2] - origins[racket_ids, 2]
            target = torch.tensor([SCENARIOS[i]["racket_z"] for i in racket_ids.tolist()], device=dev)
            hold_pos[0, racket_ids, 2] += target - sweet_z
            hold_robots(all_ids)
        if k < T0:
            set_shuttle(all_ids, park, torch.zeros((n, 3), device=dev))
        elif k == T0:
            for i, s in enumerate(SCENARIOS):
                if s["kind"] in ("racket", "drop"):
                    set_rally_state(i, s["server"], -1, s["server"])
                else:
                    set_rally_state(i, 1 - s["last"], s["last"], s["last"])
            inject = torch.tensor([i for i, s in enumerate(SCENARIOS) if s["kind"] != "racket"], device=dev)
            set_shuttle(
                inject,
                torch.tensor([SCENARIOS[i]["pos"] for i in inject.tolist()], device=dev, dtype=torch.float32),
                torch.tensor([SCENARIOS[i]["vel"] for i in inject.tolist()], device=dev, dtype=torch.float32),
            )
            sweet, normal = throw_at_racket(racket_ids)
            for j, i in enumerate(racket_ids.tolist()):
                throw_log[i] = dict(sweet_local=sweet[j], normal=normal[j])
        else:
            for i, s in enumerate(SCENARIOS):
                info = hit_info[i]
                if s.get("rethrow") and info and results[i] is None and "rethrow_step" not in info:
                    if k - T0 >= info["step"] + RETHROW_DELAY:
                        throw_at_racket(torch.tensor([i], device=dev))
                        info["rethrow_step"] = k - T0
        env.step(actions)
        if k >= T0:
            for i in racket_ids.tolist():
                if hit_info[i] is None and results[i] is None and int(base.last_hitter[i]) == 0:
                    hit_info[i] = dict(
                        step=k - T0, shuttle_z=float(shuttle.data.root_pos_w[i, 2] - origins[i, 2])
                    )
                    if SCENARIOS[i].get("clear_after_hit"):
                        # Move the racket away so a natural re-contact cannot turn this into scenario 3.
                        hold_pos[0, i] = torch.tensor((*FAR_0, root_z0), device=dev)
            alive = torch.tensor([r is None for r in results], device=dev)
            if not alive.any():
                break

    net_check = net_pass_through(base, robots, shuttle, origins, n, dev, hold_robots, set_shuttle, all_ids)
    report = evaluate(results, hit_info, throw_log, net_check)
    print(json.dumps(report, indent=1))
    if args_cli.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.json_out)), exist_ok=True)
        with open(args_cli.json_out, "w") as f:
            json.dump(report, f, indent=1)
    env.close()
    return 0 if (report["all_passed"] or args_cli.informational) else 1


def net_pass_through(base, robots, shuttle, origins, n, dev, hold_robots, set_shuttle, all_ids):
    """Scenario 8: shuttle flies through the net collider region; it must not bounce."""
    from pxr import UsdPhysics

    stage = base.sim.stage
    authored = {}
    for env_i in (0, n - 1):
        root = stage.GetPrimAtPath(f"/World/envs/env_{env_i}/NetCollision")
        prims, pending = [], [root]
        while pending:
            p = pending.pop()
            if not p.IsValid():
                continue
            if p.HasAPI(UsdPhysics.FilteredPairsAPI):
                targets = UsdPhysics.FilteredPairsAPI(p).GetFilteredPairsRel().GetTargets()
                prims.append(
                    dict(prim=str(p.GetPath()), collider=p.HasAPI(UsdPhysics.CollisionAPI), targets=[str(t) for t in targets])
                )
            pending.extend(p.GetChildren())
        authored[f"env_{env_i}"] = prims

    hold_robots(all_ids)
    start = torch.tensor([-0.5, 0.0, 1.15], device=dev).expand(n, 3).clone()
    set_shuttle(all_ids, start, torch.tensor([4.0, 0.0, 0.0], device=dev).expand(n, 3))
    try:
        shuttle.permanent_wrench_composer.reset()
    except Exception:  # noqa: BLE001
        pass
    min_vx = torch.full((n,), torch.inf, device=dev)
    for _ in range(50):
        base.scene.write_data_to_sim()
        base.sim.step(render=False)
        base.scene.update(dt=base.physics_dt)
        min_vx = torch.minimum(min_vx, shuttle.data.root_lin_vel_w[:, 0])
    x_end = shuttle.data.root_pos_w[:, 0] - origins[:, 0]
    passed = (x_end > 0.3) & (min_vx > 3.0)
    return dict(
        x_end=x_end.tolist(),
        min_vx=min_vx.tolist(),
        passed_per_env=passed.tolist(),
        passed=bool(passed.all()),
        filtered_pairs_authored=authored,
    )


def evaluate(results, hit_info, throw_log, net_check):
    rows, all_ok = [], True
    for i, s in enumerate(SCENARIOS):
        r, h, exp = results[i], hit_info[i], s["expect"]
        checks = {}
        if "hit_by" in exp:
            checks["hit_by"] = h is not None
        if "winner" in exp:
            checks["winner"] = r is not None and r["resolved"] and r["winner"] == exp["winner"]
        for flag in ("shf", "dhf"):
            if flag in exp:
                got = (r or {}).get(flag)
                if got is None and r is None:
                    got = False  # not resolved yet -> no fault raised
                checks[flag] = got == exp[flag]
        ok = all(checks.values())
        all_ok &= ok
        rows.append(dict(scenario=s["name"], ok=ok, checks=checks, expect=exp, result=r, hit=h, throw=throw_log.get(i)))
    rows.append(dict(scenario="8_net_no_bounce", ok=net_check["passed"], net=net_check))
    all_ok &= net_check["passed"]
    print("\n==== P0 rule scenarios: task=%s package=%s" % (args_cli.task, PACKAGE_FILE))
    for row in rows:
        res = row.get("result") or {}
        print(
            f"{'PASS' if row['ok'] else 'FAIL'}  {row['scenario']:<26} winner={res.get('winner')} "
            f"shf={res.get('shf')} dhf={res.get('dhf')} step={res.get('step')} hit={row.get('hit')}"
        )
    print(f"ALL PASSED: {all_ok}\n")
    return dict(task=args_cli.task, package=PACKAGE_FILE, all_passed=bool(all_ok), rows=rows)


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)
