"""Grid test of a receive checkpoint in the training env (11-11).

Every incoming ball comes from ``core.build_grid_table``: launches from the whole opponent half, spread over
landing cells (5 x-bands net -> back line, 5 y-bands, 3 apex tiers). One ball per episode from the ready pose
(= the first ball of a play rally), deterministic policy, S3 settings (ball ends when the return lands).
Per ball: touched, return won (lands in), lost after touching, fell, distance to the station at arrival.
``--serves f1 f2 ...`` evaluates recorded serves instead (one "cell" per file) to calibrate against play.

  python scripts/eval_grid.py --headless --checkpoint <ckpt> --dt_ms 2 --num_envs 512 --out grid.json
"""

import argparse
import json
import sys
import time

import torch  # noqa: F401  (before Kit, see HANDOVER 2.6)
import tensordict  # noqa: F401
import rsl_rl  # noqa: F401
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="BadmintonLab-Game-Receive")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--dt_ms", type=int, default=2)
parser.add_argument("--per_cell", type=int, default=100)
parser.add_argument("--serves", nargs="*", default=[])
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--max_steps", type=int, default=20000)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import badmintonlab_game.tasks  # noqa: E402,F401
from badmintonlab_game.tasks.manager_based.badmintonlab_game import receive as _receive  # noqa: E402,F401
from badmintonlab_game.tasks.manager_based.badmintonlab_game.physics import apply_physics  # noqa: E402
from badmintonlab_game.tasks.manager_based.badmintonlab_game.receive import core  # noqa: E402
from badmintonlab_game.tasks.manager_based.badmintonlab_game.receive import receive_env as renv  # noqa: E402
from badmintonlab_game.tasks.manager_based.badmintonlab_game.receive.agents import ReceiveRunner, ReceiveVecEnvWrapper  # noqa: E402
from badmintonlab_game.tasks.manager_based.badmintonlab_game.serve import player_cfg  # noqa: E402
from badmintonlab_game.tasks.manager_based.badmintonlab_game.serve.wrapper import hold_action  # noqa: E402

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


def main():
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    apply_physics(env_cfg, args_cli.dt_ms)
    env_cfg.feeder_v4 = True
    env_cfg.ball_gap_s = 1.0
    env_cfg.balls_per_episode = 1
    env_cfg.ball_timeout_s = 4.0
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    dev = u.device

    # ---- the test table
    if args_cli.serves:
        tabs, cells, names = [], [], []
        for i, p in enumerate(args_cli.serves):
            d = torch.load(p, map_location="cpu")
            t, _ = core.table_from_serves_v4(d["pos"][:600], d["vel"][:600], device=dev)
            tabs.append(t)
            cells.append(torch.full((len(t),), i, device=dev, dtype=torch.long))
            names.append(p.replace("\\", "/").split("/")[-1].replace(".pt", ""))
        table, cell = core.concat_tables(tabs), torch.cat(cells)
        extra = dict(n_cells=len(tabs), names=names, valid=torch.ones(len(table), dtype=torch.bool, device=dev))
    else:
        table, cell, extra = core.build_grid_table(per_cell=args_cli.per_cell, device="cpu")
        table = core.FeederTable(*(getattr(table, k).to(dev) for k in ("launch_pos", "launch_vel", "int_pos", "t_arr", "d_need", "station")))
        cell = cell.to(dev)
        extra = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in extra.items()}
    n_rows = len(table)
    print(f"[GRID] rows {n_rows} cells {extra['n_cells']}", flush=True)

    # ---- feed the rows in order, one per episode; remember which row each court is playing
    u.table = table
    u.real = None
    u.real_frac = 0.0
    u.level = -1  # not the V4 mix: _inject uses core.sample_rows over _bounds
    u._bounds = (0, n_rows)
    cursor = {"next": 0}
    cur_rows = torch.full((u.num_envs,), -1, dtype=torch.long, device=dev)
    last_rows = {"rows": None}
    orig_sample = core.sample_rows

    def sample_in_order(n, lo, hi, prev_frac, device, generator=None):
        k = cursor["next"]
        rows = torch.arange(k, k + n, device=device)
        cursor["next"] = k + n
        rows = torch.where(rows < n_rows, rows, torch.full_like(rows, -1))
        last_rows["rows"] = rows
        return rows.clamp(min=0)

    renv.core.sample_rows = sample_in_order
    orig_inject = u._inject

    def inject(ids):
        if ids.numel() == 0:
            return
        last_rows["rows"] = None
        orig_inject(ids)
        if last_rows["rows"] is not None:
            cur_rows[ids] = last_rows["rows"]

    u._inject = inject

    wrapped = ReceiveVecEnvWrapper(env, _ready_action(u, 1), clip_actions=agent_cfg.clip_actions)
    names = u.action_manager.get_term("player_0")._joint_names
    leg_mask = torch.tensor([any(k in n for k in LEG) for n in names])
    runner = ReceiveRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device, leg_mask=leg_mask, critic_warmup=0)
    runner.recv_curriculum = None
    runner.load(args_cli.checkpoint, load_optimizer=False)
    u.set_stage("S2")
    u.set_s3(True, 4.0)
    u.approach_cap = 2.4
    # restart every court so the first balls also come from the table in order
    cursor["next"] = 0
    env.reset()
    obs = wrapped.get_observations().to(runner.device)
    policy = runner.alg.policy
    policy.eval()

    res = {k: torch.zeros(n_rows, device=dev) for k in ("done", "touched", "won", "lost", "fell", "arr_d", "arrived")}
    t0 = time.time()
    steps = 0
    with torch.inference_mode():
        while steps < args_cli.max_steps:
            rows = cur_rows.clone()
            obs, _, _, _ = wrapped.step(policy.act_inference(obs))
            steps += 1
            live = rows >= 0
            arr = live & u._arrive
            if arr.any():
                root_xy = u._robots[0].data.root_pos_w[:, :2] - u.scene.env_origins[:, :2]
                d = torch.linalg.norm(u.station - root_xy, dim=-1)
                res["arr_d"].index_add_(0, rows[arr], d[arr])
                res["arrived"].index_add_(0, rows[arr], torch.ones_like(d[arr]))
            fell = live & (u._fell_low | u._fell_tilt)
            ended = live & u._ended_now & ~fell
            fin = fell | ended
            if fin.any():
                r = rows[fin]
                res["done"].index_add_(0, r, torch.ones_like(r, dtype=torch.float))
                res["fell"].index_add_(0, r, fell[fin].float())
                res["touched"].index_add_(0, r, u._touched_end[fin].float())
                res["won"].index_add_(0, r, (ended & u._won_now & u._touched_end)[fin].float())
                res["lost"].index_add_(0, r, (ended & u._touched_end & ~u._won_now)[fin].float())
                cur_rows[fin & (cur_rows == rows)] = -1
            if steps % 500 == 0:
                print(f"[GRID] step {steps} done {int((res['done'] > 0).sum())}/{n_rows} {time.time() - t0:.0f}s", flush=True)
            if bool((res["done"] > 0).all()) or (cursor["next"] >= n_rows and bool((cur_rows < 0).all())):
                break
    renv.core.sample_rows = orig_sample
    ok = res["done"] > 0
    per_row = {k: v.tolist() for k, v in res.items()}
    out = dict(checkpoint=args_cli.checkpoint, dt_ms=args_cli.dt_ms, rows=n_rows, finished=int(ok.sum()), steps=steps,
               wall_s=time.time() - t0, cell=cell.tolist(), n_cells=extra["n_cells"], names=extra.get("names"),
               valid=extra["valid"].tolist(), per_row=per_row,
               grid=dict(x=list(core.GRID_X_EDGES), y=list(core.GRID_Y_EDGES), apex=list(core.GRID_APEX_EDGES)))  # fmt: skip
    if "speed" in extra:
        out.update(speed=extra["speed"].tolist(), apex=extra["apex"].tolist(), land=extra["land"].tolist(),
                   launch=table.launch_pos.tolist(), t_hit=extra["t_hit"].tolist())  # fmt: skip
    tot = {k: float(res[k][ok].sum() / ok.sum()) for k in ("touched", "won", "lost", "fell")}
    print("[GRID_EVAL]" + json.dumps(dict(finished=int(ok.sum()), rows=n_rows, **tot)), flush=True)
    with open(args_cli.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
