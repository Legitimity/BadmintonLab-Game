"""Inner process for ``eval_serve.py``: runs inside the play package, reads only play's info and state.

play resets a resolved court inside ``env.step`` before it returns, and its ``_reset_idx``
clears ``just_hit``. Reading the robot or ``just_hit`` after ``step`` therefore sees the
next rally on those courts. We wrap ``_reset_idx`` (bookkeeping only, play's rules are
untouched) to snapshot the last state of a rally before it is cleared.
"""

import argparse
import json
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--play_root", required=True)
parser.add_argument("--model0", required=True)
parser.add_argument("--model1", required=True)
parser.add_argument("--ours", type=int, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--rallies", type=int, default=200)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--watchdog_s", type=float, default=60.0)
parser.add_argument("--fall_height", type=float, default=0.45)
parser.add_argument("--upright_cos", type=float, default=0.70, help="Fall also if projected-gravity z > -this (trunk tilt > ~45 deg), as in training. Use -1 for height only.")
parser.add_argument("--probe_out", default=None, help="P5 V2 3.1: traces of our receiver in the 0.5 s before a fall before its hit.")
parser.add_argument("--probe_window_s", type=float, default=0.5)
parser.add_argument("--probe_control_t", type=float, default=1.74, help="Control traces: receive rallies still up at this time.")
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import badmintonlab_game  # noqa: E402
import badmintonlab_game.tasks  # noqa: E402,F401

assert args.play_root.replace("\\", "/").lower() in badmintonlab_game.__file__.replace("\\", "/").lower(), (
    f"not the play package: {badmintonlab_game.__file__}"
)


def _load(path, device):
    if path == "zero":
        return lambda obs: torch.zeros(obs.shape[0], 23, device=device)
    return torch.jit.load(path, map_location=device).eval()


PROBE_FEATURES = ("t", "root_z", "vx", "vy", "vz", "speed_xy", "tilt_deg", "lean_fwd_deg", "lean_side_deg", "ang_speed",
                  "tau_max", "tau_mean", "tau_ratio_max", "act_norm", "act_absmax", "ball_dx", "ball_dy", "ball_z")


class FallProbe:
    """Ring buffer of our robot's state per court (``PROBE_FEATURES``; canonical frame of our side: x towards the
    net; lean from the projected gravity, + = forward / left); traces end at the step before the event."""

    def __init__(self, u, robot, shuttle, me, sign_me, step_dt):
        self.u, self.robot, self.shuttle, self.sign = u, robot, shuttle, sign_me
        self.k = max(int(round(args.probe_window_s / step_dt)), 1)
        n = u.num_envs
        self.buf = torch.zeros(self.k, n, len(PROBE_FEATURES), device=u.device)
        self.filled = torch.zeros(n, dtype=torch.long, device=u.device)
        self.pos = 0
        self.falls, self.controls = [], []
        self.control_done = torch.zeros(n, dtype=torch.bool, device=u.device)
        lim = getattr(robot.data, "joint_effort_limits", None)
        self.limits = None if lim is None else lim.clamp(min=1e-6)

    def trace(self, i):
        idx = [(self.pos - self.k + j) % self.k for j in range(self.k)]
        return self.buf[idx, i].tolist()

    def emit(self, events, kind):
        out = self.falls if kind == "fall" else self.controls
        for i, t_rel in events:
            if len(out) < 400 and int(self.filled[i]) >= self.k:
                out.append(dict(court=i, t=round(t_rel, 3), trace=[[round(v, 4) for v in row] for row in self.trace(i)]))

    def record(self, t_rel, actions):
        d, s = self.robot.data, self.sign
        org = self.u.scene.env_origins
        v = d.root_lin_vel_w
        g = d.projected_gravity_b
        tau = d.applied_torque.abs()
        ratio = (tau / self.limits).amax(-1) if self.limits is not None else torch.zeros_like(tau[:, 0])
        rel = self.shuttle.data.root_pos_w - d.root_pos_w
        row = torch.stack((
            t_rel, d.root_pos_w[:, 2] - org[:, 2], s * v[:, 0], s * v[:, 1], v[:, 2], v[:, :2].norm(dim=-1),
            torch.rad2deg(torch.acos((-g[:, 2]).clamp(-1, 1))), torch.rad2deg(torch.asin(g[:, 0].clamp(-1, 1))),
            torch.rad2deg(torch.asin(g[:, 1].clamp(-1, 1))), d.root_ang_vel_b.norm(dim=-1), tau.amax(-1), tau.mean(-1), ratio,
            actions.norm(dim=-1), actions.abs().amax(-1), s * rel[:, 0], s * rel[:, 1],
            self.shuttle.data.root_pos_w[:, 2] - org[:, 2]), dim=-1)
        self.buf[self.pos] = row
        self.pos = (self.pos + 1) % self.k
        self.filled += 1

    def reset(self, done):
        self.filled[done] = 0
        self.control_done[done] = False

    def summary(self):
        def agg(evts):
            if not evts:
                return None
            x = torch.tensor([e["trace"] for e in evts])  # (E, K, F)
            k = x.shape[1]
            at = {"-0.50s": 0, "-0.25s": k // 2, "-0.10s": max(k - 5, 0), "-0.02s": k - 1}
            return {name: {f: round(float(x[:, j, fi].mean()), 3) for fi, f in enumerate(PROBE_FEATURES)}
                    for name, j in at.items()} | {"n": len(evts),
                                                  "max_over_window": {f: round(float(x[:, :, fi].amax(1).mean()), 3)
                                                                      for fi, f in enumerate(PROBE_FEATURES)}}

        return dict(features=PROBE_FEATURES, window_steps=self.k, falls=agg(self.falls), controls=agg(self.controls))


def main():
    task = "BadmintonLab-Game-Play"
    cfg = parse_env_cfg(task, num_envs=args.num_envs)
    cfg.seed = args.seed
    env = gym.make(task, cfg=cfg)
    obs, _ = env.reset()
    u = env.unwrapped
    device = u.device
    pols = [_load(args.model0, device), _load(args.model1, device)]
    me, other = args.ours, 1 - args.ours
    n = u.num_envs
    keys = ("rallies", "legal", "served_in", "served_in_standing", "height", "double", "no_hit", "net", "out", "fell", "watchdog", "hits_ours")
    stats = {"serve": {k: 0 for k in keys}, "receive": {k: 0 for k in keys}}
    dims = [int(obs["player_0"].shape[-1]), int(obs["player_1"].shape[-1])]
    server = u.next_hitter.clone()
    ours_hit = torch.zeros(n, dtype=torch.bool, device=device)
    ours_hits = torch.zeros(n, dtype=torch.long, device=device)
    fell = torch.zeros(n, dtype=torch.bool, device=device)
    started = torch.zeros(n, device=device)
    sim_t = 0.0
    total = 0
    robot = u.scene[f"robot_{me}"]
    origin_z = u.scene.env_origins[:, 2]
    # Snapshots taken inside play's step, just before a resolved court is reset.
    pre_z = torch.zeros(n, device=device)
    pre_hit = torch.zeros(n, dtype=torch.bool, device=device)
    pre_gz = torch.full((n,), -1.0, device=device)
    pre_hit_opp = torch.zeros(n, dtype=torch.bool, device=device)
    pre_x = torch.zeros(n, device=device)
    shuttle = u.scene["shuttle"]
    origin_x = u.scene.env_origins[:, 0]
    sign_me = 1.0 if me == 0 else -1.0  # canonical x of our side: own half at x < 0
    original_reset = u._reset_idx

    def reset_with_snapshot(env_ids):
        ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)
        pre_z[ids] = robot.data.root_pos_w[ids, 2] - origin_z[ids]
        pre_hit[ids] = u.just_hit[ids, me]
        pre_gz[ids] = robot.data.projected_gravity_b[ids, 2]
        pre_hit_opp[ids] = u.just_hit[ids, other]
        pre_x[ids] = sign_me * (shuttle.data.root_pos_w[ids, 0] - origin_x[ids])
        return original_reset(env_ids)

    u._reset_idx = reset_with_snapshot
    probe = FallProbe(u, robot, shuttle, me, sign_me, u.step_dt) if args.probe_out else None
    fall_t = torch.full((n,), -1.0, device=device)
    hit_t = torch.full((n,), -1.0, device=device)
    timing = {"serve": {"len": [], "hit": [], "fall": [], "fall_after_hit": []}, "receive": {"len": [], "hit": [], "fall": [], "fall_after_hit": []}}
    extra = ("fell_before_hit", "fell_after_hit", "crossed", "return_in", "return_in_standing", "opp_hit_after")
    recv_keys = ("opp_served", "r_touch", "r_crossed", "r_in", "r_fell", "r_fell_after_hit")
    keys = keys + extra
    for s_ in stats.values():
        s_.update({k: 0 for k in extra + recv_keys})
    crossed = torch.zeros(n, dtype=torch.bool, device=device)
    opp_after = torch.zeros(n, dtype=torch.bool, device=device)
    opp_first = torch.zeros(n, dtype=torch.bool, device=device)
    wall = time.time()
    while total < args.rallies * n:
        with torch.inference_mode():
            a0 = pols[0](obs["player_0"]).to(device=device, dtype=torch.float32)
            a1 = pols[1](obs["player_1"]).to(device=device, dtype=torch.float32)
            obs, info = env.step(torch.cat((a0, a1), dim=-1))
        sim_t += u.step_dt
        resolved = info["rally_resolved"]
        # On resolved courts the live state already belongs to the next rally: use the snapshots.
        new_hit = torch.where(resolved, pre_hit, u.just_hit[:, me])
        z = torch.where(resolved, pre_z, robot.data.root_pos_w[:, 2] - origin_z)
        hit_t = torch.where(new_hit & (hit_t < 0), torch.full_like(hit_t, sim_t), hit_t)
        ours_hit |= new_hit
        ours_hits += new_hit.long()
        gz = torch.where(resolved, pre_gz, robot.data.projected_gravity_b[:, 2])
        now_fell = (z < args.fall_height) | (gz > -args.upright_cos)
        if probe is not None:
            # 3.1: emit the buffered 0.5 s (up to the previous step) at the first fall of our receiver before its
            # hit, and control traces of receive rallies still up at probe_control_t; then buffer this step.
            with torch.inference_mode():
                recv_role = server != me
                t_rel = sim_t - started
                first = now_fell & (fall_t < 0) & recv_role & ~ours_hit
                probe.emit([(i, float(t_rel[i])) for i in first.nonzero().flatten().tolist()], "fall")
                ctl = (recv_role & ~fell & ~now_fell & ~ours_hit & ~resolved & ~probe.control_done
                       & (t_rel >= args.probe_control_t))
                probe.emit([(i, float(t_rel[i])) for i in ctl.nonzero().flatten().tolist()], "control")
                probe.control_done |= ctl
                probe.record(t_rel, a0 if me == 0 else a1)
        fall_t = torch.where(now_fell & (fall_t < 0), torch.full_like(fall_t, sim_t), fall_t)
        fell |= now_fell
        # Receive metrics: the opponent served (touched first); after our first hit the shuttle reached the
        # other half; the opponent hit it again after our hit.
        opp_hit = torch.where(resolved, pre_hit_opp, u.just_hit[:, other])
        opp_first |= opp_hit & ~ours_hit & ~new_hit
        opp_after |= ours_hit & ~new_hit & opp_hit
        x_c = torch.where(resolved, pre_x, sign_me * (shuttle.data.root_pos_w[:, 0] - origin_x))
        crossed |= ours_hit & (x_c > 0.0)
        stuck = (sim_t - started) > args.watchdog_s
        done = resolved | stuck
        if done.any():
            for i in done.nonzero().flatten().tolist():
                role = "serve" if int(server[i]) == me else "receive"
                s = stats[role]
                s["rallies"] += 1
                total += 1
                if stuck[i] and not info["rally_resolved"][i]:
                    s["watchdog"] += 1
                    continue
                won = int(info["rally_winner"][i]) == me
                height = bool(info["serve_height_fault"][i]) and role == "serve" and not won
                double = bool(info["double_hit_fault"][i]) and not won
                s["height"] += int(height)
                s["double"] += int(double)
                # A high serve never registers as a hit, so it is a fault, not a miss.
                s["legal"] += int(bool(ours_hit[i]) and not double)
                s["no_hit"] += int(not ours_hit[i] and not height)
                s["served_in"] += int(won and bool(ours_hit[i]))
                # Same definition as the training metric Serve/in: a won serve only counts if the server is still up.
                s["served_in_standing"] += int(won and bool(ours_hit[i]) and not bool(fell[i]))
                s["fell"] += int(fell[i])
                s["hits_ours"] += int(ours_hits[i])
                s["crossed"] += int(bool(crossed[i]))
                # Return in: it crossed and either landed in (rally won without an opponent hit) or was played back.
                s["return_in"] += int(bool(crossed[i]) and (won or bool(opp_after[i])))
                s["return_in_standing"] += int(bool(crossed[i]) and (won or bool(opp_after[i])) and not bool(fell[i]))
                s["opp_hit_after"] += int(bool(opp_after[i]))
                if role == "receive" and bool(opp_first[i]):
                    # Receive metrics of 6.1, per legal opponent serve (the opponent touched the shuttle first).
                    s["opp_served"] += 1
                    s["r_touch"] += int(bool(ours_hit[i]) and not double)
                    s["r_crossed"] += int(bool(crossed[i]))
                    s["r_in"] += int(bool(crossed[i]) and (won or bool(opp_after[i])) and not bool(fell[i]))
                    s["r_fell"] += int(fell[i])
                    s["r_fell_after_hit"] += int(bool(fell[i]) and hit_t[i] >= 0 and fall_t[i] >= hit_t[i])
                t0 = float(started[i])
                tl = timing[role]
                tl["len"].append(sim_t - t0)
                if hit_t[i] >= 0:
                    tl["hit"].append(float(hit_t[i]) - t0)
                if fall_t[i] >= 0:
                    tl["fall"].append(float(fall_t[i]) - t0)
                    after = hit_t[i] >= 0 and fall_t[i] >= hit_t[i]
                    s["fell_after_hit"] += int(after)
                    s["fell_before_hit"] += int(not after)
                    if after:
                        tl["fall_after_hit"].append(float(fall_t[i] - hit_t[i]))
            # Rally state of the resolved courts was reset by play; capture the new server.
            server[done] = u.next_hitter[done]
            ours_hit[done] = False
            ours_hits[done] = 0
            crossed[done] = False
            opp_after[done] = False
            opp_first[done] = False
            fell[done] = False
            fall_t[done] = -1.0
            hit_t[done] = -1.0
            started[done] = sim_t
            if probe is not None:
                probe.reset(done)
    def summary(xs):
        if not xs:
            return None
        xs = sorted(xs)
        q = lambda f: round(xs[min(len(xs) - 1, int(f * len(xs)))], 3)  # noqa: E731
        return {"n": len(xs), "p10": q(0.1), "p50": q(0.5), "p90": q(0.9)}

    for role, s in stats.items():
        r = max(s["rallies"], 1)
        s["rates"] = {k: s[k] / r for k in keys if k != "rallies"}
        # Seconds from the rally start (hit, fall, rally length) and from our hit to the fall.
        s["timing_s"] = {k: summary(v) for k, v in timing[role].items()}
    rv = stats["receive"]
    m = max(rv["opp_served"], 1)
    rv["recv"] = {"recv_rallies": rv["opp_served"], "recv_touch": rv["r_touch"] / m, "recv_crossed": rv["r_crossed"] / m,
                  "recv_in": rv["r_in"] / m, "recv_fell": rv["r_fell"] / m, "recv_fell_after_hit": rv["r_fell_after_hit"] / m}
    out = dict(obs_dims=dims, num_envs=n, seed=args.seed, wall_s=time.time() - wall, **stats)
    if probe is not None:
        with open(args.probe_out, "w") as f:
            json.dump(dict(summary=probe.summary(), falls=probe.falls, controls=probe.controls,
                           step_dt=u.step_dt, model0=args.model0, model1=args.model1, ours=me), f)
        out["probe"] = probe.summary()
    print("[SERVE_EVAL]" + json.dumps(out), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Kit can hang in close() after an exception; print and exit hard instead.
        import os
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        app.close()
