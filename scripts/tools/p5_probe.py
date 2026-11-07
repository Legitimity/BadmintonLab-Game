"""P5 probe: run the receive env with a TorchScript policy (or hold actions) and print feeder rallies.

    python scripts/tools/p5_probe.py --headless --num_envs 8 --steps 300 [--policy exports/p3_final/policy.pt] \
        [env.p5.verify_mode=true]

Per court and control step (``--trace`` courts): shuttle position (feeder frame), launch flag, receiver root
height / projected gravity, racket-ball distance, fall and rule state. At the end: feeder landing check per level.
"""

import argparse
import json
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--policy", default="exports/p3_final/policy.pt", help="TorchScript, or 'hold' (ready-pose action).")
parser.add_argument("--trace", type=int, default=2, help="Courts to trace.")
parser.add_argument("--every", type=int, default=5)
parser.add_argument("--natural_share", type=float, default=0.0)
parser.add_argument("--level", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import badmintonlab_game.tasks  # noqa: E402,F401
from badmintonlab_game.tasks.manager_based.badmintonlab_game.serve.serve_reward import to_canonical  # noqa: E402
from badmintonlab_game.tasks.manager_based.badmintonlab_game.serve.rules import predict_flight  # noqa: E402


@hydra_task_config("BadmintonLab-Game-Receive", "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.p5.natural_share = args.natural_share
    env_cfg.p5.level = args.level
    env = gym.make("BadmintonLab-Game-Receive", cfg=env_cfg)
    u = env.unwrapped
    obs, _ = env.reset()
    dev = u.device
    n = u.num_envs
    if args.policy == "hold":
        pol = None
    else:
        pol = torch.jit.load(args.policy, map_location=dev).eval()
    adim = u.single_agent_action_dim
    ar = torch.arange(n, device=dev)
    dmin = torch.full((n,), 9.0, device=dev)
    rk_at = torch.zeros(n, 3, device=dev)
    sh_at = torch.zeros(n, 3, device=dev)
    rallies = []
    returns = []
    for t in range(args.steps):
        # Closest racket-ball distance of the receiver during the incoming flight (server frame).
        srv = u.initial_hitter
        rk = to_canonical(torch.stack([r.data.body_link_pos_w[:, i] for r, i in zip(u._robots, u._sweet_spot_ids)], 1)[ar, 1 - srv]
                          - u.scene.env_origins, srv)
        sh_now = to_canonical(u._shuttle.data.root_pos_w - u.scene.env_origins, srv)
        d = torch.linalg.norm(rk - sh_now, dim=-1)
        live = u.recv_reward.active & ~u.recv_reward.hit
        better = live & (d < dmin)
        dmin = torch.where(better, d, dmin)
        rk_at = torch.where(better[:, None], rk, rk_at)
        sh_at = torch.where(better[:, None], sh_now, sh_at)
        tgt_snapshot = u.recv_reward.tgt.clone()
        if t in (5, 15):
            idle = ~u.recv_reward.active & ~u.natural
            if bool(idle.any()):
                rr = rk[idle]
                root_c = to_canonical(torch.stack([r.data.root_pos_w for r in u._robots], 1)[ar, 1 - srv] - u.scene.env_origins, srv)[idle]
                print(f"[PROBE_REST] t={t} n={int(idle.sum())} racket mean {rr.mean(0).tolist()} std {rr.std(0).tolist()} "
                      f"root mean {root_c.mean(0).tolist()}", flush=True)
        need_snapshot = u.recv_reward.d_need.clone()
        hit_snapshot = u.recv_reward.hit.clone()
        active_snapshot = u.recv_reward.active.clone()
        with torch.inference_mode():
            if pol is None:
                act = torch.zeros(n, 2 * adim, device=dev)
            else:
                act = torch.cat((pol(obs["player_0"]), pol(obs["player_1"])), -1)
            pre_nat = u.natural.clone()
            obs, _, term, trunc, info = env.step(act)
        # Returns: the step after the receiver's first hit (outcome_pending set this step), receiver frame.
        pend = u.recv_reward.outcome_pending & ~pre_nat
        if bool(pend.any()):
            ids = pend.nonzero().flatten()
            rcv = 1 - u.initial_hitter[ids]
            pc = to_canonical(u._shuttle.data.root_pos_w[ids] - u.scene.env_origins[ids], rcv)
            vc = to_canonical(u._shuttle.data.root_lin_vel_w[ids], rcv)
            sp = u.serve_reward.p
            nz, land = predict_flight(pc, vc, sp.drag_over_mass, sp.gravity, dt=0.01, max_steps=400)
            for j in range(ids.numel()):
                v = vc[j]
                el = float(torch.rad2deg(torch.atan2(v[2], torch.linalg.norm(v[:2]))))
                returns.append(dict(pos=[round(x, 2) for x in pc[j].tolist()], vel=[round(x, 2) for x in v.tolist()],
                                    speed=round(float(v.norm()), 2), el=round(el, 1), net_z=round(float(nz[j]), 2),
                                    land=[round(x, 2) for x in land[j].tolist()], face=round(float(u.recv_reward.c_el[ids[j]]), 1), hd=round(float(u.recv_reward.c_hd[ids[j]]), 1),
                                    vc=round(float(u.recv_reward.c_vc[ids[j]]), 2)))
        done_all = (term | trunc) & active_snapshot & ~pre_nat
        for i in done_all.nonzero().flatten().tolist():
            rallies.append(dict(dmin=round(float(dmin[i]), 3), need=round(float(need_snapshot[i]), 3),
                                hit=int(hit_snapshot[i]), tgt=[round(x, 2) for x in tgt_snapshot[i].tolist()],
                                racket=[round(x, 2) for x in rk_at[i].tolist()], ball=[round(x, 2) for x in sh_at[i].tolist()]))
        dmin = torch.where(term | trunc, torch.full_like(dmin, 9.0), dmin)
        if args.trace > 0 and (t % args.every == 0 or bool((term | trunc)[: args.trace].any())):
            for i in range(min(args.trace, n)):
                s = int(u.initial_hitter[i])
                r = 1 - s
                sh = to_canonical((u._shuttle.data.root_pos_w[i] - u.scene.env_origins[i])[None], torch.tensor([s], device=dev))[0]
                root = u._robots[r].data.root_pos_w[i] - u.scene.env_origins[i]
                feeder_root = u._robots[s].data.root_pos_w[i] - u.scene.env_origins[i]
                gz = float(u._robots[r].data.projected_gravity_b[i, 2])
                gzf = float(u._robots[s].data.projected_gravity_b[i, 2])
                done = bool(term[i] | trunc[i])
                terms = {k: bool(u.termination_manager.get_term(k)[i]) for k in u.termination_manager.active_terms}
                print(f"t={t:3d} c{i} nat={int(pre_nat[i])} srv={s} L={int(u.feeder_row_level[i])} launched={int(u.launched[i])} "
                      f"shuttle_c=({sh[0]:+.2f},{sh[1]:+.2f},{sh[2]:.2f}) recv_root=({root[0]:+.2f},{root[1]:+.2f},{root[2]:.2f}) "
                      f"gz={gz:+.2f} feeder_root_z={feeder_root[2]:.2f} gzf={gzf:+.2f} active={int(u.recv_reward.active[i])} "
                      f"hit={int(u.recv_reward.hit[i])} fall={u.serve_reward.fall_given[i].tolist()} "
                      f"done={int(done)} {[k for k, v in terms.items() if v] if done else ''}", flush=True)
    if rallies:
        ds = sorted(r["dmin"] for r in rallies)
        q = lambda f: ds[min(len(ds) - 1, int(f * len(ds)))]  # noqa: E731
        print(f"[PROBE_DMIN] feeder rallies {len(ds)}: closest racket-ball distance p10 {q(0.1):.3f} p50 {q(0.5):.3f} "
              f"p90 {q(0.9):.3f}; < 0.15 m {sum(d < 0.15 for d in ds) / len(ds):.3f}, < 0.3 m {sum(d < 0.3 for d in ds) / len(ds):.3f}; "
              f"hit {sum(r['hit'] for r in rallies) / len(ds):.3f}", flush=True)
        for r in rallies[:12]:
            print("[PROBE_RALLY]" + json.dumps(r), flush=True)
    if returns:
        import statistics as stt
        med = lambda k: stt.median(r[k] for r in returns)  # noqa: E731
        fwd = sum(r["vel"][0] > 0 for r in returns) / len(returns)
        over = sum(r["net_z"] > 1.55 for r in returns) / len(returns)
        print(f"[PROBE_RET] n={len(returns)} speed p50 {med('speed'):.2f} el p50 {med('el'):.1f} face p50 {med('face'):.1f} "
              f"vc p50 {med('vc'):.2f} |hd| p50 {stt.median(abs(r['hd']) for r in returns):.1f} |hd|<25 {sum(abs(r['hd']) < 25 for r in returns) / len(returns):.3f} vx>0 (towards net) {fwd:.3f} pred net_z>1.55 {over:.3f} net_z p50 {med('net_z'):.2f} "
              f"pos z p50 {stt.median(r['pos'][2] for r in returns):.2f} pos x p50 {stt.median(r['pos'][0] for r in returns):.2f}", flush=True)
        for r in returns[:15]:
            print("[PROBE_RET1]" + json.dumps(r), flush=True)
    ver = u.pop_verify()
    print("[PROBE_VERIFY]" + json.dumps(ver), flush=True)
    print("[PROBE_RECV]" + json.dumps(u.recv_reward.pop_iteration()), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
