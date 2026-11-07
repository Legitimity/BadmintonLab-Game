"""P5 monitor (P5_RECEIVE_DESIGN 5.2, 6). Run with env_swanlab's Python (needs tensorboard).

Every ``--poll_s`` seconds:
  * play evaluation (deterministic, sequential, under ``<play>/.eval.lock``) of every ``model_<k>.pt`` with k a
    multiple of ``--eval_every`` and of the final checkpoint: serve hold (zero opponent) and receive against
    ``--recv_opps`` (default P3 final, model_test, local baseline), both sides;
  * 5.2: serve hold standing-in below the P3 baseline by >= 2 points -> natural share +0.10 (max 0.7), written
    to ``<run>/p5_control.json`` (the runner reads it every 10 iterations);
  * 6.2 stops (one-sided, only on getting worse): serve standing-in below baseline by >= 5 points twice in a
    row; fell_after_hit > 10% in any evaluation; NaN loss; EMA timeout > 2%. There is deliberately no
    train-vs-play rule (P3: it fired when play was better than the noisy training metric);
  * 6.3 early finish: all B gates met in two consecutive evaluations;
  * 7.1 alerts (no action): L0 >= 1000 iterations with f_touch < 0.3, Teach/mse rising, receiver falls > 10%,
    receive tunnel > 25%, level max reached.
``--no_stop``: alerts only. ``--eval_export DIR``: one-shot evaluation of an export, then exit.
"""

import argparse
import fcntl
import glob
import json
import math
import os
import re
import signal
import subprocess
import time
from datetime import datetime

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="/root/autodl-tmp/BadmintonLab-Game-V3L")
ap.add_argument("--play", default="/root/autodl-tmp/BadmintonLab-Game-Play")
ap.add_argument("--isaac_python", default="/root/autodl-tmp/env_isaaclab/bin/python")
ap.add_argument("--run_name", default="p5_main")
ap.add_argument("--out", default="/root/autodl-tmp/logs/v3l/p5")
ap.add_argument("--eval_every", type=int, default=500)
ap.add_argument("--serve_rallies", type=int, default=32, help="Per court and side (64 courts -> ~1000 serves per side).")
ap.add_argument("--recv_rallies", type=int, default=32, help="Per court and side; roughly half are receive rallies.")
ap.add_argument("--baseline_rallies", type=int, default=16)
ap.add_argument("--eval_num_envs", type=int, default=64)
ap.add_argument("--p3_final", default="/root/autodl-tmp/BadmintonLab-Game-V3L/exports/p3_final")
ap.add_argument("--baseline", default="/root/autodl-tmp/policy.pt")
ap.add_argument("--recv_opps", default="p3,model_test,baseline")
ap.add_argument("--serve_base", type=float, default=0.868, help="P3 final standing-in, zero opponent (REPORT_P3_F2 8).")
ap.add_argument("--done_file", default=None)
ap.add_argument("--poll_s", type=float, default=120.0)
ap.add_argument("--no_stop", action="store_true")
ap.add_argument("--no_share_control", action="store_true", help="Do not write p5_control.json (pilot, feeder only).")
ap.add_argument("--eval_export", default=None)
ap.add_argument("--eval_tag", default="oneshot")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
STATUS = os.path.join(args.out, f"P5_STATUS_{args.run_name}.txt")
STATE = os.path.join(args.out, f"monitor_state_{args.run_name}.json")
STOP_FILE = os.path.join(args.out, f"STOP_REQUESTED_{args.run_name}")
PLAY_CFGS = [
    "source/badmintonlab_game/badmintonlab_game/tasks/manager_based/badmintonlab_game/player0_cfg.py",
    "source/badmintonlab_game/badmintonlab_game/tasks/manager_based/badmintonlab_game/player1_cfg.py",
]
LOCK = os.path.join(args.play, ".eval.lock")
OPP = {"zero": "zero", "p3": args.p3_final, "model_test": "model_test", "baseline": args.baseline}
GATES = dict(serve_drop=0.02, fault=0.02, touch=0.40, crossed=0.20, fah=0.05)


def say(msg):
    line = f"[{datetime.now():%F %T}] {msg}"
    print(line, flush=True)
    with open(STATUS, "a") as f:
        f.write(line + "\n")


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"evaluated": [], "alerts": [], "stop": None, "history": {}, "natural_share": None}


def save_state(st):
    json.dump(st, open(STATE, "w"), indent=1)


def run_dirs():
    return sorted(glob.glob(os.path.join(args.repo, "logs/rsl_rl/badmintonlab_receive", f"*_{args.run_name}")))


def scalars():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    out = {}
    for d in run_dirs():
        for ev in glob.glob(os.path.join(d, "events*")):
            acc = EventAccumulator(ev, size_guidance={"scalars": 0})
            acc.Reload()
            for tag in acc.Tags()["scalars"]:
                if tag.startswith(("Recv/", "P5Curr/", "Teach/", "Policy/", "Serve/", "Loss/")):
                    s = out.setdefault(tag, {})
                    for e in acc.Scalars(tag):
                        s[e.step] = e.value
    return out


def window_mean(series, end_step, n=50):
    steps = [s for s in sorted(series) if s <= end_step][-n:]
    return sum(series[s] for s in steps) / len(steps) if steps else None


def checkpoints():
    plain = {}
    for d in run_dirs():
        for f in glob.glob(os.path.join(d, "model_*.pt")):
            m = re.fullmatch(r"model_(\d+)\.pt", os.path.basename(f))
            if m:
                plain[int(m.group(1))] = f
    return plain


def alert(st, key, msg):
    if key in st["alerts"]:
        return False
    st["alerts"].append(key)
    say("ALERT " + msg)
    return True


def request_stop(st, reason):
    if st.get("stop"):
        return
    st["stop"] = reason
    save_state(st)
    if args.no_stop:
        say(f"STOP (dry run, not sent): {reason}")
        return
    with open(STOP_FILE, "w") as f:
        json.dump({"reason": reason, "time": f"{datetime.now():%F %T}"}, f)
    pids = subprocess.run(["pgrep", "-f", f"train_p5.py.*--run_name {args.run_name}( |$)"],
                          capture_output=True, text=True).stdout.split()
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except (ProcessLookupError, ValueError):
            pass
    say(f"STOP requested: {reason} (SIGTERM to {pids or 'no running trainer'})")


def play_eval(exp, opp, side, out, rallies, num_envs):
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES")
    with open(LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            try:
                r = subprocess.run(
                    [args.isaac_python, "scripts/eval_serve.py", "--play_root", args.play, "--ours", exp, "--opponent", opp,
                     "--side", str(side), "--num_envs", str(num_envs), "--rallies", str(rallies),
                     "--python", args.isaac_python, "--out", out],
                    cwd=args.repo, env=env, stdout=open(out.replace(".json", ".log"), "w"), stderr=subprocess.STDOUT,
                    timeout=3600)
                code = r.returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
            subprocess.run(["git", "-C", args.play, "checkout", "--", *PLAY_CFGS])
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    return code


def eval_export(exp, tag):
    """Serve hold (zero) + receive (``--recv_opps``), both sides. Returns the per-opponent summary."""
    res = {}
    plan = [("zero", args.serve_rallies)] + [(o, args.baseline_rallies if o == "baseline" else args.recv_rallies)
                                             for o in args.recv_opps.split(",") if o]
    for opp, rallies in plan:
        if opp == "baseline" and not os.path.exists(args.baseline):
            say(f"  baseline {args.baseline} missing: skipped")
            continue
        rows = []
        for side in (0, 1):
            out = os.path.join(args.out, f"eval_{tag}_{opp}_side{side}.json")
            t0 = time.time()
            code = play_eval(exp, OPP[opp], side, out, rallies, args.eval_num_envs)
            if code != 0 or not os.path.exists(out):
                say(f"  eval {tag} {opp} side{side} failed ({code})")
                continue
            d = json.load(open(out))
            s, r = d["serve"], d["receive"]
            rec = r.get("recv", {})
            rows.append(dict(side=side, serve_n=s["rallies"], serve_in=s["rates"]["served_in_standing"],
                             serve_fault=s["rates"]["height"] + s["rates"]["double"], serve_fah=s["rates"]["fell_after_hit"],
                             **rec))
            say(f"  eval {tag} opp={opp} side={side} serve n={s['rallies']} in_st={rows[-1]['serve_in']:.3f} "
                f"fault={rows[-1]['serve_fault']:.3f} fah={rows[-1]['serve_fah']:.3f} | recv n={rec.get('recv_rallies')} "
                f"touch={rec.get('recv_touch', 0):.3f} crossed={rec.get('recv_crossed', 0):.3f} in={rec.get('recv_in', 0):.3f} "
                f"fell={rec.get('recv_fell', 0):.3f} fah={rec.get('recv_fell_after_hit', 0):.3f} ({time.time() - t0:.0f} s)")
        if rows:
            def avg(key, weight):
                w = [r.get(weight, 0) or 0 for r in rows]
                return sum(r.get(key, 0) * x for r, x in zip(rows, w)) / max(sum(w), 1)

            res[opp] = dict(sides=rows, serve_in=avg("serve_in", "serve_n"), serve_fault=avg("serve_fault", "serve_n"),
                            serve_fah=max(r["serve_fah"] for r in rows), touch=avg("recv_touch", "recv_rallies"),
                            crossed=avg("recv_crossed", "recv_rallies"), recv_in=avg("recv_in", "recv_rallies"),
                            recv_fell=avg("recv_fell", "recv_rallies"),
                            recv_fah=max(r.get("recv_fell_after_hit", 0) for r in rows),
                            recv_n=sum(r.get("recv_rallies", 0) for r in rows))
    return res


def judge(st, k, res):
    hist = st["history"]
    hist[str(k)] = res
    z = res.get("zero")
    keys = sorted(int(x) for x in hist)
    prev = hist.get(str(keys[-2])) if len(keys) >= 2 else None
    if z is not None:
        drop = args.serve_base - z["serve_in"]
        say(f"  it={k}: serve hold standing-in {z['serve_in']:.3f} (P3 base {args.serve_base:.3f}, drop {drop:+.3f})")
        # 5.2 natural share
        if drop >= 0.02 and not args.no_share_control:
            cur = st.get("natural_share")
            if cur is None:
                cur = 0.5
            new = min(cur + 0.10, 0.7)
            if new > cur:
                st["natural_share"] = new
                d = run_dirs()[-1]
                json.dump({"natural_share": new, "it": k}, open(os.path.join(d, "p5_control.json"), "w"))
                say(f"  5.2: natural share {cur:.2f} -> {new:.2f} ({d}/p5_control.json)")
        # 6.2 serve drop twice in a row
        pz = (prev or {}).get("zero")
        if drop >= 0.05 and pz is not None and args.serve_base - pz["serve_in"] >= 0.05:
            request_stop(st, f"6.2: serve standing-in {z['serve_in']:.3f} / {pz['serve_in']:.3f} below base by >= 5 points twice")
    fah = max([v["serve_fah"] for v in res.values()] + [v["recv_fah"] for v in res.values() if v.get("recv_n")])
    if fah > 0.10:
        request_stop(st, f"6.2: fell_after_hit {fah:.3f} > 0.10 at it={k}")
    # 6.3 early finish
    ok = gates_ok(res)
    say(f"  it={k}: B gates {'met' if ok else 'not met'}")
    if ok and prev is not None and gates_ok(prev):
        request_stop(st, f"6.3: B gates met in two consecutive evaluations (it {keys[-2]}, {k})")


def gates_ok(res):
    z = res.get("zero")
    if z is None or "p3" not in res or "model_test" not in res:
        return False
    touch = (res["p3"]["touch"] + res["model_test"]["touch"]) / 2
    crossed = (res["p3"]["crossed"] + res["model_test"]["crossed"]) / 2
    fah = max([v["serve_fah"] for v in res.values()] + [v["recv_fah"] for v in res.values() if v.get("recv_n")])
    return (z["serve_in"] >= args.serve_base - GATES["serve_drop"] and z["serve_fault"] <= GATES["fault"]
            and touch >= GATES["touch"] and crossed >= GATES["crossed"] and fah <= GATES["fah"])


def check_training(st, sc):
    lvl = sc.get("P5Curr/level", {})
    if not lvl:
        return
    last = max(lvl)
    st["last_iter"] = last
    level = int(round(lvl[last]))
    iters = int(sc.get("P5Curr/iters_in_level", {}).get(last, 0))
    ema = lambda k: sc.get(f"P5Curr/ema_{k}", {}).get(last)  # noqa: E731
    share = sc.get("P5Curr/natural_share", {}).get(last)
    if share is not None and st.get("natural_share") is None:
        st["natural_share"] = share
    for name in ("Loss/value_function", "Loss/surrogate", "Teach/mse"):
        v = sc.get(name, {}).get(last)
        if v is not None and not math.isfinite(v) and name != "Teach/mse":
            request_stop(st, f"6.2: {name} is {v} at it={last}")
    to = window_mean(sc.get("Recv/timeout", {}), last, 200)
    if to is not None and to > 0.02 and last >= 200:
        request_stop(st, f"6.2: timeout rate {to:.3f} > 0.02 (mean of 200 iterations) at it={last}")
    if level == 0 and iters >= 1000 and (ema("f_touch") or 0) < 0.3:
        alert(st, "L0_1000", f"7.1: L0 {iters} iterations, EMA f_touch={ema('f_touch')} < 0.3 (check targets / racket_near)")
    fell = window_mean(sc.get("Recv/f_fell", {}), last, 100)
    if fell is not None and fell > 0.10 and last >= 300:
        alert(st, f"fell_{last // 1000}", f"7.1: receiver fall (feeder) {fell:.3f} > 0.10 (doc: approach cap 0.7 m/s)")
    tun = window_mean(sc.get("Recv/f_tunnel", {}), last, 100)
    if tun is not None and tun > 0.25 and last >= 300:
        alert(st, f"tunnel_{last // 1000}", f"risk 5: receive tunnel {tun:.3f} > 0.25")
    mse = sc.get("Teach/mse", {})
    if last >= 1000 and mse:
        a, b = window_mean(mse, last - 500, 200), window_mean(mse, last, 200)
        if a and b and b > 2.0 * a and b > 0.05:
            alert(st, f"mse_{last // 1000}", f"7.1: Teach/mse rising {a:.3g} -> {b:.3g} (doc: beta floor 0.1 -> 0.3 if serve drops)")
    for k in sorted(lvl):
        if k > 0 and int(round(lvl[k])) != int(round(lvl.get(max(s for s in lvl if s < k), lvl[k]))):
            alert(st, f"promo_{int(round(lvl[k]))}", f"curriculum: level L{int(round(lvl[k]))} from iteration {k}")


def main():
    if args.eval_export:
        res = eval_export(args.eval_export, args.eval_tag)
        json.dump(res, open(os.path.join(args.out, f"eval_{args.eval_tag}_summary.json"), "w"), indent=1)
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "sides"} for k, v in res.items()}, indent=1))
        return
    if not args.done_file:
        ap.error("--done_file is required")
    st = load_state()
    say(f"P5 monitor start: run_name={args.run_name}, eval every {args.eval_every}, stop {'off' if args.no_stop else 'on'}, "
        f"serve base {args.serve_base}, recv opps {args.recv_opps}")
    while True:
        done = os.path.exists(args.done_file)
        todo = []
        try:
            sc = scalars()
            ck = checkpoints()
            check_training(st, sc)
            todo = sorted(k for k in ck if k > 0 and k % args.eval_every == 0 and k not in st["evaluated"])
            if done and ck and max(ck) not in st["evaluated"] and max(ck) not in todo:
                todo.append(max(ck))
            for k in todo:
                if not done and st.get("stop") and k != max(ck):
                    continue
                exp = os.path.join(args.repo, "exports", f"{args.run_name}_{k}")
                log = os.path.join(args.out, f"export_{args.run_name}_{k}.log")
                r = subprocess.run([args.isaac_python, "scripts/export_torchscript.py", "--checkpoint", ck[k], "--out", exp],
                                   cwd=args.repo, stdout=open(log, "w"), stderr=subprocess.STDOUT)
                if r.returncode != 0:
                    say(f"export {k} failed, see {log}")
                else:
                    say(f"evaluate checkpoint {k}: {ck[k]}")
                    judge(st, k, eval_export(exp, f"{args.run_name}_{k}"))
                st["evaluated"].append(k)
                save_state(st)
            save_state(st)
        except Exception as e:  # noqa: BLE001
            say(f"monitor error: {e!r}")
        if done and not todo:
            say("monitor finished")
            return
        time.sleep(30 if done else args.poll_s)


if __name__ == "__main__":
    main()
