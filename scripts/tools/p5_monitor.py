"""P5 V2 monitor (P5_RECEIVE_DESIGN_V2 3.4, 6.1, 7). Run with env_swanlab's Python (needs tensorboard).

Every ``--poll_s`` seconds, on the checkpoints ``model_<k>.pt`` of ``--run_name`` (sequential, play evaluations under
``<play>/.eval.lock``):
  * serve hold, every ``--hold_every`` (250): play, zero opponent, deterministic, both sides, ``--hold_rallies``
    per court (64 courts x 8 -> ~250 serves per side);
  * full play evaluation, every ``--eval_every`` (500) and the final checkpoint: serve hold (zero, ~500 serves per
    side), receive against ``p3_final`` and the local baseline (``exports/baseline_14500``), ~500 receive rallies
    per side each; the serve hold of this evaluation replaces the small one at the same iteration;
  * training-env deterministic evaluation (``p3_det_eval.py`` on the Receive task), same iterations as the full
    play evaluation.
Stops (one-sided, only on getting worse):
  * serve hold standing-in < ``--hold_min`` (0.80) in two consecutive serve-hold measurements;
  * play worse than training by >= ``--train_gap`` (10 points): play serve standing-in (zero) against the
    training ``Serve/in`` (mean of the 50 iterations before the checkpoint). Only this direction;
  * NaN losses; timeout rate > 2 % (mean of 200 steady iterations); fell after hit > 10 % in a play evaluation;
  * 7.2 early finish (``--early_finish``): the expected tier met in two consecutive full evaluations.
Alerts (no action): 6.1 checks at 600 / 1200 / 1800 iterations, stage promotions and ``max_*`` promotions,
actor-normaliser inputs beyond +-5 (risk 4), Teach/mse rising.
``--no_stop``: alerts only. ``--eval_export DIR``: one-shot full evaluation of an export, then exit.
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
ap.add_argument("--run_name", default="p5v2_M")
ap.add_argument("--out", default="/root/autodl-tmp/logs/v3l/p5")
ap.add_argument("--hold_every", type=int, default=250)
ap.add_argument("--hold_rallies", type=int, default=8, help="Per court and side, zero opponent (64 courts -> ~250 serves).")
ap.add_argument("--eval_every", type=int, default=500)
ap.add_argument("--serve_rallies", type=int, default=16, help="Per court and side (64 courts -> ~500 serves per side).")
ap.add_argument("--recv_rallies", type=int, default=16, help="Per court and side; about half are receive rallies.")
ap.add_argument("--baseline_rallies", type=int, default=24, help="The baseline touches ~50-60 % of its serves.")
ap.add_argument("--eval_num_envs", type=int, default=64)
ap.add_argument("--det_num_envs", type=int, default=1024)
ap.add_argument("--det_iters", type=int, default=10)
ap.add_argument("--p3_final", default="/root/autodl-tmp/BadmintonLab-Game-V3L/exports/p3_final")
ap.add_argument("--baseline", default="/root/autodl-tmp/BadmintonLab-Game-V3L/exports/baseline_14500/policy.pt")
ap.add_argument("--recv_opps", default="p3,baseline")
ap.add_argument("--serve_base", type=float, default=0.868, help="P3 final standing-in, zero opponent (REPORT_P3_F2 8).")
ap.add_argument("--hold_min", type=float, default=0.80)
ap.add_argument("--train_gap", type=float, default=0.10)
ap.add_argument("--early_finish", action="store_true")
ap.add_argument("--done_file", default=None)
ap.add_argument("--poll_s", type=float, default=120.0)
ap.add_argument("--no_stop", action="store_true")
ap.add_argument("--no_det", action="store_true")
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
# 7.2 tiers: minimum (B, simple receive) and expected.
TIER = dict(serve_in=0.85, touch=0.30, fah=0.05, fell_before=0.10, baseline_crossed=0.15)
TAGS = ("Recv/", "P5Curr/", "Teach/", "Policy/", "Serve/", "Loss/", "Critic/", "Norm/", "Reward/")


def say(msg):
    line = f"[{datetime.now():%F %T}] {msg}"
    print(line, flush=True)
    with open(STATUS, "a") as f:
        f.write(line + "\n")


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"held": [], "evaluated": [], "alerts": [], "stop": None, "hold": {}, "history": {}, "det": {}}


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
                if tag.startswith(TAGS):
                    s = out.setdefault(tag, {})
                    for e in acc.Scalars(tag):
                        s[e.step] = e.value
    return out


def window_mean(series, end_step, n=50):
    steps = [s for s in sorted(series) if s <= end_step][-n:]
    return sum(series[s] for s in steps) / len(steps) if steps else None


def at(series, step):
    """Value at the last logged step <= ``step``."""
    steps = [s for s in sorted(series) if s <= step]
    return series[steps[-1]] if steps else None


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


def eval_export(exp, tag, plan):
    """``plan`` = [(opponent, rallies per court)]; both sides. Returns the per-opponent summary."""
    res = {}
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
            n_opp = max(rec.get("recv_rallies", 0) or 0, 1)
            rows.append(dict(side=side, serve_n=s["rallies"], serve_in=s["rates"]["served_in_standing"],
                             serve_fault=s["rates"]["height"] + s["rates"]["double"], serve_fah=s["rates"]["fell_after_hit"],
                             recv_fell_before=(r.get("r_fell", 0) - r.get("r_fell_after_hit", 0)) / n_opp,
                             opp_touch_share=r.get("opp_served", 0) / max(r.get("rallies", 0), 1), **rec))
            say(f"  eval {tag} opp={opp} side={side} serve n={s['rallies']} in_st={rows[-1]['serve_in']:.3f} "
                f"fault={rows[-1]['serve_fault']:.3f} fah={rows[-1]['serve_fah']:.3f} | recv n={rec.get('recv_rallies')} "
                f"touch={rec.get('recv_touch', 0):.3f} crossed={rec.get('recv_crossed', 0):.3f} in={rec.get('recv_in', 0):.3f} "
                f"fell={rec.get('recv_fell', 0):.3f} fah={rec.get('recv_fell_after_hit', 0):.3f} "
                f"opp_touch={rows[-1]['opp_touch_share']:.3f} ({time.time() - t0:.0f} s)")
        if rows:
            def avg(key, weight):
                w = [r.get(weight, 0) or 0 for r in rows]
                return sum((r.get(key, 0) or 0) * x for r, x in zip(rows, w)) / max(sum(w), 1)

            res[opp] = dict(sides=rows, serve_in=avg("serve_in", "serve_n"), serve_fault=avg("serve_fault", "serve_n"),
                            serve_n=sum(r["serve_n"] for r in rows),
                            serve_fah=max(r["serve_fah"] for r in rows), touch=avg("recv_touch", "recv_rallies"),
                            crossed=avg("recv_crossed", "recv_rallies"), recv_in=avg("recv_in", "recv_rallies"),
                            recv_fell=avg("recv_fell", "recv_rallies"), recv_fell_before=avg("recv_fell_before", "recv_rallies"),
                            recv_fah=max(r.get("recv_fell_after_hit", 0) for r in rows),
                            recv_n=sum(r.get("recv_rallies", 0) for r in rows))
    return res


def det_eval(ck, k):
    out = os.path.join(args.out, f"det_{args.run_name}_{k}.json")
    log = out.replace(".json", ".log")
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES",  # the installed badmintonlab_game is another checkout
               PYTHONPATH=os.path.join(args.repo, "source/badmintonlab_game"))
    r = subprocess.run([args.isaac_python, "scripts/tools/p3_det_eval.py", "--headless", "--task", "BadmintonLab-Game-Receive",
                        "--num_envs", str(args.det_num_envs), "--iters", str(args.det_iters), "--modes", "det",
                        "--checkpoint", ck, "--out", out],
                       cwd=args.repo, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT, timeout=3600)
    if r.returncode != 0 or not os.path.exists(out):
        say(f"  det eval {k} failed ({r.returncode}), see {log}")
        return None
    d = json.load(open(out))["modes"]["det"]
    rv = d.get("recv", {})
    say(f"  det it={k}: serve in={d.get('in')} fall={d.get('fall')} | recv n_touch={rv.get('n_touch')} "
        f"f_touch={rv.get('f_touch')} n_crossed={rv.get('n_crossed')} f_crossed={rv.get('f_crossed')} "
        f"hd_med={rv.get('hd_abs_med')} vc_med={rv.get('vc_med')} st_err={rv.get('st_err_p50')} tf_err={rv.get('tf_err_p50')}")
    return d


def judge_hold(st, k, z, sc):
    """Serve hold (``z`` = zero-opponent summary at checkpoint ``k``): two consecutive < hold_min; play vs training."""
    st["hold"][str(k)] = dict(serve_in=z["serve_in"], serve_n=z["serve_n"])
    keys = sorted(int(x) for x in st["hold"])
    prev = st["hold"].get(str(keys[-2])) if len(keys) >= 2 else None
    say(f"  it={k}: serve hold standing-in {z['serve_in']:.3f} (n={z['serve_n']}, P3 base {args.serve_base:.3f}, "
        f"min {args.hold_min:.2f})")
    if z["serve_in"] < args.hold_min and prev is not None and prev["serve_in"] < args.hold_min:
        request_stop(st, f"3.4: serve hold {z['serve_in']:.3f} / {prev['serve_in']:.3f} < {args.hold_min} twice (it {keys[-2]}, {k})")
    tr = window_mean(sc.get("Serve/in", {}), k, 50)
    if tr is not None:
        gap = tr - z["serve_in"]
        say(f"  it={k}: training Serve/in {tr:.3f} vs play {z['serve_in']:.3f} (play worse by {gap:+.3f})")
        if gap >= args.train_gap:
            request_stop(st, f"3.4: play serve standing-in {z['serve_in']:.3f} below training {tr:.3f} by >= {args.train_gap}")


def tier(res):
    z = res.get("zero")
    if z is None or "p3" not in res or "baseline" not in res:
        return None
    touch = (res["p3"]["touch"] + res["baseline"]["touch"]) / 2
    fah = max([v["serve_fah"] for v in res.values()] + [v["recv_fah"] for v in res.values() if v.get("recv_n")])
    before = max(v["recv_fell_before"] for v in res.values() if v.get("recv_n"))
    low = z["serve_in"] >= TIER["serve_in"] and touch >= TIER["touch"] and fah <= TIER["fah"] and before <= TIER["fell_before"]
    if not low:
        return None
    return "expected" if res["baseline"]["crossed"] >= TIER["baseline_crossed"] else "minimum"


def judge_full(st, k, res):
    hist = st["history"]
    hist[str(k)] = res
    keys = sorted(int(x) for x in hist)
    prev = hist.get(str(keys[-2])) if len(keys) >= 2 else None
    vals = [v["serve_fah"] for v in res.values()] + [v["recv_fah"] for v in res.values() if v.get("recv_n")]
    if vals and max(vals) > 0.10:
        request_stop(st, f"V1 6.2: fell_after_hit {max(vals):.3f} > 0.10 at it={k}")
    t = tier(res)
    say(f"  it={k}: 7.2 tier {t or 'none'}" + "".join(
        f" | {o}: touch={res[o]['touch']:.3f} crossed={res[o]['crossed']:.3f} fell_before={res[o]['recv_fell_before']:.3f}"
        for o in ("p3", "baseline") if o in res))
    if args.early_finish and t == "expected" and prev is not None and tier(prev) == "expected":
        request_stop(st, f"7.2: expected tier met in two consecutive evaluations (it {keys[-2]}, {k})")


def check_training(st, sc):
    stage = sc.get("P5Curr/stage", {})
    if not stage:
        return
    last = max(stage)
    st["last_iter"] = last
    for name in ("Loss/value_function", "Loss/surrogate"):
        v = sc.get(name, {}).get(last)
        if v is not None and not math.isfinite(v):
            request_stop(st, f"6.2: {name} is {v} at it={last}")
    ts = sc.get("Recv/timeout", {})
    first = min(ts) if ts else 0
    steady = {k: v for k, v in ts.items() if k >= first + 50}
    to = window_mean(steady, last, 200) if len(steady) >= 200 else None
    if to is not None and to > 0.02:
        request_stop(st, f"6.2: timeout rate {to:.3f} > 0.02 (mean of 200 iterations) at it={last}")
    steps = sorted(stage)
    names = ("R0", "R1", "R2")
    for a, b in zip(steps, steps[1:]):
        if int(round(stage[b])) != int(round(stage[a])):
            n_in = at(sc.get("P5Curr/iters_in_stage", {}), a)
            alert(st, f"stage_{int(round(stage[b]))}", f"curriculum: {names[int(round(stage[b]))]} from iteration {b} "
                  f"(previous stage ran {n_in:.0f} iterations{', max reached' if n_in and n_in >= 1499 else ''})")
    ema = lambda k, s=last: at(sc.get(f"P5Curr/ema_{k}", {}), s)  # noqa: E731
    if last >= 600:
        s600 = at(stage, 600)
        e3, e6 = ema("st_err_p50", 300), ema("st_err_p50", 600)
        t3, t6 = ema("tf_err_p50", 300), ema("tf_err_p50", 600)
        ok = (s600 or 0) >= 1 or (None not in (e3, e6, t3, t6) and e6 < e3 and t6 < t3)
        alert(st, "check_600", f"6.1 it 600: stage {names[int(round(s600 or 0))]}, st_err {e3} -> {e6}, tf_err {t3} -> {t6}: "
              f"{'ok' if ok else 'NOT MET (doc: recv_approach to 3.0)'}")
    if last >= 1200:
        nt, ft = ema("n_touch", 1200), ema("f_touch", 1200)
        ev = window_mean(sc.get("Critic/explained_var_recv", {}), 1200, 50)
        ok = (nt or 0) >= 0.3 and (ft or 0) >= 0.3
        alert(st, "check_1200", f"6.1 it 1200: touch natural {nt} feeder {ft}, explained_var_recv {ev}: "
              f"{'ok' if ok else 'NOT MET'}")
    for name in ("serve", "recv"):
        o = window_mean(sc.get(f"Norm/over5_{name}", {}), last, 50)
        if o is not None and o > 0.01:
            alert(st, f"norm_{name}_{last // 1000}", f"risk 4: {o:.3%} of the {name} actor inputs beyond +-5 after the frozen normaliser")
    mse = sc.get("Teach/mse", {})
    if len(mse) >= 1000:
        a, b = window_mean(mse, last - 500, 200), window_mean(mse, last, 200)
        if a and b and b > 2.0 * a and b > 0.05:
            alert(st, f"mse_{last // 1000}", f"Teach/mse rising {a:.3g} -> {b:.3g}")


def export(k, ck):
    exp = os.path.join(args.repo, "exports", f"{args.run_name}_{k}")
    log = os.path.join(args.out, f"export_{args.run_name}_{k}.log")
    r = subprocess.run([args.isaac_python, "scripts/export_torchscript.py", "--checkpoint", ck, "--out", exp],
                       cwd=args.repo, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    if r.returncode != 0:
        say(f"export {k} failed, see {log}")
        return None
    return exp


def full_plan():
    plan = [("zero", args.serve_rallies)]
    plan += [(o, args.baseline_rallies if o == "baseline" else args.recv_rallies) for o in args.recv_opps.split(",") if o]
    return plan


def main():
    if args.eval_export:
        res = eval_export(args.eval_export, args.eval_tag, full_plan())
        json.dump(res, open(os.path.join(args.out, f"eval_{args.eval_tag}_summary.json"), "w"), indent=1)
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "sides"} for k, v in res.items()}, indent=1))
        return
    if not args.done_file:
        ap.error("--done_file is required")
    st = load_state()
    say(f"P5 V2 monitor start: run_name={args.run_name}, hold every {args.hold_every} (min {args.hold_min}), full every "
        f"{args.eval_every}, stop {'off' if args.no_stop else 'on'}, recv opps {args.recv_opps}")
    while True:
        done = os.path.exists(args.done_file)
        todo = []
        try:
            sc = scalars()
            ck = checkpoints()
            check_training(st, sc)
            final = max(ck) if (done and ck) else None
            todo = sorted(k for k in ck if k > 0 and k not in st["held"] and
                          (k % args.hold_every == 0 or k % args.eval_every == 0 or k == final))
            for k in todo:
                if st.get("stop") and not done:
                    break
                full = k % args.eval_every == 0 or k == final
                exp = export(k, ck[k])
                if exp is not None:
                    tag = f"{args.run_name}_{k}"
                    say(f"evaluate checkpoint {k} ({'full' if full else 'serve hold'}): {ck[k]}")
                    plan = full_plan() if full else [("zero", args.hold_rallies)]
                    res = eval_export(exp, tag, plan)
                    if "zero" in res:
                        judge_hold(st, k, res["zero"], sc)
                    if full:
                        judge_full(st, k, res)
                        if not args.no_det:
                            d = det_eval(ck[k], k)
                            if d is not None:
                                st["det"][str(k)] = d
                        st["evaluated"].append(k)
                st["held"].append(k)
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
