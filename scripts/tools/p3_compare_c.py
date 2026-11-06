"""Stage C branch comparison (P3_LAUNCH_F2 4.3). Run with env_swanlab's Python (TensorBoard reader).

Primary metric: mean ``Serve/in`` over iterations ``P+lo .. P+hi`` (default 1300..1500) of each branch, same
iterations for both, so the GPU share does not matter. Play check: the first ``model_<k>.pt`` with
``k >= P+hi`` of each branch, zero opponent, both sides, ~2000 serve rallies per side (64 courts), via
``p3_monitor.py --eval_export`` (holds the play lock). Verdict:

  * a branch with window ``fall`` > 0.05 or ``fault`` > 0.02 (training or play) is out
  * ``in(c7) - in(c8) >= +margin`` -> c7, ``<= -margin`` -> c8, otherwise c8 (one change less)
  * play check opposite to the training verdict with a play gap > 0.03 -> no verdict (user decides)

Also reports a second window at the end of the iterations both branches reached (informational) and the
seconds per iteration of each branch. Writes ``c_compare.json`` and appends to the status file.
"""

import argparse
import glob
import json
import os
import re
import subprocess
from datetime import datetime

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="/root/autodl-tmp/BadmintonLab-Game-V3L")
ap.add_argument("--isaac_python", default="/root/autodl-tmp/env_isaaclab/bin/python")
ap.add_argument("--runs", default="c7=p3_c7,c8=p3_c8", help="label=run_name; the first is the variant, the second the base.")
ap.add_argument("--P", type=int, required=True, help="Iteration of model_passed_B_<P>.pt.")
ap.add_argument("--lo", type=int, default=1300)
ap.add_argument("--hi", type=int, default=1500)
ap.add_argument("--margin", type=float, default=0.02)
ap.add_argument("--play_conflict", type=float, default=0.03)
ap.add_argument("--fall_max", type=float, default=0.05)
ap.add_argument("--fault_max", type=float, default=0.02)
ap.add_argument("--no_play", action="store_true")
ap.add_argument("--play_num_envs", type=int, default=64)
ap.add_argument("--play_rallies", type=int, default=64, help="Per court, both roles: 64 x 64 -> ~2000 serves per side.")
ap.add_argument("--out", default="/root/autodl-tmp/logs/v3l/p3")
args = ap.parse_args()

STATUS = os.path.join(args.out, "P3_STATUS.txt")
TAGS = ("Serve/in", "Serve/crossed", "Serve/fall", "Serve/fault", "Serve/tunnel", "Serve/no_hit", "Serve/vc_mean",
        "Serve/el_mean", "Serve/param_v_close_max", "Curriculum/ema_in", "Curriculum/stage")


def say(msg):
    line = f"[{datetime.now():%F %T}] compare_c: {msg}"
    print(line, flush=True)
    with open(STATUS, "a") as f:
        f.write(line + "\n")


def run_dirs(name):
    return sorted(glob.glob(os.path.join(args.repo, "logs/rsl_rl/badmintonlab_serve", f"*_{name}")))


def series(name):
    out, wall = {}, {}
    for d in run_dirs(name):
        for ev in glob.glob(os.path.join(d, "events*")):
            acc = EventAccumulator(ev, size_guidance={"scalars": 0})
            acc.Reload()
            for tag in acc.Tags()["scalars"]:
                if tag in TAGS:
                    for e in acc.Scalars(tag):
                        out.setdefault(tag, {})[e.step] = e.value
                        if tag == "Serve/in":
                            wall[e.step] = e.wall_time
    return out, wall


def mean(s, lo, hi):
    xs = [v for k, v in s.items() if lo <= k <= hi]
    return (sum(xs) / len(xs), len(xs)) if xs else (None, 0)


def sec_per_iter(wall):
    steps = sorted(wall)
    dts = [wall[b] - wall[a] for a, b in zip(steps, steps[1:]) if b == a + 1]
    dts = sorted(d for d in dts if d > 0)
    return dts[len(dts) // 2] if dts else None  # median: evaluation pauses do not count


def window(sc, lo, hi):
    return {t.split("/", 1)[1]: (lambda m: None if m[0] is None else round(m[0], 4))(mean(sc.get(t, {}), lo, hi))
            for t in TAGS if t.startswith("Serve/")} | {"n_iters": mean(sc.get("Serve/in", {}), lo, hi)[1]}


def play_check(label, name, k_min):
    ck = {}
    for d in run_dirs(name):
        for f in glob.glob(os.path.join(d, "model_*.pt")):
            m = re.fullmatch(r"model_(\d+)\.pt", os.path.basename(f))
            if m and int(m.group(1)) >= k_min:
                ck[int(m.group(1))] = f
    if not ck:
        say(f"{label}: no model_<k>.pt with k >= {k_min}, play check skipped")
        return None
    k = min(ck)
    exp = os.path.join(args.repo, "exports", f"{name}_{k}")
    r = subprocess.run([args.isaac_python, "scripts/export_torchscript.py", "--checkpoint", ck[k], "--out", exp],
                       cwd=args.repo, capture_output=True, text=True)
    if r.returncode != 0:
        say(f"{label}: export of {ck[k]} failed ({r.returncode})")
        return None
    tag = f"ccmp_{label}_{k}"
    swpy = os.path.join(os.path.dirname(os.path.realpath(__file__)), "p3_monitor.py")
    r = subprocess.run([os.environ.get("SWANLAB_PYTHON", "/root/autodl-tmp/env_swanlab/bin/python"), swpy,
                        "--eval_export", exp, "--eval_opps", "zero", "--eval_num_envs", str(args.play_num_envs),
                        "--eval_rallies", str(args.play_rallies), "--eval_tag", tag, "--out", args.out,
                        "--run_name", name], cwd=args.repo, capture_output=True, text=True)
    rows = {}
    for side in (0, 1):
        f = os.path.join(args.out, f"{tag}_zero_side{side}.json")
        if os.path.exists(f):
            s = json.load(open(f))["serve"]
            rows[side] = dict(rallies=s["rallies"], **{k_: round(v, 4) for k_, v in s["rates"].items()})
    if len(rows) < 2:
        say(f"{label}: play check incomplete ({r.returncode}): {r.stdout[-300:]}")
        return None
    n = sum(r_["rallies"] for r_ in rows.values())
    avg = lambda key: sum(r_[key] * r_["rallies"] for r_ in rows.values()) / n  # noqa: E731
    return dict(iteration=k, export=exp, sides=rows, rallies=n,
                served_in_standing=round(avg("served_in_standing"), 4),
                fault=round(avg("height") + avg("double"), 4), fell_after_hit=round(avg("fell_after_hit"), 4))


def main():
    labels = [x.split("=") for x in args.runs.split(",")]
    (lv, nv), (lb, nb) = labels
    lo, hi = args.P + args.lo, args.P + args.hi
    res = dict(time=f"{datetime.now():%F %T}", P=args.P, window=[lo, hi], margin=args.margin, branches={})
    data = {}
    for label, name in labels:
        sc, wall = series(name)
        data[label] = sc
        last = max(sc.get("Serve/in", {0: 0}))
        res["branches"][label] = dict(run_name=name, last_iter=last, sec_per_iter=sec_per_iter(wall),
                                      window=window(sc, lo, hi))
    common_end = min(b["last_iter"] for b in res["branches"].values())
    res["end_window"] = [common_end - 199, common_end]
    for label in res["branches"]:
        res["branches"][label]["end_window"] = window(data[label], common_end - 199, common_end)
    spi = [b["sec_per_iter"] for b in res["branches"].values()]
    if all(spi) and max(spi) > 1.25 * min(spi):
        say(f"ALERT seconds per iteration differ by > 25%: {dict(zip(res['branches'], spi))} (comparison by iterations stays valid)")
    short = [lab for lab, b in res["branches"].items() if b["window"]["n_iters"] < 0.9 * (hi - lo + 1)]
    if short:
        res.update(verdict=None, reason=f"window {lo}-{hi} incomplete for {short}")
        return finish(res)
    if not args.no_play:
        for label, name in labels:
            res["branches"][label]["play"] = play_check(label, name, hi)
    out = []
    for label, b in res["branches"].items():
        w, p = b["window"], b.get("play") or {}
        bad = []
        if w["fall"] is not None and w["fall"] > args.fall_max:
            bad.append(f"train fall {w['fall']}")
        if w["fault"] is not None and w["fault"] > args.fault_max:
            bad.append(f"train fault {w['fault']}")
        if p and p["fell_after_hit"] > args.fall_max:
            bad.append(f"play fell_after_hit {p['fell_after_hit']}")
        if p and p["fault"] > args.fault_max:
            bad.append(f"play fault {p['fault']}")
        b["disqualified"] = bad
        if bad:
            out.append(label)
    d_train = res["branches"][lv]["window"]["in"] - res["branches"][lb]["window"]["in"]
    res["delta_in_train"] = round(d_train, 4)
    pv, pb = res["branches"][lv].get("play"), res["branches"][lb].get("play")
    d_play = round(pv["served_in_standing"] - pb["served_in_standing"], 4) if pv and pb else None
    res["delta_in_play"] = d_play
    if len(out) == 2:
        res.update(verdict=None, reason=f"both branches out: { {k: res['branches'][k]['disqualified'] for k in out} }")
    elif len(out) == 1:
        win = lb if out[0] == lv else lv
        res.update(verdict=win, reason=f"{out[0]} out: {res['branches'][out[0]]['disqualified']}")
    else:
        win = lv if d_train >= args.margin else lb
        why = (f"in({lv})-in({lb}) = {d_train:+.4f} " + (f">= +{args.margin}" if d_train >= args.margin else
               f"<= -{args.margin}" if d_train <= -args.margin else f"within {args.margin}: base branch (one change less)"))
        if d_play is not None and abs(d_play) > args.play_conflict and (d_play > 0) != (win == lv):
            res.update(verdict=None, reason=f"play check opposite: train {why}, play delta {d_play:+.4f} > {args.play_conflict}; user decides")
        else:
            res.update(verdict=win, reason=why + ("" if d_play is None else f"; play delta {d_play:+.4f}"))
        if abs(d_train) < args.margin + 0.005:
            res["note"] = "difference close to the margin; one seed per branch, not enough to call a real difference"
    finish(res)


def finish(res):
    path = os.path.join(args.out, "c_compare.json")
    json.dump(res, open(path, "w"), indent=1)
    for label, b in res["branches"].items():
        w = b["window"]
        say(f"{label} ({b['run_name']}): last it {b['last_iter']}, {b['sec_per_iter']} s/it, window in={w['in']} "
            f"fall={w['fall']} fault={w['fault']} tunnel={w['tunnel']} vc={w['vc_mean']} v_close_max={w['param_v_close_max']}"
            + (f"; play in_standing={b['play']['served_in_standing']} n={b['play']['rallies']}" if b.get("play") else ""))
    say(f"verdict: {res.get('verdict')} ({res.get('reason')}) -> {path}")


if __name__ == "__main__":
    main()
