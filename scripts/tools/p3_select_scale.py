#!/usr/bin/env python3
"""E1 / E1b summary, selection and reward-fix verdict (P3_SPEEDUP_PLAN 4.2, P3_REWARD_FIX_AND_E1B 4.2-4.4).

Runs are given as ``label=run_name``; courts and mini-batches come from the run's ``run_meta.json``.

Selection (among ``--runs``; ``--controls`` are only reported)
  1. first wall time at which the 5-min mean of ``Serve/in`` reaches ``--target`` (0.6)
  2. runs within 10 % of the fastest -> the one with the shortest iteration time
  3. eliminated: 5-min ``Serve/fall`` > 0.05 at the end of the run
  4. eliminated (E1b): 5-min ``el_mean`` > 65 deg for >= 10 consecutive minutes
  5. (E1b) ``--seed_pair a,b``: both 2048:16 seeds differ by > 30 % in time to target -> pick the faster of the
     two directly (seed noise exceeds the scale effect)
  no run reaches the target -> highest 5-min ``Serve/in`` at the common end time
Verdict (``--verdict b1,b2,b3,ctrl``): effective / insufficient / interferes / unclear (E1b 4.4).
Prints "courts:mini_batches:run_name" on stdout; tables go to stderr and ``--out`` (json).
"""
import argparse
import glob
import json
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="/root/autodl-tmp/BadmintonLab-Game-V3L")
ap.add_argument("--runs", required=True, help="'label=run_name ...' candidates")
ap.add_argument("--controls", default="", help="'label=run_name ...' reported only (e.g. old-reward E1)")
ap.add_argument("--mem_dir", default="/root/autodl-tmp/logs/v3l/p3")
ap.add_argument("--target", type=float, default=0.6)
ap.add_argument("--fall_max", type=float, default=0.05)
ap.add_argument("--trap_deg", type=float, default=65.0)
ap.add_argument("--trap_run_min", type=float, default=10.0)
ap.add_argument("--window", type=float, default=300.0, help="seconds")
ap.add_argument("--seed_pair", default=None, help="labels of the two 2048:16 seeds, e.g. b1,b4")
ap.add_argument("--verdict", default=None, help="labels 'fixed2048,probe8192,other8192,control2048', e.g. b1,b2,b3,e1_2048")
ap.add_argument("--out", default=None)
args = ap.parse_args()

MARKS = (5, 10, 20, 30, 45)
TAGS = ("Serve/in", "Serve/fall", "Serve/crossed", "Serve/el_mean", "Serve/launch_el_mean", "Perf/total_fps",
        "Perf/collection time", "Perf/learning_time")


def pairs(s):
    return [tuple(x.split("=", 1)) for x in s.split()] if s else []


def load(run):
    dirs = sorted(glob.glob(os.path.join(args.repo, "logs/rsl_rl/badmintonlab_serve", f"*_{run}")))
    if not dirs:
        return None, None
    tags = {}
    for ev in glob.glob(os.path.join(dirs[-1], "events*")):
        acc = EventAccumulator(ev, size_guidance={"scalars": 0})
        acc.Reload()
        for tag in TAGS:
            if tag in acc.Tags()["scalars"]:
                tags.setdefault(tag, []).extend((e.wall_time, e.step, e.value) for e in acc.Scalars(tag))
    for v in tags.values():
        v.sort()
    meta = {}
    try:
        meta = json.load(open(os.path.join(dirs[-1], "run_meta.json"))).get("params", {})
    except (OSError, ValueError):
        pass
    return (tags if tags.get("Serve/in") else None), meta


def wmean(series, t, t0):
    """Mean of values with wall time in (t - window, t]; ``t`` relative to ``t0``."""
    vals = [v for w, _, v in series if t - args.window < w - t0 <= t]
    return sum(vals) / len(vals) if vals else None


def first_reach(series, t0, t_end, thr):
    for w, _, _ in series:
        t = w - t0
        if t >= min(args.window, t_end) and (wmean(series, t, t0) or 0) >= thr:
            return t
    return None


def peak_mem(run):
    try:
        vals = [int(x) for x in open(os.path.join(args.mem_dir, f"{run}.mem")).read().split() if x.isdigit()]
        return max(vals) if vals else None
    except OSError:
        return None


def trapped(series, t0, t_end):
    """Minutes (1-min grid) with the 5-min el_mean above trap_deg: total and longest consecutive run."""
    total = run = best = 0
    m = 1
    while m * 60 <= t_end + 1:
        v = wmean(series, m * 60, t0)
        if v is not None and v > args.trap_deg:
            total += 1
            run += 1
            best = max(best, run)
        else:
            run = 0
        m += 1
    return total, best


def summarize(label, run):
    tags, meta = load(run)
    if tags is None:
        print(f"[select] {label} ({run}): no data", file=sys.stderr)
        return None
    s_in = tags["Serve/in"]
    t0 = s_in[0][0] - 1e-6
    t_end = s_in[-1][0] - t0
    at = lambda tag: {m: (wmean(tags.get(tag, []), m * 60, t0) if m * 60 <= t_end + 30 else None) for m in MARKS}  # noqa: E731
    mean = lambda tag: (sum(v for _, _, v in tags[tag]) / len(tags[tag])) if tags.get(tag) else None  # noqa: E731
    t_hit = first_reach(s_in, t0, t_end, args.target)
    t_cross = first_reach(tags.get("Serve/crossed", []), t0, t_end, 0.3)
    trap_total, trap_run = trapped(tags.get("Serve/el_mean", []), t0, t_end)
    return dict(
        label=label, run=run, courts=meta.get("num_envs"), mini_batches=meta.get("num_mini_batches"),
        seed=meta.get("seed"), iters=len(s_in), minutes=round(t_end / 60, 1),
        iter_s=round(t_end / max(len(s_in) - 1, 1), 2),
        collect_s=mean("Perf/collection time"), learn_s=mean("Perf/learning_time"), steps_per_s=mean("Perf/total_fps"),
        mem_mib=peak_mem(run),
        t_hit_min=None if t_hit is None else round(t_hit / 60, 1),
        t_cross03_min=None if t_cross is None else round(t_cross / 60, 1),
        in_end=wmean(s_in, t_end, t0), fall_end=wmean(tags.get("Serve/fall", []), t_end, t0) if tags.get("Serve/fall") else None,
        trapped_min=trap_total, trapped_run_min=trap_run,
        in_at=at("Serve/in"), fall_at=at("Serve/fall"), el_at=at("Serve/el_mean"), launch_el_at=at("Serve/launch_el_mean"),
        _t_end=t_end, _t0=t0, _s_in=s_in,
    )


fmt = lambda x: "-" if x is None else (f"{x:.3f}" if isinstance(x, float) and abs(x) < 10 else (f"{x:.1f}" if isinstance(x, float) else str(x)))  # noqa: E731


def table(rows, title):
    print(f"== {title}", file=sys.stderr)
    print("label       courts:mb  iters iter_s  steps/s memMiB hit0.6 cross.3 in_end fall_end trap(tot/run) "
          + " ".join(f"in@{m}" for m in MARKS) + " | " + " ".join(f"el@{m}" for m in MARKS) + " | "
          + " ".join(f"lel@{m}" for m in MARKS), file=sys.stderr)
    for r in rows:
        print(f"{r['label']:<11} {str(r['courts']) + ':' + str(r['mini_batches']):<10} {r['iters']:>5} {fmt(r['iter_s']):>6} "
              f"{fmt(r['steps_per_s']):>8} {fmt(r['mem_mib']):>6} {fmt(r['t_hit_min']):>6} {fmt(r['t_cross03_min']):>7} "
              f"{fmt(r['in_end']):>6} {fmt(r['fall_end']):>8} {r['trapped_min']:>5}/{r['trapped_run_min']:<6} "
              + " ".join(f"{fmt(r['in_at'][m]):>5}" for m in MARKS) + " | "
              + " ".join(f"{fmt(r['el_at'][m]):>5}" for m in MARKS) + " | "
              + " ".join(f"{fmt(r['launch_el_at'][m]):>6}" for m in MARKS), file=sys.stderr)


rows = [r for r in (summarize(lab, run) for lab, run in pairs(args.runs)) if r]
ctrl = [r for r in (summarize(lab, run) for lab, run in pairs(args.controls)) if r]
if ctrl:
    table(ctrl, "controls")
table(rows, "candidates")
by = {r["label"]: r for r in rows + ctrl}

# ---------------------------------------------------------------- selection
choice, why = None, ""
ok = []
for r in rows:
    if r["fall_end"] is not None and r["fall_end"] > args.fall_max:
        print(f"[select] {r['label']} eliminated: fall {r['fall_end']:.3f} > {args.fall_max}", file=sys.stderr)
    elif r["trapped_run_min"] >= args.trap_run_min:
        print(f"[select] {r['label']} eliminated: el_mean > {args.trap_deg} for {r['trapped_run_min']} consecutive min", file=sys.stderr)
    else:
        ok.append(r)
if args.seed_pair:
    a, b = (by.get(x) for x in args.seed_pair.split(","))
    if a and b:
        ta, tb = a["t_hit_min"], b["t_hit_min"]
        spread = None if (ta is None or tb is None) else abs(ta - tb) / max(min(ta, tb), 1e-6)
        if (ta is None) != (tb is None) or (spread is not None and spread > 0.30):
            pick = [r for r in (a, b) if r in ok and r["t_hit_min"] is not None]
            if pick:
                choice = min(pick, key=lambda r: r["t_hit_min"])
                why = f"seed rule: {a['label']} {ta} min vs {b['label']} {tb} min differ > 30% -> 2048:16 directly"
if choice is None:
    hit = [r for r in ok if r["t_hit_min"] is not None]
    if hit:
        best = min(r["t_hit_min"] for r in hit)
        close = [r for r in hit if r["t_hit_min"] <= best * 1.10]
        choice = min(close, key=lambda r: (r["iter_s"], r["t_hit_min"]))
        why = f"in>={args.target} at {choice['t_hit_min']} min (fastest {best} min, {len(close)} within 10%, shortest iteration)"
    elif ok:
        t_common = min(r["_t_end"] for r in ok)
        choice = max(ok, key=lambda r: wmean(r["_s_in"], t_common, r["_t0"]) or 0)
        why = f"no run reached {args.target}; highest in at common end {t_common / 60:.1f} min"
if choice:
    print(f"[select] {choice['label']} ({choice['courts']}:{choice['mini_batches']}): {why}", file=sys.stderr)

# ---------------------------------------------------------------- verdict (E1b 4.4)
verdict, vwhy = None, ""
if args.verdict:
    f2048, probe, other, c2048 = (by.get(x) for x in args.verdict.split(","))
    normal = lambda r: r is not None and r["t_hit_min"] is not None and r["trapped_min"] < 10  # noqa: E731
    if f2048 is None or probe is None:
        verdict, vwhy = "unclear", "missing runs"
    elif c2048 and c2048["t_hit_min"] and (f2048["t_hit_min"] is None or f2048["t_hit_min"] > 1.3 * c2048["t_hit_min"]):
        verdict, vwhy = "interferes", f"{f2048['label']} hit {f2048['t_hit_min']} min vs control {c2048['t_hit_min']} min (> +30%)"
    elif (probe["in_end"] or 0) >= 0.5 and probe["trapped_min"] < 10:
        verdict, vwhy = "effective", f"{probe['label']} in_end={probe['in_end']:.3f}, trapped {probe['trapped_min']} min"
    elif normal(f2048) and normal(other):
        verdict, vwhy = "insufficient", (f"{probe['label']} in_end={fmt(probe['in_end'])}, trapped {probe['trapped_min']} min; "
                                         f"{f2048['label']} and {other['label']} normal")
    else:
        verdict, vwhy = "unclear", (f"{probe['label']} in_end={fmt(probe['in_end'])} trapped {probe['trapped_min']} min; "
                                    f"{other['label'] if other else '?'} normal={normal(other)}")
    print(f"[verdict] {verdict}: {vwhy}", file=sys.stderr)

if args.out:
    clean = lambda rs: [{k: v for k, v in r.items() if not k.startswith("_")} for r in rs]  # noqa: E731
    json.dump(dict(candidates=clean(rows), controls=clean(ctrl), choice=choice and choice["label"],
                   choice_run=choice and choice["run"], reason=why, verdict=verdict, verdict_reason=vwhy),
              open(args.out, "w"), indent=1)
if choice is None:
    sys.exit(1)
print(f"{choice['courts']}:{choice['mini_batches']}:{choice['run']}")
