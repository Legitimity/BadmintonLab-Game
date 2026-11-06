#!/usr/bin/env python3
"""E1 selection (P3_SPEEDUP_PLAN 4.2): pick "courts:mini_batches" from wall-clock aligned learning curves.

Rules
  1. main metric: first wall time at which the 5-min mean of ``Serve/in`` reaches ``--target`` (0.6)
  2. groups within 10 % of the fastest time -> the one with the shortest iteration time wins
  3. a group whose 5-min mean ``Serve/fall`` exceeds 0.05 at the end of its run is eliminated
  no group reaches the target -> highest 5-min ``Serve/in`` at the common end time
Prints "courts:mini_batches" on stdout; the table goes to stderr and ``--out`` (json).
"""
import argparse
import glob
import json
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="/root/autodl-tmp/BadmintonLab-Game-V3L")
ap.add_argument("--groups", default="2048:16 8192:64 8192:32 4096:32")
ap.add_argument("--name", default="p3e1_{n}_mb{mb}", help="run name pattern")
ap.add_argument("--mem_dir", default="/root/autodl-tmp/logs/v3l/p3")
ap.add_argument("--target", type=float, default=0.6)
ap.add_argument("--fall_max", type=float, default=0.05)
ap.add_argument("--window", type=float, default=300.0, help="seconds")
ap.add_argument("--out", default=None)
args = ap.parse_args()

MARKS = (5, 10, 20, 30, 45)


def load(run):
    dirs = sorted(glob.glob(os.path.join(args.repo, "logs/rsl_rl/badmintonlab_serve", f"*_{run}")))
    if not dirs:
        return None
    tags = {}
    for ev in glob.glob(os.path.join(dirs[-1], "events*")):
        acc = EventAccumulator(ev, size_guidance={"scalars": 0})
        acc.Reload()
        for tag in ("Serve/in", "Serve/fall", "Perf/total_fps", "Perf/collection time", "Perf/learning_time"):
            if tag in acc.Tags()["scalars"]:
                tags.setdefault(tag, []).extend((e.wall_time, e.step, e.value) for e in acc.Scalars(tag))
    for v in tags.values():
        v.sort()
    return tags if tags.get("Serve/in") else None


def wmean(series, t, t0):
    """Mean of values with wall time in (t - window, t]; ``t`` is relative to ``t0``."""
    vals = [v for w, _, v in series if t - args.window < w - t0 <= t]
    return sum(vals) / len(vals) if vals else None


def peak_mem(run):
    try:
        vals = [int(x) for x in open(os.path.join(args.mem_dir, f"{run}.mem")).read().split() if x.isdigit()]
        return max(vals) if vals else None
    except OSError:
        return None


rows = []
for g in args.groups.split():
    n, mb = g.split(":")
    run = args.name.format(n=n, mb=mb)
    tags = load(run)
    if tags is None:
        print(f"[select] {run}: no data", file=sys.stderr)
        continue
    s_in, s_fall = tags["Serve/in"], tags.get("Serve/fall", [])
    t0 = s_in[0][0] - 1e-6
    t_end = s_in[-1][0] - t0
    t_hit = None
    for w, _, _ in s_in:
        t = w - t0
        if t >= min(args.window, t_end) and (wmean(s_in, t, t0) or 0) >= args.target:
            t_hit = t
            break
    iters = len(s_in)
    coll = [v for _, _, v in tags.get("Perf/collection time", [])]
    learn = [v for _, _, v in tags.get("Perf/learning_time", [])]
    fps = [v for _, _, v in tags.get("Perf/total_fps", [])]
    rows.append(
        dict(
            group=g,
            run=run,
            iters=iters,
            minutes=round(t_end / 60, 1),
            iter_s=round(t_end / max(iters - 1, 1), 2),
            collect_s=round(sum(coll) / len(coll), 2) if coll else None,
            learn_s=round(sum(learn) / len(learn), 2) if learn else None,
            steps_per_s=round(sum(fps) / len(fps)) if fps else None,
            mem_mib=peak_mem(run),
            t_hit_min=None if t_hit is None else round(t_hit / 60, 1),
            fall_end=wmean(s_fall, t_end, t0) if s_fall else None,
            in_at={m: wmean(s_in, m * 60, t0) if m * 60 <= t_end + 30 else None for m in MARKS},
            fall_at={m: wmean(s_fall, m * 60, t0) if m * 60 <= t_end + 30 else None for m in MARKS},
            _t_end=t_end,
            _t0=t0,
            _s_in=s_in,
        )
    )

fmt = lambda x: "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
print("group      iters  iter_s  coll_s  learn_s  steps/s  memMiB  hit0.6  fall_end  " + "  ".join(f"in@{m}" for m in MARKS), file=sys.stderr)
for r in rows:
    print(
        f"{r['group']:<10} {r['iters']:>5}  {fmt(r['iter_s']):>6}  {fmt(r['collect_s']):>6}  {fmt(r['learn_s']):>7}  "
        f"{fmt(r['steps_per_s']):>7}  {fmt(r['mem_mib']):>6}  {fmt(r['t_hit_min']):>6}  {fmt(r['fall_end']):>8}  "
        + "  ".join(f"{fmt(r['in_at'][m]):>5}" for m in MARKS),
        file=sys.stderr,
    )

ok = [r for r in rows if r["fall_end"] is None or r["fall_end"] <= args.fall_max]
for r in rows:
    if r not in ok:
        print(f"[select] {r['group']} eliminated: fall {r['fall_end']:.3f} > {args.fall_max}", file=sys.stderr)
choice, why = None, ""
hit = [r for r in ok if r["t_hit_min"] is not None]
if hit:
    best = min(r["t_hit_min"] for r in hit)
    close = [r for r in hit if r["t_hit_min"] <= best * 1.10]
    choice = min(close, key=lambda r: (r["iter_s"], r["t_hit_min"]))
    why = f"in>={args.target} at {choice['t_hit_min']} min (fastest {best} min, {len(close)} within 10%, shortest iteration)"
elif ok:
    t_common = min(r["_t_end"] for r in ok)
    choice = max(ok, key=lambda r: wmean(r["_s_in"], t_common, r["_t0"]) or 0)
    why = f"no group reached {args.target}; highest in at common end {t_common / 60:.1f} min"
if choice:
    print(f"[select] {choice['group']}: {why}", file=sys.stderr)
if args.out:
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    json.dump(dict(rows=clean, choice=choice and choice["group"], reason=why), open(args.out, "w"), indent=1)
if choice is None:
    sys.exit(1)
print(choice["group"])
