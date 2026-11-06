"""P3 monitor (P3_PLAN_V2 4.6, 4.7, 5). Run with a Python that has ``tensorboard`` (env_swanlab).

Every few minutes:
  * every new ``model_<k>.pt`` with k a multiple of ``--eval_every``: export, then evaluate in play
    (both sides x zero / model_test), sequentially; play's player cfgs are restored by git after each run;
  * checks the "stop and report" conditions of plan section 5 and the 4.6 symptoms, and logs them.
    It never changes weights or stops training: decisions are left to a human.

Stops when ``--done_file`` exists and nothing is left to evaluate.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import time
from datetime import datetime

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="/root/autodl-tmp/BadmintonLab-Game-V3L")
ap.add_argument("--play", default="/root/autodl-tmp/BadmintonLab-Game-Play")
ap.add_argument("--isaac_python", default="/root/autodl-tmp/env_isaaclab/bin/python")
ap.add_argument("--run_name", default="p3_s1")
ap.add_argument("--out", default="/root/autodl-tmp/logs/v3l/p3")
ap.add_argument("--eval_every", type=int, default=1000)
ap.add_argument("--eval_rallies", type=int, default=30, help="Per court; ~16 courts -> >= 200 serve rallies.")
ap.add_argument("--done_file", required=True)
ap.add_argument("--poll_s", type=float, default=300.0)
ap.add_argument("--orig_max", default="A=1500,B=4000,C=4000", help="Plan max iterations, for reporting only.")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
STATUS = os.path.join(args.out, "P3_STATUS.txt")
STATE = os.path.join(args.out, "monitor_state.json")
PLAY_CFGS = [
    "source/badmintonlab_game/badmintonlab_game/tasks/manager_based/badmintonlab_game/player0_cfg.py",
    "source/badmintonlab_game/badmintonlab_game/tasks/manager_based/badmintonlab_game/player1_cfg.py",
]
ORIG_MAX = {k: int(v) for k, v in (x.split("=") for x in args.orig_max.split(","))}
STAGES = ("A", "B", "C")


def say(msg):
    line = f"[{datetime.now():%F %T}] {msg}"
    print(line, flush=True)
    with open(STATUS, "a") as f:
        f.write(line + "\n")


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"evaluated": [], "alerts": []}


def save_state(st):
    json.dump(st, open(STATE, "w"), indent=1)


def run_dirs():
    return sorted(glob.glob(os.path.join(args.repo, "logs/rsl_rl/badmintonlab_serve", f"*_{args.run_name}")))


def scalars():
    """Merged scalar series over all (resumed) run directories: tag -> {step: value}."""
    out = {}
    for d in run_dirs():
        for ev in glob.glob(os.path.join(d, "events*")):
            acc = EventAccumulator(ev, size_guidance={"scalars": 0})
            acc.Reload()
            for tag in acc.Tags()["scalars"]:
                if tag.startswith(("Serve/", "Curriculum/")):
                    s = out.setdefault(tag, {})
                    for e in acc.Scalars(tag):
                        s[e.step] = e.value
    return out


def window_mean(series, end_step, n=50):
    steps = [s for s in sorted(series) if s <= end_step][-n:]
    return sum(series[s] for s in steps) / len(steps) if steps else None


def checkpoints():
    found = {}
    for d in run_dirs():
        for f in glob.glob(os.path.join(d, "model_*.pt")):
            m = re.fullmatch(r"model_(\d+)\.pt", os.path.basename(f))
            if m:
                found[int(m.group(1))] = f
    return found


def alert(st, key, msg):
    if key in st["alerts"]:
        return
    st["alerts"].append(key)
    say("ALERT " + msg)


def check_training(st, sc):
    if "Curriculum/stage" not in sc:
        return
    stage_s, iters_s = sc["Curriculum/stage"], sc.get("Curriculum/iters_in_stage", {})
    last = max(stage_s)
    stage = STAGES[int(round(stage_s[last]))]
    iters = int(iters_s.get(last, 0))
    ema = {k.split("_", 1)[1]: v[last] for k, v in sc.items() if k.startswith("Curriculum/ema_") and last in v}
    # Stage history (promotions)
    prev = None
    for s in sorted(stage_s):
        cur = int(round(stage_s[s]))
        if prev is not None and cur != prev:
            alert(st, f"promo_{cur}_{s}", f"curriculum promoted to stage {STAGES[cur]} at iteration {s}")
        prev = cur
    if iters >= ORIG_MAX[stage]:
        alert(st, f"origmax_{stage}", f"stage {stage} passed the plan's max {ORIG_MAX[stage]} iterations without "
              f"promotion (limit relaxed by request; continuing). EMA: { {k: round(v, 3) for k, v in ema.items() if k in ('legal','crossed','in','fault','fall','tunnel','vc_mean','el_mean')} }")
    if stage == "B" and iters >= 1500 and ema.get("in", 1.0) < 0.40:
        alert(st, "B1500_in40", f"plan 5: stage B {iters} iterations, EMA in={ema['in']:.3f} < 0.40")
    if iters >= 300 and ema.get("tunnel", 0) > 0.25:
        vc = ema.get("vc_mean", 0)
        alert(st, f"tunnel_{stage}", f"plan 4.6: EMA tunnel={ema['tunnel']:.3f} > 0.25 with vc_mean={vc:.2f} m/s "
              + ("(> 8: suggests tunnel weight -3 or v_close_max 7.5)" if vc > 8 else "(< 7: not speed-driven; report to local)" if vc < 7 else ""))
    if iters >= 300 and ema.get("timeout", 0) > 0.01:
        alert(st, f"timeout_{stage}", f"plan 4.6: EMA timeout={ema['timeout']:.3f} > 0.01")
    if stage == "B" and iters >= 600 and ema.get("in", 1.0) < 0.50:
        alert(st, "B600_in50", f"plan 4.6: stage B {iters} iterations, EMA in={ema['in']:.3f} < 0.50 (plan suggests fall weight -7)")
    el = ema.get("el_mean")
    if iters >= 300 and el is not None and ema.get("legal", 0) > 0.3 and not (30.0 <= el <= 70.0):
        alert(st, f"el_{stage}", f"plan 5: EMA el_mean={el:.1f} deg outside 30-70 with legal={ema['legal']:.2f}")


def evaluate(st, k, ckpt, sc):
    exp = os.path.join(args.repo, "exports", f"{args.run_name}_{k}")
    log = os.path.join(args.out, f"export_{k}.log")
    r = subprocess.run([args.isaac_python, "scripts/export_torchscript.py", "--checkpoint", ckpt, "--out", exp],
                       cwd=args.repo, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    if r.returncode != 0:
        say(f"export {k} failed (exit {r.returncode}), see {log}")
        return
    rows = []
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES")
    for opp in ("zero", "model_test"):
        for side in (0, 1):
            out = os.path.join(args.out, f"eval_{k}_{opp}_side{side}.json")
            r = subprocess.run(
                [args.isaac_python, "scripts/eval_serve.py", "--play_root", args.play, "--ours", exp, "--opponent", opp,
                 "--side", str(side), "--num_envs", "16", "--rallies", str(args.eval_rallies),
                 "--python", args.isaac_python, "--out", out],
                cwd=args.repo, env=env, stdout=open(out.replace(".json", ".log"), "w"), stderr=subprocess.STDOUT,
                timeout=3600,
            )
            subprocess.run(["git", "-C", args.play, "checkout", "--", *PLAY_CFGS])
            if r.returncode != 0 or not os.path.exists(out):
                say(f"  eval {k} {opp} side{side} failed (exit {r.returncode})")
                continue
            s = json.load(open(out))["serve"]
            rt = s["rates"]
            rows.append((opp, side, s["rallies"], rt))
            say(f"  eval it={k} opp={opp} side={side} n={s['rallies']} legal={rt['legal']:.3f} in={rt['served_in']:.3f} "
                f"fault={rt['height'] + rt['double']:.3f} no_hit={rt['no_hit']:.3f} fell_after_hit={rt['fell_after_hit']:.3f} "
                f"hit_t50={((s.get('timing_s') or {}).get('hit') or {}).get('p50')}")
    if not rows:
        return
    train_in = window_mean(sc.get("Serve/in", {}), k)
    train_fall = window_mean(sc.get("Serve/fall", {}), k)
    ev_in = sum(r[3]["served_in"] for r in rows) / len(rows)
    ev_fah = max(r[3]["fell_after_hit"] for r in rows)
    say(f"  it={k}: train(in={train_in}, fall={train_fall}) vs play mean in={ev_in:.3f}, max fell_after_hit={ev_fah:.3f}")
    if train_in is not None and abs(train_in - ev_in) > 0.10:
        alert(st, f"gap_in_{k}", f"plan 5: it={k} train in {train_in:.3f} vs play {ev_in:.3f} differ by > 10 points")
    if train_fall is not None and train_fall < 0.10 and ev_fah > 0.30:
        alert(st, f"gap_fall_{k}", f"plan 5: it={k} train fall {train_fall:.3f} < 0.10 but play fell_after_hit {ev_fah:.3f} > 0.30")
    m1 = all(r[3]["legal"] >= 0.90 and r[3]["served_in"] >= 0.85 and r[3]["height"] + r[3]["double"] <= 0.02
             and r[3]["fell_after_hit"] <= 0.10 for r in rows) and len(rows) == 4
    if m1:
        alert(st, f"M1_{k}", f"M1 reached in play at it={k} (export {exp})")


def main():
    st = load_state()
    say(f"monitor start: run_name={args.run_name}, eval every {args.eval_every}")
    while True:
        done = os.path.exists(args.done_file)
        try:
            sc = scalars()
            check_training(st, sc)
            ck = checkpoints()
            todo = sorted(k for k in ck if k > 0 and k % args.eval_every == 0 and k not in st["evaluated"])
            if done and ck:
                last = max(ck)
                if last not in st["evaluated"] and last not in todo:
                    todo.append(last)  # final checkpoint
            for k in todo:
                say(f"evaluate checkpoint {k}: {ck[k]}")
                evaluate(st, k, ck[k], sc)
                st["evaluated"].append(k)
                save_state(st)
            save_state(st)
        except Exception as e:  # keep monitoring
            say(f"monitor error: {e!r}")
        if done and not todo:
            say("monitor finished")
            return
        time.sleep(30 if done else args.poll_s)


if __name__ == "__main__":
    main()
