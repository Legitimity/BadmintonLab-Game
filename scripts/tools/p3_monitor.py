"""P3 monitor (P3_PLAN_V2 4.6, 4.7, 5 and the review of the run flow). Run with env_swanlab's Python.

Every ``--poll_s`` seconds:
  * evaluates in play (export, both sides x zero / model_test, sequential; play's cfgs restored by git):
    every ``model_<k>.pt`` with k a multiple of ``--eval_every``, the ``model_finished_<k>.pt`` written
    when stage C passes, and the final checkpoint;
  * automatic stop (SIGTERM to the trainer: it saves ``latest.pt`` and ``model_<it>.pt`` after the current
    iteration; ``STOP_REQUESTED`` tells the queue not to restart):
      - plan 5: stage B >= 1500 iterations with EMA in < 0.40
      - plan 5: train ``Serve/in`` vs play standing-in (zero opponent) differ by > 10 points
      - plan 5: train fall < 0.10 while play fell_after_hit > 0.30
      - early stop: M1 met in play (both sides, both opponents)
      - early stop: 1000 iterations after stage C passed
  * alerts only (no action): plan 4.6 symptoms, el_mean, the plan's original stage maxima (relaxed on
    request), and ``in`` stuck below the gate (the trigger for considering CaT).

Exits when ``--done_file`` exists and nothing is left to evaluate.
"""

import argparse
import glob
import json
import os
import re
import signal
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
ap.add_argument("--eval_rallies", type=int, default=30, help="Per court; 16 courts -> >= 200 serve rallies.")
ap.add_argument("--done_file", required=True)
ap.add_argument("--poll_s", type=float, default=120.0)
ap.add_argument("--after_c_iters", type=int, default=1000)
ap.add_argument("--orig_max", default="A=1500,B=4000,C=4000", help="Plan maxima, alert only.")
ap.add_argument("--no_stop", action="store_true", help="Alerts only (testing).")
ap.add_argument("--test_stop_at", type=int, default=None, help="Testing: request a stop once this iteration is logged.")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
STATUS = os.path.join(args.out, "P3_STATUS.txt")
STATE = os.path.join(args.out, "monitor_state.json")
STOP_FILE = os.path.join(args.out, "STOP_REQUESTED")
PLAY_CFGS = [
    "source/badmintonlab_game/badmintonlab_game/tasks/manager_based/badmintonlab_game/player0_cfg.py",
    "source/badmintonlab_game/badmintonlab_game/tasks/manager_based/badmintonlab_game/player1_cfg.py",
]
ORIG_MAX = {k: int(v) for k, v in (x.split("=") for x in args.orig_max.split(","))}
STAGES = ("A", "B", "C")
GATE_IN = {"A": None, "B": 0.60, "C": 0.85}


def say(msg):
    line = f"[{datetime.now():%F %T}] {msg}"
    print(line, flush=True)
    with open(STATUS, "a") as f:
        f.write(line + "\n")


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"evaluated": [], "alerts": [], "stop": None}


def save_state(st):
    json.dump(st, open(STATE, "w"), indent=1)


def run_dirs():
    return sorted(glob.glob(os.path.join(args.repo, "logs/rsl_rl/badmintonlab_serve", f"*_{args.run_name}")))


def scalars():
    """Scalar series merged over all (resumed) run directories: tag -> {step: value}."""
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
    """iteration -> path of plain ``model_<k>.pt``; and the iteration where stage C passed (or None)."""
    plain, finished = {}, None
    for d in run_dirs():
        for f in glob.glob(os.path.join(d, "model_*.pt")):
            b = os.path.basename(f)
            m = re.fullmatch(r"model_(\d+)\.pt", b)
            if m:
                plain[int(m.group(1))] = f
            m = re.fullmatch(r"model_finished_(\d+)\.pt", b)
            if m:
                finished = int(m.group(1))
                plain.setdefault(finished, f)
    return plain, finished


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
    pids = subprocess.run(["pgrep", "-f", f"train_serve.py.*--run_name {args.run_name}( |$)"],
                          capture_output=True, text=True).stdout.split()
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except (ProcessLookupError, ValueError):
            pass
    say(f"STOP requested: {reason} (SIGTERM to {pids or 'no running trainer'}; checkpoint written after the current iteration)")


def check_training(st, sc, finished_it):
    if "Curriculum/stage" not in sc:
        return
    stage_s, iters_s = sc["Curriculum/stage"], sc.get("Curriculum/iters_in_stage", {})
    last = max(stage_s)
    stage = STAGES[int(round(stage_s[last]))]
    iters = int(iters_s.get(last, 0))
    ema = {k.split("_", 1)[1]: v[last] for k, v in sc.items() if k.startswith("Curriculum/ema_") and last in v}
    if args.test_stop_at is not None and last >= args.test_stop_at:
        request_stop(st, f"test stop at iteration {last}")
    short = {k: round(v, 3) for k, v in ema.items() if k in ("legal", "crossed", "in", "fault", "fall", "tunnel", "vc_mean", "el_mean", "launch_el_mean")}
    prev = None
    for s in sorted(stage_s):
        cur = int(round(stage_s[s]))
        if prev is not None and cur != prev:
            alert(st, f"promo_{cur}_{s}", f"curriculum promoted to stage {STAGES[cur]} at iteration {s}")
        prev = cur
    if finished_it is not None:
        alert(st, f"finished_{finished_it}", f"stage C passed at iteration {finished_it}; early stop at +{args.after_c_iters} or M1")
        if last >= finished_it + args.after_c_iters:
            request_stop(st, f"early stop: {last - finished_it} iterations after stage C passed (it {last})")
    # -- automatic stop (plan 5)
    if stage == "B" and iters >= 1500 and ema.get("in", 1.0) < 0.40:
        request_stop(st, f"plan 5: stage B {iters} iterations, EMA in={ema['in']:.3f} < 0.40; EMA {short}")
    # -- alerts only
    if iters >= ORIG_MAX[stage]:
        alert(st, f"origmax_{stage}", f"stage {stage} passed the plan's max {ORIG_MAX[stage]} iterations without promotion "
              f"(relaxed on request, continuing). EMA {short}")
    if iters >= 300 and ema.get("tunnel", 0) > 0.25:
        vc = ema.get("vc_mean", 0)
        alert(st, f"tunnel_{stage}", f"plan 4.6: EMA tunnel={ema['tunnel']:.3f} > 0.25 with vc_mean={vc:.2f} m/s "
              + ("(> 8: plan suggests tunnel weight -3 or v_close_max 7.5)" if vc > 8 else "(< 7: not speed-driven)" if vc < 7 else ""))
    if iters >= 300 and ema.get("timeout", 0) > 0.01:
        alert(st, f"timeout_{stage}", f"plan 4.6: EMA timeout={ema['timeout']:.3f} > 0.01")
    if stage == "B" and iters >= 600 and ema.get("in", 1.0) < 0.50:
        alert(st, "B600_in50", f"plan 4.6: stage B {iters} iterations, EMA in={ema['in']:.3f} < 0.50 (plan: fall -10 -> -7, or CaT)")
    el = ema.get("el_mean")
    if iters >= 300 and el is not None and ema.get("legal", 0) > 0.3 and not (30.0 <= el <= 70.0):
        alert(st, f"el_{stage}", f"EMA el_mean={el:.1f} deg outside 30-70 with legal={ema['legal']:.2f}")
    lel = ema.get("launch_el_mean")
    if iters >= 300 and lel is not None and ema.get("legal", 0) > 0.3 and not (25.0 <= lel <= 65.0):
        alert(st, f"launch_el_{stage}", f"EMA launch_el_mean={lel:.1f} deg outside 25-65 with legal={ema['legal']:.2f}")
    # "in" pressed down and not recovering: below the gate and < 2 points gained over the last 500 iterations.
    gate = GATE_IN[stage]
    s_in = sc.get("Curriculum/ema_in", {})
    if gate is not None and iters >= 800 and s_in:
        past = [s for s in s_in if s <= last - 500]
        if past and ema.get("in", 1.0) < gate and ema["in"] - s_in[max(past)] < 0.02:
            alert(st, f"plateau_{stage}_{last // 1000}", f"stage {stage}: EMA in={ema['in']:.3f} below gate {gate} and "
                  f"+{ema['in'] - s_in[max(past)]:.3f} over 500 iterations (consider CaT); EMA {short}")


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
            try:
                r = subprocess.run(
                    [args.isaac_python, "scripts/eval_serve.py", "--play_root", args.play, "--ours", exp, "--opponent", opp,
                     "--side", str(side), "--num_envs", "16", "--rallies", str(args.eval_rallies),
                     "--python", args.isaac_python, "--out", out],
                    cwd=args.repo, env=env, stdout=open(out.replace(".json", ".log"), "w"), stderr=subprocess.STDOUT,
                    timeout=3600,
                )
                code = r.returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
            subprocess.run(["git", "-C", args.play, "checkout", "--", *PLAY_CFGS])
            if code != 0 or not os.path.exists(out):
                say(f"  eval {k} {opp} side{side} failed ({code})")
                continue
            s = json.load(open(out))["serve"]
            rt = s["rates"]
            rows.append((opp, side, s["rallies"], rt))
            say(f"  eval it={k} opp={opp} side={side} n={s['rallies']} legal={rt['legal']:.3f} in={rt['served_in']:.3f} "
                f"in_standing={rt.get('served_in_standing', float('nan')):.3f} fault={rt['height'] + rt['double']:.3f} "
                f"no_hit={rt['no_hit']:.3f} fell_after_hit={rt['fell_after_hit']:.3f} "
                f"hit_t50={((s.get('timing_s') or {}).get('hit') or {}).get('p50')}")
    if not rows:
        return
    train_in = window_mean(sc.get("Serve/in", {}), k)
    train_fall = window_mean(sc.get("Serve/fall", {}), k)
    zero = [r for r in rows if r[0] == "zero"]
    ev_in = sum(r[3].get("served_in_standing", r[3]["served_in"]) for r in zero) / len(zero) if zero else None
    ev_fah = max(r[3]["fell_after_hit"] for r in rows)
    say(f"  it={k}: train(in={train_in}, fall={train_fall}) vs play(zero opp) standing-in={ev_in}, max fell_after_hit={ev_fah:.3f}")
    if train_in is not None and ev_in is not None and abs(train_in - ev_in) > 0.10:
        request_stop(st, f"plan 5: it={k} train in {train_in:.3f} vs play standing-in {ev_in:.3f} differ by > 10 points")
    if train_fall is not None and train_fall < 0.10 and ev_fah > 0.30:
        request_stop(st, f"plan 5: it={k} train fall {train_fall:.3f} < 0.10 but play fell_after_hit {ev_fah:.3f} > 0.30")
    m1 = len(rows) == 4 and all(
        r[3]["legal"] >= 0.90 and r[3]["served_in"] >= 0.85 and r[3]["height"] + r[3]["double"] <= 0.02
        and r[3]["fell_after_hit"] <= 0.10 for r in rows)
    if m1:
        alert(st, f"M1_{k}", f"M1 reached in play at it={k} (export {exp})")
        request_stop(st, f"early stop: M1 reached in play at it={k}")


def main():
    st = load_state()
    say(f"monitor start: run_name={args.run_name}, eval every {args.eval_every}, auto stop {'off' if args.no_stop else 'on'}")
    while True:
        done = os.path.exists(args.done_file)
        todo = []
        try:
            sc = scalars()
            ck, finished_it = checkpoints()
            check_training(st, sc, finished_it)
            todo = sorted(k for k in ck if k > 0 and (k % args.eval_every == 0 or k == finished_it) and k not in st["evaluated"])
            if done and ck and max(ck) not in st["evaluated"] and max(ck) not in todo:
                todo.append(max(ck))  # final checkpoint
            for k in todo:
                if not done and st.get("stop") and k != max(ck):
                    continue  # a stop is pending: only the final checkpoint matters now
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
