"""Monitor of one P3_FINETUNE_DT1MS run: play evaluation (1 ms, zero opponent, deterministic, both sides) every
``--eval_every`` iterations, stop rules of sections 3 (F1) and 4.3 (F2).

For each ``model_<it>.pt`` of the newest ``*_<run_name>`` directory with ``(it - start) % eval_every == 0`` (and the
last checkpoint once ``--done_file`` exists) it exports ``exports/<run_name>_<it>`` and runs ``eval_serve.py`` for
side 0 and side 1 under ``<play>/.eval.lock``. Results go to ``<out>/state.json`` and ``<out>/STATUS.txt``.

Stop rules (SIGTERM to the training process, which saves and exits; ``<out>/STOP_REQUESTED`` holds the reason):
  f1: best in >= base + 0.03 and two evaluations in a row without a new best -> converged;
      in < base - 0.05 twice in a row -> degraded.
  f2: in < ref - 0.04 twice in a row -> degraded.
"""

import argparse
import fcntl
import glob
import json
import os
import re
import signal
import subprocess
import time

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="/root/autodl-tmp/BadmintonLab-Game-V3L")
ap.add_argument("--play", default="/root/autodl-tmp/BadmintonLab-Game-Play")
ap.add_argument("--isaac_python", default="/root/autodl-tmp/env_isaaclab/bin/python")
ap.add_argument("--run_name", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--mode", choices=("f1", "f2", "none"), default="none")
ap.add_argument("--start_iter", type=int, required=True)
ap.add_argument("--eval_every", type=int, default=250)
ap.add_argument("--base", type=float, default=None, help="f1: E0 play in of p3_final.")
ap.add_argument("--ref", type=float, default=None, help="f2: play in of p3_dt_best.")
ap.add_argument("--select_floor", type=float, default=0.97,
                help="Section 4.5 (revised): play standing in >= this to be selectable (was max(0.85, ref - 0.02)).")
ap.add_argument("--num_envs", type=int, default=256)
ap.add_argument("--rallies", type=int, default=2,
                help="Per court, serve and receive rallies together: 256 x 2 = 512 per side, ~256 of them serves.")
ap.add_argument("--done_file", required=True)
ap.add_argument("--poll_s", type=float, default=60.0)
ap.add_argument("--no_stop", action="store_true")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
STATE = os.path.join(args.out, "state.json")
STATUS = os.path.join(args.out, "STATUS.txt")
STOP_FILE = os.path.join(args.out, "STOP_REQUESTED")
LOCK = os.path.join(args.play, ".eval.lock")
LOG_ROOT = os.path.join(args.repo, "logs", "rsl_rl", "badmintonlab_serve")


def say(msg):
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(STATUS, "a") as f:
        f.write(line + "\n")


def load_state():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {"evals": {}, "stopped": None}


def save_state(st):
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATE)


def run_dir():
    dirs = sorted(d for d in glob.glob(os.path.join(LOG_ROOT, f"*_{args.run_name}")) if os.path.isdir(d))
    return dirs[-1] if dirs else None


def checkpoints(d):
    out = {}
    for p in glob.glob(os.path.join(d, "model_*.pt")):
        m = re.fullmatch(r"model_(\d+)\.pt", os.path.basename(p))
        if m:
            out[int(m.group(1))] = p
    return out


def play_eval(exp, side, out):
    with open(LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            r = subprocess.run(
                [args.isaac_python, "scripts/eval_serve.py", "--play_root", args.play, "--ours", exp, "--opponent", "zero",
                 "--side", str(side), "--num_envs", str(args.num_envs), "--rallies", str(args.rallies), "--out", out],
                cwd=args.repo, capture_output=True, text=True)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    if r.returncode != 0:
        say(f"  eval failed side {side}: {r.stdout[-800:]} {r.stderr[-800:]}")
    return r.returncode


def evaluate(it, ckpt):
    exp = os.path.join(args.repo, "exports", f"{args.run_name}_{it}")
    t0 = time.time()
    r = subprocess.run([args.isaac_python, "scripts/export_torchscript.py", "--checkpoint", ckpt, "--out", exp],
                       cwd=args.repo, capture_output=True, text=True)
    if r.returncode != 0:
        say(f"export {it} failed: {r.stderr[-800:]}")
        return None
    sides = []
    for side in (0, 1):
        out = os.path.join(args.out, f"play_{it}_side{side}.json")
        if play_eval(exp, side, out) != 0:
            # One retry (a rare Isaac abort was seen once in P3).
            if play_eval(exp, side, out) != 0:
                return None
        with open(out) as f:
            sides.append(json.load(f)["serve"])
    n = sum(s["rallies"] for s in sides)
    pooled = {k: sum(s[k] for s in sides) / max(n, 1) for k in
              ("served_in_standing", "height", "double", "no_hit", "net", "out", "fell", "fell_after_hit")}
    pooled["fault"] = pooled["height"] + pooled["double"]
    res = dict(it=it, rallies=n, ckpt=ckpt, export=exp, wall_s=round(time.time() - t0, 1),
               side_in=[round(s["served_in_standing"] / max(s["rallies"], 1), 4) for s in sides],
               **{k: round(v, 4) for k, v in pooled.items()})
    # Medians of the in serves, averaged over the two sides (each side's p50 weighted by its count).
    qual = {}
    for k in ("land_x", "land_ay", "tof", "v_close", "exit_v", "exit_el", "d_recv_racket"):
        parts = [((s.get("quality") or {}).get(k) or {}) for s in sides]
        parts = [p for p in parts if p.get("n")]
        if parts:
            qual[k] = round(sum(p["p50"] * p["n"] for p in parts) / sum(p["n"] for p in parts), 3)
    res["q50"] = qual
    return res


def valid(e):
    return e["fault"] <= 0.01 and e["fell_after_hit"] <= 0.02


def request_stop(st, reason):
    st["stopped"] = reason
    save_state(st)
    say(f"STOP: {reason}")
    if args.no_stop:
        return
    with open(STOP_FILE, "w") as f:
        f.write(reason + "\n")
    r = subprocess.run(["pgrep", "-f", f"train_serve.py.*--run_name {args.run_name}( |$)"], capture_output=True, text=True)
    for pid in r.stdout.split():
        try:
            os.kill(int(pid), signal.SIGTERM)
            say(f"  SIGTERM -> {pid}")
        except ProcessLookupError:
            pass


def check_rules(st):
    evals = [st["evals"][k] for k in sorted(st["evals"], key=int)]
    if not evals or st["stopped"] or args.mode == "none":
        return
    ins = [e["served_in_standing"] for e in evals]
    if args.mode == "f1" and args.base is not None:
        if len(ins) >= 2 and ins[-1] < args.base - 0.05 and ins[-2] < args.base - 0.05:
            request_stop(st, f"f1_degraded: {ins[-2]:.3f}, {ins[-1]:.3f} < base {args.base:.3f} - 0.05")
            return
        best_at = max(range(len(ins)), key=lambda i: ins[i])
        if ins[best_at] >= args.base + 0.03 and len(ins) - 1 - best_at >= 2:
            request_stop(st, f"f1_converged: best {ins[best_at]:.3f} at {evals[best_at]['it']}, two evals without a new best")
    if args.mode == "f2" and args.ref is not None:
        if len(ins) >= 2 and ins[-1] < args.ref - 0.04 and ins[-2] < args.ref - 0.04:
            request_stop(st, f"f2_degraded: {ins[-2]:.3f}, {ins[-1]:.3f} < ref {args.ref:.3f} - 0.04")


def main():
    st = load_state()
    say(f"monitor {args.run_name}: mode {args.mode}, start {args.start_iter}, every {args.eval_every}, base {args.base}, ref {args.ref}")
    while True:
        done = os.path.exists(args.done_file)
        d = run_dir()
        if d is not None:
            cks = checkpoints(d)
            last = max(cks) if cks else None
            todo = [it for it in sorted(cks) if str(it) not in st["evals"] and it > args.start_iter
                    and ((it - args.start_iter) % args.eval_every == 0 or (done and it == last))]
            for it in todo:
                res = evaluate(it, cks[it])
                if res is None:
                    st["evals"][str(it)] = {"it": it, "failed": True, "served_in_standing": -1.0, "fault": 1.0,
                                            "fell_after_hit": 1.0}
                else:
                    st["evals"][str(it)] = res
                    say(f"eval {it}: in {res['served_in_standing']:.3f} (sides {res['side_in']}) fault {res['fault']:.3f} "
                        f"no_hit {res['no_hit']:.3f} net {res['net']:.3f} out {res['out']:.3f} "
                        f"fell_after_hit {res['fell_after_hit']:.3f} n {res['rallies']} q50 {res['q50']} ({res['wall_s']} s)")
                ok = [e for e in st["evals"].values() if not e.get("failed") and valid(e)]
                if ok:
                    b = max(ok, key=lambda e: e["served_in_standing"])
                    st["best"] = {"it": b["it"], "in": b["served_in_standing"], "ckpt": b["ckpt"], "export": b["export"]}
                    # Section 4.5 (revised): in >= select_floor, then the largest landing-to-racket distance
                    # (within 0.2 m: the higher in).
                    floor = args.select_floor
                    cand = [e for e in ok if e["served_in_standing"] >= floor and "d_recv_racket" in e.get("q50", {})]
                    if cand:
                        dmax = max(e["q50"]["d_recv_racket"] for e in cand)
                        near = [e for e in cand if e["q50"]["d_recv_racket"] >= dmax - 0.2]
                        c = max(near, key=lambda e: (e["served_in_standing"], e["q50"]["d_recv_racket"]))
                        st["select"] = {"it": c["it"], "in": c["served_in_standing"], "d": c["q50"]["d_recv_racket"],
                                        "ckpt": c["ckpt"], "export": c["export"]}
                save_state(st)
                check_rules(st)
            if done and not todo:
                say(f"done; best {st.get('best')}; select {st.get('select')}")
                return
        elif done:
            say("done without a run directory")
            return
        time.sleep(args.poll_s)


if __name__ == "__main__":
    main()
