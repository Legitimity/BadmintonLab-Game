"""Evaluate exported models in the real ``play`` environment and report serve metrics (design doc 6.6).

Runs inside a ``play`` checkout. It installs our player configs into play's
``player0_cfg.py`` / ``player1_cfg.py`` only for the duration of the run and
restores the originals afterwards; play's rule code is never modified.

    python scripts/eval_serve.py --play_root D:/rl-badminton-eval/BadmintonLab-Game-Play \
        --ours exports/serve_1000 --opponent model_test --side 0 --num_envs 16 --rallies 200

``--opponent`` is ``model_test`` (the course model with play's default config), ``zero``
(our config, zero action: tests our serve alone), a ``.pt`` file (TorchScript with play's default config,
e.g. the baseline ``/root/autodl-tmp/policy.pt``) or a directory with policy.pt + player_cfg.py.
The ``receive`` block has ``crossed`` / ``return_in`` / ``return_in_standing`` (our return reached the other
half and landed in or was played back) besides the serve counters.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "_eval_serve_inner.py"


def cfg_for_side(src: Path, side: int) -> str:
    """Our player_cfg is side-generic: only the SIDE constant changes."""
    text = src.read_text(encoding="utf-8")
    marker = "SIDE = 0\n"
    assert marker in text, "player_cfg.py must define SIDE = 0"
    return text.replace(marker, f"SIDE = {side}\n", 1)


def resolve_opponent(opponent: str, play_root: Path, ours: Path, opp_side: int) -> tuple[str | None, str]:
    """(player cfg text for the opponent's side or None = play's default config stays, model path or "zero")."""
    if opponent == "zero":
        return cfg_for_side(ours / "player_cfg.py", opp_side), "zero"
    if opponent == "model_test":
        return None, str(play_root / "model_test.pt")
    if opponent.endswith(".pt"):
        return None, str(Path(opponent).resolve())  # e.g. the course baseline (120-dim, play's default config)
    opp = Path(opponent).resolve()
    if not (opp / "player_cfg.py").exists():
        return None, str(opp / "policy.pt")  # exported baseline directory without our config
    return cfg_for_side(opp / "player_cfg.py", opp_side), str(opp / "policy.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--play_root", type=Path, required=True)
    ap.add_argument("--ours", type=Path, required=True, help="Directory with policy.pt and player_cfg.py")
    ap.add_argument("--opponent", default="model_test")
    ap.add_argument("--side", type=int, choices=(0, 1), default=0)
    ap.add_argument("--num_envs", type=int, default=16)
    ap.add_argument("--rallies", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--watchdog_s", type=float, default=60.0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--probe_out", type=Path, default=None, help="Fall traces of our receiver (P5 V2 3.1).")
    a = ap.parse_args()
    a.ours = a.ours.resolve()  # the inner process runs with cwd = play root
    if a.out is not None:
        a.out = Path(a.out).resolve()

    task_dir = a.play_root / "source/badmintonlab_game/badmintonlab_game/tasks/manager_based/badmintonlab_game"
    targets = {0: task_dir / "player0_cfg.py", 1: task_dir / "player1_cfg.py"}
    backup = Path(tempfile.mkdtemp(prefix="play_cfg_backup_"))
    for s, p in targets.items():
        shutil.copy2(p, backup / p.name)
    opp_side = 1 - a.side
    try:
        targets[a.side].write_text(cfg_for_side(a.ours / "player_cfg.py", a.side), encoding="utf-8")
        opp_cfg, opp_model = resolve_opponent(a.opponent, a.play_root, a.ours, opp_side)
        if opp_cfg is not None:
            targets[opp_side].write_text(opp_cfg, encoding="utf-8")
        models = {a.side: str(a.ours / "policy.pt"), opp_side: opp_model}
        env = dict(os.environ, PYTHONPATH=str(a.play_root / "source/badmintonlab_game") + os.pathsep + os.environ.get("PYTHONPATH", ""))
        cmd = [
            a.python,
            str(RUNNER),
            "--play_root", str(a.play_root),
            "--model0", models[0],
            "--model1", models[1],
            "--ours", str(a.side),
            "--num_envs", str(a.num_envs),
            "--rallies", str(a.rallies),
            "--seed", str(a.seed),
            "--watchdog_s", str(a.watchdog_s),
            "--headless",
        ]
        if a.probe_out is not None:
            cmd += ["--probe_out", str(a.probe_out.resolve())]
        out = subprocess.run(cmd, env=env, cwd=a.play_root, capture_output=True, text=True)
        sys.stderr.write(out.stderr[-4000:])
        line = [ln for ln in out.stdout.splitlines() if ln.startswith("[SERVE_EVAL]")]
        if out.returncode != 0 or not line:
            print(out.stdout[-4000:])
            raise SystemExit(f"evaluation failed (exit {out.returncode})")
        result = json.loads(line[-1][len("[SERVE_EVAL]"):])
        result.update(ours=str(a.ours), opponent=a.opponent, side=a.side)
        print(json.dumps(result, indent=1))
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    finally:
        for s, p in targets.items():
            shutil.copy2(backup / p.name, p)
        shutil.rmtree(backup, ignore_errors=True)


if __name__ == "__main__":
    main()
