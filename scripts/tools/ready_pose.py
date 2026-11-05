"""P2 initial pose from the swing search (CPU only).

Reads ``docs/swing_search.json`` and writes the reset pose into ``player_cfg.py``:
the trunk pose (READY_ROOT) and the arm/waist hold pose (READY_JOINTS). The hold
pose is the search's ``q0``. The swing target ``qc`` and trigger step are recorded
in ``docs/ready_pose.json`` for the behaviour-cloning fallback (design doc 7.4).

    python scripts/tools/ready_pose.py --candidate 0 [--write]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAYER_CFG = ROOT / "source/badmintonlab_game/badmintonlab_game/tasks/manager_based/badmintonlab_game/serve/player_cfg.py"
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402
import swing_search as ss  # noqa: E402

DEFAULT_ROOT_Z = 0.70


def check_pose(c: dict) -> dict:
    """Static checks on the ready pose (no physics): racket height, ball path clearance."""
    q0 = np.array(c["q0"])[None]
    _, _, _, sweet, r, _ = ss.kinematics(q0)
    rot = ss._rpy(0, 0, c["yaw"])
    sweet_w = np.array([*c["pl"]["xy"], ss.HIP_Z]) + rot @ sweet[0]
    # Racket box corners: none may sit in the drop column above 1.15 m (an early touch is a high serve).
    corners = []
    for sx in (-0.005, 0.005):
        for sy in (-0.07, 0.07):
            for sz in (-0.10, 0.10):
                corners.append(sweet_w + rot @ r[0] @ np.array([sx, sy, sz]))
    corners = np.array(corners)
    horiz = np.linalg.norm(corners[:, :2] - ss.BALL_XY, axis=1)
    in_column_high = bool(((horiz < 0.11) & (corners[:, 2] > 1.10)).any())
    return dict(
        sweet_world=sweet_w.round(3).tolist(),
        sweet_to_ball_xy=float(np.linalg.norm(sweet_w[:2] - ss.BALL_XY)),
        racket_top_z=float(corners[:, 2].max()),
        racket_in_drop_column_above_1_10=in_column_high,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", type=int, default=0)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    data = json.loads((ROOT / "docs/swing_search.json").read_text(encoding="utf-8"))
    c = data["results"][a.candidate]
    root = dict(x=round(c["trunk"]["x"], 3), y=round(c["trunk"]["y"], 3), yaw=round(c["trunk"]["yaw"], 3))
    joints = {name: round(float(v), 3) for name, v in zip(data["joints"], c["q0"])}
    report = dict(candidate=a.candidate, root=root, joints=joints, swing_target=dict(zip(data["joints"], c["qc"])),
                  trigger_ctrl_step=c["pl"]["trig"] / 4, score=c["score"], speculative=c.get("speculative"),
                  contact=c["pl"], checks=check_pose(c))
    print(json.dumps(report, indent=1))
    (ROOT / "docs/ready_pose.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    if report["checks"]["racket_in_drop_column_above_1_10"]:
        raise SystemExit("ready pose puts the racket in the drop path above 1.10 m: an early touch would be a fault")
    if a.write:
        text = PLAYER_CFG.read_text(encoding="utf-8")
        text = re.sub(r"READY_ROOT = dict\([^)]*\)", f"READY_ROOT = dict(x={root['x']}, y={root['y']}, yaw={root['yaw']})", text)
        body = "".join(f'    "{k}": {v},\n' for k, v in joints.items())
        text = re.sub(r"READY_JOINTS = \{[^}]*\}", "READY_JOINTS = {\n" + body + "}", text)
        PLAYER_CFG.write_text(text, encoding="utf-8")
        print(f"wrote {PLAYER_CFG}")


if __name__ == "__main__":
    main()
