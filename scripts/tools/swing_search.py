"""CPU-only feasibility search for a scripted serve (no Isaac, no GPU).

Model
-----
* Hips are fixed; the waist and the right arm (5 DoF) are simulated with URDF
  inertias, armature, gravity and the course delayed PD actuator, including the
  URDF torque limit and the speed-dependent torque clip.
* The shuttle uses the course drag model. The racket contact is a discrete
  200 Hz sphere/box overlap with restitution 0.75 and Coulomb friction.

Limits: reaction forces on the feet and balance are ignored, so the speed is an
upper bound for the whole robot. PhysX contact offsets are not modelled, so the
tunnelling estimate is conservative. Isaac measurements supersede this script.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / "robots" / "T1" / "T1_23dof.urdf"
DT = 0.005
DECIMATION = 4
G = 9.81
DRAG = 0.00110 / 0.00519
BALL_R = 0.025
BALL_XY = np.array([-2.55, 0.0])
BALL_Z0 = 2.2
NET_X = 0.0
HALF_LEN = 6.7
HALF_W = 2.59
TRUNK_Z = 0.70

W = 2 * math.pi * 4.0
# armature, URDF effort (the clip that really applies), v_max, v_knee, cfg effort (used for action scale)
_ARM = (0.0282528, 18.0, 17.59, 7.85, 38.3)
_WST = (0.0478125, 25.0, 14.66, 1.88, 68.0)
JOINTS = ["Waist", "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw"]
PARAM = [_WST, _ARM, _ARM, _ARM, _ARM]
ARMATURE = np.array([p[0] for p in PARAM])
KP = ARMATURE * W**2
KD = 2 * 1.5 * ARMATURE * W
TAU = np.array([p[1] for p in PARAM])
VMAX = np.array([p[2] for p in PARAM])
VKNEE = np.array([p[3] for p in PARAM])
SCALE = np.array([0.5 * p[4] for p in PARAM]) / KP
LIM = np.array([[-1.57, 1.57], [-3.29, 1.1868], [0.0, 1.577], [-2.234, 2.234], [-1.6844, 2.1402]])
DEFAULT = np.array([0.0, 0.2, 1.3, 0.0, 0.5])


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _load():
    root = ET.parse(URDF).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    links = {k.get("name"): k for k in root.findall("link")}

    def origin(e):
        o = e.find("origin")
        xyz = np.array(list(map(float, o.get("xyz", "0 0 0").split())))
        rpy = list(map(float, o.get("rpy", "0 0 0").split()))
        return xyz, _rpy(*rpy)

    def inertial(name):
        i = links[name].find("inertial")
        c, rc = origin(i)
        a = i.find("inertia").attrib
        ic = np.array(
            [
                [float(a["ixx"]), float(a["ixy"]), float(a["ixz"])],
                [float(a["ixy"]), float(a["iyy"]), float(a["iyz"])],
                [float(a["ixz"]), float(a["iyz"]), float(a["izz"])],
            ]
        )
        return float(i.find("mass").get("value")), c, rc @ ic @ rc.T

    def shift(inertia, mass, d):
        return inertia + mass * ((d @ d) * np.eye(3) - np.outer(d, d))

    chain = []
    for name in JOINTS[1:]:
        j = joints[name]
        xyz, r0 = origin(j)
        axis = np.array(list(map(float, j.find("axis").get("xyz").split())))
        chain.append(dict(xyz=xyz, r0=r0, axis=axis, link=inertial(j.find("child").get("link"))))
    # The racket is rigid on the hand: lump it into the last link.
    xr, rr = origin(joints["right_hand_to_racket"])
    mr, cr, ir = inertial("racket_link")
    mh, ch, ih = chain[-1]["link"]
    c_r, i_r = xr + rr @ cr, rr @ ir @ rr.T
    m = mh + mr
    com = (mh * ch + mr * c_r) / m
    chain[-1]["link"] = (m, com, shift(ih, mh, ch - com) + shift(i_r, mr, c_r - com))
    sweet = xr + rr @ np.array([0.0, 0.0, 0.52])
    # Trunk carries the head and left arm, which are lumped at the trunk.
    mt, ct, it = inertial("Trunk")
    extra = sum(inertial(n)[0] for n in ("H1", "H2", "AL1", "AL2", "AL3", "left_hand_link"))
    xw, _ = origin(joints["Waist"])
    return dict(chain=chain, rr=rr, sweet=sweet, trunk=(mt + extra, ct, it + extra * 0.03 * np.eye(3)), waist=xw)


MODEL = _load()
HIP_Z = TRUNK_Z + float(MODEL["waist"][2])


def _rot(axis, t):
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    s, c = np.sin(t)[:, None, None], np.cos(t)[:, None, None]
    return np.eye(3)[None] + s * k[None] + (1 - c) * (k @ k)[None]


def kinematics(q):
    """FK with the waist joint at the origin and hips facing +x. Trunk is the URDF root."""
    n = q.shape[0]
    r = _rot(np.array([0.0, 0.0, 1.0]), -q[:, 0])
    p = np.einsum("nij,j->ni", r, -MODEL["waist"])
    axes = [np.tile(np.array([0.0, 0.0, -1.0]), (n, 1))]
    origins = [np.zeros((n, 3))]
    mt, ct, it = MODEL["trunk"]
    links = [(mt, p + np.einsum("nij,j->ni", r, ct), r, it, 1)]
    for i, jc in enumerate(MODEL["chain"]):
        p = p + np.einsum("nij,j->ni", r, jc["xyz"])
        r = r @ jc["r0"]
        axes.append(np.einsum("nij,j->ni", r, jc["axis"]))
        origins.append(p.copy())
        r = r @ _rot(jc["axis"], q[:, i + 1])
        m, c, inertia = jc["link"]
        links.append((m, p + np.einsum("nij,j->ni", r, c), r, inertia, i + 2))
    axes = np.stack(axes, 1)
    origins = np.stack(origins, 1)
    sweet = p + np.einsum("nij,j->ni", r, MODEL["sweet"])
    r_racket = r @ MODEL["rr"]
    jac = np.cross(axes, sweet[:, None] - origins)
    return links, axes, origins, sweet, r_racket, jac


def mass_gravity(q):
    links, axes, origins, _, _, _ = kinematics(q)
    n = q.shape[0]
    mass = np.zeros((n, 5, 5))
    grav = np.zeros((n, 5))
    for m, c, r, inertia, nj in links:
        jv = np.zeros((n, 5, 3))
        jw = np.zeros((n, 5, 3))
        jv[:, :nj] = np.cross(axes[:, :nj], c[:, None] - origins[:, :nj])
        jw[:, :nj] = axes[:, :nj]
        iw = r @ inertia @ r.transpose(0, 2, 1)
        mass += m * jv @ jv.transpose(0, 2, 1) + jw @ iw @ jw.transpose(0, 2, 1)
        grav += m * G * jv[:, :, 2]
    mass += np.diag(ARMATURE)[None]
    return mass, grav


def _coriolis(q, qd, mass):
    e = 1e-5
    mp, _ = mass_gravity(q + e * qd)
    h = np.einsum("nij,nj->ni", (mp - mass) / e, qd)
    for i in range(5):
        dq = np.zeros(5)
        dq[i] = e
        mi, _ = mass_gravity(q + dq)
        h[:, i] -= 0.5 * np.einsum("ni,nij,nj->n", qd, (mi - mass) / e, qd)
    return h


def hold_target(q0):
    """PD target that holds q0 still: the course PD has no gravity compensation."""
    _, g = mass_gravity(q0)
    return q0 + g / KP


def simulate(q0, q_cmd, delays, trigger, steps):
    """Roll out explicit PD. delays: (n, 5) physics steps; trigger: (n,) physics step of the switch."""
    q, qd = q0.copy(), np.zeros_like(q0)
    hold = hold_target(q0)
    out_q, out_s, out_r, out_v = [], [], [], []
    for k in range(steps):
        tgt = np.where(k - delays >= trigger[:, None], q_cmd, hold)
        limit = np.clip(TAU * (VMAX - np.abs(qd)) / np.maximum(VMAX - VKNEE, 1e-6), 0.0, TAU)
        tau = np.clip(KP * (tgt - q) - KD * qd, -limit, limit)
        mass, g = mass_gravity(q)
        h = _coriolis(q, qd, mass)
        qdd = np.linalg.solve(mass, (tau - g - h)[..., None])[..., 0]
        qd = qd + DT * qdd
        q = q + DT * qd
        over = (q < LIM[:, 0]) | (q > LIM[:, 1])
        q = np.clip(q, LIM[:, 0], LIM[:, 1])
        qd = np.where(over, 0.0, qd)
        _, _, _, s, r, jac = kinematics(q)
        out_q.append(q.copy()), out_s.append(s), out_r.append(r), out_v.append(np.einsum("nji,nj->ni", jac, qd))
    return map(np.stack, (out_q, out_s, out_r, out_v))


def _drop_table():
    z, v, t = BALL_Z0, 0.0, 0.0
    h = 0.0005
    ts, zs, vs = [0.0], [z], [0.0]
    while z > 0.03:
        v += (-G + DRAG * v * v) * h
        z += v * h
        t += h
        ts.append(t), zs.append(z), vs.append(v)
    return np.array(ts), np.array(zs), np.array(vs)


T_TAB, Z_TAB, V_TAB = _drop_table()


def flight(pos, vel, dt=0.002):
    """Serve outcome under the play rules. The server is on the negative-x half."""
    p, v = np.array(pos, float), np.array(vel, float)
    net_z = None
    for _ in range(5000):
        if p[2] <= 0.03:
            break
        v = v + dt * (np.array([0, 0, -G]) - DRAG * np.linalg.norm(v) * v)
        pn = p + dt * v
        if p[0] < NET_X <= pn[0] and net_z is None:
            a = (NET_X - p[0]) / (pn[0] - p[0])
            net_z, net_y = p[2] + a * (pn[2] - p[2]), p[1] + a * (pn[1] - p[1])
            if net_z < 1.524 + (1.55 - 1.524) * min(abs(net_y) / 3.05, 1.0):
                return dict(ok=False, why="net", x=float(pn[0]), y=float(pn[1]))
        p = pn
        if abs(p[0]) > HALF_LEN or abs(p[1]) > HALF_W:
            return dict(ok=False, why="out", x=float(p[0]), y=float(p[1]))
    if p[0] < NET_X:
        return dict(ok=False, why="short", x=float(p[0]), y=float(p[1]))
    return dict(ok=True, why="in", x=float(p[0]), y=float(p[1]), net_z=float(net_z))


def _impact(vb, vr, n, e=0.75, mu=0.275):
    rel = vb - vr
    vn = rel @ n
    if vn >= 0:
        return vb
    jn = -(1 + e) * vn
    vt = rel - vn * n
    st = np.linalg.norm(vt)
    return vb + jn * n - (0.0 if st < 1e-9 else min(st, mu * jn)) * vt / max(st, 1e-9)


def _overlap(ball, p, r, offset=0.0):
    """Returns (touching, position in the racket frame) for the racket box."""
    rel = r.T @ (ball - p)
    half = np.array([0.005, 0.07, 0.10])
    dist = np.linalg.norm(np.maximum(np.abs(rel) - half, 0.0))
    return dist < BALL_R + offset, rel


def serve_episode(q0, qc, yaw, xy, trig, d_arm, d_waist, steps, offset=0.0):
    """One scripted serve with the real ball. Returns the result under the play rules."""
    delays = np.array([[d_waist, d_arm, d_arm, d_arm, d_arm]])
    _, s, r, v = simulate(q0[None], qc[None], delays, np.array([trig]), steps)
    rot = _rpy(0, 0, yaw)
    pos = np.array([xy[0], xy[1], HIP_Z]) + (rot @ s[:, 0, :, None])[..., 0]
    r_w = rot @ r[:, 0]
    v_w = (rot @ v[:, 0, :, None])[..., 0]
    ball = np.array([BALL_XY[0], BALL_XY[1], BALL_Z0])
    vb = np.zeros(3)
    res = dict(result="miss", d_arm=d_arm, d_waist=d_waist)
    hit_step = None
    was_touching = False
    last_side = None
    for k in range(steps):
        vb = vb + DT * (np.array([0, 0, -G]) - DRAG * np.linalg.norm(vb) * vb)
        ball = ball + DT * vb
        if ball[2] <= 0.03:
            break
        touching, rel = _overlap(ball, pos[k], r_w[k], offset)
        side = rel[0] if abs(rel[0]) > 1e-9 else 1.0
        if not was_touching and last_side is not None:
            # Resolve on the face the ball was on before the step (the two faces are 1 cm apart).
            side = last_side
        if not touching:
            last_side = rel[0]
        if touching and not was_touching:
            if hit_step is not None:
                return dict(res, result="double_hit", dt_ms=(k - hit_step) * 5)
            if ball[2] > 1.15:
                return dict(res, result="high_serve", z=float(ball[2]))
            n = r_w[k][:, 0] * math.copysign(1.0, side)
            vn = float(v_w[k] @ n)
            back_side = (n[0] <= 0) or (vb @ n) > 0 or vn < 0
            vb = _impact(vb, v_w[k], n)
            hit_step = k
            res.update(
                result="hit",
                z=float(ball[2]),
                vn=vn,
                speed=float(np.linalg.norm(v_w[k])),
                el=float(math.degrees(math.asin(np.clip(n[2], -1, 1)))),
                off=[float(rel[1]), float(rel[2])],
                back_side=bool(back_side),
                out=vb.tolist(),
            )
            res.update({"flight_" + kk: vv for kk, vv in flight(ball, vb).items()})
        was_touching = touching
    return res


def score_episode(res):
    if res["result"] != "hit":
        return 0.0
    return 1.0 if res.get("flight_ok") and not res.get("back_side") else 0.0


# Target and ceiling of the racket normal speed at contact.
VN_RANGE = [5.5, 8.0]


def place_contact(q0, qc, yaw, ref_arm=5, ref_waist=5, steps=70):
    """Choose the hip placement and trigger step so the nominal delay hits the sweet spot."""
    delays = np.array([[ref_waist, ref_arm, ref_arm, ref_arm, ref_arm]])
    _, s, r, v = simulate(q0[None], qc[None], delays, np.array([0]), steps)
    rot = _rpy(0, 0, yaw)
    s_w = (rot @ s[:, 0, :, None])[..., 0]
    r_w = rot @ r[:, 0]
    v_w = (rot @ v[:, 0, :, None])[..., 0]
    n = r_w[:, :, 0] * np.sign(r_w[:, 0, 0])[:, None]
    vn = np.einsum("ti,ti->t", v_w, n)
    el = np.degrees(np.arcsin(np.clip(n[:, 2], -1, 1)))
    hd = np.degrees(np.arctan2(n[:, 1], n[:, 0]))
    z = s_w[:, 2] + HIP_Z
    ok = (z > 0.60) & (z < 1.08) & (el > 30) & (el < 72) & (np.abs(hd) < 25) & (vn > 2.5) & (vn < VN_RANGE[1])
    if not ok.any():
        return None
    k = int(np.argmin(np.where(ok, np.abs(vn - VN_RANGE[0]), np.inf)))
    t_ball = float(np.interp(-z[k], -Z_TAB, T_TAB))
    trig_phys = int(round((t_ball / DT - (k + 1)) / DECIMATION)) * DECIMATION
    if trig_phys < 8:
        return None
    return dict(
        k=k,
        trig=trig_phys,
        xy=(BALL_XY - s_w[k, :2]).tolist(),
        z=float(z[k]),
        vn=float(vn[k]),
        el=float(el[k]),
        hd=float(hd[k]),
        z0=float(s_w[0, 2] + HIP_Z),
    )


GRID = tuple((a, w) for a in range(2, 9) for w in (2, 8))


def evaluate(c, steps=175, grid=GRID, offset=0.0):
    q0, qc = np.array(c["q0"]), np.array(c["qc"])
    pl = place_contact(q0, qc, c["yaw"])
    if pl is None:
        return None
    rows = [serve_episode(q0, qc, c["yaw"], pl["xy"], pl["trig"], a, w, steps, offset) for a, w in grid]
    return dict(pl=pl, rows=rows, score=float(np.mean([score_episode(r) for r in rows])))


def sample(rng, n, base=None, sigma=0.15):
    if base is None:
        q0 = rng.uniform(LIM[:, 0], LIM[:, 1], (n, 5))
        q0[:, 0] = rng.uniform(-0.9, 0.9, n)
        qc = q0 + rng.uniform(-3.0, 3.0, (n, 5))
        qc[:, 0] = np.clip(qc[:, 0], -1.2, 1.2)
        yaw = rng.uniform(-0.6, 0.6, n)
    else:
        idx = rng.integers(0, len(base), n)
        q0 = np.array([base[i]["q0"] for i in idx]) + rng.normal(0, sigma, (n, 5))
        qc = np.array([base[i]["qc"] for i in idx]) + rng.normal(0, sigma * 2, (n, 5))
        yaw = np.array([base[i]["yaw"] for i in idx]) + rng.normal(0, sigma, n)
    q0 = np.clip(q0, 0.9 * LIM[:, 0], 0.9 * LIM[:, 1])
    return q0, qc, np.clip(yaw, -0.7, 0.7)


def run(args):
    rng = np.random.default_rng(args.seed)
    pop = []
    for it in range(args.iters):
        base = pop[:30] if pop and rng.random() < 0.8 else None
        q0, qc, yaw = sample(rng, args.batch, base)
        for i in range(args.batch):
            c = dict(q0=q0[i].tolist(), qc=qc[i].tolist(), yaw=float(yaw[i]))
            pl = place_contact(q0[i], qc[i], float(yaw[i]))
            # The trunk must roughly face the opponent: the course orientation cost starts at 90 deg.
            if pl is None or pl["z0"] > 0.95 or abs(float(yaw[i]) - q0[i, 0]) > 1.2:
                continue
            ev = evaluate(c)
            if ev is None:
                continue
            c.update(ev)
            c["rank"] = ev["score"] - 0.01 * abs(pl["vn"] - VN_RANGE[0])
            pop.append(c)
        pop = sorted(pop, key=lambda x: -x["rank"])[:200]
        if pop:
            t = pop[0]
            print(f"iter {it} pop {len(pop)} best score {t['score']:.2f} vn {t['pl']['vn']:.2f} z {t['pl']['z']:.2f} el {t['pl']['el']:.0f}", flush=True)
        else:
            print(f"iter {it} no candidates", flush=True)
    return pop


def trunk_pose(c):
    """Trunk (URDF root) pose at reset for player 0, from the hip placement."""
    q_w = c["q0"][0]
    hip_yaw = c["yaw"]
    trunk_yaw = hip_yaw - q_w
    w = MODEL["waist"]
    x = c["pl"]["xy"][0] - math.cos(trunk_yaw) * w[0] + math.sin(trunk_yaw) * w[1]
    y = c["pl"]["xy"][1] - math.sin(trunk_yaw) * w[0] - math.cos(trunk_yaw) * w[1]
    return dict(x=x, y=y, yaw=trunk_yaw)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vn_target", type=float, default=5.5)
    ap.add_argument("--vn_max", type=float, default=8.0)
    ap.add_argument("--out", default="docs/swing_search.json")
    a = ap.parse_args()
    VN_RANGE[:] = [a.vn_target, a.vn_max]
    population = run(a)
    final = []
    for cand in population[:60]:
        if any(np.linalg.norm(np.array(cand["q0"]) - np.array(s["q0"])) < 0.25 for s in final):
            continue
        cand["trunk"] = trunk_pose(cand)
        cand["speculative"] = evaluate(cand, offset=0.02)["score"]
        final.append(cand)
        if len(final) >= 8:
            break
    Path(a.out).parent.mkdir(exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(dict(joints=JOINTS, scale=SCALE.tolist(), default=DEFAULT.tolist(), results=final), f, indent=1)
    for r in final:
        print(f"score {r['score']:.2f} spec {r['speculative']:.2f} vn {r['pl']['vn']:.2f} z {r['pl']['z']:.2f} el {r['pl']['el']:.0f} ctrl_trig {r['pl']['trig'] / 4:.0f} trunk {np.round(list(r['trunk'].values()), 2)}")
        print("  q0", np.round(r["q0"], 2), "qc", np.round(r["qc"], 2))
        print("  ", [(x["d_arm"], x["d_waist"], x.get("flight_why") or x["result"], round(x.get("flight_x", 0), 1), round(x.get("vn", 0), 1)) for x in r["rows"]])
