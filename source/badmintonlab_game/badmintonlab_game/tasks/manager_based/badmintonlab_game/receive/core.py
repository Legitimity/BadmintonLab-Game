"""Pure-torch pieces of the receive task (no Isaac imports), so they can be CPU-tested.

Canonical frame: the receiver is on the negative-x half, the net is the plane x = 0,
the feeder (opponent) launches from x > 0 towards -x. For side 0 this is the env-local frame.

Design doc: D:/Legitimity/badminton/P5_RECEIVE_EXPERT_V3.md (V3.1, close to arXiv 2511.11218 v4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

DRAG_OVER_MASS = 0.00110 / 0.00519
GRAVITY = 9.81
HIT_Z = 1.0  # intercept = first descending crossing of this height
BAND_X = (-5.0, -0.5)  # receiver half, 0.5-5.0 m from the net
BAND_Y = 2.4
NET_CLEAR = 1.60  # legal feed: crosses the net at least 5 cm above its edge height
# Ready pose of the side-0 robot (serve/player_cfg.READY_ROOT) and the sweet-spot offset in the root
# (Trunk) frame at that pose, from URDF forward kinematics (P5 V3.1 4.2). Verified in Isaac at smoke time.
READY_ROOT_XY = (-3.251, 0.605)
READY_YAW = -0.043
O_READY = (0.825, -0.521)


def predict_intercept(
    pos: torch.Tensor,
    vel: torch.Tensor,
    dt: float = 0.03,
    steps: int = 70,
    hit_z: float = HIT_Z,
    band_x: tuple[float, float] = BAND_X,
    band_y: float = BAND_Y,
    drag: float = DRAG_OVER_MASS,
    g: float = GRAVITY,
):
    """First descending crossing of ``hit_z`` inside the band. Fixed steps, no host sync.

    Returns ``(point (N,3), time (N,), valid (N,) bool)``. Same midpoint drag law as
    ``rules.predict_flight``. A copy of this function lives in ``receive/player_cfg.py``
    (the submission file must be self-contained); a test checks they agree.
    """
    p = pos.clone()
    v = vel.clone()
    n = p.shape[0]
    gv = torch.tensor((0.0, 0.0, -g), device=p.device, dtype=p.dtype)
    found = torch.zeros(n, dtype=torch.bool, device=p.device)
    alive = p[:, 2] > 0.03
    pt = torch.zeros_like(p)
    t_hit = torch.zeros(n, device=p.device, dtype=p.dtype)
    for k in range(steps):
        speed = torch.linalg.norm(v, dim=-1, keepdim=True)
        vm = v + 0.5 * dt * (gv - drag * speed * v)
        speed_m = torch.linalg.norm(vm, dim=-1, keepdim=True)
        vn = v + dt * (gv - drag * speed_m * vm)
        pn = p + dt * vm
        cross = alive & ~found & (p[:, 2] >= hit_z) & (pn[:, 2] < hit_z)
        a = (p[:, 2] - hit_z) / (p[:, 2] - pn[:, 2]).clamp(min=1e-9)
        q = p + a[:, None] * (pn - p)
        pt = torch.where(cross[:, None], q, pt)
        t_hit = torch.where(cross, (k + a) * dt, t_hit)
        found = found | cross
        p, v = pn, vn
        alive = alive & (p[:, 2] > 0.03)
    valid = found & (pt[:, 0] >= band_x[0]) & (pt[:, 0] <= band_x[1]) & (pt[:, 1].abs() <= band_y)
    return pt, t_hit, valid


# ------------------------------------------------------------------ S1 V4 (11-09): free hitting height
# The hitting point is chosen among the descending crossings of several heights. The choice depends only
# on the trajectory (not on the robot), with a score that is invariant to a common time shift, so the
# target does not switch while the shuttle flies: score = d_k - V_REF t_k + Z_PREF |z_k - 1.0|, where
# d_k is the distance from the ready root to the station of crossing k. A copy lives in
# receive/player_cfg.py (``_predict_intercept``); a test checks they agree.
HIT_ZS = (0.8, 0.9, 1.0, 1.1, 1.2)  # 0.7/1.3 tried offline: half the balls went to 0.7 (too low for the swing)
V_REF = 1.5  # m of walking per s of extra flight time
Z_PREF = 0.3  # m-equivalent per m of height away from 1.0 (prefer the trained height on ties)
REACT = 0.2
REACH_SLACK = 0.3  # r_track is flat within 0.3 m


def _ready_offset_xy(yaw: float = READY_YAW, o_ready=O_READY) -> tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return c * o_ready[0] - s * o_ready[1], s * o_ready[0] + c * o_ready[1]


def predict_intercept_multi(
    pos: torch.Tensor,
    vel: torch.Tensor,
    dt: float = 0.03,
    steps: int = 70,
    hit_zs: tuple = HIT_ZS,
    band_x: tuple[float, float] = BAND_X,
    band_y: float = BAND_Y,
    drag: float = DRAG_OVER_MASS,
    g: float = GRAVITY,
):
    """Hitting point with a free height (V4). Returns ``(point (N,3), time (N,), valid (N,))``."""
    p = pos.clone()
    v = vel.clone()
    n = p.shape[0]
    zs = torch.tensor(hit_zs, device=p.device, dtype=p.dtype)
    k_n = zs.numel()
    gv = torch.tensor((0.0, 0.0, -g), device=p.device, dtype=p.dtype)
    found = torch.zeros(n, k_n, dtype=torch.bool, device=p.device)
    alive = p[:, 2] > 0.03
    pt = torch.zeros(n, k_n, 3, device=p.device, dtype=p.dtype)
    t_hit = torch.zeros(n, k_n, device=p.device, dtype=p.dtype)
    for k in range(steps):
        speed = torch.linalg.norm(v, dim=-1, keepdim=True)
        vm = v + 0.5 * dt * (gv - drag * speed * v)
        speed_m = torch.linalg.norm(vm, dim=-1, keepdim=True)
        vn = v + dt * (gv - drag * speed_m * vm)
        pn = p + dt * vm
        z0, z1 = p[:, 2:3], pn[:, 2:3]
        cross = alive[:, None] & ~found & (z0 >= zs) & (z1 < zs)
        a = (z0 - zs) / (z0 - z1).clamp(min=1e-9)
        q = p[:, None, :] + a[..., None] * (pn - p)[:, None, :]
        pt = torch.where(cross[..., None], q, pt)
        t_hit = torch.where(cross, (k + a) * dt, t_hit)
        found = found | cross
        p, v = pn, vn
        alive = alive & (p[:, 2] > 0.03)
    ok = found & (pt[..., 0] >= band_x[0]) & (pt[..., 0] <= band_x[1]) & (pt[..., 1].abs() <= band_y)
    ox, oy = _ready_offset_xy()
    sx = pt[..., 0] - ox - READY_ROOT_XY[0]
    sy = pt[..., 1] - oy - READY_ROOT_XY[1]
    d = torch.sqrt(sx * sx + sy * sy)
    score = d - V_REF * t_hit + Z_PREF * (zs - 1.0).abs()
    score = torch.where(ok, score, torch.full_like(score, 1e9))
    best = score.argmin(dim=1)
    idx = torch.arange(n, device=p.device)
    return pt[idx, best], t_hit[idx, best], ok.any(dim=1)


def required_speed(d_need: torch.Tensor, t_arr: torch.Tensor) -> torch.Tensor:
    """Mean walking speed needed after the reaction time, with the flat r_track zone as slack."""
    return (d_need - REACH_SLACK).clamp(min=0.0) / (t_arr - REACT).clamp(min=0.05)


def simulate_feed(pos: torch.Tensor, vel: torch.Tensor, dt: float = 0.01, steps: int = 400):
    """Net-crossing height (going -x), landing point, for feeds launched from x > 0."""
    p = pos.clone()
    v = vel.clone()
    n = p.shape[0]
    gv = torch.tensor((0.0, 0.0, -GRAVITY), device=p.device, dtype=p.dtype)
    net_z = torch.full((n,), -torch.inf, device=p.device, dtype=p.dtype)
    alive = torch.ones(n, dtype=torch.bool, device=p.device)
    for _ in range(steps):
        speed = torch.linalg.norm(v, dim=-1, keepdim=True)
        vm = v + 0.5 * dt * (gv - DRAG_OVER_MASS * speed * v)
        speed_m = torch.linalg.norm(vm, dim=-1, keepdim=True)
        vn = v + dt * (gv - DRAG_OVER_MASS * speed_m * vm)
        pn = torch.where(alive[:, None], p + dt * vm, p)
        crossing = alive & (p[:, 0] > 0.0) & (pn[:, 0] <= 0.0)
        a = p[:, 0] / torch.where(crossing, p[:, 0] - pn[:, 0], torch.ones_like(p[:, 0]))
        net_z = torch.where(crossing, p[:, 2] + a * (pn[:, 2] - p[:, 2]), net_z)
        v = torch.where(alive[:, None], vn, v)
        p = pn
        alive = alive & (p[:, 2] > 0.03)
    return net_z, p[:, :2]


def station_from_intercept(int_xy: torch.Tensor, yaw: float = READY_YAW, o_ready=O_READY) -> torch.Tensor:
    """Root position that puts the ready-pose sweet spot on the intercept, facing ``yaw``. Fixed per ball."""
    c, s = math.cos(yaw), math.sin(yaw)
    ox = c * o_ready[0] - s * o_ready[1]
    oy = s * o_ready[0] + c * o_ready[1]
    return int_xy - torch.tensor((ox, oy), device=int_xy.device, dtype=int_xy.dtype)


def r_track(d: torch.Tensor) -> torch.Tensor:
    """Paper eq. (1): exp(-4 max(d - 0.3, 0))."""
    return torch.exp(-4.0 * (d - 0.3).clamp(min=0.0))


# ------------------------------------------------------------------ S2 (paper eq. (2), (3))
def target_normal(elev_deg: float = 50.0, device="cpu") -> torch.Tensor:
    """n* for side 0: facing the net (+x), tilted up by ``elev_deg``."""
    e = math.radians(elev_deg)
    return torch.tensor((math.cos(e), 0.0, math.sin(e)), device=device)


def quat_axes_xy(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Body x and y axes in the world frame; quaternions (w, x, y, z)."""
    w, x, y, z = q.unbind(-1)
    ax = torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)), dim=-1)
    ay = torch.stack((2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)), dim=-1)
    return ax, ay


def face_error(normal: torch.Tensor, n_star: torch.Tensor) -> torch.Tensor:
    """Angle (rad) between the racket plate normal and n*. The plate has two faces, so |cos| is used."""
    c = (normal * n_star).sum(-1).abs().clamp(max=1.0)
    return torch.acos(c)


def r_swing(e_pos: torch.Tensor, e_ori: torch.Tensor, v_n: torch.Tensor, sigma_p: float, sigma_r: float,
            sigma_v: float = 3.0, v_cap: float = 8.0, w_v: float = 0.3, gated: bool = True) -> torch.Tensor:
    """Paper eq. (2): exp(-|e_pos|^2/sp) exp(-e_ori/sr) + 0.3 (1 - exp(-max(0, v.n*)/sv)).

    ``e_pos``: (N,) distance sweet spot - hitting target, m. ``v_n``: sweet-spot speed along n*.
    ``v_cap`` (11-08, 1 ms physics): the old 3.5 m/s cap was a tunnelling guard (closing speed >6.4 m/s
    passed through the plate at 5 ms); at 1 ms that limit is gone, so the cap only bounds outliers.
    ``gated`` (11-08 V3.3, our adaptation): the speed term is multiplied by the pose term, so a fast swing
    only pays where the racket meets the target. The additive paper form was exploited in the S2 probe
    (normal racket speed 0.4 -> 6.2 m/s in 270 iterations with the sweet-spot error flat at ~0.36 m).
    """
    pose = torch.exp(-(e_pos**2) / sigma_p) * torch.exp(-e_ori / sigma_r)
    v = v_n.clamp(min=0.0, max=v_cap)
    speed = w_v * (1.0 - torch.exp(-v / sigma_v))
    return pose * (1.0 + speed) if gated else pose + speed


def r_racket_style(axis_y: torch.Tensor) -> torch.Tensor:
    """Paper eq. (3): (y_ee . y_world)^2, y_ee = plate short edge (URDF box 0.01 x 0.14 x 0.2: y)."""
    return axis_y[..., 1] ** 2


def anneal(it: int, start: float, end: float, n: int) -> float:
    """Linear from ``start`` (it = 0) to ``end`` (it >= n)."""
    a = min(max(it / max(n, 1), 0.0), 1.0)
    return start + a * (end - start)


def anneal_exp(it: int, start: float, end: float, tau: float) -> float:
    """end + (start - end) exp(-it / tau): fast first, then slow (V3.3; sigma_p 2.0 -> 0.3 at ~420 iters)."""
    return end + (start - end) * math.exp(-max(it, 0) / max(tau, 1e-6))


def blend_weights(a: dict, b: dict, alpha: float) -> dict:
    alpha = min(max(alpha, 0.0), 1.0)
    keys = set(a) | set(b)
    return {k: (1 - alpha) * a.get(k, 0.0) + alpha * b.get(k, 0.0) for k in keys}


@dataclass
class FeederTable:
    launch_pos: torch.Tensor
    launch_vel: torch.Tensor
    int_pos: torch.Tensor
    t_arr: torch.Tensor
    d_need: torch.Tensor
    station: torch.Tensor

    def __len__(self) -> int:
        return self.d_need.shape[0]


def build_feeder_table(
    n: int = 200_000,
    seed: int = 0,
    device="cpu",
    t_min: float = 0.8,
    v_walk: float = 0.9,
    react: float = 0.2,
    speed=(7.0, 18.0),
    elev_deg=(10.0, 65.0),
    azim_deg=(-20.0, 20.0),
    height=(0.5, 1.15),
) -> FeederTable:
    """Monte-Carlo feeds, filtered like the paper (intercept in the hitting region, flight >= 0.8 s)
    plus a walking-reachability bound. Sorted by the distance the receiver must move (``d_need``)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def u(lo, hi):
        return lo + (hi - lo) * torch.rand(n, generator=gen)

    pos = torch.stack((u(2.25, 2.85), u(-0.3, 0.3), u(*height)), dim=-1)
    sp, el, az = u(*speed), torch.deg2rad(u(*elev_deg)), torch.deg2rad(u(*azim_deg))
    vel = torch.stack((-sp * torch.cos(el) * torch.cos(az), sp * torch.cos(el) * torch.sin(az), sp * torch.sin(el)), dim=-1)
    pos, vel = pos.to(device), vel.to(device)
    net_z, land = simulate_feed(pos, vel)
    legal = (net_z >= NET_CLEAR) & (land[:, 0] <= 0.0) & (land[:, 0] >= -6.7) & (land[:, 1].abs() <= 2.59)
    pt, t, ok = predict_intercept(pos, vel)
    station = station_from_intercept(pt[:, :2])
    ready = torch.tensor(READY_ROOT_XY, device=pos.device)
    d_need = torch.linalg.norm(station - ready, dim=-1)
    keep = legal & ok & (t >= t_min) & (d_need <= v_walk * (t - react) + 0.3)
    order = torch.argsort(d_need[keep])
    sel = lambda x: x[keep][order]  # noqa: E731
    return FeederTable(sel(pos), sel(vel), sel(pt), sel(t), sel(d_need), sel(station))


def table_from_serves(pos: torch.Tensor, vel: torch.Tensor, device="cpu") -> tuple[FeederTable, dict]:
    """Feeder table from serves recorded in play (``scripts/tools/record_serves.py``), receiver frame.

    Same intercept / station definitions as the synthetic table, but **no time or reachability filter**:
    these are the balls the receiver will actually face. Only serves without a valid intercept in the
    band are dropped. Sorted by ``d_need``.
    """
    pos, vel = pos.to(device).float(), vel.to(device).float()
    pt, t, ok = predict_intercept(pos, vel)
    station = station_from_intercept(pt[:, :2])
    ready = torch.tensor(READY_ROOT_XY, device=pos.device)
    d_need = torch.linalg.norm(station - ready, dim=-1)
    keep = ok & (t > 0.1)
    order = torch.argsort(d_need[keep])
    sel = lambda x: x[keep][order]  # noqa: E731
    info = dict(n_in=int(pos.shape[0]), n_kept=int(keep.sum()), d_need_p50=float(d_need[keep].median()) if keep.any() else float("nan"),
                t_p50=float(t[keep].median()) if keep.any() else float("nan"))  # fmt: skip
    return FeederTable(sel(pos), sel(vel), sel(pt), sel(t), sel(d_need), sel(station)), info


def build_feeder_table_v4(
    n: int = 300_000,
    seed: int = 0,
    device="cpu",
    t_min: float = 0.7,
    v_max: float = 2.4,
    launch_x=(2.0, 3.0),
    launch_y=(-0.4, 0.4),
    height=(0.6, 1.15),
    speed=(5.0, 16.0),
    elev_deg=(20.0, 62.0),
    azim_deg=(-25.0, 25.0),
) -> FeederTable:
    """S1 V4 (11-09): wide Monte-Carlo feeds, free hitting height, no reachability filter.

    The launch box covers the recorded serves (contact x 2.2-2.4, z 0.74-0.84, 8-13 m/s, 32-53 deg).
    Kept: legal (net clearance, singles court), an intercept in the band, flight >= ``t_min`` and a required
    speed <= ``v_max``. ``d_need`` keeps its meaning (distance ready root -> station); the curriculum key
    is ``required_speed(d_need, t_arr)``. Sorted by that speed.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def u(lo, hi):
        return lo + (hi - lo) * torch.rand(n, generator=gen)

    pos = torch.stack((u(*launch_x), u(*launch_y), u(*height)), dim=-1)
    sp, el, az = u(*speed), torch.deg2rad(u(*elev_deg)), torch.deg2rad(u(*azim_deg))
    vel = torch.stack((-sp * torch.cos(el) * torch.cos(az), sp * torch.cos(el) * torch.sin(az), sp * torch.sin(el)), dim=-1)
    pos, vel = pos.to(device), vel.to(device)
    net_z, land = simulate_feed(pos, vel)
    legal = (net_z >= NET_CLEAR) & (land[:, 0] <= 0.0) & (land[:, 0] >= -6.7) & (land[:, 1].abs() <= 2.59)
    pt, t, ok = predict_intercept_multi(pos, vel)
    station = station_from_intercept(pt[:, :2])
    ready = torch.tensor(READY_ROOT_XY, device=pos.device)
    d_need = torch.linalg.norm(station - ready, dim=-1)
    v_req = required_speed(d_need, t)
    keep = legal & ok & (t >= t_min) & (v_req <= v_max)
    order = torch.argsort(v_req[keep])
    sel = lambda x: x[keep][order]  # noqa: E731
    return FeederTable(sel(pos), sel(vel), sel(pt), sel(t), sel(d_need), sel(station))


def table_from_serves_v4(pos: torch.Tensor, vel: torch.Tensor, device="cpu") -> tuple[FeederTable, dict]:
    """Recorded serves with the free-height intercept; no time / reachability filter (sorted by speed)."""
    pos, vel = pos.to(device).float(), vel.to(device).float()
    pt, t, ok = predict_intercept_multi(pos, vel)
    station = station_from_intercept(pt[:, :2])
    ready = torch.tensor(READY_ROOT_XY, device=pos.device)
    d_need = torch.linalg.norm(station - ready, dim=-1)
    keep = ok & (t > 0.1)
    v_req = required_speed(d_need, t)
    order = torch.argsort(v_req[keep])
    sel = lambda x: x[keep][order]  # noqa: E731
    vk = v_req[keep]
    info = dict(n_in=int(pos.shape[0]), n_kept=int(keep.sum()), v_req_p50=float(vk.median()) if keep.any() else float("nan"),
                d_need_p50=float(d_need[keep].median()) if keep.any() else float("nan"),
                t_p50=float(t[keep].median()) if keep.any() else float("nan"))  # fmt: skip
    return FeederTable(sel(pos), sel(vel), sel(pt), sel(t), sel(d_need), sel(station)), info


# Speed levels (V4): upper edges of the required speed per level, m/s. The last level is open (<= v_max).
# 11-10: the old open top band (>= 1.5 m/s) was split in three (band rate stuck at 0.01-0.04 for 700 iters).
SPEED_EDGES = (0.25, 0.5, 0.8, 1.1, 1.5, 1.8, 2.1, float("inf"))
APPROACH_CAPS = (0.6, 0.9, 1.2, 1.5, 1.9, 2.1, 2.4, 2.6)  # approach reward cap per level, m/s


def speed_level(v_req: torch.Tensor, edges=SPEED_EDGES) -> torch.Tensor:
    e = torch.tensor(edges[:-1], device=v_req.device, dtype=v_req.dtype)
    return torch.bucketize(v_req, e, right=True)


def speed_level_bounds(v_sorted: torch.Tensor, level: int, edges=SPEED_EDGES) -> tuple[int, int]:
    """(lo, hi) rows of a table sorted by required speed: [lo, hi) is this level's band."""
    if level in (LEVEL_ALL_V4, LEVEL_MIX_V4):
        return 0, int(v_sorted.numel())
    lo = 0 if level == 0 else int(torch.searchsorted(v_sorted, torch.tensor([edges[level - 1]], device=v_sorted.device, dtype=v_sorted.dtype)).item())
    edge = edges[level]
    hi = int(v_sorted.numel()) if math.isinf(edge) else int(torch.searchsorted(v_sorted, torch.tensor([edge], device=v_sorted.device, dtype=v_sorted.dtype)).item())
    return lo, max(hi, lo + 1)


LEVEL_ALL_V4 = 99  # uniform over the whole V4 table
# S2 V4 (11-09): feeds mixed over the speed levels (harder levels weighted up, the easy ones kept so the
# footwork does not decay; V3.4 lost its station rate 0.73 -> 0.08 when only the swing was rewarded).
LEVEL_MIX_V4 = 98
S2_MIX_V4 = (0.05, 0.05, 0.10, 0.15, 0.30, 0.15, 0.12, 0.08)


def mix_row_weights(v_sorted: torch.Tensor, mix=S2_MIX_V4, edges=SPEED_EDGES) -> torch.Tensor:
    """Per-row sampling weights so that level k gets ``mix[k]`` of the feeds (uniform inside a level)."""
    lv = speed_level(v_sorted, edges)
    counts = torch.bincount(lv, minlength=len(edges)).to(v_sorted.dtype)
    m = torch.tensor(mix, device=v_sorted.device, dtype=v_sorted.dtype)
    m = torch.where(counts > 0, m, torch.zeros_like(m))
    m = m / m.sum()
    return (m / counts.clamp(min=1.0))[lv]


@dataclass
class SpeedCurriculum:
    """S1 V4 (11-09): footwork with a required-speed curriculum.

    The paper samples S1 targets uniformly from one filtered corpus and moves on when the stage objective
    has converged; it gives no in-stage schedule. Our robot starts from p3_final's fixed serve stance, so we
    grow the difficulty instead of starting wide: level k feeds 80% from its speed band and 20% from the
    easier rows (``sample_rows``). Promotion: EMA of the **band's own** station rate >= ``promote[k]`` and
    falls per ball <= ``level_fall`` after ``level_min`` iterations. Gate (stage passed), at the top level:
    band station rate >= ``gate_band``, overall station rate >= ``gate_station``, heading ok >= ``gate_face``,
    falls per ball <= ``gate_fall``.
    """

    level: int = 0
    stage: str = "S1"
    n_levels: int = len(SPEED_EDGES)
    promote: tuple = (0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3)
    level_ema: int = 100
    level_min: int = 150
    level_fall: float = 0.03
    gate_ema: int = 200
    gate_min: int = 300  # iterations at the top level before the gate is judged
    gate_band: float = 0.2  # 11-10: top band is now >= 2.1 m/s
    # 11-09 22:30: judged on the synthetic balls only. The overall rate mixes 30% real serves (station ~0 at
    # this point), and at the top level 80% of the synthetic feeds are L5: 0.45 overall was unreachable.
    gate_station: float = 0.35
    gate_station_key: str = "src_syn_station_ok"
    gate_face: float = 0.5
    gate_fall: float = 0.03
    gate_max: int = 8000
    iters_in_level: int = 0
    iters_in_stage: int = 0
    ema: dict = field(default_factory=dict)
    gate_ema_vals: dict = field(default_factory=dict)
    passed: bool = False
    stop_reason: str | None = None
    history: list = field(default_factory=list)
    events: list = field(default_factory=list)
    version: str = "v4"

    @property
    def top(self) -> int:
        return self.n_levels - 1

    def _band(self, store: dict) -> float:
        return store.get(f"L{self.level}_station_ok", 0.0)

    def update(self, iteration: int, rates: dict) -> None:
        self.events = []
        self.iters_in_level += 1
        self.iters_in_stage += 1
        ReceiveCurriculum._ema(self.ema, rates, self.level_ema)
        if self.level == self.top:
            ReceiveCurriculum._ema(self.gate_ema_vals, rates, self.gate_ema)
        e = self.ema
        if self.level < self.top and self.iters_in_level >= self.level_min:
            if self._band(e) >= self.promote[self.level] and e.get("fall_ball", 1.0) <= self.level_fall:
                self.history.append(dict(event=f"level_{self.level + 1}", iteration=iteration, ema=dict(e)))
                self.level += 1
                self.iters_in_level = 0
                self.ema = {}
                self.events.append(f"level{self.level}")
        if self.level == self.top and not self.passed and self.iters_in_level >= self.gate_min:
            g = self.gate_ema_vals
            if (
                self._band(g) >= self.gate_band
                and g.get(self.gate_station_key, 0.0) >= self.gate_station
                and g.get("face_ok", 0.0) >= self.gate_face
                and g.get("fall_ball", 1.0) <= self.gate_fall
            ):
                self.passed = True
                self.history.append(dict(event="passed_S1", iteration=iteration, ema=dict(g)))
                self.events.append("passed_S1")
        for m in (500, 1000, 1500, 2000, 3000, 4000, 5000, 6000):
            if self.iters_in_stage == m:
                self.events.append(f"s1_{m}")
        if not self.passed and self.iters_in_stage >= self.gate_max:
            self.stop_reason = "S1_max_iters"

    def approach_cap(self) -> float:
        return APPROACH_CAPS[min(self.level, len(APPROACH_CAPS) - 1)]

    def log_values(self) -> dict:
        out = {"Curriculum/level": float(self.level), "Curriculum/passed": float(self.passed), "Curriculum/approach_cap": self.approach_cap()}
        out.update({f"Curriculum/ema_{k}": v for k, v in self.ema.items() if k in ("station_ok", "fall_ball", "v_to_station", f"L{self.level}_station_ok")})
        out.update({f"Curriculum/gate_{k}": v for k, v in self.gate_ema_vals.items() if k in ("station_ok", "fall_ball", "face_ok", f"L{self.level}_station_ok")})
        return out

    def state_dict(self) -> dict:
        keys = ("level", "stage", "iters_in_level", "iters_in_stage", "ema", "gate_ema_vals", "passed", "history", "version")
        return {k: getattr(self, k) for k in keys}

    def load_state_dict(self, state: dict) -> None:
        if state.get("version") != "v4":
            raise ValueError("not a V4 S1 curriculum state")
        for k, v in state.items():
            setattr(self, k, v)


def concat_tables(tables: list) -> FeederTable:
    return FeederTable(*[torch.cat([getattr(t, f) for t in tables]) for f in ("launch_pos", "launch_vel", "int_pos", "t_arr", "d_need", "station")])


LEVEL_EDGES = (0.4, 0.8, 1.4, float("inf"))
LEVEL_ALL = 4  # uniform over the whole filtered table (the paper samples the filtered set uniformly)


def ball_level(d_need: torch.Tensor) -> torch.Tensor:
    """Distance band 0-3 of each feed (L0 <0.4, L1 <0.8, L2 <1.4, L3 >=1.4 m)."""
    edges = torch.tensor(LEVEL_EDGES[:-1], device=d_need.device, dtype=d_need.dtype)
    return torch.bucketize(d_need, edges, right=True)


def level_bounds(table: FeederTable, level: int) -> tuple[int, int]:
    """(lo, hi): rows [lo, hi) have this level's d_need band; rows [0, lo) are the easier levels.

    Note: at L3 80% of the feeds are >= 1.4 m away (the S1 gate was judged on that mix, not on the
    natural distribution). ``LEVEL_ALL`` returns (0, len): uniform over the table."""
    if level == LEVEL_ALL:
        return 0, len(table)
    d = table.d_need
    lo = 0 if level == 0 else int(torch.searchsorted(d, torch.tensor([LEVEL_EDGES[level - 1]], device=d.device)).item())
    edge = LEVEL_EDGES[level]
    hi = len(table) if math.isinf(edge) else int(torch.searchsorted(d, torch.tensor([edge], device=d.device)).item())
    return lo, max(hi, lo + 1)


def sample_rows(n: int, lo: int, hi: int, prev_frac: float, device, generator=None) -> torch.Tensor:
    """80% from the current band [lo, hi), 20% from the easier rows [0, lo) (all from the band at level 0)."""
    r = torch.rand(n, device=device, generator=generator)
    band = lo + (r * (hi - lo)).long().clamp(max=hi - 1)
    if lo == 0:
        return band
    r2 = torch.rand(n, device=device, generator=generator)
    easy = (r2 * lo).long().clamp(max=lo - 1)
    use_easy = torch.rand(n, device=device, generator=generator) < prev_frac
    return torch.where(use_easy, easy, band)


# ------------------------------------------------------------------ curriculum (S1 only for now)
@dataclass
class ReceiveCurriculum:
    """Feeder levels L0-L3 by ``d_need``; the S1 gate is judged on L3 only (V3.1 5.1, 7.3).

    ``rates`` keys: ``station_ok`` (fraction of balls whose arrival-time station error < 0.35 m),
    ``face_ok`` (fraction with trunk heading error < 30 deg), ``fall`` (falls per finished episode).
    """

    level: int = 0
    stage: str = "S1"
    level_ema: int = 100
    level_min: int = 100
    level_station: float = 0.4
    level_fall: float = 0.03  # 11-08: per fed ball, = spec "fall < 3%"
    gate_ema: int = 200
    gate_min: int = 300
    gate_max: int = 4000  # 11-08: raised from 2000 (L1 alone took ~800 iters)
    gate_station: float = 0.5
    gate_face: float = 0.5
    gate_fall: float = 0.03  # 11-08: per fed ball, = spec
    iters_in_level: int = 0
    iters_in_stage: int = 0
    ema: dict = field(default_factory=dict)
    gate_ema_vals: dict = field(default_factory=dict)
    passed: bool = False
    stop_reason: str | None = None
    history: list = field(default_factory=list)
    events: list = field(default_factory=list)

    @staticmethod
    def _ema(store: dict, rates: dict, n: int) -> None:
        a = 2.0 / (n + 1.0)
        for k, v in rates.items():
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            store[k] = v if k not in store else (1 - a) * store[k] + a * v

    def update(self, iteration: int, rates: dict) -> None:
        self.events = []
        self.iters_in_level += 1
        self.iters_in_stage += 1
        self._ema(self.ema, rates, self.level_ema)
        if self.level == 3:
            self._ema(self.gate_ema_vals, rates, self.gate_ema)
        if self.level < 3 and self.iters_in_level >= self.level_min:
            e = self.ema
            if e.get("station_ok", 0.0) >= self.level_station and e.get("fall_ball", 1.0) <= self.level_fall:
                self.history.append(dict(event=f"level_{self.level + 1}", iteration=iteration, ema=dict(e)))
                self.level += 1
                self.iters_in_level = 0
                self.ema = {}
                self.events.append(f"level{self.level}")
        if self.level == 3 and not self.passed and self.iters_in_stage >= self.gate_min and self.iters_in_level >= self.level_min:
            g = self.gate_ema_vals
            if (
                g.get("station_ok", 0.0) >= self.gate_station
                and g.get("face_ok", 0.0) >= self.gate_face
                and g.get("fall_ball", 1.0) <= self.gate_fall
            ):
                self.passed = True
                self.history.append(dict(event="passed_S1", iteration=iteration, ema=dict(g)))
                self.events.append("passed_S1")
        if not self.passed and self.iters_in_stage >= self.gate_max:
            self.stop_reason = "S1_max_iters"

    def log_values(self) -> dict:
        out = {"Curriculum/level": float(self.level), "Curriculum/passed": float(self.passed)}
        out.update({f"Curriculum/ema_{k}": v for k, v in self.ema.items()})
        out.update({f"Curriculum/gate_{k}": v for k, v in self.gate_ema_vals.items()})
        return out

    def state_dict(self) -> dict:
        keys = ("level", "stage", "iters_in_level", "iters_in_stage", "ema", "gate_ema_vals", "passed", "history")
        return {k: getattr(self, k) for k in keys}

    def load_state_dict(self, state: dict) -> None:
        for k, v in state.items():
            setattr(self, k, v)


@dataclass
class S2Curriculum:
    """Stage S2, precision-guided swing (paper IV-A stage 2; P5 V3.1 5.2, 11-08 revision).

    - reward weights blend S1 -> S2 over ``blend_iters``;
    - sigma_p 2.0 -> 0.1, sigma_r 8.0 -> 1.0 (paper: "progressively tighten"); V3.3: exponential with time
      constant ``anneal_tau`` (220: sigma_p 0.30 / sigma_r 1.7 at 500 iters, 0.12 / 1.1 at 1000). The linear
      2000-iteration schedule of the probe left sigma_p ~1.7 at 300 iters (no position gradient);
      ``anneal_iters`` is now only the earliest iteration at which the gate is judged;
    - feeds uniform over the filtered table (``LEVEL_ALL``; the paper samples targets uniformly, no distance
      curriculum);
    - gate (EMA): hit rate >= ``gate_hit``, hit-instant sweet-spot error p50 < ``gate_epos``, falls per ball
      < ``gate_fall``; only after the annealing has finished (judged at the final tolerances).
    """

    s1_weights: dict = field(default_factory=dict)
    s2_weights: dict = field(default_factory=dict)
    stage: str = "S2"
    level: int = LEVEL_ALL
    blend_iters: int = 300
    anneal_iters: int = 1000
    anneal_tau: float = 220.0
    sigma_p: tuple = (2.0, 0.1)
    sigma_r: tuple = (8.0, 1.0)
    gate_ema: int = 200
    gate_min: int = 500
    gate_max: int = 4000
    gate_hit: float = 0.4
    gate_epos: float = 0.15
    gate_fall: float = 0.03
    gate_hit_key: str = "hit"  # S2 V4: "first_all_hit" (first ball of an episode, = a play rally)
    iters_in_stage: int = 0
    ema: dict = field(default_factory=dict)
    passed: bool = False
    stop_reason: str | None = None
    history: list = field(default_factory=list)
    events: list = field(default_factory=list)

    def params(self) -> dict:
        it = self.iters_in_stage
        return dict(
            weights=blend_weights(self.s1_weights, self.s2_weights, it / max(self.blend_iters, 1)),
            sigma_p=anneal_exp(it, *self.sigma_p, self.anneal_tau),
            sigma_r=anneal_exp(it, *self.sigma_r, self.anneal_tau),
        )

    def update(self, iteration: int, rates: dict) -> None:
        self.events = []
        self.iters_in_stage += 1
        ReceiveCurriculum._ema(self.ema, rates, self.gate_ema)
        e = self.ema
        ready = self.iters_in_stage >= max(self.gate_min, self.anneal_iters)
        if ready and not self.passed and (
            e.get(self.gate_hit_key, 0.0) >= self.gate_hit and e.get("epos_p50", 1.0) < self.gate_epos and e.get("fall_ball", 1.0) <= self.gate_fall
        ):
            self.passed = True
            self.history.append(dict(event="passed_S2", iteration=iteration, ema=dict(e)))
            self.events.append("passed_S2")
        for m in (500, 1000, 1500, 2000, 3000):
            if self.iters_in_stage == m:
                self.events.append(f"s2_{m}")  # periodic checkpoints for play evaluation
        if not self.passed and self.iters_in_stage >= self.gate_max:
            self.stop_reason = "S2_max_iters"

    def log_values(self) -> dict:
        p = self.params()
        out = {"Curriculum/passed": float(self.passed), "Curriculum/sigma_p": p["sigma_p"], "Curriculum/sigma_r": p["sigma_r"],
               "Curriculum/blend": min(self.iters_in_stage / max(self.blend_iters, 1), 1.0)}  # fmt: skip
        out.update({f"Curriculum/ema_{k}": v for k, v in self.ema.items() if not k.startswith("L")})
        return out

    def state_dict(self) -> dict:
        keys = ("stage", "level", "iters_in_stage", "ema", "passed", "history")
        return {k: getattr(self, k) for k in keys}

    def load_state_dict(self, state: dict) -> None:
        if state.get("stage") != "S2":
            raise ValueError(f"not an S2 curriculum state: {state.get('stage')}")
        for k, v in state.items():
            setattr(self, k, v)


# ------------------------------------------------------------------ warm start from p3_final
def load_p3_torchscript(path: str) -> dict:
    """Normalizer and MLP weights of an exported ``Sequential(normalizer, mlp)`` policy."""
    m = torch.jit.load(path, map_location="cpu")
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    return sd


@torch.no_grad()
def warm_start_actor(policy, ts_state: dict, extra_mean: torch.Tensor, extra_std: torch.Tensor) -> None:
    """Copy p3_final into an RSL-RL ActorCritic whose actor input has extra columns (zero-initialised).

    With zero columns the new actor equals p3_final on the first 95 inputs, whatever the extra inputs are.
    """
    w0 = ts_state["1.0.weight"]
    n_old = w0.shape[1]
    first = policy.actor[0]
    n_new = first.in_features
    first.weight.zero_()
    first.weight[:, :n_old] = w0
    first.bias.copy_(ts_state["1.0.bias"])
    lin = [m for m in policy.actor if isinstance(m, torch.nn.Linear)]
    keys = ("1.2", "1.4", "1.6")
    for layer, k in zip(lin[1:], keys):
        layer.weight.copy_(ts_state[f"{k}.weight"])
        layer.bias.copy_(ts_state[f"{k}.bias"])
    norm = policy.actor_obs_normalizer
    mean = torch.cat((ts_state["0._mean"].reshape(-1), extra_mean.reshape(-1)))
    std = torch.cat((ts_state["0._std"].reshape(-1), extra_std.reshape(-1)))
    assert mean.numel() == n_new, (mean.numel(), n_new)
    norm._mean.copy_(mean[None])
    norm._std.copy_(std[None])
    norm._var.copy_((std**2)[None])
    norm.count.fill_(int(ts_state["0.count"]))
    norm.until = 1  # frozen: count >= until


@torch.no_grad()
def set_log_std(log_std: torch.Tensor, leg_mask: torch.Tensor, leg: float, other: float) -> None:
    log_std.copy_(torch.where(leg_mask, torch.full_like(log_std, math.log(leg)), torch.full_like(log_std, math.log(other))))


@torch.no_grad()
def clamp_log_std(log_std: torch.Tensor, leg_mask: torch.Tensor, leg=(0.25, 0.6), other=(0.15, 0.5)) -> None:
    lo = torch.where(leg_mask, torch.full_like(log_std, math.log(leg[0])), torch.full_like(log_std, math.log(other[0])))
    hi = torch.where(leg_mask, torch.full_like(log_std, math.log(leg[1])), torch.full_like(log_std, math.log(other[1])))
    log_std.copy_(torch.maximum(torch.minimum(log_std, hi), lo))
