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


LEVEL_EDGES = (0.4, 0.8, 1.4, float("inf"))


def level_bounds(table: FeederTable, level: int) -> tuple[int, int]:
    """(lo, hi): rows [lo, hi) have this level's d_need band; rows [0, lo) are the easier levels."""
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
    level_fall: float = 0.02  # 11-08: per ball (was 0.05 per 6-ball episode)
    gate_ema: int = 200
    gate_min: int = 300
    gate_max: int = 4000  # 11-08: raised from 2000 (L1 alone took ~800 iters)
    gate_station: float = 0.5
    gate_face: float = 0.5
    gate_fall: float = 0.015  # 11-08: per ball
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
