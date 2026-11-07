"""P5 feeder ("serve machine") and intercept targets (P5_RECEIVE_DESIGN 2.2). Pure torch, no Isaac import.

Frame: the canonical frame of the feeder (= the server of a natural rally): own half at x<0, net at x=0,
the receiver on x>0. The receiver's ready root is the mirror of ``READY_ROOT``: ``RECV_HOME``.

* ``scan_trajectories`` integrates shuttle flights with the training drag law and records the net
  crossing height, the landing point and the **intercept target**: the point inside the height band
  ``[band_lo, band_hi]`` (and ``x > band_x_min``) with the smallest horizontal distance to ``RECV_HOME``,
  its time and velocity. ``d_need = max(0, that distance - REACH)`` (``analysis/p5_reach.py``).
* ``FeederTable`` is the offline grid (speed x elevation x azimuth x launch height x lateral offset),
  legal rows only (net clearance >= ``net_margin`` over the edge height, landing inside the receiver's
  singles court), sorted by ``d_need`` so a curriculum level is an index range.

The table uses semi-implicit Euler at the physics step (what PhysX does with a force held over the step),
so its landing points match the simulator; ``scan_trajectories`` with ``scheme="midpoint"`` and a coarse
step is the cheap online variant for natural serves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

RECV_HOME = (3.251, -0.605)  # receiver root, feeder frame (mirror of player_cfg.READY_ROOT)
REACH = 0.85  # horizontal sweet-spot reach from the root, ready pose
LAUNCH_X = -2.55  # x of the drop point (mirrored per side by reset_shuttle_drop_state)
DROP_Z = 2.2


@dataclass
class FeederParams:
    speed: tuple = (7.0, 18.0, 0.5)  # (lo, hi, step) m/s
    elevation: tuple = (10.0, 65.0, 2.5)  # deg
    azimuth: tuple = (-20.0, 20.0, 2.5)  # deg
    z_launch: tuple = (0.6, 1.15, 0.11)  # m
    y_launch: tuple = (-0.5, 0.5, 0.25)  # m
    net_margin: float = 0.05  # over the net edge height
    band_lo: float = 0.5
    band_hi: float = 1.4
    band_x_min: float = 0.3
    dt: float = 0.005
    max_time: float = 4.0
    levels: tuple = (0.3, 0.8, 1.5, 2.5)  # d_need upper bound of L0..L3
    new_share: float = 0.7  # share of a level's samples from its own d_need band
    floor: float = 0.03  # landing detection height
    half_length: float = 6.7
    singles_half_width: float = 2.59
    net_height: float = 1.55
    home: tuple = field(default=RECV_HOME)
    reach: float = REACH


def _grid(lo, hi, step):
    n = int(round((hi - lo) / step)) + 1
    return torch.linspace(lo, hi, n, dtype=torch.float64)


def scan_trajectories(pos, vel, drag_over_mass: float, gravity: float, p: FeederParams, dt: float | None = None,
                      scheme: str = "euler", max_time: float | None = None, check_every: int = 20) -> dict:
    """(R, 3) start states -> dict of (R, ...) tensors: ``net_z`` (-inf: never crossed forward),
    ``land`` (xy), ``t_land``, ``tgt`` (3), ``t_tgt``, ``v_tgt`` (3), ``d_min`` (inf: never in the band),
    ``d_need``, ``tgt_valid``. Times are from the start state."""
    dt = p.dt if dt is None else dt
    steps = int(round((p.max_time if max_time is None else max_time) / dt))
    dev, dt_ = pos.device, pos.dtype
    rows = pos.shape[0]
    P, V = pos.clone(), vel.clone()
    g = torch.tensor((0.0, 0.0, -gravity), device=dev, dtype=dt_)
    home = torch.tensor(p.home, device=dev, dtype=dt_)
    active = torch.ones(rows, dtype=torch.bool, device=dev)
    net_z = torch.full((rows,), -math.inf, device=dev, dtype=dt_)
    t_land = torch.full((rows,), math.inf, device=dev, dtype=dt_)
    d_min = torch.full((rows,), math.inf, device=dev, dtype=dt_)
    tgt = torch.zeros(rows, 3, device=dev, dtype=dt_)
    v_tgt = torch.zeros(rows, 3, device=dev, dtype=dt_)
    t_tgt = torch.zeros(rows, device=dev, dtype=dt_)
    for k in range(steps):
        if k % check_every == 0 and not bool(active.any()):
            break
        if scheme == "euler":
            speed = torch.linalg.norm(V, dim=-1, keepdim=True)
            vn = V + dt * (g - drag_over_mass * speed * V)
            pn = P + dt * vn
        else:
            speed = torch.linalg.norm(V, dim=-1, keepdim=True)
            vm = V + 0.5 * dt * (g - drag_over_mass * speed * V)
            vn = V + dt * (g - drag_over_mass * torch.linalg.norm(vm, dim=-1, keepdim=True) * vm)
            pn = P + dt * vm
        pn = torch.where(active[:, None], pn, P)
        vn = torch.where(active[:, None], vn, V)
        crossing = active & (P[:, 0] < 0.0) & (pn[:, 0] >= 0.0)
        dx = pn[:, 0] - P[:, 0]
        a = -P[:, 0] / torch.where(crossing, dx, torch.ones_like(dx))
        net_z = torch.where(crossing, P[:, 2] + a * (pn[:, 2] - P[:, 2]), net_z)
        P, V = pn, vn
        t = (k + 1) * dt
        band = active & (P[:, 2] >= p.band_lo) & (P[:, 2] <= p.band_hi) & (P[:, 0] > p.band_x_min)
        d = torch.linalg.norm(P[:, :2] - home, dim=-1)
        better = band & (d < d_min)
        d_min = torch.where(better, d, d_min)
        tgt = torch.where(better[:, None], P, tgt)
        v_tgt = torch.where(better[:, None], V, v_tgt)
        t_tgt = torch.where(better, torch.full_like(t_tgt, t), t_tgt)
        landed = active & (P[:, 2] <= p.floor)
        t_land = torch.where(landed, torch.full_like(t_land, t), t_land)
        active = active & ~landed
    valid = torch.isfinite(d_min)
    d_need = torch.where(valid, torch.clamp(d_min - p.reach, min=0.0), torch.full_like(d_min, math.inf))
    return dict(net_z=net_z, land=P[:, :2].clone(), t_land=t_land, tgt=tgt, t_tgt=t_tgt, v_tgt=v_tgt,
                d_min=d_min, d_need=d_need, tgt_valid=valid)


def legal_mask(scan: dict, p: FeederParams) -> torch.Tensor:
    land = scan["land"]
    return ((scan["net_z"] >= p.net_height + p.net_margin) & (land[:, 0] > 0.0) & (land[:, 0] <= p.half_length)
            & (land[:, 1].abs() <= p.singles_half_width) & torch.isfinite(scan["t_land"]) & scan["tgt_valid"])


def launch_velocity(speed, elev_deg, azim_deg):
    e, a = torch.deg2rad(elev_deg), torch.deg2rad(azim_deg)
    return torch.stack((speed * torch.cos(e) * torch.cos(a), speed * torch.cos(e) * torch.sin(a), speed * torch.sin(e)), -1)


def drop_time(z_launch, drag_over_mass: float, gravity: float, dt: float, z0: float = DROP_Z):
    """Physics steps of the drop from rest at ``z0`` until z <= z_launch (semi-implicit Euler), and the speed."""
    z = torch.full_like(z_launch, z0)
    v = torch.zeros_like(z_launch)
    k = torch.zeros_like(z_launch)
    done = torch.zeros_like(z_launch, dtype=torch.bool)
    for i in range(2000):
        v = torch.where(done, v, v + dt * (-gravity + drag_over_mass * v * v))
        z = torch.where(done, z, z + dt * v)
        now = ~done & (z <= z_launch)
        k = torch.where(now, torch.full_like(k, i + 1), k)
        done |= now
        if bool(done.all()):
            break
    return k, -v


COLUMNS = ("speed", "elev", "azim", "z", "y", "vx", "vy", "vz", "land_x", "land_y", "net_z", "t_land",
           "tgt_x", "tgt_y", "tgt_z", "t_tgt", "vt_x", "vt_y", "vt_z", "d_need")


class FeederTable:
    """Legal launches sorted by ``d_need``; ``sample(level, n)`` draws row indices on the table's device."""

    def __init__(self, data: torch.Tensor, p: FeederParams, total_rows: int):
        self.p = p
        self.data = data  # (R, len(COLUMNS)) float32
        self.total_rows = total_rows
        d = data[:, COLUMNS.index("d_need")]
        self.cuts = [0] + [int(torch.searchsorted(d.contiguous(), torch.tensor(float(x), device=d.device), right=True))
                           for x in p.levels]

    def col(self, name: str) -> torch.Tensor:
        return self.data[:, COLUMNS.index(name)]

    def __len__(self) -> int:
        return self.data.shape[0]

    @classmethod
    def build(cls, p: FeederParams, drag_over_mass: float, gravity: float, device="cpu") -> "FeederTable":
        grids = [_grid(*r) for r in (p.speed, p.elevation, p.azimuth, p.z_launch, p.y_launch)]
        mesh = torch.cartesian_prod(*grids).to(device=device, dtype=torch.float64)
        sp, el, az, z, y = mesh.unbind(-1)
        pos = torch.stack((torch.full_like(z, LAUNCH_X), y, z), -1)
        vel = launch_velocity(sp, el, az)
        scan = scan_trajectories(pos, vel, drag_over_mass, gravity, p)
        ok = legal_mask(scan, p)
        cols = torch.stack((sp, el, az, z, y, vel[:, 0], vel[:, 1], vel[:, 2], scan["land"][:, 0], scan["land"][:, 1],
                            scan["net_z"], scan["t_land"], scan["tgt"][:, 0], scan["tgt"][:, 1], scan["tgt"][:, 2],
                            scan["t_tgt"], scan["v_tgt"][:, 0], scan["v_tgt"][:, 1], scan["v_tgt"][:, 2], scan["d_need"]), -1)
        cols = cols[ok]
        cols = cols[torch.argsort(cols[:, COLUMNS.index("d_need")])]
        return cls(cols.float(), p, total_rows=mesh.shape[0])

    def level_counts(self) -> list[int]:
        return [self.cuts[i + 1] - self.cuts[i] for i in range(len(self.p.levels))]

    def sample(self, level, n: int, generator=None) -> torch.Tensor:
        """``level``: int or (n,) long tensor. 70% from the level's own band, 30% from all lower bands."""
        dev = self.data.device
        lv = torch.as_tensor(level, device=dev, dtype=torch.long).expand(n) if not torch.is_tensor(level) or level.dim() == 0 \
            else level.to(dev)
        cuts = torch.tensor(self.cuts, device=dev, dtype=torch.long)
        lo_band, hi_band = cuts[lv], cuts[lv + 1]
        empty = hi_band <= lo_band
        # L0 (nothing below it) always draws from its own band; an empty band falls back to the lower ones.
        own = ((torch.rand(n, device=dev, generator=generator) < self.p.new_share) | (lo_band == 0)) & ~(empty & (lo_band > 0))
        lo = torch.where(own, lo_band, torch.zeros_like(lo_band))
        hi = torch.where(own, hi_band, lo_band)
        u = torch.rand(n, device=dev, generator=generator)
        idx = lo + (u * (hi - lo).clamp(min=1).float()).long()
        return torch.minimum(idx, (hi - 1).clamp(min=0))
