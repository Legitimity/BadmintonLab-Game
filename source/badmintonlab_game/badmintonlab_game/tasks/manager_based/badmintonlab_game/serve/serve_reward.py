"""Serve-stage rewards and metrics (design doc V3 section 7.2). Pure torch, no Isaac import.

The environment calls :meth:`ServeRewardComputer.capture_contact` once per physics step
and :meth:`ServeRewardComputer.end_step` once per control step, before the reward
managers run. Every reward term is then a thin getter (``reward_terms.serve_term``) with
weight 1.0: the stage weights live here so the curriculum has a single place to change
them and the positive-reward cap can see the weighted values.

Conventions:

* Values are already weighted. Event terms are multiplied by ``1/step_dt`` so the
  RewardManager's ``* dt`` cancels and the number is the reward of one event.
* ``hold_ready`` is dense: its weight is a cost per second, like the default costs.
* The canonical frame of a side has its own half at x<0 and the net at x=0. Side 1 is
  rotated 180 degrees about z, as in ``play``.
* Only the server's first legal hit is scored. The receiver gets no result reward in M1.

Numbers marked P1 must be replaced by the measured values before the long run.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .rules import predict_flight

TERMS = (
    "hold_ready",
    "legal_contact",
    "swing_speed",
    "face_quality",
    "outcome_pred",
    "serve_fault",
    "no_hit",
    "fall",
    "serve_result",
)
POSITIVE_TERMS = ("legal_contact", "swing_speed", "face_quality", "outcome_pred")
EVENT_TERMS = tuple(t for t in TERMS if t != "hold_ready")
STAT_KEYS = (
    "rallies",
    "legal",
    "in",
    "crossed",
    "height",
    "double",
    "no_hit",
    "net",
    "out",
    "fall",
    "return_touch",
    "timeout",
    "vn_sum",
    "el_sum",
    "hd_abs_sum",
    "outcome_sum",
    "contacts",
)


@dataclass
class ServeRewardParams:
    """Targets and shaping constants. Override with hydra, e.g. ``env.serve_reward.v_target=5.5``."""

    v_target: float = 6.0  # P1: racket normal speed at contact, m/s
    elevation_target_deg: float = 55.0  # P1: face normal elevation at contact
    heading_target_deg: float = 0.0
    face_sigma_deg: float = 20.0
    x_target: float = 3.0  # P1: landing x behind the net, m
    x_sigma: float = 2.0
    net_height: float = 1.55  # conservative: the edge height
    net_margin: float = 0.20
    net_scale: float = 0.5
    y_safe: float = 2.0
    y_scale: float = 0.5
    # Not in V3: without it a long serve that lands out still earns about half of outcome_pred.
    x_long_safe: float = 6.4
    x_long_scale: float = 0.3
    no_hit_height: float = 0.5
    fall_height: float = 0.45
    hold_sigma2: float = 0.1  # rad^2, on the mean squared joint error
    positive_cap: float = 8.0  # per rally, on the weighted positive terms
    end_on_return: bool = True  # M1: the rally ends when the receiver touches a legal serve
    flight_dt: float = 0.02
    drag_over_mass: float = 0.00110 / 0.00519
    gravity: float = 9.81
    initial_stage: str = "A"  # weights come from STAGE_WEIGHTS; the curriculum changes them at runtime


# V3 table 7.2. Signs are part of the weight.
STAGE_WEIGHTS = {
    "A": dict(hold_ready=-0.5, legal_contact=2.0, swing_speed=0.0, face_quality=0.0, outcome_pred=1.0,
              serve_fault=-5.0, no_hit=-3.0, fall=-5.0, serve_result=10.0),
    "B": dict(hold_ready=-0.2, legal_contact=1.0, swing_speed=2.0, face_quality=1.0, outcome_pred=4.0,
              serve_fault=-5.0, no_hit=-3.0, fall=-5.0, serve_result=10.0),
    "C": dict(hold_ready=-0.1, legal_contact=0.5, swing_speed=1.0, face_quality=0.5, outcome_pred=4.0,
              serve_fault=-5.0, no_hit=-3.0, fall=-5.0, serve_result=10.0),
}


def quat_rotate_x_axis(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Body x axis in the world frame; Isaac Lab quaternions are (w, x, y, z)."""
    w, x, y, z = quat_wxyz.unbind(-1)
    return torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)), dim=-1)


def to_canonical(vec: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
    """Rotate world/court vectors of ``side`` 1 by 180 degrees about z. ``side``: (...,) long."""
    sign = (1 - 2 * side).to(vec.dtype).unsqueeze(-1)
    return torch.cat((vec[..., :2] * sign, vec[..., 2:]), dim=-1)


def contact_features(normal_c: torch.Tensor, racket_vel_c: torch.Tensor):
    """Striking-face normal speed, elevation and heading (degrees) in the canonical frame.

    The plate has two faces; the striking face is the one moving along its normal.
    """
    vn_signed = (normal_c * racket_vel_c).sum(-1)
    n = normal_c * torch.where(vn_signed >= 0, 1.0, -1.0).unsqueeze(-1)
    vn = vn_signed.abs()
    elevation = torch.rad2deg(torch.asin(n[..., 2].clamp(-1.0, 1.0)))
    heading = torch.rad2deg(torch.atan2(n[..., 1], n[..., 0]))
    return vn, elevation, heading


def face_quality(elevation, heading, p: ServeRewardParams) -> torch.Tensor:
    de = (elevation - p.elevation_target_deg) / p.face_sigma_deg
    dh = (heading - p.heading_target_deg) / p.face_sigma_deg
    return torch.exp(-de * de) * torch.exp(-dh * dh)


def outcome_score(net_z, landing_xy, p: ServeRewardParams) -> torch.Tensor:
    """V3 ``outcome_pred`` in [0, 1] plus the long-serve guard. ``net_z=-inf`` scores 0."""
    clearance = net_z - p.net_height
    s1 = torch.exp(-torch.clamp(p.net_margin - clearance, min=0.0) / p.net_scale)
    s1 = torch.where(torch.isfinite(net_z), s1, torch.zeros_like(s1))
    x, y = landing_xy[..., 0], landing_xy[..., 1]
    s2 = torch.exp(-torch.clamp(y.abs() - p.y_safe, min=0.0) / p.y_scale)
    s3 = torch.exp(-(((x - p.x_target) / p.x_sigma) ** 2))
    s4 = torch.exp(-torch.clamp(x - p.x_long_safe, min=0.0) / p.x_long_scale)
    return s1 * s2 * (0.5 + 0.5 * s3) * s4


@dataclass
class StepInputs:
    """Everything ``end_step`` reads, after the rules have processed the control step."""

    server: torch.Tensor  # (N,) long, the opening hitter
    just_hit: torch.Tensor  # (N, 2) bool, valid hits in this control step
    hit_count: torch.Tensor  # (N, 2) long, valid hits this rally including this step
    just_resolved: torch.Tensor  # (N,) bool
    winner: torch.Tensor  # (N,) long, -1 if unresolved
    height_fault: torch.Tensor  # (N,) bool, latched for the rally
    double_fault: torch.Tensor  # (N,) bool, latched for the rally
    net_fault: torch.Tensor  # (N,) bool, this step
    out_fault: torch.Tensor  # (N,) bool, this step
    shuttle_pos: torch.Tensor  # (N, 3) court-local
    shuttle_vel: torch.Tensor  # (N, 3) world
    root_z: torch.Tensor  # (N, 2) court-local root height
    joint_pos: torch.Tensor  # (N, 2, J)


class ServeRewardComputer:
    def __init__(self, num_envs: int, device, params: ServeRewardParams, q_ready: torch.Tensor,
                 server_hold_mask: torch.Tensor, step_dt: float):
        n = num_envs
        self.p = params
        self.device = device
        self.step_dt = float(step_dt)
        self.q_ready = q_ready.to(device).float()
        full = torch.ones_like(server_hold_mask, dtype=torch.float32, device=device)
        # Role masks: index 0 is the server's, index 1 the receiver's.
        self.hold_masks = torch.stack((server_hold_mask.to(device).float(), full))
        self.weights = {k: 0.0 for k in TERMS}
        self.set_weights(STAGE_WEIGHTS[params.initial_stage])
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=device)  # noqa: E731
        self.values = {k: z(n, 2) for k in TERMS}
        # Contact features captured at the physics step of the server's first legal hit.
        self.c_vn, self.c_el, self.c_hd = z(n), z(n), z(n)
        self.c_have = z(n, dt=torch.bool)
        self.outcome_pending = z(n, dt=torch.bool)
        # Per-rally latches.
        self.pos_acc = z(n, 2)
        self.no_hit_given = z(n, dt=torch.bool)
        self.fall_given = z(n, 2, dt=torch.bool)
        self.result_given = z(n, dt=torch.bool)
        self.crossed = z(n, dt=torch.bool)
        self.fell_server = z(n, dt=torch.bool)
        self.return_done = z(n, dt=torch.bool)
        self.done_now = z(n, dt=torch.bool)
        self.iteration = torch.zeros(len(STAT_KEYS), device=device)

    # ------------------------------------------------------------------ weights
    def set_weights(self, weights: dict) -> None:
        unknown = set(weights) - set(TERMS)
        if unknown:
            raise KeyError(f"Unknown serve reward terms: {sorted(unknown)}")
        self.weights.update({k: float(v) for k, v in weights.items()})

    # ------------------------------------------------------------------ physics rate
    def capture_contact(self, new_hits: torch.Tensor, server: torch.Tensor, racket_quat_w: torch.Tensor,
                        racket_vel_w: torch.Tensor) -> None:
        """``new_hits``: (N, 2) valid hits that appeared in this physics step.

        ``racket_quat_w``/``racket_vel_w``: (N, 2, 4)/(N, 2, 3) sweet-spot pose and velocity.
        """
        idx = server.unsqueeze(1)
        hit = new_hits.gather(1, idx).squeeze(1) & ~self.c_have
        q = racket_quat_w.gather(1, idx[..., None].expand(-1, 1, 4)).squeeze(1)
        v = racket_vel_w.gather(1, idx[..., None].expand(-1, 1, 3)).squeeze(1)
        vn, el, hd = contact_features(to_canonical(quat_rotate_x_axis(q), server), to_canonical(v, server))
        self.c_vn = torch.where(hit, vn, self.c_vn)
        self.c_el = torch.where(hit, el, self.c_el)
        self.c_hd = torch.where(hit, hd, self.c_hd)
        self.c_have |= hit

    # ------------------------------------------------------------------ control rate
    def end_step(self, s: StepInputs) -> None:
        p, w = self.p, self.weights
        n = s.server.shape[0]
        ar = torch.arange(n, device=self.device)
        server, receiver = s.server, 1 - s.server
        server_hit_now = s.just_hit[ar, server]
        server_hits = s.hit_count[ar, server] > 0
        ev = 1.0 / self.step_dt

        role_vals = {k: torch.zeros(n, 2, device=self.device) for k in TERMS}  # column 0 server, 1 receiver

        # Dense: keep the ready pose. The server's swing joints are masked out, and the server
        # is free once it has hit (V3: "before the hit or while not serving").
        q_err = s.joint_pos - self.q_ready
        active = (~server_hits, torch.ones_like(server_hits))
        for role, who in ((0, server), (1, receiver)):
            err = q_err[ar, who] * self.hold_masks[role]
            mse = (err * err).sum(-1) / self.hold_masks[role].sum().clamp(min=1.0)
            cost = 1.0 - torch.exp(-mse / p.hold_sigma2)
            role_vals["hold_ready"][:, role] = w["hold_ready"] * cost * active[role].float()

        # Server's first legal contact, with the features captured at the physics step.
        contact = server_hit_now & self.c_have
        role_vals["legal_contact"][:, 0] = w["legal_contact"] * contact.float()
        role_vals["swing_speed"][:, 0] = w["swing_speed"] * contact.float() * torch.clamp(self.c_vn / p.v_target, max=1.0)
        role_vals["face_quality"][:, 0] = w["face_quality"] * contact.float() * face_quality(self.c_el, self.c_hd, p)

        # Outcome prediction one control step after the hit, unless the rally already ended on a fault.
        due = self.outcome_pending & ~(s.just_resolved & (s.winner != server))
        score = torch.zeros(n, device=self.device)
        if bool(due.any()):
            ids = due.nonzero(as_tuple=False).squeeze(-1)
            pos_c = to_canonical(s.shuttle_pos[ids], server[ids])
            vel_c = to_canonical(s.shuttle_vel[ids], server[ids])
            net_z, land = predict_flight(pos_c, vel_c, p.drag_over_mass, p.gravity, dt=p.flight_dt,
                                         max_steps=int(4.0 / p.flight_dt))
            score[ids] = outcome_score(net_z, land, p)
        role_vals["outcome_pred"][:, 0] = w["outcome_pred"] * score
        self.outcome_pending = server_hit_now.clone()

        # Positive cap per rally, applied to the weighted positive terms together.
        pos = sum(role_vals[k].clamp(min=0.0) for k in POSITIVE_TERMS)
        allowed = (p.positive_cap - self.pos_acc_roles(server)).clamp(min=0.0)
        scale = torch.where(pos > 0, torch.clamp(allowed / pos.clamp(min=1e-9), max=1.0), torch.ones_like(pos))
        for k in POSITIVE_TERMS:
            role_vals[k] = torch.where(role_vals[k] > 0, role_vals[k] * scale, role_vals[k])
        self._add_pos_acc(server, pos * scale)

        # Events of the serve.
        server_fault = s.just_resolved & (s.winner == receiver) & (s.height_fault | s.double_fault)
        role_vals["serve_fault"][:, 0] = w["serve_fault"] * server_fault.float()

        low = s.shuttle_pos[:, 2] < p.no_hit_height
        no_hit = low & ~server_hits & ~self.no_hit_given & ~s.height_fault
        self.no_hit_given |= no_hit
        role_vals["no_hit"][:, 0] = w["no_hit"] * no_hit.float()

        fell = (s.root_z < p.fall_height) & ~self.fall_given
        self.fall_given |= fell
        fell_roles = torch.stack((fell[ar, server], fell[ar, receiver]), dim=1)
        role_vals["fall"] = w["fall"] * fell_roles.float()
        self.fell_server |= fell_roles[:, 0]

        # Crossing the net plane after the serve, without a net fault.
        x_c = s.shuttle_pos[:, 0] * (1 - 2 * server).float()
        self.crossed |= server_hits & (x_c > 0.0) & ~s.net_fault

        return_now = s.just_hit[ar, receiver] & server_hits & ~self.result_given
        self.return_done = return_now & p.end_on_return
        success = ((s.just_resolved & (s.winner == server) & server_hits) | return_now) & ~self.result_given
        failure = s.just_resolved & (s.winner == receiver) & ~self.result_given & ~success
        self.result_given |= success | failure
        role_vals["serve_result"][:, 0] = w["serve_result"] * (success.float() - failure.float())

        # Back to per-side columns, events scaled to cancel the RewardManager's dt.
        for k in TERMS:
            rv = role_vals[k] * (ev if k in EVENT_TERMS else 1.0)
            out = torch.empty_like(rv)
            out[ar, server] = rv[:, 0]
            out[ar, receiver] = rv[:, 1]
            self.values[k] = out

        self.done_now = s.just_resolved | return_now
        self._stats(s, contact, score, due, success, return_now)

    # ------------------------------------------------------------------ bookkeeping
    def pos_acc_roles(self, server: torch.Tensor) -> torch.Tensor:
        ar = torch.arange(server.shape[0], device=self.device)
        return torch.stack((self.pos_acc[ar, server], self.pos_acc[ar, 1 - server]), dim=1)

    def _add_pos_acc(self, server: torch.Tensor, add_roles: torch.Tensor) -> None:
        ar = torch.arange(server.shape[0], device=self.device)
        self.pos_acc[ar, server] += add_roles[:, 0]
        self.pos_acc[ar, 1 - server] += add_roles[:, 1]

    def _stats(self, s, contact, score, due, success, return_now) -> None:
        ar = torch.arange(s.server.shape[0], device=self.device)
        done = self.done_now
        server_hits = s.hit_count[ar, s.server] > 0
        f = lambda m: m.float().sum()  # noqa: E731
        add = {
            "rallies": f(done),
            "legal": f(done & server_hits & ~s.double_fault),
            "in": f(done & success),
            "crossed": f(done & self.crossed),
            "height": f(done & s.height_fault & (s.winner == 1 - s.server)),
            "double": f(done & s.double_fault & (s.winner == 1 - s.server)),
            "no_hit": f(done & ~server_hits & ~s.height_fault),
            "net": f(done & s.net_fault & server_hits),
            "out": f(done & s.out_fault & server_hits),
            "fall": f(done & self.fell_server),
            "return_touch": f(return_now),
            "vn_sum": (self.c_vn * contact.float()).sum(),
            "el_sum": (self.c_el * contact.float()).sum(),
            "hd_abs_sum": (self.c_hd.abs() * contact.float()).sum(),
            "outcome_sum": (score * due.float()).sum(),
            "contacts": f(contact),
        }
        zero = torch.zeros((), device=self.device)
        self.iteration += torch.stack([add.get(k, zero) for k in STAT_KEYS])

    def count_timeouts(self, timed_out: torch.Tensor) -> None:
        """Rallies cut by the episode timeout, e.g. a shuttle resting on a robot. Call before ``reset``."""
        t = (timed_out & ~self.done_now).float().sum()
        self.iteration[STAT_KEYS.index("timeout")] += t
        self.iteration[STAT_KEYS.index("rallies")] += t

    def reset(self, env_ids: torch.Tensor) -> None:
        for t in (self.c_have, self.outcome_pending, self.no_hit_given, self.result_given, self.crossed,
                  self.fell_server, self.return_done):
            t[env_ids] = False
        self.fall_given[env_ids] = False
        self.pos_acc[env_ids] = 0.0
        self.c_vn[env_ids] = 0.0
        self.c_el[env_ids] = 0.0
        self.c_hd[env_ids] = 0.0

    @staticmethod
    def rates(counts) -> dict:
        """Python floats from the count vector (one device sync). ``vn_mean`` etc. are per contact."""
        c = dict(zip(STAT_KEYS, counts.tolist() if hasattr(counts, "tolist") else list(counts)))
        r = max(c["rallies"], 1.0)
        k = max(c["contacts"], 1.0)
        out = {key: c[key] / r for key in STAT_KEYS if key not in ("rallies", "vn_sum", "el_sum", "hd_abs_sum",
                                                                    "outcome_sum", "contacts")}
        out["fault"] = out["height"] + out["double"]
        out.update(vn_mean=c["vn_sum"] / k, el_mean=c["el_sum"] / k, hd_abs_mean=c["hd_abs_sum"] / k,
                   outcome_mean=c["outcome_sum"] / k, rallies=c["rallies"])
        return out

    def pop_iteration(self) -> dict:
        out = self.rates(self.iteration)
        self.iteration.zero_()
        return out


# ---------------------------------------------------------------------- manager terms
def serve_term(env, term: str, side: int) -> torch.Tensor:
    """RewardManager getter; use weight 1.0 (the stage weight is already applied)."""
    return env.serve_reward.values[term][:, side]


def serve_return_done(env) -> torch.Tensor:
    """M1 termination: the receiver touched a legal serve. The serve counts as a success."""
    return env.serve_reward.return_done
