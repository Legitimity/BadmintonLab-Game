"""P5 receive rewards and metrics (P5_RECEIVE_DESIGN 3). Pure torch, no Isaac import.

Only the **receiver** (``1 - server``) is scored, around its first legal hit. The structure follows the
serve: contact features captured at the physics step (``capture_contact``), pass-through detection
(``capture_tunnel``), one ``end_step`` per control step that fills ``values`` (already weighted; events
scaled by ``1/step_dt`` like the serve terms). Scoring functions are the serve ones
(``swing_score``, the elevation part of ``face_quality``, ``outcome_pred_score``, ``launch_score``) with the
receive targets; the face heading and the launch heading use wider receive-only factors (pilot A).

The incoming ball is announced by the environment with ``set_incoming``: the feeder launch, or the step
after a natural serve. Targets are in the server's (feeder's) canonical frame (``feeder.RECV_HOME``).

The receiver's fall is scored by the serve computer's ``fall`` term (both roles); this module only asks
for the termination (``fall_done``). ``recv_result`` is outside the positive cap, like ``serve_result``.

V2 (P5_RECEIVE_DESIGN_V2 4): stages R0 (footwork and turn: ``recv_approach``, ``recv_station``,
``recv_torso_face``, ``recv_racket_near``), R1 (touch), R2 (return); ``recv_face``'s heading tolerance is set
from outside (``set_face_tolerance``). ``pop_iteration`` adds the R0 gate statistics at the arrival time
(``st_err_p50`` m, ``tf_err_p50`` deg; ``tf_lit_p50`` = the doc's literal torso angle, logged only).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import torch

from .feeder import RACKET_BEARING, RECV_HOME, ROOT_FROM_RACKET
from .rules import predict_flight
from .serve_reward import (
    face_elevation_quality,
    ServeRewardParams,
    contact_features,
    launch_score,
    outcome_pred_score,
    quat_rotate_x_axis,
    swing_score,
    to_canonical,
)

RECV_TERMS = (
    "recv_approach",
    "recv_station",
    "recv_torso_face",
    "recv_racket_near",
    "recv_contact",
    "recv_swing",
    "recv_face",
    "recv_outcome_pred",
    "recv_launch_dir",
    "recv_result",
    "recv_miss",
    "recv_tunnel",
    "recv_post_stand",
)
RECV_POSITIVE = ("recv_approach", "recv_station", "recv_torso_face", "recv_racket_near", "recv_contact", "recv_swing",
                 "recv_face", "recv_outcome_pred", "recv_launch_dir")
RECV_DENSE = ("recv_approach", "recv_station", "recv_torso_face", "recv_racket_near", "recv_post_stand")
RECV_EVENTS = tuple(t for t in RECV_TERMS if t not in RECV_DENSE)

# V2 4.2. recv_post_stand (receiver upright after its hit) is a posture term: reward group G_body, not G_recv,
# constant over the stages.
RECV_STAGES = ("R0", "R1", "R2")
RECV_STAGE_WEIGHTS = {
    "R0": dict(recv_approach=2.0, recv_station=1.0, recv_torso_face=1.0, recv_racket_near=0.5, recv_contact=0.0,
               recv_face=0.0, recv_swing=0.0, recv_outcome_pred=0.0, recv_launch_dir=0.0, recv_result=0.0,
               recv_miss=-1.0, recv_tunnel=0.0, recv_post_stand=0.5),
    "R1": dict(recv_approach=1.0, recv_station=0.5, recv_torso_face=0.5, recv_racket_near=1.0, recv_contact=2.0,
               recv_face=0.5, recv_swing=0.3, recv_outcome_pred=1.0, recv_launch_dir=0.5, recv_result=5.0,
               recv_miss=-2.0, recv_tunnel=-1.0, recv_post_stand=0.5),
    "R2": dict(recv_approach=0.3, recv_station=0.2, recv_torso_face=0.2, recv_racket_near=0.5, recv_contact=1.0,
               recv_face=1.0, recv_swing=0.5, recv_outcome_pred=4.0, recv_launch_dir=1.0, recv_result=10.0,
               recv_miss=-3.0, recv_tunnel=-2.0, recv_post_stand=0.5),
}
HD_LOG_DEG = 25.0  # hd25_share is always "|heading| < 25 deg", whatever the annealed tolerance

# Per-rally receive statistics; prefix f_ = feeder rallies, n_ = natural rallies (only rallies with a legal
# incoming ball count).
_RALLY_KEYS = ("rallies", "touch", "crossed", "in", "fell", "tunnel", "miss", "timeout", "need_sum")
RECV_STAT_KEYS = tuple(f"{t}_{k}" for t in ("f", "n") for k in _RALLY_KEYS) + (
    "contacts", "vc_sum", "el_sum", "outcome_sum", "outcome_n", "approach_sum", "approach_n", "launch_el_sum")


@dataclass
class ReceiveParams:
    approach_lo: float = -0.3  # m/s, clamp of the root speed towards the stand point
    approach_hi: float = 1.0
    approach_scale: float = 1.0
    approach_min_need: float = 0.3  # only rallies with d_need above this
    # Stand point = target + ROOT_FROM_RACKET: the root position that puts the resting racket on the target
    # (the doc's "target moved back 0.6 m towards the home" assumed a racket in front of the root).
    arrive_radius: float = 0.15  # inside: full approach score (no sign flip on arrival)
    near_sigma: float = 0.15
    # Pilot: with sigma 0.15 alone the term is ~0 at the P3 receiver's typical 0.9 m miss distance. A wide kernel
    # (share ``near_wide_share``) gives a gradient from ~1 m: near = (1-s) exp(-(d/0.15)^2) + s exp(-(d/0.6)^2).
    near_sigma_wide: float = 0.6
    near_wide_share: float = 0.5
    near_time: float = 0.4  # racket_near once the ball is less than this from the target
    elevation_target_deg: float = 52.0
    x_target: float = 2.0
    # A' (pilot A + drag model): from the measured contact point ~2.6 m behind the net a return needs >= 8 m/s
    # launch speed to clear the net and land in; doc 1.2's "slow swing (6.5 / 8.0) suffices" assumed 1.5 m.
    v_close_target: float = 7.5  # swing_score of the return only (the serve keeps its own values)
    v_close_max: float = 9.0
    # Horizontal heading of the striking face for recv_face (receiver frame, 0 = towards the net): full score
    # within heading_flat_deg, Gaussian of width heading_sigma_deg outside. Replaces face_quality's 15 deg heading
    # Gaussian, which scored ~0 at the pilot's 50-75 deg and gave no gradient.
    # V2: heading_flat_deg is the annealed tolerance (R1 90 -> 45, R2 45 -> 25), set by the curriculum.
    heading_flat_deg: float = 25.0
    heading_sigma_deg: float = 25.0
    # V2 recv_station: exp(-d^2 / station_sigma^2), d = |root - stand point|, from near_time before the arrival.
    station_sigma: float = 0.3
    # V2 recv_torso_face: exp(-(theta / torso_sigma_deg)^2). "racket": theta is the yaw error w.r.t. the yaw that
    # points the resting racket bearing (feeder.RACKET_BEARING, -36.5 deg) at the intercept target, so the P3 pose
    # at the stand point scores 1 (consistent with the stand point, which assumes no turn). "literal": the torso's
    # forward axis against the root -> target direction (the doc's wording); ~0.4 at the stand point.
    torso_sigma_deg: float = 40.0
    torso_face_mode: str = "racket"
    torso_min_dist: float = 0.2  # closer than this to the target: theta = 0 (bearing undefined)
    positive_cap: float = 8.0
    terminate_on_fall: bool = True
    flight_dt: float = 0.02
    initial_stage: str = "R0"
    # Pilot A: launch_score ignores the heading and P3's swing sends the return sideways (vy ~ +3, vx ~ 1 m/s,
    # never over the net) at near-full launch_dir. For the return it is scaled by cos(heading)^power.
    launch_heading_power: float = 2.0


@dataclass
class ReceiveStepInputs:
    server: torch.Tensor  # (N,) long; the feeder in feeder rallies
    just_hit: torch.Tensor  # (N, 2) bool
    just_resolved: torch.Tensor  # (N,) bool
    winner: torch.Tensor  # (N,) long
    net_fault: torch.Tensor  # (N,) bool, this step
    shuttle_pos: torch.Tensor  # (N, 3) court-local
    shuttle_vel: torch.Tensor  # (N, 3) world
    root_pos: torch.Tensor  # (N, 2, 3) court-local
    root_vel: torch.Tensor  # (N, 2, 3) world
    racket_pos: torch.Tensor  # (N, 2, 3) court-local sweet spot
    root_z: torch.Tensor  # (N, 2)
    gravity_z: torch.Tensor  # (N, 2)
    fall_given: torch.Tensor  # (N, 2) bool, latched falls (serve computer)
    time_s: torch.Tensor  # (N,) episode time after this step
    natural: torch.Tensor  # (N,) bool
    root_yaw: torch.Tensor | None = None  # (N, 2) world yaw of the root (= trunk); None: torso terms off


class ReceiveRewardComputer:
    def __init__(self, num_envs: int, device, params: ReceiveParams, serve_params: ServeRewardParams, step_dt: float):
        n = num_envs
        self.p = params
        self.device = device
        self.step_dt = float(step_dt)
        # The serve scoring with the receive targets (face 52 deg, landing 2.0 m behind the net).
        self.sp = dataclasses.replace(serve_params, elevation_target_deg=params.elevation_target_deg,
                                      x_target=params.x_target, v_close_target=params.v_close_target,
                                      v_close_max=params.v_close_max)
        self.weights = {k: 0.0 for k in RECV_TERMS}
        self.set_weights(RECV_STAGE_WEIGHTS[params.initial_stage])
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=device)  # noqa: E731
        self.values = {k: z(n, 2) for k in RECV_TERMS}
        self.home = torch.tensor(RECV_HOME, device=device)
        self.root_from_racket = torch.tensor(ROOT_FROM_RACKET, device=device)
        # Incoming ball (server frame), set by the environment.
        self.active = z(n, dt=torch.bool)
        self.tgt = z(n, 3)
        self.t_arrive = z(n)  # episode time at the target
        self.d_need = z(n)
        self.level = torch.full((n,), -1.0, device=device)
        # Receiver contact and rally latches.
        self.c_have = z(n, dt=torch.bool)
        self.c_vc, self.c_el, self.c_hd = z(n), z(n), z(n)
        self.hit = z(n, dt=torch.bool)
        self.outcome_pending = z(n, dt=torch.bool)
        self.pos_acc = z(n)
        self.crossed = z(n, dt=torch.bool)
        self.result_given = z(n, dt=torch.bool)
        self.succeeded = z(n, dt=torch.bool)
        self.miss_given = z(n, dt=torch.bool)
        self.fell = z(n, dt=torch.bool)
        self.tunnel_given = z(n, dt=torch.bool)
        self.tunnel_now = z(n, dt=torch.bool)
        self.prev_face_pos, self.prev_face_n = z(n, 3), z(n, 3)
        self.prev_face_valid = z(n, dt=torch.bool)
        self.fall_done = z(n, dt=torch.bool)
        self.counted = z(n, dt=torch.bool)
        self.natural = torch.ones(n, dtype=torch.bool, device=device)
        self.arr_given = z(n, dt=torch.bool)
        self.iteration = torch.zeros(len(RECV_STAT_KEYS), device=device)
        self._contact_log: list[torch.Tensor] = []  # (k, 2): |heading|, v_close of receiver contacts
        self._arrive_log: list[torch.Tensor] = []  # (k, 3): stand error m, |torso error| deg, |literal error| deg

    def set_weights(self, weights: dict) -> None:
        unknown = set(weights) - set(RECV_TERMS)
        if unknown:
            raise KeyError(f"Unknown receive reward terms: {sorted(unknown)}")
        self.weights.update({k: float(v) for k, v in weights.items()})

    def set_face_tolerance(self, deg: float) -> None:
        """Flat part of the recv_face heading factor (V2 annealing)."""
        self.p.heading_flat_deg = float(deg)

    def torso_errors(self, yaw_c: torch.Tensor, root_xy: torch.Tensor, tgt_xy: torch.Tensor):
        """(racket-compensated, literal) torso yaw errors in degrees, server frame inputs."""
        d = tgt_xy - root_xy
        bearing = torch.atan2(d[:, 1], d[:, 0])
        far = torch.linalg.norm(d, dim=-1) >= self.p.torso_min_dist
        wrap = lambda a: torch.remainder(a + math.pi, 2.0 * math.pi) - math.pi  # noqa: E731
        lit = torch.where(far, wrap(yaw_c - bearing), torch.zeros_like(yaw_c))
        rac = torch.where(far, wrap(yaw_c - (bearing - RACKET_BEARING)), torch.zeros_like(yaw_c))
        return torch.rad2deg(rac), torch.rad2deg(lit)

    def face_score(self, elevation, heading):
        """Elevation part of ``face_quality`` (core + tail) times the wide heading factor."""
        off = torch.clamp(heading.abs() - self.p.heading_flat_deg, min=0.0) / self.p.heading_sigma_deg
        return face_elevation_quality(elevation, self.sp) * torch.exp(-off * off)

    # ------------------------------------------------------------------ incoming ball
    def set_incoming(self, mask: torch.Tensor, tgt: torch.Tensor, t_arrive: torch.Tensor, d_need: torch.Tensor,
                     level: torch.Tensor | None = None) -> None:
        """``mask`` (N,) rallies whose legal incoming ball starts now; the other arguments are (N, ...)."""
        m = mask & ~self.active
        self.active |= m
        self.tgt = torch.where(m[:, None], tgt, self.tgt)
        self.t_arrive = torch.where(m, t_arrive, self.t_arrive)
        self.d_need = torch.where(m, d_need, self.d_need)
        if level is not None:
            self.level = torch.where(m, level.float(), self.level)

    # ------------------------------------------------------------------ physics rate
    def capture_contact(self, new_hits: torch.Tensor, server: torch.Tensor, racket_quat_w: torch.Tensor,
                        racket_vel_w: torch.Tensor, shuttle_vel_in_w: torch.Tensor) -> None:
        """``shuttle_vel_in_w``: (N, 3) shuttle velocity of the physics step *before* the contact."""
        recv = (1 - server).unsqueeze(1)
        hit = new_hits.gather(1, recv).squeeze(1) & self.active & ~self.hit & ~self.c_have
        q = racket_quat_w.gather(1, recv[..., None].expand(-1, 1, 4)).squeeze(1)
        v = racket_vel_w.gather(1, recv[..., None].expand(-1, 1, 3)).squeeze(1)
        r = recv.squeeze(1)
        n_c, v_c, vin_c = to_canonical(quat_rotate_x_axis(q), r), to_canonical(v, r), to_canonical(shuttle_vel_in_w, r)
        vc, el, hd = contact_features(n_c, v_c, vin_c)
        self.c_vc = torch.where(hit, vc, self.c_vc)
        self.c_el = torch.where(hit, el, self.c_el)
        self.c_hd = torch.where(hit, hd, self.c_hd)
        self.c_have |= hit

    def capture_tunnel(self, server: torch.Tensor, racket_pos_w: torch.Tensor, racket_quat_w: torch.Tensor,
                       shuttle_pos_w: torch.Tensor, prev_shuttle_pos_w: torch.Tensor, contact: torch.Tensor) -> None:
        """Same test as ``ServeRewardComputer.capture_tunnel``, receiver's face, incoming ball, before its hit."""
        idx = (1 - server).unsqueeze(1)
        pos = racket_pos_w.gather(1, idx[..., None].expand(-1, 1, 3)).squeeze(1)
        n = quat_rotate_x_axis(racket_quat_w.gather(1, idx[..., None].expand(-1, 1, 4)).squeeze(1))
        d_now = ((shuttle_pos_w - pos) * n).sum(-1)
        d_prev_same = ((prev_shuttle_pos_w - pos) * n).sum(-1)
        d_prev_old = ((prev_shuttle_pos_w - self.prev_face_pos) * self.prev_face_n).sum(-1)
        d_prev = torch.where(self.prev_face_valid, d_prev_old, d_prev_same)
        self.prev_face_pos.copy_(pos)
        self.prev_face_n.copy_(n)
        self.prev_face_valid.fill_(True)
        rel = shuttle_pos_w - pos
        in_plane = torch.linalg.norm(rel - d_now.unsqueeze(-1) * n, dim=-1)
        crossed = (torch.sign(d_now) != torch.sign(d_prev)) & (d_prev != 0)
        touched = contact.gather(1, idx).squeeze(1)
        flag = crossed & (in_plane < self.sp.tunnel_face_radius) & ~touched & self.active & ~self.c_have & ~self.hit
        self.tunnel_now |= flag & ~self.tunnel_given

    # ------------------------------------------------------------------ control rate
    def end_step(self, s: ReceiveStepInputs) -> None:
        p, w, sp = self.p, self.weights, self.sp
        n = s.server.shape[0]
        ar = torch.arange(n, device=self.device)
        server, recv = s.server, 1 - s.server
        ev = 1.0 / self.step_dt
        vals = {k: torch.zeros(n, device=self.device) for k in RECV_TERMS}
        down = s.fall_given[ar, recv]
        pre = self.active & ~self.hit & ~down

        # Approach: root velocity towards the stand point (server frame), rallies that need a move.
        root_c = to_canonical(s.root_pos[ar, recv], server)
        vel_c = to_canonical(s.root_vel[ar, recv], server)
        stand = self.tgt[:, :2] + self.root_from_racket
        to_stand = stand - root_c[:, :2]
        dist = torch.linalg.norm(to_stand, dim=-1)
        toward = (vel_c[:, :2] * to_stand).sum(-1) / dist.clamp(min=1e-6)
        approach = torch.clamp(toward, p.approach_lo, p.approach_hi) / p.approach_scale
        approach = torch.where(dist < p.arrive_radius, torch.ones_like(approach), approach)
        do_approach = pre & (self.d_need > p.approach_min_need)
        vals["recv_approach"] = w["recv_approach"] * approach * do_approach.float()

        # V2: at the stand point from near_time before the arrival (rallies that need a move), torso turned to
        # the intercept target before the hit.
        near = pre & (self.t_arrive - s.time_s < p.near_time)
        vals["recv_station"] = w["recv_station"] * torch.exp(-(dist / p.station_sigma) ** 2) * (near & do_approach).float()
        if s.root_yaw is not None:
            yaw_c = s.root_yaw[ar, recv] + math.pi * server.float()
            th_rac, th_lit = self.torso_errors(yaw_c, root_c[:, :2], self.tgt[:, :2])
        else:
            th_rac = th_lit = torch.full((n,), 180.0, device=self.device)
        th = th_rac if p.torso_face_mode == "racket" else th_lit
        torso_ok = (pre & (s.root_yaw is not None)).float()
        vals["recv_torso_face"] = w["recv_torso_face"] * torch.exp(-(th / p.torso_sigma_deg) ** 2) * torso_ok

        # Racket near the ball once the ball is less than near_time from the target.
        d_rb = torch.linalg.norm(s.racket_pos[ar, recv] - s.shuttle_pos, dim=-1)
        kern = ((1.0 - p.near_wide_share) * torch.exp(-(d_rb / p.near_sigma) ** 2)
                + p.near_wide_share * torch.exp(-(d_rb / p.near_sigma_wide) ** 2))
        vals["recv_racket_near"] = w["recv_racket_near"] * kern * near.float()

        # First legal hit of the receiver.
        hit_now = s.just_hit[ar, recv] & self.active & ~self.hit
        # R0 gate statistics: stand error and torso errors at the arrival time (or at the hit, if earlier).
        arrive = (self.active & ~self.arr_given & (self.d_need > p.approach_min_need) & ~down
                  & ((s.time_s >= self.t_arrive) | hit_now))
        self.arr_given |= arrive
        # Unfiltered with a flag column: no host sync per step; filtered in pop_iteration.
        self._arrive_log.append(torch.stack((dist, th_rac.abs(), th_lit.abs(), arrive.float()), -1))
        contact = hit_now & self.c_have
        vals["recv_contact"] = w["recv_contact"] * contact.float()
        vals["recv_swing"] = w["recv_swing"] * contact.float() * swing_score(self.c_vc, sp)
        vals["recv_face"] = w["recv_face"] * contact.float() * self.face_score(self.c_el, self.c_hd)
        tunnel = self.tunnel_now & ~self.tunnel_given
        self.tunnel_given |= tunnel
        vals["recv_tunnel"] = w["recv_tunnel"] * tunnel.float()

        # Outcome prediction and launch direction of the return, the step after the hit.
        due = self.outcome_pending & ~(s.just_resolved & (s.winner != recv))
        score = torch.zeros(n, device=self.device)
        launch = torch.zeros(n, device=self.device)
        launch_el = torch.zeros(n, device=self.device)
        if bool(due.any()):
            ids = due.nonzero(as_tuple=False).squeeze(-1)
            pos_c = to_canonical(s.shuttle_pos[ids], recv[ids])
            v_c = to_canonical(s.shuttle_vel[ids], recv[ids])
            net_z, land = predict_flight(pos_c, v_c, sp.drag_over_mass, sp.gravity, dt=sp.flight_dt,
                                         max_steps=int(4.0 / sp.flight_dt))
            score[ids] = outcome_pred_score(net_z, land, sp)
            ls, launch_el[ids] = launch_score(v_c, sp)
            cos_h = (v_c[:, 0] / torch.linalg.norm(v_c[:, :2], dim=-1).clamp(min=1e-6)).clamp(min=0.0)
            launch[ids] = ls * cos_h ** p.launch_heading_power
        vals["recv_outcome_pred"] = w["recv_outcome_pred"] * score
        vals["recv_launch_dir"] = w["recv_launch_dir"] * launch
        self.outcome_pending = hit_now.clone()
        self.hit |= hit_now

        # Positive cap per rally on the process terms.
        pos = sum(vals[k].clamp(min=0.0) for k in RECV_POSITIVE)
        allowed = (p.positive_cap - self.pos_acc).clamp(min=0.0)
        scale = torch.where(pos > 0, torch.clamp(allowed / pos.clamp(min=1e-9), max=1.0), torch.ones_like(pos))
        for k in RECV_POSITIVE:
            vals[k] = torch.where(vals[k] > 0, vals[k] * scale, vals[k])
        self.pos_acc += pos * scale

        # After the hit: stand, crossing of the return, result.
        standing = (s.root_z[ar, recv] > sp.stand_height) & (s.gravity_z[ar, recv] < -sp.upright_cos) & ~down
        vals["recv_post_stand"] = w["recv_post_stand"] * (self.hit & standing).float()
        x_r = s.shuttle_pos[:, 0] * (1 - 2 * recv).float()
        self.crossed |= self.hit & (x_r > 0.0) & ~s.net_fault
        # Success: the return is touched by the server, or wins the rally; the receiver is up.
        touched_back = s.just_hit[ar, server] & self.hit & ~hit_now
        won = (s.just_resolved & (s.winner == recv) & self.hit) | touched_back
        success = won & ~down & ~self.result_given
        self.result_given |= won
        self.succeeded |= success
        vals["recv_result"] = w["recv_result"] * success.float()
        miss = s.just_resolved & (s.winner == server) & self.active & ~self.hit & ~self.miss_given & ~down
        self.miss_given |= miss
        vals["recv_miss"] = w["recv_miss"] * miss.float()

        fell_now = down & ~self.fell
        self.fell |= down
        self.fall_done = fell_now & ~s.just_resolved & p.terminate_on_fall

        for k in RECV_TERMS:
            rv = vals[k] * (ev if k in RECV_EVENTS else 1.0)
            out = torch.zeros(n, 2, device=self.device)
            out[ar, recv] = rv
            self.values[k] = out
        self._success_now = success
        self._step_stats(contact, score, due, launch_el, approach, do_approach)

    # ------------------------------------------------------------------ statistics
    def _step_stats(self, contact, score, due, launch_el, approach, do_approach) -> None:
        if bool(contact.any()):
            self._contact_log.append(torch.stack((self.c_hd[contact].abs(), self.c_vc[contact]), -1))
        add = {
            "contacts": contact.float().sum(),
            "vc_sum": (self.c_vc * contact.float()).sum(),
            "el_sum": (self.c_el * contact.float()).sum(),
            "outcome_sum": (score * due.float()).sum(),
            "outcome_n": due.float().sum(),
            "launch_el_sum": (launch_el * due.float()).sum(),
            "approach_sum": (approach * do_approach.float()).sum(),
            "approach_n": do_approach.float().sum(),
        }
        for k, v in add.items():
            self.iteration[RECV_STAT_KEYS.index(k)] += v

    def count_rallies(self, ended: torch.Tensor, timed_out: torch.Tensor) -> None:
        """Per-rally receive statistics; call once per control step after the terminations, before reset."""
        done = ended & self.active & ~self.counted
        self.counted |= done
        timeout = done & timed_out
        f = lambda m: m.float().sum()  # noqa: E731
        for prefix, kind in (("f", ~self.natural), ("n", self.natural)):
            d = done & kind
            add = {"rallies": f(d), "touch": f(d & self.hit), "crossed": f(d & self.crossed),
                   "in": f(d & self.succeeded), "fell": f(d & self.fell),
                   "tunnel": f(d & self.tunnel_given), "miss": f(d & self.miss_given), "timeout": f(timeout & kind),
                   "need_sum": (self.d_need * d.float()).sum()}
            for k, v in add.items():
                self.iteration[RECV_STAT_KEYS.index(f"{prefix}_{k}")] += v

    def reset(self, env_ids: torch.Tensor) -> None:
        for t in (self.active, self.c_have, self.hit, self.outcome_pending, self.crossed, self.result_given, self.succeeded,
                  self.miss_given, self.fell, self.tunnel_given, self.tunnel_now, self.prev_face_valid, self.fall_done,
                  self.counted, self.arr_given):
            t[env_ids] = False
        for t in (self.c_vc, self.c_el, self.c_hd, self.pos_acc, self.d_need, self.t_arrive):
            t[env_ids] = 0.0
        self.tgt[env_ids] = 0.0
        self.level[env_ids] = -1.0

    @staticmethod
    def rates(counts) -> dict:
        c = dict(zip(RECV_STAT_KEYS, counts.tolist() if hasattr(counts, "tolist") else list(counts)))
        out = {}
        for prefix in ("f", "n"):
            r = max(c[f"{prefix}_rallies"], 1.0)
            for k in _RALLY_KEYS:
                if k == "rallies":
                    out[f"{prefix}_rallies"] = c[f"{prefix}_rallies"]
                elif k == "need_sum":
                    out[f"{prefix}_need_mean"] = c[f"{prefix}_need_sum"] / r
                else:
                    out[f"{prefix}_{k}"] = c[f"{prefix}_{k}"] / r
        tot = max(c["f_rallies"] + c["n_rallies"], 1.0)
        for k in ("touch", "crossed", "in", "fell", "tunnel", "miss", "timeout"):
            out[k] = (c[f"f_{k}"] + c[f"n_{k}"]) / tot
        out["rallies"] = c["f_rallies"] + c["n_rallies"]
        k = max(c["contacts"], 1.0)
        out.update(vc_mean=c["vc_sum"] / k, el_mean=c["el_sum"] / k,
                   outcome_mean=c["outcome_sum"] / max(c["outcome_n"], 1.0),
                   launch_el_mean=c["launch_el_sum"] / max(c["outcome_n"], 1.0),
                   approach_mean=c["approach_sum"] / max(c["approach_n"], 1.0))
        return out

    def pop_iteration(self) -> dict:
        out = self.rates(self.iteration)
        self.iteration.zero_()
        if self._contact_log:
            c = torch.cat(self._contact_log)
            out.update(hd_abs_med=float(c[:, 0].median()), hd25_share=float((c[:, 0] < HD_LOG_DEG).float().mean()),
                       vc_med=float(c[:, 1].median()))
        else:
            out.update(hd_abs_med=float("nan"), hd25_share=float("nan"), vc_med=float("nan"))
        self._contact_log = []
        nan = float("nan")
        out.update(st_err_p50=nan, tf_err_p50=nan, tf_lit_p50=nan, arr_n=0.0)
        if self._arrive_log:
            a = torch.cat(self._arrive_log)
            a = a[a[:, 3] > 0]
            if a.shape[0] > 0:
                med = a[:, :3].median(dim=0).values.tolist()
                out.update(st_err_p50=med[0], tf_err_p50=med[1], tf_lit_p50=med[2], arr_n=float(a.shape[0]))
        self._arrive_log = []
        out["face_tol"] = float(self.p.heading_flat_deg)
        return out


REWARD_GROUPS = ("serve", "recv", "body")
# V2 2.1: the P3 C serve terms (RewardManager names, prefix serve_); everything that is neither these nor a recv_*
# term (fall, post_stand, hold_ready, the parent's default costs) and recv_post_stand is G_body.
SERVE_GROUP_TERMS = tuple(f"serve_{t}" for t in ("legal_contact", "swing_speed", "face_quality", "outcome_pred",
                                                  "launch_dir", "serve_result", "serve_fault", "no_hit", "tunnel"))
BODY_RECV_TERMS = ("recv_post_stand",)


def reward_group(term_name: str) -> int:
    """Index in ``REWARD_GROUPS`` of a RewardManager term."""
    if term_name in SERVE_GROUP_TERMS:
        return 0
    if term_name.startswith("recv_") and term_name not in BODY_RECV_TERMS:
        return 1
    return 2


def group_mask(server: torch.Tensor, natural: torch.Tensor) -> torch.Tensor:
    """(N, 2, 3) applicability per side: serve = server of a natural rally, recv = receiver (both rally types),
    body = every learned sample (all but the feeder)."""
    n = server.shape[0]
    side = torch.arange(2, device=server.device).expand(n, 2)
    is_srv = side == server.unsqueeze(1)
    nat = natural.unsqueeze(1)
    return torch.stack((is_srv & nat, ~is_srv, ~(is_srv & ~nat)), dim=-1).float()


def recv_term(env, term: str, side: int) -> torch.Tensor:
    """RewardManager getter; weight 1.0 (the stage weight is already applied)."""
    return env.recv_reward.values[term][:, side]


def recv_fall_done(env) -> torch.Tensor:
    """P5: the receiver fell (any rally type)."""
    return env.recv_reward.fall_done
