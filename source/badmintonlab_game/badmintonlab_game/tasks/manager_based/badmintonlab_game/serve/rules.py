"""Pure-torch port of the ``play`` branch match rules.

This module has no Isaac dependency so it can be differential-tested on the CPU
against the real ``play`` source (see ``tests/test_rules_vs_play.py``). The logic
and statement order mirror ``play/dual_agent_env.py`` ``_update_hits`` and
``_update_rally_result``; do not "simplify" them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CourtGeometry:
    """The subset of ``BadmintonCourtCfg`` used by the rules."""

    net_x: float = 0.0
    net_center_height: float = 1.524
    net_edge_height: float = 1.55
    doubles_half_width: float = 3.05
    singles_half_width: float = 2.59
    half_length: float = 6.7
    use_singles_bounds: bool = True

    @classmethod
    def from_cfg(cls, cfg) -> "CourtGeometry":
        return cls(
            net_x=cfg.net_x,
            net_center_height=cfg.net_center_height,
            net_edge_height=cfg.net_edge_height,
            doubles_half_width=cfg.doubles_half_width,
            singles_half_width=cfg.singles_half_width,
            half_length=cfg.half_length,
            use_singles_bounds=cfg.use_singles_bounds,
        )


def net_height_at_y(y: torch.Tensor, court: CourtGeometry) -> torch.Tensor:
    ratio = torch.clamp(y.abs() / max(court.doubles_half_width, 1e-6), max=1.0)
    return court.net_center_height + (court.net_edge_height - court.net_center_height) * ratio


def is_out_of_bounds(pos: torch.Tensor, court: CourtGeometry) -> torch.Tensor:
    half_w = court.singles_half_width if court.use_singles_bounds else court.doubles_half_width
    return (pos[:, 0].abs() > court.half_length) | (pos[:, 1].abs() > half_w)


def is_net_fault(prev: torch.Tensor, curr: torch.Tensor, court: CourtGeometry) -> torch.Tensor:
    """``play`` crossing test: includes a shuttle that lands exactly on the net plane."""
    dx_prev = prev[:, 0] - court.net_x
    dx_curr = curr[:, 0] - court.net_x
    crossed_x = ((dx_prev < 0.0) & (dx_curr >= 0.0)) | ((dx_prev > 0.0) & (dx_curr <= 0.0))
    dx = curr[:, 0] - prev[:, 0]
    safe_dx = torch.where(dx.abs() > 1e-8, dx, torch.ones_like(dx))
    t = torch.clamp((court.net_x - prev[:, 0]) / safe_dx, 0.0, 1.0)
    y_cross = prev[:, 1] + t * (curr[:, 1] - prev[:, 1])
    z_cross = prev[:, 2] + t * (curr[:, 2] - prev[:, 2])
    within = (prev[:, 1] + t * (curr[:, 1] - prev[:, 1])).abs() <= court.doubles_half_width
    return crossed_x & within & (z_cross < net_height_at_y(y_cross, court))


class MatchRules:
    """Per-physics-step match state. All tensors are mutated in place."""

    def __init__(self, num_envs: int, device, court: CourtGeometry, max_serve_height: float = 1.15):
        n = num_envs
        self.court = court
        self.max_serve_height = max_serve_height
        self.just_hit = torch.zeros((n, 2), dtype=torch.bool, device=device)
        self.next_hitter = torch.zeros(n, dtype=torch.long, device=device)
        self.last_hitter = torch.full((n,), -1, dtype=torch.long, device=device)
        self.rally_resolved = torch.zeros(n, dtype=torch.bool, device=device)
        self.rally_winner = torch.full((n,), -1, dtype=torch.long, device=device)
        self.in_contact = torch.zeros((n, 2), dtype=torch.bool, device=device)
        self.serve_height_fault = torch.zeros(n, dtype=torch.bool, device=device)
        self.double_hit_fault = torch.zeros(n, dtype=torch.bool, device=device)
        self.shuttle_landed = torch.zeros(n, dtype=torch.bool, device=device)
        self.prev_pos = torch.zeros((n, 3), device=device)
        self.curr_pos = torch.zeros((n, 3), device=device)
        # Events accumulated over one control step, for rewards and metrics.
        self.step_net_fault = torch.zeros(n, dtype=torch.bool, device=device)
        self.step_out_fault = torch.zeros(n, dtype=torch.bool, device=device)
        self.step_just_resolved = torch.zeros(n, dtype=torch.bool, device=device)
        self.hit_height = torch.zeros((n, 2), device=device)

    def begin_control_step(self) -> None:
        self.just_hit.zero_()
        self.step_net_fault.zero_()
        self.step_out_fault.zero_()
        self.step_just_resolved.zero_()

    def detect_landing(self, local_z: torch.Tensor, threshold: float) -> None:
        self.shuttle_landed |= local_z <= threshold

    def update_hits(self, contacts: torch.Tensor, height: torch.Tensor) -> None:
        """``contacts``: (N, 2) bool racket-shuttle contact in this physics step."""
        was_resolved = self.rally_resolved.clone()
        new_contacts = contacts & ~self.in_contact
        self.in_contact.copy_(contacts)
        for side in range(2):
            hit = new_contacts[:, side] & ~self.rally_resolved
            wrong_turn = hit & (self.next_hitter != side)
            high_serve = hit & (self.last_hitter < 0) & (height > self.max_serve_height)
            self.double_hit_fault |= wrong_turn
            self.serve_height_fault |= high_serve
            self.rally_winner[wrong_turn | high_serve] = 1 - side
            self.rally_resolved |= wrong_turn | high_serve
            valid = hit & ~self.rally_resolved
            self.just_hit[:, side] |= valid
            self.hit_height[valid, side] = height[valid]
            self.next_hitter[valid] = 1 - side
            self.last_hitter[valid] = side
        self.step_just_resolved |= self.rally_resolved & ~was_resolved

    def update_rally_result(self, local_pos_world: torch.Tensor, env_origins: torch.Tensor) -> None:
        was_resolved = self.rally_resolved.clone()
        self.prev_pos.copy_(self.curr_pos)
        self.curr_pos.copy_(local_pos_world)
        local = self.curr_pos - env_origins
        prev_local = self.prev_pos - env_origins
        net_fault = is_net_fault(prev_local, local, self.court)
        out_fault = is_out_of_bounds(local, self.court) & (self.last_hitter >= 0)
        active = ~self.rally_resolved
        shot_fault = active & (net_fault | out_fault)
        hitter = torch.where(self.last_hitter >= 0, self.last_hitter, self.next_hitter)
        self.rally_winner[shot_fault] = 1 - hitter[shot_fault]
        landed = active & self.shuttle_landed & ~shot_fault
        landing_side = (local[:, 0] >= self.court.net_x).long()
        self.rally_winner[landed] = 1 - landing_side[landed]
        self.rally_resolved |= net_fault | out_fault | self.shuttle_landed
        self.step_net_fault |= shot_fault & net_fault
        self.step_out_fault |= shot_fault & out_fault
        self.step_just_resolved |= self.rally_resolved & ~was_resolved

    def reset(self, env_ids: torch.Tensor, opening_hitter: torch.Tensor, shuttle_pos_w: torch.Tensor) -> None:
        self.just_hit[env_ids] = False
        self.next_hitter[env_ids] = opening_hitter
        self.last_hitter[env_ids] = -1
        self.rally_resolved[env_ids] = False
        self.rally_winner[env_ids] = -1
        self.shuttle_landed[env_ids] = False
        self.in_contact[env_ids] = False
        self.serve_height_fault[env_ids] = False
        self.double_hit_fault[env_ids] = False
        self.prev_pos[env_ids] = shuttle_pos_w
        self.curr_pos[env_ids] = shuttle_pos_w
        self.hit_height[env_ids] = 0.0


def predict_flight(
    pos: torch.Tensor,
    vel: torch.Tensor,
    drag_over_mass: float,
    gravity: float = 9.81,
    dt: float = 0.01,
    max_steps: int = 400,
    net_x: float = 0.0,
    floor: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorised flight prediction in the canonical frame (the striker is at x<0).

    Returns ``(net_z, landing_xy)``. ``net_z`` is ``-inf`` if the shuttle never crosses
    the net plane going forward. Same drag law as ``_apply_shuttle_forces``.
    """
    p, v = pos.clone(), vel.clone()
    net_z = torch.full((p.shape[0],), -torch.inf, device=p.device, dtype=p.dtype)
    active = torch.ones(p.shape[0], dtype=torch.bool, device=p.device)
    g = torch.tensor((0.0, 0.0, -gravity), device=p.device, dtype=p.dtype)
    for _ in range(max_steps):
        if not active.any():
            break
        # Midpoint rule: at dt=0.01 this stays within ~1 cm of a fine integration.
        speed = torch.linalg.norm(v, dim=-1, keepdim=True)
        vm = v + 0.5 * dt * (g - drag_over_mass * speed * v)
        speed_m = torch.linalg.norm(vm, dim=-1, keepdim=True)
        vn = v + dt * (g - drag_over_mass * speed_m * vm)
        pn = torch.where(active[:, None], p + dt * vm, p)
        v = torch.where(active[:, None], vn, v)
        crossing = active & (p[:, 0] < net_x) & (pn[:, 0] >= net_x)
        if crossing.any():
            a = (net_x - p[crossing, 0]) / (pn[crossing, 0] - p[crossing, 0])
            net_z[crossing] = p[crossing, 2] + a * (pn[crossing, 2] - p[crossing, 2])
        p = pn
        active = active & (p[:, 2] > floor)
    return net_z, p[:, :2]
