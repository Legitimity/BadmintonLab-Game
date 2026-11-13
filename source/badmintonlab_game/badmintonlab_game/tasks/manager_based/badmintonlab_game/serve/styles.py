"""Serve style presets (P3_SERVE_STYLES.md): reward parameters and stage-C weights per style.

Canonical frame of the server: net at x = 0, the receiver's half at x > 0, signed y (the receiver's ready racket
is at y ~ -0.07, its body at y ~ -0.6). Each style is trained as its own policy from p3_final.
"""

from __future__ import annotations

# Common to every style: P3 scaffolding that locks the serve style is switched off (face_quality fixes the face
# heading and elevation, launch_dir the launch elevation, swing_speed rewards only faster swings); the x-landing
# factor of outcome_pred is dropped (outcome_pred keeps the net margin and the out guards).
COMMON_PARAMS = dict(
    v_close_max=12.0,
    outcome_land_mix=0.0,
    positive_cap=12.0,
    y_safe=2.25,
    net_margin=0.12,
)
COMMON_WEIGHTS = dict(
    face_quality=0.0,
    launch_dir=0.0,
    swing_speed=0.0,
    outcome_pred=2.0,
    flight_time=0.0,
    land_target=0.0,
    net_clear=0.0,
    apex=0.0,
)

STYLES = {
    # Flat and fast: low over the net, short flight, deep centre.
    "s1_flat": dict(
        params=dict(land_x_t=4.2, land_y_t=0.0, land_sigma=0.8, net_clear_t=0.15, net_clear_sigma=0.25,
                    flight_time_ref=1.0, flight_time_scale=0.4),
        weights=dict(land_target=3.0, net_clear=2.0, flight_time=3.0),
    ),
    # Mid-depth to the sideline on the receiver's racket (+y) side.
    "s2_wide_fh": dict(
        params=dict(land_x_t=3.0, land_y_t=2.0),
        weights=dict(land_target=5.0),
    ),
    # Mid-depth to the sideline on the receiver's body (-y) side.
    "s3_wide_bh": dict(
        params=dict(land_x_t=3.0, land_y_t=-2.0),
        weights=dict(land_target=5.0),
    ),
    # Short, just over the net, centre: the receiver has to run in about 2.3 m.
    "s4_short": dict(
        params=dict(land_x_t=0.9, land_y_t=0.0, net_clear_t=0.15, net_clear_sigma=0.3),
        weights=dict(land_target=5.0, net_clear=1.0),
    ),
    # Later: deep corner with a high arc (apex), short and wide.
    "s5_deep_corner": dict(
        params=dict(land_x_t=5.8, land_y_t=-1.8, apex_sign=1.0, apex_scale=2.5),
        weights=dict(land_target=5.0, apex=1.0),
    ),
    # P7 (11-13): flat long serves to the receiver's back court. Physics (play drag, serve point 2.55 m from the net,
    # contact <= 1.15 m): a landing at 4.4-5.6 m with apex < 2.5 m needs >= 16 m/s at ~24 deg. The closing-speed
    # cap is raised (1-2 ms physics tunnels only near 30 m/s); a low arc is rewarded by apex (sign -1), flight_time
    # stays off (s1_flat lesson). The landing target moves out by a schedule (TARGET_SCHEDULES), not at once.
    "l1_flat_long": dict(
        params=dict(land_x_t=5.0, land_y_t=0.0, land_sigma=0.6, land_tail=2.0, apex_sign=-1.0, apex_scale=1.2, apex_ref=3.2,
                    net_clear_t=0.3, net_clear_sigma=0.3, x_long_safe=6.2, v_close_max=15.0, v_close_penalty_scale=3.0),
        weights=dict(land_target=5.0, apex=2.0, net_clear=1.0),
    ),
    # Back court on the receiver's racket (+y) side; started from st_s2_wide_fh (already hits +y at 13 m/s).
    "l2_flat_long_wide": dict(
        params=dict(land_x_t=4.8, land_y_t=1.6, land_sigma=0.6, land_tail=2.0, apex_sign=-1.0, apex_scale=1.2, apex_ref=3.2,
                    net_clear_t=0.3, net_clear_sigma=0.3, x_long_safe=6.2, y_safe=2.25, v_close_max=15.0,
                    v_close_penalty_scale=3.0),
        weights=dict(land_target=5.0, apex=2.0, net_clear=1.0),
    ),
    "s6_short_wide": dict(
        params=dict(land_x_t=0.9, land_y_t=1.8, net_clear_t=0.15, net_clear_sigma=0.3),
        weights=dict(land_target=5.0, net_clear=1.0),
    ),
}


# P7 target schedules (Liu et al. 2025 sigma annealing, easy-to-hard targets): list of steps
# (land_x_t, land_sigma, apex weight). The runner moves to the next step when, over the last ``window``
# iterations, the training predicted-in rate >= ``gate_in`` and the mean landing x is within ``gate_dx`` of the
# current target (and at least ``min_iters`` iterations were spent in the step).
TARGET_SCHEDULES = {
    "l1_flat_long": dict(steps=[(3.8, 1.0, 1.0), (4.4, 0.8, 2.0), (5.0, 0.6, 2.0)], gate_in=0.95, gate_dx=0.6,
                         window=50, min_iters=150),
    "l2_flat_long_wide": dict(steps=[(3.6, 1.0, 1.0), (4.2, 0.8, 2.0), (4.8, 0.6, 2.0)], gate_in=0.95, gate_dx=0.6,
                              window=50, min_iters=150),
}


class TargetSchedule:
    """Moves ``land_x_t`` / ``land_sigma`` / the apex weight along a style's TARGET_SCHEDULES entry."""

    def __init__(self, name: str):
        cfg = TARGET_SCHEDULES[name]
        self.steps = cfg["steps"]
        self.gate_in, self.gate_dx = cfg["gate_in"], cfg["gate_dx"]
        self.window, self.min_iters = cfg["window"], cfg["min_iters"]
        self.step = 0
        self.in_step = 0
        self.hist: list[tuple[float, float]] = []
        self.events: list[str] = []

    def current(self) -> tuple[float, float, float]:
        return self.steps[self.step]

    def update(self, rates: dict) -> bool:
        """Feed one iteration's rates (pred_in, land_x_mean); True when the step changed."""
        self.events = []
        self.in_step += 1
        pi, lx = rates.get("pred_in"), rates.get("land_x_mean")
        if pi is None or lx is None or lx != lx:
            return False
        self.hist = (self.hist + [(float(pi), float(lx))])[-self.window:]
        if self.step + 1 >= len(self.steps) or self.in_step < self.min_iters or len(self.hist) < self.window:
            return False
        m_in = sum(h[0] for h in self.hist) / len(self.hist)
        m_x = sum(h[1] for h in self.hist) / len(self.hist)
        if m_in >= self.gate_in and abs(m_x - self.steps[self.step][0]) <= self.gate_dx:
            self.step += 1
            self.in_step = 0
            self.hist = []
            self.events.append(f"target_step_{self.step}")
            return True
        return False

    def state_dict(self) -> dict:
        return dict(step=self.step, in_step=self.in_step)

    def load_state_dict(self, st: dict) -> None:
        self.step, self.in_step = int(st.get("step", 0)), int(st.get("in_step", 0))


def style_config(name: str) -> tuple[dict, dict]:
    """(reward parameters, stage-C weight overrides) of a style."""
    if name not in STYLES:
        raise KeyError(f"unknown serve style {name!r}; known: {sorted(STYLES)}")
    s = STYLES[name]
    return {**COMMON_PARAMS, **s["params"]}, {**COMMON_WEIGHTS, **s["weights"]}
