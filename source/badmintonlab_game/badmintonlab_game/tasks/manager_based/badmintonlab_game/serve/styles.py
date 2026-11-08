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
    "s6_short_wide": dict(
        params=dict(land_x_t=0.9, land_y_t=1.8, net_clear_t=0.15, net_clear_sigma=0.3),
        weights=dict(land_target=5.0, net_clear=1.0),
    ),
}


def style_config(name: str) -> tuple[dict, dict]:
    """(reward parameters, stage-C weight overrides) of a style."""
    if name not in STYLES:
        raise KeyError(f"unknown serve style {name!r}; known: {sorted(STYLES)}")
    s = STYLES[name]
    return {**COMMON_PARAMS, **s["params"]}, {**COMMON_WEIGHTS, **s["weights"]}
