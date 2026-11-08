"""P3_FINETUNE_DT1MS: physics-step helpers, 2-D landing factor, flight time (CPU only, no Isaac)."""

import math

import torch

from _serve_pkg import load

_m = load("rules", "serve_reward")
rules, sr = _m["rules"], _m["serve_reward"]


def test_actuator_delay_steps_keep_10_to_40_ms():
    # booster.py imports Isaac Lab; the conversion rule is restated here and checked against play's numbers.
    def steps(dt):
        return round(0.010 / dt), round(0.040 / dt)

    assert steps(0.005) == (2, 8)
    assert steps(0.002) == (5, 20)
    assert steps(0.001) == (10, 40)
    for dt in (0.001, 0.002, 0.005):
        assert abs(round(0.02 / dt) * dt - 0.02) < 1e-12


def test_predict_flight_time_matches_vacuum_drop():
    pos = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    vel = torch.zeros(1, 3, dtype=torch.float64)
    _, _, t = rules.predict_flight(pos, vel, 0.0, 9.81, dt=0.001, max_steps=4000, return_time=True, check_every=7)
    assert abs(float(t[0]) - math.sqrt(2 * (1.0 - 0.03) / 9.81)) < 2e-3
    nz, land = rules.predict_flight(pos, vel, 0.0, 9.81, dt=0.001, max_steps=4000, check_every=7)
    assert land.shape == (1, 2) and not torch.isfinite(nz).any()


def test_landing_score_default_is_x_only():
    p = sr.ServeRewardParams()
    xy = torch.tensor([[1.6, 0.0], [1.6, 2.0], [3.1, 1.0]])
    s = sr.landing_score(xy, p)
    assert torch.allclose(s[:2], torch.ones(2))
    assert abs(float(s[2]) - math.exp(-1.0)) < 1e-6


def test_landing_score_two_dimensional_uses_abs_y():
    p = sr.ServeRewardParams(x_target=1.0, x_sigma=0.8, y_target=1.7, y_sigma=0.5)
    xy = torch.tensor([[1.0, 1.7], [1.0, -1.7], [1.0, 1.2], [1.8, 1.7]])
    s = sr.landing_score(xy, p)
    assert torch.allclose(s[:2], torch.ones(2))
    assert abs(float(s[2]) - math.exp(-1.0)) < 1e-6
    assert abs(float(s[3]) - math.exp(-1.0)) < 1e-6


def test_flight_time_score_and_predicted_in():
    p = sr.ServeRewardParams()
    t = torch.tensor([0.8, 1.0, 1.3, 1.6, 2.0])
    assert torch.allclose(sr.flight_time_score(t, p), torch.tensor([1.0, 1.0, 0.5, 0.0, 0.0]))
    nz = torch.tensor([2.0, 1.5, 2.0, 2.0, -math.inf])
    land = torch.tensor([[2.0, 0.0], [2.0, 0.0], [6.8, 0.0], [2.0, 2.7], [2.0, 0.0]])
    assert sr.predicted_in(nz, land, p).tolist() == [True, False, False, False, False]


def test_flight_time_term_is_positive_and_off_by_default():
    assert "flight_time" in sr.TERMS and "flight_time" in sr.POSITIVE_TERMS
    assert all(w["flight_time"] == 0.0 for w in sr.STAGE_WEIGHTS.values())
