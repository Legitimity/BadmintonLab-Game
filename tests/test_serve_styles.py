"""P3_SERVE_STYLES: land_target / net_clear / apex terms, apex prediction and the style presets (CPU only)."""

import math

import pytest
import torch

from _serve_pkg import load

_m = load("rules", "serve_reward", "styles")
rules, sr, styles = _m["rules"], _m["serve_reward"], _m["styles"]


def test_new_terms_are_positive_and_off_by_default():
    for t in ("land_target", "net_clear", "apex"):
        assert t in sr.TERMS and t in sr.POSITIVE_TERMS
        assert all(w[t] == 0.0 for w in sr.STAGE_WEIGHTS.values())


def test_land_target_has_gradient_far_from_target_and_needs_the_net():
    p = sr.ServeRewardParams(land_x_t=3.0, land_y_t=2.0)
    xy = torch.tensor([[3.0, 2.0], [3.0, 0.0], [1.5, -0.5], [3.0, 1.0]])
    nz = torch.full((4,), 2.0)
    s = sr.land_target_score(xy, p, nz)
    assert abs(float(s[0]) - 1.0) < 1e-6
    # 2 m and ~2.9 m away: the tail still ranks them (the old Gaussian was ~0 for both).
    assert float(s[1]) > float(s[2]) > 0.1
    assert float(s[3]) > float(s[1])
    # Signed y: the mirror landing is far from the target.
    assert float(sr.land_target_score(torch.tensor([[3.0, -2.0]]), p)[0]) < 0.2
    # Net ball (never crossed) or under the edge: 0.
    bad = sr.land_target_score(xy[:2], p, torch.tensor([-math.inf, 1.5]))
    assert torch.equal(bad, torch.zeros(2))


def test_land_target_abs_y_and_out_guard():
    p = sr.ServeRewardParams(land_x_t=3.0, land_y_t=2.0, land_abs_y=True)
    s = sr.land_target_score(torch.tensor([[3.0, -2.0], [3.0, 3.5]]), p)
    assert abs(float(s[0]) - 1.0) < 1e-6
    assert float(s[1]) < 0.1


def test_net_clear_score():
    p = sr.ServeRewardParams(net_clear_t=0.15, net_clear_sigma=0.25)
    nz = torch.tensor([1.70, 1.95, 1.50, -math.inf])
    s = sr.net_clear_score(nz, p)
    assert abs(float(s[0]) - 1.0) < 1e-5
    assert 0.0 < float(s[1]) < float(s[0])
    assert float(s[2]) == 0.0 and float(s[3]) == 0.0


def test_apex_score_sign():
    p = sr.ServeRewardParams(apex_sign=1.0, apex_scale=2.0)
    s = sr.apex_score(torch.tensor([2.0, 4.0, 5.0]), torch.tensor([1.0, 1.0, 1.0]), p)
    assert torch.allclose(s, torch.tensor([0.5, 1.0, 1.0]))
    p.apex_sign = -1.0
    assert float(sr.apex_score(torch.tensor([2.0]), torch.tensor([1.0]), p)[0]) == 0.0


def test_predict_flight_apex_vacuum():
    pos = torch.tensor([[-2.0, 0.0, 1.0], [-2.0, 0.0, 1.0]], dtype=torch.float64)
    vel = torch.tensor([[5.0, 0.0, 4.0], [5.0, 0.0, -1.0]], dtype=torch.float64)
    nz, land, t, zmax = rules.predict_flight(pos, vel, 0.0, 9.81, dt=0.001, max_steps=4000, return_apex=True)
    assert abs(float(zmax[0]) - (1.0 + 16.0 / (2 * 9.81))) < 5e-3
    assert abs(float(zmax[1]) - 1.0) < 1e-6
    nz2, land2, t2 = rules.predict_flight(pos, vel, 0.0, 9.81, dt=0.001, max_steps=4000, return_time=True)
    assert torch.equal(land, land2) and torch.equal(t, t2)


def test_outcome_land_mix_zero_drops_the_landing_factor():
    p = sr.ServeRewardParams(outcome_land_mix=0.0, net_margin=0.12)
    nz = torch.tensor([2.0, 2.0])
    xy = torch.tensor([[1.6, 0.0], [4.5, 0.0]])
    s = sr.outcome_score(nz, xy, p)
    assert torch.allclose(s, torch.ones(2))
    p.outcome_land_mix = 0.5
    s = sr.outcome_score(nz, xy, p)
    assert float(s[0]) > float(s[1])


@pytest.mark.parametrize("name", sorted(styles.STYLES))
def test_styles_use_known_params_and_terms(name):
    sp, sw = styles.style_config(name)
    p = sr.ServeRewardParams()
    for k in sp:
        assert hasattr(p, k), k
    for k in sw:
        assert k in sr.STAGE_WEIGHTS["C"], k
    assert sw["face_quality"] == 0.0 and sw["launch_dir"] == 0.0 and sw["swing_speed"] == 0.0
    assert abs(sp["land_y_t"]) < sp["y_safe"]


def test_unknown_style():
    with pytest.raises(KeyError):
        styles.style_config("nope")


def test_p7_flat_long_presets_and_schedule():
    for name in ("l1_flat_long", "l2_flat_long_wide"):
        sp, sw = styles.style_config(name)
        assert sp["apex_sign"] == -1.0 and sp["v_close_max"] == 15.0
        assert sw["flight_time"] == 0.0 and sw["land_target"] > 0 and sw["apex"] > 0
        assert all(hasattr(sr.ServeRewardParams(), k) for k in sp)
    ts = styles.TargetSchedule("l1_flat_long")
    assert ts.current()[0] == 3.8
    # far from the target: no step; close and in: steps after min_iters
    for _ in range(300):
        assert not ts.update({"pred_in": 0.99, "land_x_mean": 2.5})
    for i in range(200):
        moved = ts.update({"pred_in": 0.99, "land_x_mean": 3.6})
        if moved:
            break
    assert ts.step == 1 and ts.current()[0] == 4.4
    # low in-rate blocks the next step
    for _ in range(400):
        ts.update({"pred_in": 0.80, "land_x_mean": 4.4})
    assert ts.step == 1
    st = ts.state_dict()
    ts2 = styles.TargetSchedule("l1_flat_long")
    ts2.load_state_dict(st)
    assert ts2.current() == ts.current()
    # Old form: with sign -1 an apex above the contact point (every serve over the net) always scored 0.
    z0 = torch.tensor([1.0, 1.0, 1.0])
    ap = torch.tensor([2.0, 2.6, 4.0])
    assert float(sr.apex_score(ap, z0, sr.ServeRewardParams(apex_sign=-1.0, apex_scale=1.5)).max()) == 0.0
    # apex_ref (P7): low arcs score, high arcs 0, monotone.
    sp, _ = styles.style_config("l1_flat_long")
    p = sr.ServeRewardParams(apex_sign=sp["apex_sign"], apex_scale=sp["apex_scale"], apex_ref=sp["apex_ref"])
    s = sr.apex_score(ap, z0, p)
    assert float(s[0]) == pytest.approx(1.0) and 0.0 < float(s[1]) < 1.0 and float(s[2]) == 0.0
    # default (apex_ref < 0) unchanged for the high-arc styles
    assert float(sr.apex_score(torch.tensor([3.5]), torch.tensor([1.0]), sr.ServeRewardParams(apex_sign=1.0, apex_scale=2.5))[0]) == pytest.approx(1.0)
