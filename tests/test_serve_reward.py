"""CPU tests for the P3 serve reward and curriculum. No Isaac import.

Loads ``serve_reward.py``, ``rules.py`` and ``curriculum.py`` by file into a fake package,
so the Isaac-dependent ``serve/__init__`` chain is never imported.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SERVE = ROOT / "source/badmintonlab_game/badmintonlab_game/tasks/manager_based/badmintonlab_game/serve"
DT = 0.02


def _load_pkg():
    pkg = types.ModuleType("_serve")
    pkg.__path__ = [str(SERVE)]
    sys.modules["_serve"] = pkg
    mods = {}
    for name in ("rules", "serve_reward", "curriculum"):
        spec = importlib.util.spec_from_file_location(f"_serve.{name}", SERVE / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"_serve.{name}"] = mod
        spec.loader.exec_module(mod)
        mods[name] = mod
    return mods


M = _load_pkg()
R, C, RULES = M["serve_reward"], M["curriculum"], M["rules"]
J = 23


def make(n=4, stage="B", **kw):
    p = R.ServeRewardParams(initial_stage=stage, **kw)
    hold = torch.ones(J)
    hold[:5] = 0.0
    return R.ServeRewardComputer(n, "cpu", p, torch.zeros(J), hold, DT)


def inputs(n=4, **over):
    d = dict(
        server=torch.zeros(n, dtype=torch.long),
        just_hit=torch.zeros(n, 2, dtype=torch.bool),
        hit_count=torch.zeros(n, 2, dtype=torch.long),
        just_resolved=torch.zeros(n, dtype=torch.bool),
        winner=torch.full((n,), -1, dtype=torch.long),
        height_fault=torch.zeros(n, dtype=torch.bool),
        double_fault=torch.zeros(n, dtype=torch.bool),
        net_fault=torch.zeros(n, dtype=torch.bool),
        out_fault=torch.zeros(n, dtype=torch.bool),
        shuttle_pos=torch.tensor([[-2.55, 0.0, 1.0]]).repeat(n, 1),
        shuttle_vel=torch.zeros(n, 3),
        root_z=torch.full((n, 2), 0.68),
        joint_pos=torch.zeros(n, 2, J),
    )
    d.update(over)
    return R.StepInputs(**d)


def quat_for_normal(elev_deg, head_deg):
    """Quaternion (w,x,y,z) whose body x axis has the given elevation and heading."""
    e, h = math.radians(elev_deg), math.radians(head_deg)
    # yaw h about z, then pitch -e about y.
    qz = torch.tensor([math.cos(h / 2), 0, 0, math.sin(h / 2)])
    qy = torch.tensor([math.cos(-e / 2), 0, math.sin(-e / 2), 0])
    w1, x1, y1, z1 = qz
    w2, x2, y2, z2 = qy
    return torch.tensor([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def ev(c, term, side=0):
    """Event value in reward units (undo the 1/dt scaling)."""
    return c.values[term][:, side] * DT


# ------------------------------------------------------------------ geometry helpers
def test_quat_x_axis_and_contact_features():
    q = quat_for_normal(55, 10)
    n = R.quat_rotate_x_axis(q[None])[0]
    assert math.degrees(math.asin(n[2])) == pytest.approx(55, abs=1e-4)
    assert math.degrees(math.atan2(n[1], n[0])) == pytest.approx(10, abs=1e-4)
    # The face moving along -n is the striking face: same speed, flipped normal.
    vn, el, hd = R.contact_features(n[None], (-6.0 * n)[None])
    assert vn.item() == pytest.approx(6.0, abs=1e-5)
    assert el.item() == pytest.approx(-55, abs=1e-3)
    vn, el, hd = R.contact_features(n[None], (6.0 * n)[None])
    assert el.item() == pytest.approx(55, abs=1e-3) and hd.item() == pytest.approx(10, abs=1e-3)


def test_canonical_frame_side1():
    v = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.equal(R.to_canonical(v, torch.tensor([1])), torch.tensor([[-1.0, -2.0, 3.0]]))
    assert torch.equal(R.to_canonical(v, torch.tensor([0])), v)


# ------------------------------------------------------------------ outcome prediction
def _launch(speed, elev_deg):
    e = math.radians(elev_deg)
    return torch.tensor([[-2.55, 0.0, 0.7]]), torch.tensor([[speed * math.cos(e), 0.0, speed * math.sin(e)]])


def test_outcome_score_orders_serves():
    p = R.ServeRewardParams()
    scores = {}
    for name, (spd, el) in {"short": (3.0, 50), "good": (10.0, 45), "long": (30.0, 25)}.items():
        pos, vel = _launch(spd, el)
        net_z, land = RULES.predict_flight(pos, vel, p.drag_over_mass, dt=p.flight_dt, max_steps=200)
        scores[name] = R.outcome_score(net_z, land, p).item()
        scores[name + "_x"] = land[0, 0].item()
    assert scores["short"] < 0.2, scores
    assert scores["good"] > 0.6, scores
    assert 0 < scores["good_x"] < 6.7, scores
    # A long serve is predicted out and must score much less than a good one (guard added to V3).
    assert scores["long_x"] > 6.7 and scores["long"] < 0.1, scores


def test_outcome_backward_ball_scores_zero():
    p = R.ServeRewardParams()
    net_z, land = RULES.predict_flight(torch.tensor([[-2.55, 0, 1.0]]), torch.tensor([[-3.0, 0, 2.0]]), p.drag_over_mass)
    assert R.outcome_score(net_z, land, p).item() == 0.0


def test_flight_dt_matches_fine_integration():
    p = R.ServeRewardParams()
    pos, vel = _launch(10.0, 45)
    nz1, l1 = RULES.predict_flight(pos, vel, p.drag_over_mass, dt=p.flight_dt, max_steps=400)
    nz2, l2 = RULES.predict_flight(pos, vel, p.drag_over_mass, dt=0.001, max_steps=8000)
    assert abs(nz1 - nz2).item() < 0.03 and abs(l1 - l2).max().item() < 0.08


# ------------------------------------------------------------------ reward computer
def _hit(c, n, quat, vel, server=None):
    server = torch.zeros(n, dtype=torch.long) if server is None else server
    new = torch.zeros(n, 2, dtype=torch.bool)
    new[torch.arange(n), server] = True
    c.capture_contact(new, server, quat[None, None].repeat(n, 2, 1), vel[None, None].repeat(n, 2, 1))
    jh = torch.zeros(n, 2, dtype=torch.bool)
    jh[torch.arange(n), server] = True
    hc = jh.long()
    return jh, hc


def test_contact_terms_stage_b():
    n = 2
    c = make(n, "B", elevation_target_deg=55.0)
    q = quat_for_normal(55, 0)
    nrm = R.quat_rotate_x_axis(q[None])[0]
    jh, hc = _hit(c, n, q, 2.0 * nrm)
    c.end_step(inputs(n, just_hit=jh, hit_count=hc))
    # Closing speed = 2.0 + 4.3 * sin(55 deg) = 5.52; swing_score = 5.52 / 6.5.
    vc = 2.0 + 4.3 * math.sin(math.radians(55))
    assert c.c_vc[0].item() == pytest.approx(vc, abs=1e-4)
    assert ev(c, "legal_contact").tolist() == pytest.approx([1.0, 1.0])
    assert ev(c, "swing_speed").tolist() == pytest.approx([1.5 * vc / 6.5] * 2, abs=1e-4)
    assert ev(c, "face_quality").tolist() == pytest.approx([1.0, 1.0], abs=1e-4)
    assert ev(c, "legal_contact", 1).abs().sum() == 0  # receiver gets nothing


def test_swing_score_plateau_and_overspeed():
    p = R.ServeRewardParams()
    v = torch.tensor([0.0, 3.25, 6.5, 8.0, 9.0, 10.0, 20.0])
    s_ = R.swing_score(v, p)
    assert s_.tolist() == pytest.approx([0.0, 0.5, 1.0, 1.0, 0.5, 0.0, -1.0], abs=1e-5)


def test_closing_speed_picks_the_face_the_shuttle_meets():
    # Static racket, shuttle falling at 4.3 m/s onto a 50 deg face: v_close = 4.3 sin 50, elevation +50.
    n = torch.tensor([[math.cos(math.radians(50)), 0.0, math.sin(math.radians(50))]])
    vc, el, _ = R.contact_features(n, torch.zeros(1, 3), torch.tensor([[0.0, 0.0, -4.3]]))
    assert vc.item() == pytest.approx(4.3 * math.sin(math.radians(50)), abs=1e-5)
    assert el.item() == pytest.approx(50.0, abs=1e-3)
    vc, el, _ = R.contact_features(-n, torch.zeros(1, 3), torch.tensor([[0.0, 0.0, -4.3]]))
    assert el.item() == pytest.approx(50.0, abs=1e-3)


def test_outcome_one_step_later_and_cap():
    n = 1
    c = make(n, "B", positive_cap=5.0)
    q = quat_for_normal(55, 0)
    nrm = R.quat_rotate_x_axis(q[None])[0]
    jh, hc = _hit(c, n, q, 4.0 * nrm)  # v_close = 4 + 4.3 sin 55 = 7.5: inside the plateau
    c.end_step(inputs(n, just_hit=jh, hit_count=hc))
    first = sum(ev(c, k).item() for k in R.POSITIVE_TERMS)
    expected_first = 1.0 + 1.5 + ev(c, "face_quality").item()
    assert first == pytest.approx(expected_first, abs=1e-3)
    assert ev(c, "outcome_pred").item() == 0.0
    pos, vel = _launch(10.0, 45)
    c.end_step(inputs(n, hit_count=hc, shuttle_pos=pos, shuttle_vel=vel))
    # outcome_pred + launch_dir would be ~4 * 0.8 + 2, but only 5 - first is left under the cap of 5
    # (V2.1: the two share the cap, scaled by the same factor).
    second = ev(c, "outcome_pred").item() + ev(c, "launch_dir").item()
    assert second == pytest.approx(5.0 - first, abs=1e-4)
    assert ev(c, "launch_dir").item() > 0.0
    c.end_step(inputs(n, hit_count=hc, shuttle_pos=pos, shuttle_vel=vel))
    assert ev(c, "outcome_pred").item() == 0.0 and ev(c, "launch_dir").item() == 0.0


def test_side1_server_uses_canonical_frame():
    n = 1
    c = make(n, "B")
    server = torch.tensor([1])
    # World normal pointing to -x at the target elevation is "towards the net" for side 1.
    q = quat_for_normal(c.p.elevation_target_deg, 180)
    nrm = R.quat_rotate_x_axis(q[None])[0]
    jh, hc = _hit(c, n, q, 6.0 * nrm, server)
    c.end_step(inputs(n, server=server, just_hit=jh, hit_count=hc))
    assert ev(c, "face_quality", 1).item() == pytest.approx(1.0, abs=1e-4)
    pos, vel = _launch(10.0, 45)
    pos[:, :2] *= -1
    vel[:, :2] *= -1
    c.end_step(inputs(n, server=server, hit_count=hc, shuttle_pos=pos, shuttle_vel=vel))
    assert ev(c, "outcome_pred", 1).item() > 2.0


def test_fault_no_outcome_and_result():
    n = 1
    c = make(n, "B")
    q = quat_for_normal(55, 0)
    jh, hc = _hit(c, n, q, torch.zeros(3))
    c.end_step(inputs(n, just_hit=jh, hit_count=hc))
    pos, vel = _launch(10.0, 45)
    # Double hit next step: rally lost by the server, no outcome reward, fault and result penalties.
    c.end_step(inputs(n, hit_count=hc, just_resolved=torch.tensor([True]), winner=torch.tensor([1]),
                      double_fault=torch.tensor([True]), shuttle_pos=pos, shuttle_vel=vel))
    assert ev(c, "outcome_pred").item() == 0.0
    assert ev(c, "serve_fault").item() == pytest.approx(-5.0)
    assert ev(c, "serve_result").item() == pytest.approx(-10.0)
    stats = c.pop_iteration()
    assert stats["double"] == 1.0 and stats["legal"] == 0.0 and stats["in"] == 0.0


def test_height_fault_is_not_no_hit():
    n = 1
    c = make(n, "B")
    c.end_step(inputs(n, just_resolved=torch.tensor([True]), winner=torch.tensor([1]),
                      height_fault=torch.tensor([True]), shuttle_pos=torch.tensor([[-2.55, 0, 0.3]])))
    assert ev(c, "serve_fault").item() == -5.0 and ev(c, "no_hit").item() == 0.0
    assert c.pop_iteration()["height"] == 1.0


def test_no_hit_once_then_landing_loss():
    n = 1
    c = make(n, "A")
    low = torch.tensor([[-2.55, 0, 0.4]])
    c.end_step(inputs(n, shuttle_pos=low))
    assert ev(c, "no_hit").item() == -3.0
    c.end_step(inputs(n, shuttle_pos=low))
    assert ev(c, "no_hit").item() == 0.0
    c.end_step(inputs(n, shuttle_pos=low, just_resolved=torch.tensor([True]), winner=torch.tensor([1])))
    assert ev(c, "serve_result").item() == -10.0
    r = c.pop_iteration()
    assert r["no_hit"] == 1.0 and r["rallies"] == 1.0


def test_success_on_landing_and_on_return():
    n = 2
    c = make(n, "B")
    q = quat_for_normal(55, 0)
    jh, hc = _hit(c, n, q, torch.zeros(3))
    c.end_step(inputs(n, just_hit=jh, hit_count=hc))
    over = torch.tensor([[1.0, 0, 1.6], [1.0, 0, 1.6]])
    c.end_step(inputs(n, hit_count=hc, shuttle_pos=over))
    # env 0: lands in the receiver's court. env 1: the receiver touches it.
    rj = torch.tensor([[False, False], [False, True]])
    hc2 = hc.clone()
    hc2[1, 1] = 1
    c.end_step(inputs(n, just_hit=rj, hit_count=hc2, just_resolved=torch.tensor([True, False]),
                      winner=torch.tensor([0, -1]), shuttle_pos=over))
    assert ev(c, "serve_result").tolist() == pytest.approx([10.0, 10.0])
    assert c.return_done.tolist() == [False, True]
    r = c.pop_iteration()
    assert r["in"] == 1.0 and r["crossed"] == 1.0 and r["return_touch"] == 0.5
    # The return is not rewarded twice when the rally later resolves.
    c.end_step(inputs(n, hit_count=hc2, just_resolved=torch.tensor([False, True]), winner=torch.tensor([-1, 1])))
    assert ev(c, "serve_result").tolist() == [0.0, 0.0]


def test_hold_ready_masks_and_server_release():
    n = 1
    c = make(n, "A")
    jp = torch.zeros(n, 2, J)
    jp[:, :, :5] = 1.0  # swing joints: ignored for the server only
    c.end_step(inputs(n, joint_pos=jp))
    assert c.values["hold_ready"][0, 0].item() == 0.0
    assert c.values["hold_ready"][0, 1].item() < 0.0
    jp[:, :, 10] = 1.0
    c.end_step(inputs(n, joint_pos=jp))
    assert c.values["hold_ready"][0, 0].item() < 0.0
    hc = torch.tensor([[1, 0]])
    c.end_step(inputs(n, joint_pos=jp, hit_count=hc))
    assert c.values["hold_ready"][0, 0].item() == 0.0  # free after the hit


def test_fall_once_per_rally_both_sides_and_reset():
    n = 1
    c = make(n, "B")
    rz = torch.tensor([[0.3, 0.3]])
    c.end_step(inputs(n, root_z=rz))
    assert ev(c, "fall", 0).item() == -10.0 and ev(c, "fall", 1).item() == -10.0
    assert c.fall_done.tolist() == [True]
    c.end_step(inputs(n, root_z=rz))
    assert ev(c, "fall", 0).item() == 0.0
    c.reset(torch.tensor([0]))
    c.end_step(inputs(n, root_z=rz))
    assert ev(c, "fall", 0).item() == -10.0


def test_receiver_fall_does_not_end_the_serve():
    c = make(1, "B")
    c.end_step(inputs(1, root_z=torch.tensor([[0.68, 0.3]])))
    assert ev(c, "fall", 1).item() == -10.0 and c.fall_done.tolist() == [False]


def test_tilt_counts_as_fall():
    c = make(1, "B")
    # Trunk tilted ~60 deg: projected gravity z = -cos 60 = -0.5 > -0.70.
    c.end_step(inputs(1, gravity_z=torch.tensor([[-0.5, -1.0]])))
    assert ev(c, "fall", 0).item() == -10.0 and c.fall_done.tolist() == [True]


def test_dive_serve_does_not_pay_and_is_counted():
    """REPORT 5: every successful serve was followed by a fall. That must be a failed serve."""
    n = 1
    c = make(n, "B")
    q = quat_for_normal(55, 0)
    jh, hc = _hit(c, n, q, torch.zeros(3))
    c.end_step(inputs(n, just_hit=jh, hit_count=hc))
    # Falls while the shuttle is in flight: rally ends now, serve scored as a failure.
    c.end_step(inputs(n, hit_count=hc, root_z=torch.tensor([[0.3, 0.68]])))
    assert c.fall_done.tolist() == [True]
    assert ev(c, "serve_result").item() == -10.0
    r = c.pop_iteration()
    assert r["in"] == 0.0 and r["fall"] == 1.0 and r["fall_after_hit"] == 1.0 and r["rallies"] == 1.0


def test_win_while_down_is_not_success_when_not_terminating():
    n = 1
    c = make(n, "B", terminate_on_fall=False)
    q = quat_for_normal(55, 0)
    jh, hc = _hit(c, n, q, torch.zeros(3))
    c.end_step(inputs(n, just_hit=jh, hit_count=hc))
    c.end_step(inputs(n, hit_count=hc, root_z=torch.tensor([[0.3, 0.68]])))
    assert c.fall_done.tolist() == [False]
    c.end_step(inputs(n, hit_count=hc, root_z=torch.tensor([[0.3, 0.68]]), just_resolved=torch.tensor([True]),
                      winner=torch.tensor([0])))
    assert ev(c, "serve_result").item() == -10.0


def test_post_stand_dense_after_hit_only():
    n = 1
    c = make(n, "B")
    c.end_step(inputs(n))
    assert c.values["post_stand"][0, 0].item() == 0.0
    c.end_step(inputs(n, hit_count=torch.tensor([[1, 0]])))
    assert c.values["post_stand"][0, 0].item() == pytest.approx(1.0)  # per second, not scaled by 1/dt
    c.end_step(inputs(n, hit_count=torch.tensor([[1, 0]]), root_z=torch.tensor([[0.50, 0.68]])))
    assert c.values["post_stand"][0, 0].item() == 0.0


def test_tunnel_detection():
    n = 2
    c = make(n, "B")
    server = torch.zeros(n, dtype=torch.long)
    q = quat_for_normal(50, 0)[None, None].repeat(n, 2, 1)
    racket = torch.tensor([[-2.55, 0.0, 1.0]])[:, None].repeat(n, 2, 1)
    prev = torch.tensor([[-2.55, 0.0, 1.03], [-2.55, 0.30, 1.03]])  # env 1 is 30 cm to the side
    now = torch.tensor([[-2.55, 0.0, 0.99], [-2.55, 0.30, 0.99]])
    c.capture_tunnel(server, racket, q, now, prev, torch.zeros(n, 2, dtype=torch.bool))
    c.end_step(inputs(n))
    assert ev(c, "tunnel").tolist() == pytest.approx([-2.0, 0.0])
    # Once per rally, and a registered contact is not a tunnel.
    c.capture_tunnel(server, racket, q, now, prev, torch.zeros(n, 2, dtype=torch.bool))
    c.end_step(inputs(n))
    assert ev(c, "tunnel").tolist() == [0.0, 0.0]
    c.reset(torch.arange(n))
    c.capture_tunnel(server, racket, q, now, prev, torch.ones(n, 2, dtype=torch.bool))
    c.end_step(inputs(n))
    assert ev(c, "tunnel").tolist() == [0.0, 0.0]


def test_reward_manager_scaling():
    """RewardManager multiplies by weight*dt: events must come out as their table value."""
    n = 1
    c = make(n, "B")
    q = quat_for_normal(55, 0)
    jh, hc = _hit(c, n, q, torch.zeros(3))
    c.end_step(inputs(n, just_hit=jh, hit_count=hc))
    env = types.SimpleNamespace(serve_reward=c)
    assert (R.serve_term(env, "legal_contact", 0) * 1.0 * DT).item() == pytest.approx(1.0)


def test_timeout_counted_as_rally():
    n = 2
    c = make(n, "B")
    c.end_step(inputs(n))
    c.count_timeouts(torch.tensor([True, False]))
    r = c.pop_iteration()
    assert r["rallies"] == 1.0 and r["timeout"] == 1.0


def test_unknown_weight_rejected():
    c = make(1, "A")
    with pytest.raises(KeyError):
        c.set_weights({"nope": 1.0})


# ------------------------------------------------------------------ curriculum
def good(stage):
    return dict(rallies=500.0, fall=0.01, legal=0.95, crossed=0.9, **{"in": 0.95}, fault=0.0) if stage else {}  # noqa: E501


def test_curriculum_min_iters_promotion_blend_and_finish():
    cur = C.ServeCurriculum(ema_iters=10, blend_iters=4)
    rates = good(True)
    for it in range(199):
        cur.update(it, rates)
    assert cur.stage == "A"
    w = cur.update(199, rates)
    assert cur.stage == "B" and cur.events == ["passed_A"]
    # First blended iteration is still mostly stage A.
    assert w["hold_ready"] == pytest.approx(R.STAGE_WEIGHTS["A"]["hold_ready"])
    for i in range(4):
        w = cur.update(200 + i, rates)
    assert w == pytest.approx(R.STAGE_WEIGHTS["B"])
    for it in range(204, 2000):
        cur.update(it, rates)
        if cur.finished:
            break
    assert cur.finished and cur.stage == "C" and cur.stop_reason is None
    assert [h["stage"] for h in cur.history] == ["A", "B", "C"]


def test_curriculum_stops_at_max_without_promotion():
    cur = C.ServeCurriculum(ema_iters=10)
    bad = dict(rallies=500.0, fall=0.5, legal=0.2)
    for it in range(1500):
        cur.update(it, bad)
    assert cur.stage == "A" and cur.stop_reason == "stage_A_max_iters"


def test_curriculum_ignores_thin_iterations_and_roundtrips():
    cur = C.ServeCurriculum(ema_iters=10)
    cur.update(0, dict(rallies=10.0, fall=1.0, legal=0.0))
    assert cur.ema == {}
    cur.update(1, good(True))
    d = cur.state_dict()
    cur2 = C.ServeCurriculum()
    cur2.load_state_dict(d)
    assert cur2.state_dict() == d and cur2.stop_reason is None


def test_stage_tables():
    a, b, c = (R.STAGE_WEIGHTS[k] for k in "ABC")
    assert (a["hold_ready"], b["hold_ready"], c["hold_ready"]) == (-0.5, -0.2, -0.1)
    assert (a["outcome_pred"], b["outcome_pred"], c["outcome_pred"]) == (2.0, 4.0, 4.0)
    assert a["fall"] == b["fall"] == c["fall"] == -10.0
    # Positive shaping per serve must stay below the result: max of the event terms < serve_result.
    for w in (a, b, c):
        assert sum(w[k] for k in R.POSITIVE_TERMS) < w["serve_result"]
    assert (a["launch_dir"], b["launch_dir"], c["launch_dir"]) == (1.0, 2.0, 1.0)
    assert sum(b[k] for k in R.POSITIVE_TERMS) == pytest.approx(9.5)
    assert set(a) == set(b) == set(c) == set(R.TERMS)
    s = C.DEFAULT_STAGES
    assert (s["A"].min_iters, s["A"].max_iters, s["B"].max_iters, s["C"].max_iters) == (200, 1500, 4000, 4000)
    # P3_LAUNCH_F2 2.1: lowered for the pass-through cap (in <= ~0.77).
    assert s["B"].gate["crossed"] == (">=", 0.65)
    assert s["C"].gate["in"] == (">=", 0.75)


def test_apply_stage_gates_changes_thresholds_only():
    s = C.apply_stage_gates(C.DEFAULT_STAGES, "C.in=0.72, B.crossed=0.6,C.fall=0.04")
    assert s["C"].gate["in"] == (">=", 0.72) and s["C"].gate["fall"] == ("<=", 0.04)
    assert s["B"].gate["crossed"] == (">=", 0.6) and s["B"].gate["in"] == (">=", 0.60)
    assert C.DEFAULT_STAGES["C"].gate["in"] == (">=", 0.75)  # the defaults are not mutated
    assert s["C"].min_iters == C.DEFAULT_STAGES["C"].min_iters
    with pytest.raises(KeyError):
        C.apply_stage_gates(C.DEFAULT_STAGES, "C.legal=0.9")
    cur = C.ServeCurriculum(stage="C")
    cur.stages = s
    assert cur.stages["C"].gate["in"][1] == 0.72


def test_parse_stage_params_and_unknown_fields():
    assert C.parse_stage_params("C.v_close_max=7.0") == {"C": {"v_close_max": 7.0}}
    assert C.parse_stage_params("B.v_close_max=7.5,C.v_close_max=7") == {"B": {"v_close_max": 7.5}, "C": {"v_close_max": 7.0}}
    for bad in ("C.no_such_field=1", "D.v_close_max=7", "C.initial_stage=1"):
        with pytest.raises(KeyError):
            C.parse_stage_params(bad)


def _param_curriculum(**kw):
    cur = C.ServeCurriculum(ema_iters=10, blend_iters=4, **kw)
    cur.stage_params = C.parse_stage_params("C.v_close_max=7.0")
    cur.base_params = {"v_close_max": 8.0}
    return cur


def test_stage_param_switches_on_entering_c_with_blend():
    cur = _param_curriculum(stage="B")
    assert cur.params() == {"v_close_max": 8.0}
    rates = good(True)
    for it in range(400):
        cur.update(it, rates)
        if cur.stage == "C":
            break
    assert cur.stage == "C" and cur.blend_left == 4 and cur.events == ["passed_B"]
    # The promotion iteration itself is still the old value (f = 1), then linear like the weights.
    seen = [cur.params()["v_close_max"]]
    for i in range(4):
        cur.update(it + 1 + i, rates)
        seen.append(cur.params()["v_close_max"])
    assert seen == pytest.approx([8.0, 7.75, 7.5, 7.25, 7.0])
    f = cur.blend_left / cur.blend_iters
    assert f == 0.0
    # Weights and parameter blend with the same factor.
    cur2 = _param_curriculum(stage="B")
    for it in range(400):
        cur2.update(it, rates)
        if cur2.stage == "C":
            break
    cur2.update(it + 1, rates)
    f = cur2.blend_left / cur2.blend_iters
    assert cur2.params()["v_close_max"] == pytest.approx(f * 8.0 + (1 - f) * 7.0)
    w = cur2.weights()
    k = "hold_ready"
    assert w[k] == pytest.approx(f * R.STAGE_WEIGHTS["B"][k] + (1 - f) * R.STAGE_WEIGHTS["C"][k])


def test_stage_param_resume_in_c_is_immediate():
    src = _param_curriculum(stage="B")
    for it in range(400):
        src.update(it, good(True))
        if src.stage == "C":
            break
    d = src.state_dict()
    assert d["stage"] == "C" and d["blend_left"] == 4  # what model_passed_B_<it>.pt holds
    # Resumed mid-blend: continues the blend.
    cur = _param_curriculum()
    cur.load_state_dict(dict(d, blend_left=2))
    assert cur.params()["v_close_max"] == pytest.approx(7.5)
    # Resumed after the blend: the C value at once; without --stage_param the base value.
    cur = _param_curriculum()
    cur.load_state_dict(dict(d, blend_left=0))
    assert cur.params() == {"v_close_max": 7.0}
    plain = C.ServeCurriculum()
    plain.load_state_dict(dict(d, blend_left=0))
    assert plain.params() == {}


def _load_runner():
    spec = importlib.util.spec_from_file_location("_serve.runner", SERVE / "runner.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_runner_puts_stage_params_into_the_reward_computer():
    pytest.importorskip("rsl_rl")
    runner_mod = _load_runner()
    comp = make(2, "C")
    assert comp.p.v_close_max == 8.0
    cur = _param_curriculum()
    cur.load_state_dict(dict(_param_curriculum(stage="C").state_dict(), blend_left=0))
    fake = types.SimpleNamespace(env=types.SimpleNamespace(unwrapped=types.SimpleNamespace(serve_reward=comp)),
                                 curriculum=cur)
    runner_mod.ServeRunner.apply_stage_params(fake)
    assert comp.p.v_close_max == 7.0
    # The swing score follows the new value: 7.5 m/s is now over the limit.
    assert float(R.swing_score(torch.tensor([7.5]), comp.p)) < 1.0


def test_tunnel_detects_racket_sweeping_past_a_slow_shuttle():
    """The racket moves 5 cm through a shuttle that moves 1 cm: only the previous face plane sees it."""
    n = 1
    c = make(n, "B")
    server = torch.zeros(n, dtype=torch.long)
    q = quat_for_normal(90, 0)[None, None].repeat(n, 2, 1)  # face normal +z
    no_contact = torch.zeros(n, 2, dtype=torch.bool)
    shuttle_prev = torch.tensor([[-2.55, 0.0, 1.000]])
    shuttle_now = torch.tensor([[-2.55, 0.0, 0.990]])
    racket_prev = torch.tensor([[-2.55, 0.0, 0.980]])[:, None].repeat(n, 2, 1)  # 2 cm below the shuttle
    racket_now = torch.tensor([[-2.55, 0.0, 1.030]])[:, None].repeat(n, 2, 1)  # 4 cm above it
    c.capture_tunnel(server, racket_prev, q, shuttle_prev, shuttle_prev, no_contact)
    c.capture_tunnel(server, racket_now, q, shuttle_now, shuttle_prev, no_contact)
    c.end_step(inputs(n))
    assert ev(c, "tunnel").tolist() == pytest.approx([-2.0])


def _flight(pos, vel, p):
    return RULES.predict_flight(torch.tensor(pos), torch.tensor(vel), p.drag_over_mass, dt=p.flight_dt,
                                max_steps=int(4.0 / p.flight_dt))


# ------------------------------------------------------------------ V2.1 reward fix (P3_REWARD_FIX_AND_E1B)
def test_reach_backward_short_net_and_clear():
    p = R.ServeRewardParams()
    # Backward / dropped at the serve point: 0.
    for vel in ([-3.0, 0.0, 2.0], [0.0, 0.0, 0.0]):
        nz, land = _flight([[-2.55, 0.0, 1.0]], [vel], p)
        assert R.reach_score(nz, land, p).item() == 0.0
    # Short of the net: the further forward it lands, the higher.
    short = []
    for vx in (1.0, 2.0, 3.0):
        nz, land = _flight([[-2.55, 0.0, 1.0]], [[vx, 0.0, 1.0]], p)
        assert torch.isinf(nz).all() and land[0, 0] < 0
        short.append(R.reach_score(nz, land, p).item())
    assert 0 < short[0] < short[1] < short[2]
    # Over the net plane below the net: higher crossing, higher reach; above the short ones.
    net_z = torch.tensor([0.3, 0.9, 1.5])
    land = torch.tensor([[0.5, 0.0]] * 3)
    r = R.reach_score(net_z, land, p)
    assert short[2] < r[0] < r[1] < r[2] < 1.0
    # Continuous at the net plane: a ball landing just before the net ~ one crossing at the floor.
    a = R.reach_score(torch.tensor([-math.inf]), torch.tensor([[-1e-4, 0.0]]), p)
    b = R.reach_score(torch.tensor([0.0]), torch.tensor([[0.1, 0.0]]), p)
    assert a.item() == pytest.approx(b.item(), abs=1e-3)
    # Clearing the net by >= 20 cm: 1.
    r = R.reach_score(torch.tensor([1.75, 2.5]), torch.tensor([[1.5, 0.0], [3.0, 0.0]]), p)
    assert r.tolist() == [1.0, 1.0]


def test_outcome_pred_score_blend_and_v3_fallback():
    p = R.ServeRewardParams()
    nz, land = _flight([[-2.55, 0.0, 1.0]], [[2.0, 0.0, 1.0]], p)  # short of the net
    assert R.outcome_score(nz, land, p).item() == 0.0
    assert R.outcome_pred_score(nz, land, p).item() == pytest.approx(0.3 * R.reach_score(nz, land, p).item())
    p0 = R.ServeRewardParams(reach_lambda=0.0)
    pos, vel = _launch(10.0, 45)
    nz, land = RULES.predict_flight(pos, vel, p.drag_over_mass, dt=p.flight_dt, max_steps=200)
    assert R.outcome_pred_score(nz, land, p0).item() == R.outcome_score(nz, land, p0).item()
    # A long serve keeps its guard: the reach share cannot pay an out-long serve.
    nz, land = _flight([[-2.55, 0.0, 0.7]], [[30 * math.cos(0.44), 0.0, 30 * math.sin(0.44)]], p)
    assert land[0, 0] > 6.7 and R.outcome_pred_score(nz, land, p).item() < 0.1


def test_face_quality_tail():
    p = R.ServeRewardParams()
    p0 = R.ServeRewardParams(face_tail_weight=0.0)
    t = p.elevation_target_deg
    on = R.face_quality(torch.tensor([t]), torch.tensor([0.0]), p)
    assert on.item() == R.face_quality(torch.tensor([t]), torch.tensor([0.0]), p0).item() == 1.0
    # 0.4 * exp(-1) = 0.147 at 25 deg off (the design's "0.15", rounded); 0.08 at 40 deg.
    q25 = R.face_quality(torch.tensor([t + 25.0, t - 25.0]), torch.zeros(2), p)
    assert q25.min().item() >= 0.145 and q25.max().item() == pytest.approx(0.4 * math.exp(-1.0), abs=1e-6)
    assert R.face_quality(torch.tensor([t + 40.0]), torch.zeros(1), p).item() == pytest.approx(0.0808, abs=1e-3)
    d = torch.linspace(0.0, 60.0, 241)
    for sign in (1.0, -1.0):
        q = R.face_quality(t + sign * d, torch.zeros_like(d), p)
        assert (q[1:] <= q[:-1] + 1e-7).all()
    # The heading factor still applies to the tail.
    assert R.face_quality(torch.tensor([t + 25.0]), torch.tensor([15.0]), p).item() == pytest.approx(
        0.4 * math.exp(-1.0) * math.exp(-1.0), abs=1e-6)


def _vel(speed, el_deg, head_deg=0.0):
    e, h = math.radians(el_deg), math.radians(head_deg)
    return [speed * math.cos(e) * math.cos(h), speed * math.cos(e) * math.sin(h), speed * math.sin(e)]


def test_launch_score_values():
    p = R.ServeRewardParams()
    v = torch.tensor([_vel(8.0, 38.0), _vel(9.0, 45.0), _vel(8.0, 52.0), _vel(4.0, 45.0), _vel(8.0, 72.0),
                      _vel(8.0, 45.0, 180.0), [0.0, 0.0, 8.0]])
    s, el = R.launch_score(v, p)
    assert s[:3].tolist() == pytest.approx([1.0, 1.0, 1.0], abs=1e-5)
    assert s[3].item() == pytest.approx(0.5, abs=1e-5)  # half the reference speed
    assert s[4].item() == pytest.approx(math.exp(-1.0), abs=1e-4)  # 72 deg: 20 deg past the flat band
    assert s[5].item() == 0.0 and s[6].item() == 0.0  # backwards / straight up: vx <= 0
    assert el[1].item() == pytest.approx(45.0, abs=1e-4)


def _serve(c, vel, pos=(-2.3, 0.0, 1.0), resolved_fault=False):
    """Hit, then the launch one control step later. Returns the second step's event values."""
    n = 1
    q = quat_for_normal(55, 0)
    jh, hc = _hit(c, n, q, torch.zeros(3))
    c.end_step(inputs(n, just_hit=jh, hit_count=hc))
    first = {k: ev(c, k).item() for k in R.POSITIVE_TERMS}
    kw = dict(just_resolved=torch.tensor([True]), winner=torch.tensor([1]), double_fault=torch.tensor([True])) \
        if resolved_fault else {}
    c.end_step(inputs(n, hit_count=hc, shuttle_pos=torch.tensor([list(pos)]), shuttle_vel=torch.tensor([vel]), **kw))
    return first, {k: ev(c, k).item() for k in R.TERMS}


def test_launch_dir_event_fault_and_cap():
    c = make(1, "B", positive_cap=100.0)
    _, second = _serve(c, _vel(9.0, 45.0))
    assert second["launch_dir"] == pytest.approx(2.0, abs=1e-4)
    stats = c.pop_iteration()
    assert stats["launch_el_mean"] == pytest.approx(45.0, abs=1e-3)
    # Not paid when the rally ended on a fault at that step.
    c = make(1, "B")
    _, second = _serve(c, _vel(9.0, 45.0), resolved_fault=True)
    assert second["launch_dir"] == 0.0 and second["outcome_pred"] == 0.0
    # Counted in the positive cap together with outcome_pred.
    c = make(1, "B", positive_cap=4.0)
    first, second = _serve(c, _vel(9.0, 45.0))
    paid = sum(first.values()) + second["outcome_pred"] + second["launch_dir"]
    assert paid == pytest.approx(4.0, abs=1e-4)


# Exit velocities of the calibrated contact model (restitution 0.65, friction 0.11, shuttle falling at
# 4.3 m/s, racket moving along the face normal), contact at (-2.3, 0, 1.0):
#   face 55 deg / swing 3 m/s -> (5.20, 0, 5.19): over the net at 1.80 m, lands 1.25 m in
#   face 78 deg / swing 2 m/s -> (1.25, 0, 5.90): lands 1.2 m short of the net
#   "into the net": a steeper, slower launch (55 deg, 6.5 m/s) crossing the net plane at 1.21 m.
TYPICAL = {
    "in": [5.20, 0.0, 5.19],
    "short": [1.25, 0.0, 5.90],
    "net": [6.5 * math.cos(math.radians(55.0)), 0.0, 6.5 * math.sin(math.radians(55.0))],
}


def test_typical_serves_rank_and_spread():
    p = R.ServeRewardParams()
    nz = {k: _flight([[-2.3, 0.0, 1.0]], [v], p)[0].item() for k, v in TYPICAL.items()}
    assert nz["in"] > p.net_height + 0.2 and math.isinf(nz["short"]) and 0.0 < nz["net"] < p.net_height
    new, old = {}, {}
    for name, vel in TYPICAL.items():
        _, s = _serve(make(1, "B"), vel)
        new[name] = s["outcome_pred"] + s["launch_dir"]
        c_old = make(1, "B", reach_lambda=0.0, face_tail_weight=0.0)
        c_old.set_weights({"launch_dir": 0.0})  # V3 reward
        _, s = _serve(c_old, vel)
        old[name] = s["outcome_pred"]
    assert new["in"] > new["net"] > new["short"] > 0.0
    assert new["in"] - new["short"] > old["in"] - old["short"]
    assert old["short"] == 0.0


def test_contact_elevation_uses_face_met_by_falling_shuttle():
    """A static racket under a falling shuttle: the upper face (positive elevation) is the striking face."""
    n = 1
    c = make(n, "B")
    server = torch.zeros(n, dtype=torch.long)
    new = torch.tensor([[True, False]])
    for normal_sign in (1.0, -1.0):
        c.reset(torch.arange(n))
        q = quat_for_normal(50, 0)
        if normal_sign < 0:
            q = quat_for_normal(-50, 180)
        c.capture_contact(new, server, q[None, None].repeat(n, 2, 1), torch.zeros(n, 2, 3))
        assert float(c.c_el[0]) == pytest.approx(50.0, abs=0.5)

