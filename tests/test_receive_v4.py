"""CPU tests of S1 V4 (11-09): free hitting height, wide feeder, speed curriculum."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
RECV = ROOT / "source/badmintonlab_game/badmintonlab_game/tasks/manager_based/badmintonlab_game/receive"


@pytest.fixture(scope="module")
def core():
    spec = importlib.util.spec_from_file_location("recv_core_v4", RECV / "core.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["recv_core_v4"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def table(core):
    return core.build_feeder_table_v4(60_000, seed=5)


def _advance(core, pos, vel, t, dt=0.001):
    """Fine integration of the play drag law for ``t`` seconds."""
    g = torch.tensor((0.0, 0.0, -core.GRAVITY))
    for _ in range(int(round(t / dt))):
        s = torch.linalg.norm(vel, dim=-1, keepdim=True)
        vel = vel + dt * (g - core.DRAG_OVER_MASS * s * vel)
        pos = pos + dt * vel
    return pos, vel


def test_multi_height_point_is_a_crossing(core, table):
    pt, t, ok = core.predict_intercept_multi(table.launch_pos[:2000], table.launch_vel[:2000])
    assert ok.all()
    z = pt[:, 2]
    assert torch.all(torch.stack([(z - h).abs() < 1e-4 for h in core.HIT_ZS]).any(0))
    assert torch.allclose(pt, table.int_pos[:2000], atol=1e-5)


def test_target_stable_during_flight(core, table):
    """The chosen point must not jump while the shuttle flies (score invariant to a time shift)."""
    idx = torch.randperm(len(table), generator=torch.Generator().manual_seed(0))[:400]
    pos, vel = table.launch_pos[idx], table.launch_vel[idx]
    pt0, t0, ok0 = core.predict_intercept_multi(pos, vel)
    for lag in (0.2, 0.4):
        p, v = _advance(core, pos, vel, lag)
        pt, t, ok = core.predict_intercept_multi(p, v)
        both = ok0 & ok & (t0 > lag + 0.1)
        err = (pt[both] - pt0[both]).norm(dim=-1)
        # Discretisation (dt 0.03) can flip near-ties; require almost all to keep the same point.
        frac = (err < 0.05).float().mean()
        assert frac > 0.97, (lag, frac)
        assert ((t[both] - (t0[both] - lag)).abs() < 0.03).float().mean() > 0.97


def test_multi_never_worse_than_fixed_height(core, table):
    """The free-height score is never worse than the z = 1.0 point when that point exists."""
    pos, vel = table.launch_pos[:3000], table.launch_vel[:3000]
    p1, t1, ok1 = core.predict_intercept(pos, vel)
    pm, tm, okm = core.predict_intercept_multi(pos, vel)
    ready = torch.tensor(core.READY_ROOT_XY)
    d1 = torch.linalg.norm(core.station_from_intercept(p1[:, :2]) - ready, dim=-1)
    dm = torch.linalg.norm(core.station_from_intercept(pm[:, :2]) - ready, dim=-1)
    both = ok1 & okm
    s1 = d1 - core.V_REF * t1
    sm = dm - core.V_REF * tm + core.Z_PREF * (pm[:, 2] - 1.0).abs()
    assert torch.all(sm[both] <= s1[both] + 1e-4)


def test_v4_table_filters_and_levels(core, table):
    v = core.required_speed(table.d_need, table.t_arr)
    assert torch.all(v[1:] >= v[:-1] - 1e-6)
    assert torch.all(table.t_arr >= 0.7) and torch.all(v <= 2.4 + 1e-5)
    assert torch.all(table.int_pos[:, 0] <= -0.5) and torch.all(table.int_pos[:, 1].abs() <= 2.4)
    n_lvl = len(core.SPEED_EDGES)
    prev_hi = 0
    for k in range(n_lvl):
        lo, hi = core.speed_level_bounds(v, k)
        assert lo == prev_hi and hi > lo, (k, lo, hi)
        assert torch.all(core.speed_level(v[lo:hi]) == k)
        prev_hi = hi
    assert prev_hi == len(table)
    assert core.speed_level_bounds(v, core.LEVEL_ALL_V4) == (0, len(table))
    counts = torch.bincount(core.speed_level(v), minlength=n_lvl)
    assert counts.min() > 1000
    lo, hi = core.speed_level_bounds(v, 0)
    assert table.d_need[lo:hi].max() < 1.0  # the first level really is a short step


def test_required_speed(core):
    d = torch.tensor([0.2, 1.3, 1.3])
    t = torch.tensor([1.0, 1.2, 0.2])
    v = core.required_speed(d, t)
    assert v[0] == 0.0 and v[1] == pytest.approx(1.0) and v[2] == pytest.approx(1.0 / 0.05)


def test_speed_curriculum_promotion_and_gate(core):
    cur = core.SpeedCurriculum(level_min=5, level_ema=3, gate_min=5, gate_ema=3, gate_max=500)
    top = cur.top
    for it in range(20):
        cur.update(it, {"L0_station_ok": 0.3, "fall_ball": 0.0, "station_ok": 0.9, "face_ok": 0.9})
    assert cur.level == 0
    for it in range(20, 40):  # falls block promotion even when the band is good
        cur.update(it, {"L0_station_ok": 0.9, "fall_ball": 0.2})
    assert cur.level == 0
    it = 40
    caps = [cur.approach_cap()]
    while cur.level < top and it < 400:
        cur.update(it, {f"L{cur.level}_station_ok": 0.9, "fall_ball": 0.0, "station_ok": 0.9, "face_ok": 0.9})
        caps.append(cur.approach_cap())
        it += 1
    assert cur.level == top
    assert caps == sorted(caps) and caps[-1] == core.APPROACH_CAPS[-1]
    for _ in range(10):  # other levels' rates do not count for the band
        cur.update(it, {"L0_station_ok": 1.0, f"L{top}_station_ok": 0.1, "fall_ball": 0.0, "station_ok": 0.9, "face_ok": 0.9})
        it += 1
    assert not cur.passed
    for _ in range(10):
        cur.update(it, {f"L{top}_station_ok": 0.5, "fall_ball": 0.0, "src_syn_station_ok": 0.6, "face_ok": 0.9})
        it += 1
    assert cur.passed and "passed_S1" in [h["event"] for h in cur.history]
    c2 = core.SpeedCurriculum()
    c2.load_state_dict(cur.state_dict())
    assert c2.level == top and c2.passed
    with pytest.raises(ValueError):
        c2.load_state_dict(core.ReceiveCurriculum().state_dict())


def test_speed_curriculum_ignores_nan_band(core):
    cur = core.SpeedCurriculum(level_min=1, level_ema=1)
    for it in range(5):
        cur.update(it, {"L0_station_ok": float("nan"), "fall_ball": 0.0})
    assert cur.level == 0


def test_real_serves_v4(core):
    p = Path("D:/rl-badminton-dev/recv_logs/serves/serves_s2.pt")
    if not p.exists():
        pytest.skip("recorded serves not available")
    d = torch.load(p, weights_only=False)
    t, info = core.table_from_serves_v4(d["pos"][:300], d["vel"][:300])
    assert info["n_kept"] >= 290
    v = core.required_speed(t.d_need, t.t_arr)
    assert torch.all(v[1:] >= v[:-1] - 1e-6)
    assert 1.2 < info["v_req_p50"] < 3.0  # the wide forehand serve needs fast footwork


def test_s2_mix_weights(core, table):
    v = core.required_speed(table.d_need, table.t_arr)
    w = core.mix_row_weights(v)
    assert torch.isclose(w.sum(), torch.tensor(1.0), atol=1e-4)
    rows = torch.multinomial(w, 200_000, replacement=True, generator=torch.Generator().manual_seed(1))
    frac = torch.bincount(core.speed_level(v[rows]), minlength=len(core.SPEED_EDGES)).float() / rows.numel()
    m = torch.tensor(core.S2_MIX_V4)
    assert torch.allclose(frac, m / m.sum(), atol=0.01), frac
    assert core.speed_level_bounds(v, core.LEVEL_MIX_V4) == (0, len(table))


def test_s2_gate_on_first_ball(core):
    cur = core.S2Curriculum(s1_weights={}, s2_weights={}, anneal_iters=5, gate_min=5, gate_ema=1, gate_hit_key="first_all_hit")
    for it in range(10):  # all-ball hit rate high, first-ball low: not passed
        cur.update(it, dict(hit=0.9, first_all_hit=0.1, epos_p50=0.1, fall_ball=0.0))
    assert not cur.passed
    cur.update(10, dict(hit=0.9, first_all_hit=0.5, epos_p50=0.1, fall_ball=0.0))
    assert cur.passed


def test_s1_v4_gate_uses_synthetic_station(core):
    cur = core.SpeedCurriculum(level=len(core.SPEED_EDGES) - 1, level_min=1, gate_min=2, gate_ema=1)
    good = {f"L{cur.top}_station_ok": 0.4, "station_ok": 0.25, "src_syn_station_ok": 0.38, "face_ok": 0.9, "fall_ball": 0.01}
    for it in range(3):
        cur.update(it, good)
    assert cur.passed  # overall 0.25 (diluted by real serves) does not block the gate


# ---------------------------------------------------------------- S3 (11-10)
def test_return_outcome_score(core):
    inf = float("inf")
    net = torch.tensor([2.0, 1.60, 1.40, -inf, 2.0, 2.0, 2.0])
    land = torch.tensor([[4.0, 0.0], [4.0, 0.0], [4.0, 0.0], [4.0, 0.0], [7.5, 0.0], [4.0, 3.2], [-1.0, 0.0]])
    s = core.return_outcome_score(net, land)
    assert s[0] == pytest.approx(1.0)  # clears by > margin, lands in
    assert 0.8 < s[1] < 1.0  # clears by 5 cm: slightly below the full score
    assert s[2] < s[1]  # below the net edge
    assert s[3] == 0.0  # never crosses
    assert s[4] < 0.1 and s[5] < 0.1  # long / wide
    assert s[6] == 0.0  # lands on our side


def test_return_result_order(core):
    t = torch.tensor([True, True, False, True])
    e = torch.tensor([True, True, True, False])
    w = torch.tensor([True, False, False, True])
    r = core.return_result(t, e, w)
    assert r.tolist() == pytest.approx([1.0, -0.2, -0.5, 0.0])
    assert r[1] > r[2]  # a bad return is better than no touch


def test_s3_next_gap(core):
    touched = torch.tensor([True, True, False])
    since = torch.tensor([20, 80, 0])
    tret = torch.tensor([60, 60, 60])
    g = core.s3_next_gap(touched, since, tret, 50)
    assert g.tolist() == [40, 0, 50]


def test_s3_curriculum(core):
    cur = core.S3Curriculum(s2_weights={"swing": 20.0, "ret_pred": 0.0}, s3_weights={"swing": 15.0, "ret_pred": 10.0},
                            blend_iters=10, gate_min=5, gate_ema=1)
    p = cur.params()
    assert p["weights"]["swing"] == 20.0 and p["sigma_p"] == 0.1
    for it in range(10):
        cur.update(it, dict(first_all_hit=0.5, first_ret_over=0.1, fall_ball=0.0))
    assert not cur.passed and cur.params()["weights"]["ret_pred"] == pytest.approx(10.0)
    cur.update(10, dict(first_all_hit=0.5, first_ret_over=0.25, fall_ball=0.0))
    assert cur.passed
    c2 = core.S3Curriculum()
    c2.load_state_dict(cur.state_dict())
    assert c2.passed
    with pytest.raises(ValueError):
        c2.load_state_dict(dict(stage="S2"))


def test_s4_rally_table_and_weights(core):
    """S4 (11-12): opponent returns cover the whole half including the back court, every row has a target,
    cell weights equalize the cells and follow the grid-test weak cells."""
    tab, cell, w = core.build_rally_table(n=300_000, seed=3)
    assert len(tab) > 10_000
    assert bool((tab.t_arr >= 0.35).all())
    assert float(tab.int_pos[:, 0].min()) < -5.2  # back court has targets now (band to -6.6)
    ix = cell // 15
    share = torch.stack([w[ix == i].sum() for i in range(5)]) / w.sum()
    assert float(share.min()) > 0.08  # every distance band gets a real share
    weak = {c: (3.0 if c // 15 == 4 else 1.0) for c in range(75)}
    _, cell2, w2 = core.build_rally_table(n=300_000, seed=3, cell_w=weak)
    s2 = float(w2[(cell2 // 15) == 4].sum() / w2.sum())
    assert s2 > float(share[4]) * 1.8


def test_cell_weights_from_grid(core):
    per_row = {"done": [1, 1, 1, 1, 1, 1, 0], "won": [1, 1, 1, 1, 1, 0, 0]}
    cell = [0, 0, 0, 0, 0, 1, 2]
    w = core.cell_weights_from_grid(per_row, cell, 3, base=0.5, min_n=5)
    assert w[0] == pytest.approx(0.5)  # all won
    assert w[1] == pytest.approx(1.5) and w[2] == pytest.approx(1.5)  # too few finished balls -> max


def test_s4_curriculum_names(core):
    cur = core.S3Curriculum(s2_weights={"a": 1.0}, s3_weights={"a": 3.0}, stage="S4", blend_iters=200)
    for _ in range(250):
        cur.update(0, {})
    assert "s4_250" in cur.events
    assert cur.params()["weights"]["a"] == pytest.approx(3.0)
    st = cur.state_dict()
    cur2 = core.S3Curriculum(stage="S4")
    cur2.load_state_dict(st)
    with pytest.raises(ValueError):
        core.S3Curriculum(stage="S3").load_state_dict(st)
