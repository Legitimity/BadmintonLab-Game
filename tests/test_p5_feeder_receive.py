"""CPU tests: feeder table (legality, levels, sampling) and the receive reward (roles, events, cap, stats)."""

import math

import pytest
import torch

from _serve_pkg import load

m = load("serve_reward", "rules", "feeder", "receive")
fd, rc, sr = m["feeder"], m["receive"], m["serve_reward"]
DRAG, G = 0.00110 / 0.00519, 9.81  # k / m, as ServeRewardParams.drag_over_mass


@pytest.fixture(scope="module")
def table():
    return fd.FeederTable.build(fd.FeederParams(), DRAG, G)


def test_table_rows_are_legal_and_sorted(table):
    p = table.p
    assert len(table) > 0 and len(table) < table.total_rows
    net = table.col("net_z")
    assert float(net.min()) >= p.net_height + p.net_margin - 1e-6
    lx, ly = table.col("land_x"), table.col("land_y")
    assert float(lx.min()) > 0.0 and float(lx.max()) <= p.half_length + 1e-6
    assert float(ly.abs().max()) <= p.singles_half_width + 1e-6
    d = table.col("d_need")
    assert torch.all(d[1:] >= d[:-1])
    for i, hi in enumerate(p.levels):
        a, b = table.cuts[i], table.cuts[i + 1]
        if b > a:
            assert float(d[a:b].max()) <= hi + 1e-6
    assert sum(table.level_counts()) <= len(table)


def test_sampling_bands(table):
    torch.manual_seed(0)
    c = table.cuts
    rows = table.sample(0, 4000)
    assert int(rows.max()) < c[1]
    rows = table.sample(2, 20000)
    assert int(rows.max()) < c[3]
    own = ((rows >= c[2]) & (rows < c[3])).float().mean().item()
    assert abs(own - (table.p.new_share + (1 - table.p.new_share) * 0.0)) < 0.03
    lv = torch.tensor([0, 1, 2, 3] * 500)
    rows = table.sample(lv, lv.numel())
    hi = torch.tensor(c)[lv + 1]
    assert torch.all(rows < hi)


def test_launch_target_in_band_and_reach(table):
    tz = table.col("tgt_z")
    assert float(tz.min()) >= table.p.band_lo - 1e-6 and float(tz.max()) <= table.p.band_hi + 1e-6
    assert float(table.col("tgt_x").min()) > table.p.band_x_min - 1e-6
    assert table.p.home == fd.RACKET_HOME and table.p.reach == fd.RACKET_REACH
    home = torch.tensor(table.p.home, dtype=table.data.dtype)
    d_min = torch.linalg.norm(torch.stack((table.col("tgt_x"), table.col("tgt_y")), -1) - home, dim=-1)
    need = (d_min - table.p.reach).clamp(min=0)
    assert torch.allclose(need, table.col("d_need"), atol=1e-4)


# ---------------------------------------------------------------------- receive reward
def _computer(n=3):
    sp = sr.ServeRewardParams()
    return rc.ReceiveRewardComputer(n, "cpu", rc.ReceiveParams(), sp, step_dt=0.02), sp


def _inputs(n=3, **kw):
    z = torch.zeros
    d = dict(server=torch.zeros(n, dtype=torch.long), just_hit=z(n, 2, dtype=torch.bool),
             just_resolved=z(n, dtype=torch.bool), winner=torch.full((n,), -1, dtype=torch.long),
             net_fault=z(n, dtype=torch.bool), shuttle_pos=torch.tensor([[2.0, -0.6, 1.0]]).repeat(n, 1),
             shuttle_vel=torch.tensor([[-2.0, 0.0, -4.9]]).repeat(n, 1),
             root_pos=torch.tensor([[[-3.25, 0.6, 0.75], [3.25, -0.6, 0.75]]]).repeat(n, 1, 1),
             root_vel=z(n, 2, 3), racket_pos=torch.tensor([[[-2.6, 0.2, 1.0], [2.6, -0.2, 1.0]]]).repeat(n, 1, 1),
             root_z=torch.full((n, 2), 0.75), gravity_z=torch.full((n, 2), -1.0), fall_given=z(n, 2, dtype=torch.bool),
             time_s=torch.full((n,), 0.6), natural=torch.zeros(n, dtype=torch.bool))
    d.update(kw)
    return rc.ReceiveStepInputs(**d)


def test_only_receiver_column_and_contact_event():
    c, _ = _computer()
    n = 3
    c.natural[:] = False
    mask = torch.tensor([True, True, False])
    c.set_incoming(mask, torch.tensor([[3.0, -0.5, 0.9]]).repeat(n, 1), torch.full((n,), 1.0), torch.tensor([0.5, 0.5, 0.5]))
    # Physics-rate contact of the receiver (side 1) on court 0.
    new_hits = torch.tensor([[False, True], [False, False], [False, True]])
    q = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(n, 2, 1)
    c.capture_contact(new_hits, torch.zeros(n, dtype=torch.long), q, torch.zeros(n, 2, 3), torch.tensor([[-2.0, 0, -4.9]]).repeat(n, 1))
    jh = new_hits.clone()
    c.end_step(_inputs(just_hit=jh))
    v = c.values
    assert float(v["recv_contact"][0, 1]) == pytest.approx(2.0 / 0.02)
    assert float(v["recv_contact"][1, 1]) == 0.0 and float(v["recv_contact"][2, 1]) == 0.0  # court 2: no incoming
    for k in rc.RECV_TERMS:
        assert torch.count_nonzero(v[k][:, 0]) == 0, k  # server/feeder column never scored
    assert bool(c.hit[0]) and not bool(c.hit[1])


def test_miss_and_result_and_stats():
    c, _ = _computer()
    n = 3
    c.natural[:] = torch.tensor([False, False, True])
    c.set_incoming(torch.ones(n, dtype=torch.bool), torch.tensor([[3.0, -0.5, 0.9]]).repeat(n, 1), torch.full((n,), 1.0),
                   torch.zeros(n))
    # Court 0: the receiver hits; court 1 / 2: nothing.
    jh = torch.tensor([[False, True], [False, False], [False, False]])
    c.c_have[0] = True
    c.end_step(_inputs(just_hit=jh))
    # Next step: court 0 return crosses and wins; court 1 lands on the receiver's side (server wins); court 2 natural miss.
    pos = torch.tensor([[-1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    res = torch.ones(n, dtype=torch.bool)
    win = torch.tensor([1, 0, 0])
    c.end_step(_inputs(shuttle_pos=pos, just_resolved=res, winner=win, time_s=torch.full((n,), 0.62)))
    assert float(c.values["recv_result"][0, 1]) == pytest.approx(10.0 / 0.02)
    assert float(c.values["recv_miss"][1, 1]) == pytest.approx(-3.0 / 0.02)
    assert float(c.values["recv_miss"][2, 1]) == pytest.approx(-3.0 / 0.02)
    c.count_rallies(res, torch.zeros(n, dtype=torch.bool))
    r = c.pop_iteration()
    assert r["f_rallies"] == 2 and r["n_rallies"] == 1
    assert r["f_touch"] == 0.5 and r["f_crossed"] == 0.5 and r["f_in"] == 0.5 and r["n_miss"] == 1.0


def test_positive_cap_and_approach_sign():
    c, _ = _computer(1)
    c.set_weights({"recv_approach": 100.0})
    c.set_incoming(torch.ones(1, dtype=torch.bool), torch.tensor([[2.0, -0.6, 0.9]]), torch.full((1,), 5.0),
                   torch.full((1,), 1.0))
    total = 0.0
    for _ in range(20):
        # Moving towards the net (canonical -x for the receiver in the server frame) at 1 m/s.
        c.end_step(_inputs(1, root_vel=torch.tensor([[[0.0, 0, 0], [-1.0, 0, 0]]])))
        total += float(c.values["recv_approach"][0, 1])
    assert total == pytest.approx(c.p.positive_cap, rel=1e-5)
    c2, _ = _computer(1)
    c2.set_incoming(torch.ones(1, dtype=torch.bool), torch.tensor([[2.0, -0.6, 0.9]]), torch.full((1,), 5.0),
                    torch.full((1,), 1.0))
    c2.end_step(_inputs(1, root_vel=torch.tensor([[[0.0, 0, 0], [1.0, 0, 0]]])))  # away from the ball
    assert float(c2.values["recv_approach"][0, 1]) == pytest.approx(-0.3)


def test_fall_terminates_once():
    c, _ = _computer(1)
    c.set_incoming(torch.ones(1, dtype=torch.bool), torch.tensor([[3.0, -0.5, 0.9]]), torch.ones(1), torch.zeros(1))
    fg = torch.tensor([[False, True]])
    c.end_step(_inputs(1, fall_given=fg))
    assert bool(c.fall_done[0])
    c.end_step(_inputs(1, fall_given=fg))
    assert not bool(c.fall_done[0])
    assert not math.isnan(sum(float(v.sum()) for v in c.values.values()))


def test_stand_point_puts_resting_racket_on_target():
    c, _ = _computer(1)
    tgt = torch.tensor([[2.0, -0.3, 0.9]])
    c.set_incoming(torch.ones(1, dtype=torch.bool), tgt, torch.full((1,), 5.0), torch.full((1,), 1.0))
    stand = tgt[0, :2] + torch.tensor(fd.ROOT_FROM_RACKET)
    # Receiver root exactly at the stand point: inside arrive_radius -> full score.
    root = torch.tensor([[[-3.25, 0.6, 0.75], [stand[0], stand[1], 0.75]]])
    c.end_step(_inputs(1, root_pos=root))
    assert float(c.values["recv_approach"][0, 1]) == pytest.approx(1.0)
    racket_rest = stand + torch.tensor(fd.RACKET_HOME) - torch.tensor(fd.RECV_HOME)
    assert torch.allclose(racket_rest, tgt[0, :2], atol=1e-6)


def test_racket_near_wide_kernel():
    c, _ = _computer(1)
    c.set_incoming(torch.ones(1, dtype=torch.bool), torch.tensor([[2.5, 0.0, 1.0]]), torch.full((1,), 0.7), torch.zeros(1))
    rk = torch.tensor([[[-2.6, 0.2, 1.0], [2.0, -0.6, 1.0]]])
    sh = torch.tensor([[2.0, -0.6 + 0.9, 1.0]])  # 0.9 m from the racket
    c.end_step(_inputs(1, racket_pos=rk, shuttle_pos=sh))
    p = c.p
    want = (1 - p.near_wide_share) * math.exp(-(0.9 / p.near_sigma) ** 2) + p.near_wide_share * math.exp(-(0.9 / p.near_sigma_wide) ** 2)
    assert float(c.values["recv_racket_near"][0, 1]) == pytest.approx(want, rel=1e-4) and want > 0.05


def test_launch_dir_penalises_sideways_return():
    def run(vel):
        c, _ = _computer(1)
        c.set_incoming(torch.ones(1, dtype=torch.bool), torch.tensor([[3.0, -0.5, 0.9]]), torch.ones(1), torch.zeros(1))
        c.c_have[0] = True
        c.end_step(_inputs(1, just_hit=torch.tensor([[False, True]])))
        # Server 0 -> receiver 1; receiver frame = world rotated by 180 deg: forward return has world vx < 0.
        c.end_step(_inputs(1, shuttle_pos=torch.tensor([[2.6, -0.2, 1.0]]), shuttle_vel=torch.tensor([vel])))
        return float(c.values["recv_launch_dir"][0, 1])

    fwd = run([-5.66, 0.0, 5.66])
    side = run([-1.0, -5.57, 5.66])
    assert fwd > 0 and side < 0.05 * fwd


def test_receive_face_heading_and_swing_targets():
    c, sp = _computer(1)
    assert c.sp.v_close_target == 7.5 and c.sp.v_close_max == 9.0
    assert sp.v_close_target != 7.5 or sp.v_close_max != 9.0  # the serve parameters are untouched
    el = torch.full((5,), c.p.elevation_target_deg)
    hd = torch.tensor([0.0, 25.0, -25.0, 50.0, 75.0])
    f = c.face_score(el, hd)
    assert torch.allclose(f[:3], torch.ones(3))
    assert float(f[3]) == pytest.approx(math.exp(-1.0), rel=1e-5) and float(f[4]) == pytest.approx(math.exp(-4.0), rel=1e-5)
    # Elevation part identical to the serve's face_quality at heading 0 (with the receive elevation target).
    els = torch.tensor([20.0, 40.0, 52.0, 70.0])
    assert torch.allclose(c.face_score(els, torch.zeros(4)), sr.face_quality(els, torch.zeros(4), c.sp))
    s = sr.swing_score(torch.tensor([3.75, 7.5, 9.0, 10.0]), c.sp)
    assert torch.allclose(s, torch.tensor([0.5, 1.0, 1.0, 1.0 - 1.0 / c.sp.v_close_penalty_scale]))


def test_contact_log_medians():
    c, _ = _computer(2)
    c.set_incoming(torch.ones(2, dtype=torch.bool), torch.tensor([[3.0, -0.5, 0.9]]).repeat(2, 1), torch.ones(2), torch.zeros(2))
    c.c_have[:] = True
    c.c_hd[:] = torch.tensor([-60.0, 10.0])
    c.c_vc[:] = torch.tensor([5.0, 8.0])
    c.end_step(_inputs(2, just_hit=torch.tensor([[False, True], [False, True]])))
    r = c.pop_iteration()
    assert r["hd_abs_med"] in (10.0, 60.0) and r["hd25_share"] == 0.5 and r["vc_med"] in (5.0, 8.0)
    assert math.isnan(c.pop_iteration()["hd_abs_med"])
