"""CPU tests of the receive expert (P5 V3.1 section 11). No Isaac needed."""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
RECV = ROOT / "source/badmintonlab_game/badmintonlab_game/tasks/manager_based/badmintonlab_game/receive"
P3 = Path("D:/rl-badminton-dev/p3_sync_v4/p3_final/policy.pt")


@pytest.fixture(scope="module")
def core():
    spec = importlib.util.spec_from_file_location("recv_core", RECV / "core.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["recv_core"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cfg_predict():
    """The copy of the predictor in the submission file, executed without Isaac imports."""
    src = (RECV / "player_cfg.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == "_predict_intercept")
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "").startswith("_") for t in n.targets))]  # fmt: skip
    ns = {"torch": torch}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "player_cfg_subset", "exec"), ns)
    return ns["_predict_intercept"]


def _feeds(n=1000, seed=0):
    g = torch.Generator().manual_seed(seed)
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(n, generator=g)  # noqa: E731
    pos = torch.stack((u(1.0, 3.0), u(-0.5, 0.5), u(0.5, 2.5)), -1)
    sp, el, az = u(3, 18), torch.deg2rad(u(-10, 65)), torch.deg2rad(u(-25, 25))
    vel = torch.stack((-sp * torch.cos(el) * torch.cos(az), sp * torch.cos(el) * torch.sin(az), sp * torch.sin(el)), -1)
    return pos, vel


def test_submission_copy_matches_core(core, cfg_predict):
    pos, vel = _feeds()
    a = core.predict_intercept(pos, vel)
    b = cfg_predict(pos, vel)
    assert torch.equal(a[2], b[2])
    assert torch.allclose(a[0], b[0], atol=1e-6) and torch.allclose(a[1], b[1], atol=1e-6)


def test_intercept_matches_fine_integration(core):
    pos, vel = _feeds(300, seed=1)
    pt, t, ok = core.predict_intercept(pos, vel)
    fine, tf, okf = core.predict_intercept(pos, vel, dt=0.002, steps=1050)
    both = ok & okf
    assert both.float().mean() > 0.3
    assert (pt[both] - fine[both]).norm(dim=-1).max() < 0.03
    assert (t[both] - tf[both]).abs().max() < 0.02
    # The descent crossing is at z = 1.0.
    assert torch.allclose(pt[both, 2], torch.ones(int(both.sum())), atol=1e-5)


def test_r_track_values(core):
    d = torch.tensor([0.0, 0.3, 1.1])
    r = core.r_track(d)
    assert torch.allclose(r, torch.tensor([1.0, 1.0, math.exp(-3.2)]), atol=1e-6)


def test_station_puts_sweet_spot_on_intercept(core):
    p = torch.tensor([[-2.0, 0.3], [-3.0, -1.0]])
    s = core.station_from_intercept(p)
    c, si = math.cos(core.READY_YAW), math.sin(core.READY_YAW)
    o = torch.tensor([c * core.O_READY[0] - si * core.O_READY[1], si * core.O_READY[0] + c * core.O_READY[1]])
    assert torch.allclose(s + o, p, atol=1e-6)
    # Ready pose's own station is at distance ~0 from READY_ROOT for the ready sweet spot.
    ready_sweet = torch.tensor(core.READY_ROOT_XY) + o
    assert torch.allclose(core.station_from_intercept(ready_sweet[None])[0], torch.tensor(core.READY_ROOT_XY), atol=1e-6)


def test_feeder_table_is_filtered_and_sorted(core):
    t = core.build_feeder_table(40_000, seed=3)
    assert len(t) > 5000
    assert torch.all(t.d_need[1:] >= t.d_need[:-1])
    assert torch.all(t.t_arr >= 0.8)
    assert torch.all(t.d_need <= 0.9 * (t.t_arr - 0.2) + 0.3 + 1e-5)
    assert torch.all(t.int_pos[:, 0] <= -0.5) and torch.all(t.int_pos[:, 0] >= -5.0)
    # Intercepts recomputed from the stored launch match.
    pt, tt, ok = core.predict_intercept(t.launch_pos, t.launch_vel)
    assert ok.all() and torch.allclose(pt, t.int_pos, atol=1e-5)
    for level in range(4):
        lo, hi = core.level_bounds(t, level)
        assert 0 <= lo < hi <= len(t)
        if level < 3:
            assert t.d_need[hi - 1] < core.LEVEL_EDGES[level]
    rows = core.sample_rows(10_000, *core.level_bounds(t, 2), 0.2, "cpu")
    lo, hi = core.level_bounds(t, 2)
    easy = (rows < lo).float().mean()
    assert 0.15 < easy < 0.25 and rows.max() < hi


def test_curriculum_levels_and_gate(core):
    c = core.ReceiveCurriculum(level_min=10, level_ema=5, gate_min=20, gate_ema=5, gate_max=200)
    good = dict(station_ok=0.9, face_ok=0.9, fall=0.0, fall_ball=0.0)
    bad = dict(station_ok=0.1, face_ok=0.9, fall=0.0, fall_ball=0.0)
    for it in range(30):
        c.update(it, bad)
    assert c.level == 0 and not c.passed
    it = 30
    while c.level < 3 and it < 200:
        c.update(it, good)
        it += 1
    assert c.level == 3
    for _ in range(25):
        c.update(it, good)
        it += 1
    assert c.passed
    c2 = core.ReceiveCurriculum(gate_max=5)
    for i in range(6):
        c2.update(i, bad)
    assert c2.stop_reason == "S1_max_iters"
    s = c.state_dict()
    c3 = core.ReceiveCurriculum()
    c3.load_state_dict(s)
    assert c3.level == 3 and c3.passed


def test_curriculum_ignores_nan(core):
    c = core.ReceiveCurriculum()
    c.update(0, dict(station_ok=float("nan"), fall=0.0))
    assert "station_ok" not in c.ema


@pytest.mark.skipif(not P3.exists(), reason="p3_final not available")
def test_warm_start_equals_p3(core):
    from rsl_rl.modules import ActorCritic
    from tensordict import TensorDict

    ts = core.load_p3_torchscript(str(P3))
    obs = TensorDict({"policy": torch.zeros(1, 100), "critic": torch.zeros(1, 120)}, batch_size=[1])
    pol = ActorCritic(obs, {"policy": ["policy"], "critic": ["policy", "critic"]}, 23,
                      actor_obs_normalization=True, critic_obs_normalization=True,
                      actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
                      activation="elu", init_noise_std=0.3, noise_std_type="log")  # fmt: skip
    core.warm_start_actor(pol, ts, torch.tensor((0.6, -0.4, 0.3, 0.6, 0.6)), torch.tensor((0.8, 0.8, 0.3, 0.5, 0.5)))
    ref = torch.jit.load(str(P3))
    x95 = torch.randn(64, 95) * 2
    for extra in (torch.zeros(64, 5), torch.randn(64, 5) * 3):
        x = TensorDict({"policy": torch.cat((x95, extra), -1), "critic": torch.zeros(64, 120)}, batch_size=[64])
        out = pol.act_inference(x)
        assert (out - ref(x95)).abs().max() < 1e-5
    # Normaliser is frozen.
    pol.train()
    before = pol.actor_obs_normalizer._mean.clone()
    pol.update_normalization(TensorDict({"policy": torch.randn(32, 100) * 9, "critic": torch.zeros(32, 120)}, batch_size=[32]))
    assert torch.equal(before, pol.actor_obs_normalizer._mean)


def test_log_std_groups_and_clamp(core):
    leg = torch.tensor([i % 2 == 0 for i in range(23)])
    ls = torch.zeros(23)
    core.set_log_std(ls, leg, 0.5, 0.3)
    assert torch.allclose(ls.exp()[leg], torch.full((int(leg.sum()),), 0.5))
    assert torch.allclose(ls.exp()[~leg], torch.full((int((~leg).sum()),), 0.3))
    ls.fill_(math.log(0.05))
    core.clamp_log_std(ls, leg)
    assert torch.allclose(ls.exp()[leg].min(), torch.tensor(0.25)) and torch.allclose(ls.exp()[~leg].min(), torch.tensor(0.15))
    ls.fill_(math.log(5.0))
    core.clamp_log_std(ls, leg)
    assert torch.allclose(ls.exp()[leg].max(), torch.tensor(0.6)) and torch.allclose(ls.exp()[~leg].max(), torch.tensor(0.5))
