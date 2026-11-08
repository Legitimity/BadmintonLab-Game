"""CPU tests of stage S2 (P5 V3.1 5.2 / section 11 item 4) and the physics presets. No Isaac needed."""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "source/badmintonlab_game/badmintonlab_game/tasks/manager_based/badmintonlab_game"


@pytest.fixture(scope="module")
def core():
    spec = importlib.util.spec_from_file_location("recv_core_s2", PKG / "receive/core.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["recv_core_s2"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def physics():
    spec = importlib.util.spec_from_file_location("recv_physics", PKG / "physics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_r_swing_ends(core):
    z = torch.zeros(1)
    k = 0.3 * (1 - math.exp(-1.0))  # speed share at v = sigma_v = 3
    v = torch.tensor([3.0])
    assert torch.allclose(core.r_swing(z, z, z, 2.0, 8.0), torch.ones(1))
    far = torch.tensor([2.0])
    # Paper form (additive): speed pays even far from the target.
    assert torch.allclose(core.r_swing(z, z, v, 0.1, 1.0, gated=False), torch.tensor([1.0 + k]))
    assert float(core.r_swing(far, z, v, 0.1, 1.0, gated=False)) == pytest.approx(k, abs=1e-6)
    # V3.3 gated form: no speed reward far from the target.
    assert torch.allclose(core.r_swing(z, z, v, 0.1, 1.0), torch.tensor([1.0 + k]))
    assert float(core.r_swing(far, z, v, 0.1, 1.0)) == pytest.approx(0.0, abs=1e-6)
    assert torch.allclose(core.r_swing(z, z, -v, 0.1, 1.0), torch.ones(1))
    big = core.r_swing(z, z, torch.tensor([50.0]), 0.1, 1.0, v_cap=8.0)
    assert torch.allclose(big, torch.tensor([1.0 + 0.3 * (1 - math.exp(-8.0 / 3.0))]))
    e = torch.tensor([0.3])
    assert torch.allclose(core.r_swing(e, z, z, 2.0, 8.0), torch.tensor([math.exp(-0.09 / 2.0)]))
    assert torch.allclose(core.r_swing(e, z, z, 0.1, 1.0), torch.tensor([math.exp(-0.09 / 0.1)]))
    o = torch.tensor([0.5])
    assert torch.allclose(core.r_swing(z, o, z, 2.0, 1.0), torch.tensor([math.exp(-0.5)]))


def test_anneal_exp(core):
    assert core.anneal_exp(0, 2.0, 0.1, 220.0) == pytest.approx(2.0)
    assert core.anneal_exp(500, 2.0, 0.1, 220.0) == pytest.approx(0.296, abs=2e-3)
    assert core.anneal_exp(10_000, 2.0, 0.1, 220.0) == pytest.approx(0.1, abs=1e-6)


def test_face_error_two_faces(core):
    n = core.target_normal(50.0)
    assert torch.allclose(core.face_error(n[None], n), torch.zeros(1), atol=1e-3)
    assert torch.allclose(core.face_error(-n[None], n), torch.zeros(1), atol=1e-3)
    up = torch.tensor([[0.0, 0.0, 1.0]])
    assert abs(float(core.face_error(up, n)) - math.radians(40.0)) < 1e-4


def test_quat_axes(core):
    # 90 deg about z: x -> y, y -> -x.
    q = torch.tensor([[math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]])
    ax, ay = core.quat_axes_xy(q)
    assert torch.allclose(ax, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)
    assert torch.allclose(ay, torch.tensor([[-1.0, 0.0, 0.0]]), atol=1e-6)
    assert torch.allclose(core.r_racket_style(ay), torch.zeros(1), atol=1e-6)
    assert torch.allclose(core.r_racket_style(ax), torch.ones(1), atol=1e-6)


def test_ball_level_and_uniform(core):
    d = torch.tensor([0.1, 0.4, 0.79, 0.8, 1.39, 1.4, 3.0])
    assert core.ball_level(d).tolist() == [0, 1, 1, 2, 2, 3, 3]
    t = core.build_feeder_table(20_000, seed=3)
    assert core.level_bounds(t, core.LEVEL_ALL) == (0, len(t))


def test_s2_curriculum_blend_anneal_gate(core):
    s1 = dict(track=4.0, swing=0.0)
    s2 = dict(track=2.0, swing=20.0)
    cur = core.S2Curriculum(s1_weights=s1, s2_weights=s2, blend_iters=100, anneal_iters=200, anneal_tau=50.0,
                            gate_min=50, gate_ema=1)  # fmt: skip
    p = cur.params()
    assert p["weights"]["track"] == 4.0 and p["weights"]["swing"] == 0.0 and p["sigma_p"] == 2.0 and p["sigma_r"] == 8.0
    good = dict(hit=0.6, epos_p50=0.1, fall_ball=0.01)
    for it in range(100):
        cur.update(it, good)
    p = cur.params()
    assert p["weights"]["track"] == pytest.approx(2.0) and p["weights"]["swing"] == pytest.approx(20.0)
    assert p["sigma_p"] == pytest.approx(0.1 + 1.9 * math.exp(-2.0))
    assert not cur.passed
    for it in range(100, 200):
        cur.update(it, good)
    assert cur.passed and cur.params()["sigma_p"] == pytest.approx(0.1 + 1.9 * math.exp(-4.0))
    st = cur.state_dict()
    cur2 = core.S2Curriculum(s1_weights=s1, s2_weights=s2)
    cur2.load_state_dict(st)
    assert cur2.passed and cur2.iters_in_stage == 200
    with pytest.raises(ValueError):
        cur2.load_state_dict(dict(stage="S1"))


def test_s2_gate_not_passed_when_bad(core):
    cur = core.S2Curriculum(s1_weights={}, s2_weights={}, anneal_iters=10, gate_min=10, gate_max=30, gate_ema=1)
    for it in range(30):
        cur.update(it, dict(hit=0.2, epos_p50=0.1, fall_ball=0.0))
    assert not cur.passed and cur.stop_reason == "S2_max_iters"


class _Obj(types.SimpleNamespace):
    pass


@pytest.mark.parametrize("dt_ms,dec,lo,hi", [(1, 20, 10, 40), (2, 10, 5, 20), (5, 4, 2, 8)])
def test_apply_physics(physics, dt_ms, dec, lo, hi):
    act = lambda: _Obj(min_delay=0, max_delay=0)  # noqa: E731
    robot = lambda: _Obj(actuators={"arms": act(), "legs": act()})  # noqa: E731
    scene = _Obj(racket_contact_0=_Obj(history_length=0), racket_contact_1=_Obj(history_length=0),
                 robot_0=robot(), robot_1=robot())  # fmt: skip
    cfg = _Obj(sim=_Obj(dt=0.0, render_interval=0), decimation=0, scene=scene)
    physics.apply_physics(cfg, dt_ms)
    assert cfg.decimation == dec and cfg.sim.dt == pytest.approx(dt_ms / 1000) and cfg.sim.render_interval == dec
    assert scene.racket_contact_0.history_length == dec and scene.racket_contact_1.history_length == dec
    for r in (scene.robot_0, scene.robot_1):
        for a in r.actuators.values():
            assert (a.min_delay, a.max_delay) == (lo, hi)
    with pytest.raises(ValueError):
        physics.apply_physics(cfg, 3)


def test_defaults_match_play():
    """The training defaults equal play a5e0780 (1 ms, decimation 20, delay 10-40, sensor history 20)."""
    src = (PKG / "dual_agent_env_cfg.py").read_text(encoding="utf-8")
    assert "self.decimation = 20" in src and "self.sim.dt = 0.001" in src
    scene = (PKG / "badminton_scene_cfg.py").read_text(encoding="utf-8")
    assert "history_length=20" in scene
    booster = (PKG.parents[2] / "assets/robots/booster.py").read_text(encoding="utf-8")
    assert booster.count("min_delay=10") == 5 and booster.count("max_delay=40") == 5
    play = Path("D:/rl-badminton-eval/BadmintonLab-Game-Play/source/badmintonlab_game/badmintonlab_game/assets/robots/booster.py")
    if play.exists():
        ps = play.read_text(encoding="utf-8")
        assert ps.count("min_delay=10") == 5 and ps.count("max_delay=40") == 5
