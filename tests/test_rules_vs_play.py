"""Differential test: our rule port vs the real ``play`` source, without Isaac.

Stubs the Isaac imports, loads ``play/dual_agent_env.py`` and ``play/assets/court.py``
unchanged, and drives both implementations with the same random event streams.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest import mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PLAY_ROOT = Path(os.environ.get("PLAY_ROOT", "/root/autodl-tmp/BadmintonLab-Game-Play"))
SERVE = ROOT / "source/badmintonlab_game/badmintonlab_game/tasks/manager_based/badmintonlab_game/serve"
PLAY_PKG = PLAY_ROOT / "source/badmintonlab_game/badmintonlab_game"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rules = _load("serve_rules_under_test", SERVE / "rules.py")


def _load_play():
    stubs = {}
    for name in [
        "gymnasium",
        "isaaclab",
        "isaaclab.assets",
        "isaaclab.envs",
        "isaaclab.sensors",
        "isaaclab.sim",
        "isaaclab.utils",
        "isaaclab.utils.math",
    ]:
        stubs[name] = mock.MagicMock(name=name)
    stubs["gymnasium"].Env = type("GymEnvStub", (), {})
    stubs["isaaclab.envs"].ManagerBasedEnv = type("ManagerBasedEnvStub", (), {})
    stubs["isaaclab.utils"].configclass = lambda c: c
    stubs["isaaclab.sim"].clone = lambda f: f
    pkg = types.ModuleType("badmintonlab_game")
    assets = types.ModuleType("badmintonlab_game.assets")
    pkg.__path__, assets.__path__ = [], []
    with mock.patch.dict(sys.modules, stubs):
        sys.modules["badmintonlab_game"] = pkg
        sys.modules["badmintonlab_game.assets"] = assets
        court = _load("badmintonlab_game.assets.court", PLAY_PKG / "assets/court.py")
        env = _load(
            "play_env_under_test", PLAY_PKG / "tasks/manager_based/badmintonlab_game/dual_agent_env.py"
        )
    for k in ("badmintonlab_game", "badmintonlab_game.assets", "badmintonlab_game.assets.court"):
        sys.modules.pop(k, None)
    return env.DualAgentBadmintonEnv, court


@pytest.fixture(scope="module")
def play():
    if not PLAY_PKG.exists():
        pytest.skip("play checkout not found; set PLAY_ROOT")
    return _load_play()


class _Contact:
    def __init__(self, n):
        self.data = types.SimpleNamespace(force_matrix_w=torch.zeros((n, 1, 1, 3)))


def _make_play_self(cls, court_mod, n):
    s = types.SimpleNamespace()
    s.cfg = types.SimpleNamespace(
        hit_contact_threshold=1.0,
        max_serve_height=1.15,
        court=court_mod.BadmintonCourtCfg(),
        shuttle_landing_detection_height=0.03,
    )
    s.just_hit = torch.zeros((n, 2), dtype=torch.bool)
    s.next_hitter = torch.zeros(n, dtype=torch.long)
    s.last_hitter = torch.full((n,), -1, dtype=torch.long)
    s.rally_resolved = torch.zeros(n, dtype=torch.bool)
    s.rally_winner = torch.full((n,), -1, dtype=torch.long)
    s._in_contact = torch.zeros((n, 2), dtype=torch.bool)
    s.serve_height_fault = torch.zeros(n, dtype=torch.bool)
    s.double_hit_fault = torch.zeros(n, dtype=torch.bool)
    s.shuttle_landed = torch.zeros(n, dtype=torch.bool)
    s.prev_shuttle_pos_w = torch.zeros((n, 3))
    s.curr_shuttle_pos_w = torch.zeros((n, 3))
    s._contacts = (_Contact(n), _Contact(n))
    s._shuttle = types.SimpleNamespace(data=types.SimpleNamespace(root_pos_w=torch.zeros((n, 3))))
    s.scene = types.SimpleNamespace(env_origins=torch.zeros((n, 3)))
    return s


def _court_geometry(court_mod):
    return rules.CourtGeometry.from_cfg(court_mod.BadmintonCourtCfg())


def _assert_same(s, r, msg):
    for a, b in [
        (s.just_hit, r.just_hit),
        (s.next_hitter, r.next_hitter),
        (s.last_hitter, r.last_hitter),
        (s.rally_resolved, r.rally_resolved),
        (s.rally_winner, r.rally_winner),
        (s.serve_height_fault, r.serve_height_fault),
        (s.double_hit_fault, r.double_hit_fault),
        (s.shuttle_landed, r.shuttle_landed),
    ]:
        assert torch.equal(a, b), msg


@pytest.mark.parametrize("seed", range(5))
def test_random_streams_match_play(play, seed):
    cls, court_mod = play
    n = 256
    g = torch.Generator().manual_seed(seed)
    s = _make_play_self(cls, court_mod, n)
    r = rules.MatchRules(n, "cpu", _court_geometry(court_mod))
    opening = torch.randint(0, 2, (n,), generator=g)
    s.next_hitter[:] = opening
    r.next_hitter[:] = opening
    pos = torch.zeros((n, 3))
    pos[:, 0] = torch.where(opening == 0, -2.55, 2.55)
    pos[:, 2] = 2.2
    s.prev_shuttle_pos_w[:] = pos
    s.curr_shuttle_pos_w[:] = pos
    r.prev_pos[:] = pos
    r.curr_pos[:] = pos
    origins = torch.zeros((n, 3))
    for step in range(3000):
        # Random walk that visits the net plane, the sidelines, the floor and the serve height.
        pos = pos + torch.randn((n, 3), generator=g) * torch.tensor([0.5, 0.5, 0.25])
        jump = torch.rand(n, generator=g) < 0.02
        pos[jump] = torch.stack(
            (
                torch.rand(n, generator=g) * 14 - 7,
                torch.rand(n, generator=g) * 7 - 3.5,
                torch.rand(n, generator=g) * 3,
            ),
            dim=1,
        )[jump]
        # Exactly on the net plane sometimes: this is the play/main difference.
        pos[torch.rand(n, generator=g) < 0.01, 0] = 0.0
        contact = torch.rand((n, 2), generator=g) < 0.06
        force = contact.float() * 5.0
        for side in range(2):
            s._contacts[side].data.force_matrix_w[:, 0, 0, 0] = force[:, side]
        s._shuttle.data.root_pos_w = pos.clone()
        cls._detect_landing(s)
        r.detect_landing(pos[:, 2], 0.03)
        cls._update_hits(s)
        r.update_hits(contact, pos[:, 2])
        cls._update_rally_result(s)
        r.update_rally_result(pos.clone(), origins)
        _assert_same(s, r, f"divergence at seed {seed} step {step}")
        # Reset resolved courts, as the play env does at the end of a control step.
        if step % 4 == 3:
            s.just_hit.zero_()
            r.just_hit.zero_()
            ids = s.rally_resolved.nonzero().flatten()
            if ids.numel():
                new_open = torch.randint(0, 2, (ids.numel(),), generator=g)
                pos[ids, 0] = torch.where(new_open == 0, -2.55, 2.55)
                pos[ids, 1] = 0.0
                pos[ids, 2] = 2.2
                s.just_hit[ids] = False
                s.next_hitter[ids] = new_open
                s.last_hitter[ids] = -1
                s.rally_resolved[ids] = False
                s.rally_winner[ids] = -1
                s.shuttle_landed[ids] = False
                s._in_contact[ids] = False
                s.serve_height_fault[ids] = False
                s.double_hit_fault[ids] = False
                s.prev_shuttle_pos_w[ids] = pos[ids]
                s.curr_shuttle_pos_w[ids] = pos[ids]
                r.reset(ids, new_open, pos[ids])


def test_event_streams_visit_every_outcome(play):
    """Guard against a vacuous differential test."""
    cls, court_mod = play
    n = 512
    g = torch.Generator().manual_seed(0)
    r = rules.MatchRules(n, "cpu", _court_geometry(court_mod))
    pos = torch.zeros((n, 3))
    pos[:, 2] = 1.0
    seen = dict(high=0, double=0, net=0, out=0, land=0, hit=0)
    for _ in range(600):
        pos = pos + torch.randn((n, 3), generator=g) * torch.tensor([0.5, 0.5, 0.25])
        contact = torch.rand((n, 2), generator=g) < 0.06
        r.begin_control_step()
        r.detect_landing(pos[:, 2], 0.03)
        r.update_hits(contact, pos[:, 2])
        r.update_rally_result(pos.clone(), torch.zeros((n, 3)))
        seen["high"] += int(r.serve_height_fault.sum())
        seen["double"] += int(r.double_hit_fault.sum())
        seen["net"] += int(r.step_net_fault.sum())
        seen["out"] += int(r.step_out_fault.sum())
        seen["land"] += int((r.shuttle_landed & r.step_just_resolved).sum())
        seen["hit"] += int(r.just_hit.sum())
        ids = r.rally_resolved.nonzero().flatten()
        if ids.numel():
            pos[ids] = torch.tensor([-2.55, 0.0, 1.0])
            r.reset(ids, torch.randint(0, 2, (ids.numel(),), generator=g), pos[ids])
    assert all(v > 0 for v in seen.values()), seen


def test_sustained_contact_counts_once():
    r = rules.MatchRules(1, "cpu", rules.CourtGeometry())
    z = torch.tensor([1.0])
    for _ in range(5):
        r.update_hits(torch.tensor([[True, False]]), z)
    assert r.last_hitter.item() == 0 and not r.double_hit_fault.any()
    r.update_hits(torch.tensor([[False, False]]), z)
    r.update_hits(torch.tensor([[True, False]]), z)
    assert r.double_hit_fault.item() and r.rally_winner.item() == 1


def test_high_serve_loses_even_if_low_afterwards():
    r = rules.MatchRules(1, "cpu", rules.CourtGeometry())
    r.update_hits(torch.tensor([[True, False]]), torch.tensor([1.16]))
    assert r.serve_height_fault.item() and r.rally_winner.item() == 1 and r.rally_resolved.item()


def test_no_hit_loss_follows_landing_half():
    """Agent's correction: an unhit shuttle deflected to the other half costs that other player."""
    r = rules.MatchRules(1, "cpu", rules.CourtGeometry())
    origin = torch.zeros((1, 3))
    r.curr_pos[:] = torch.tensor([[-2.55, 0.0, 1.0]])
    r.next_hitter[:] = 0
    pos = torch.tensor([[1.0, 0.0, 0.02]])
    r.prev_pos[:] = pos
    r.curr_pos[:] = pos
    r.detect_landing(pos[:, 2], 0.03)
    r.update_rally_result(pos.clone(), origin)
    assert r.rally_winner.item() == 0  # landed on player 1's half


def test_unhit_shuttle_out_of_bounds_is_not_a_fault():
    r = rules.MatchRules(1, "cpu", rules.CourtGeometry())
    pos = torch.tensor([[-2.0, 3.0, 1.0]])
    r.prev_pos[:] = pos
    r.curr_pos[:] = pos
    r.update_rally_result(pos.clone(), torch.zeros((1, 3)))
    assert not r.rally_resolved.item()


def test_flight_prediction_matches_fine_integration():
    pos = torch.tensor([[-2.55, 0.0, 1.05], [-2.55, 0.5, 0.95]])
    vel = torch.tensor([[6.0, 0.0, 8.0], [3.0, 0.2, 3.0]])
    drag = 0.00110 / 0.00519
    coarse_net, coarse_land = rules.predict_flight(pos, vel, drag, dt=0.01)
    fine_net, fine_land = rules.predict_flight(pos, vel, drag, dt=0.0005, max_steps=8000)
    assert torch.allclose(coarse_net, fine_net, atol=0.03)
    assert torch.allclose(coarse_land, fine_land, atol=0.05)


@pytest.mark.parametrize("seed", range(3))
def test_inject_serve_matches_a_real_serve_in_play(play, seed):
    """P5 2.2: after ``inject_serve`` every ruling matches play after the feeder's real (legal) serve."""
    cls, court_mod = play
    n = 256
    g = torch.Generator().manual_seed(100 + seed)
    s = _make_play_self(cls, court_mod, n)
    r = rules.MatchRules(n, "cpu", _court_geometry(court_mod))
    feeder = torch.randint(0, 2, (n,), generator=g)
    sign = torch.where(feeder == 0, -1.0, 1.0)
    s.next_hitter[:] = feeder
    r.next_hitter[:] = feeder
    pos = torch.zeros((n, 3))
    pos[:, 0] = 2.55 * sign
    pos[:, 2] = 2.2
    for t in (s.prev_shuttle_pos_w, s.curr_shuttle_pos_w, r.prev_pos, r.curr_pos):
        t[:] = pos
    origins = torch.zeros((n, 3))
    k0 = torch.randint(5, 40, (n,), generator=g)
    ar = torch.arange(n)
    injected = 0
    for step in range(400):
        before = step < k0
        at = step == k0
        after = step > k0
        # Drop on the feeder's side down to ~1.0 m (no landing, no fault), then a random walk.
        drop = pos.clone()
        drop[:, 0] = 2.55 * sign
        drop[:, 1] = 0.0
        drop[:, 2] = 2.2 - 1.2 * (step + 1) / (k0 + 1).float()
        walk = pos + torch.randn((n, 3), generator=g) * torch.tensor([0.5, 0.5, 0.25])
        pos = torch.where((before | at)[:, None], drop, walk)
        contact = (torch.rand((n, 2), generator=g) < 0.06) & after[:, None]
        contact[ar, feeder] &= step != k0 + 1  # a real serve contact ends after its step
        play_contact = contact.clone()
        play_contact[ar, feeder] |= at
        for side in range(2):
            s._contacts[side].data.force_matrix_w[:, 0, 0, 0] = play_contact[:, side].float() * 5.0
        s._shuttle.data.root_pos_w = pos.clone()
        cls._detect_landing(s)
        r.detect_landing(pos[:, 2], 0.03)
        cls._update_hits(s)
        r.update_hits(contact, pos[:, 2])
        ids = at.nonzero().flatten()
        if ids.numel():
            r.inject_serve(ids, feeder[ids])
            injected += ids.numel()
            # The real serve was legal and is the only difference: just_hit of that step.
            assert bool(s.just_hit[ids, feeder[ids]].all()) and not bool(s.serve_height_fault[ids].any())
            s.just_hit[ids] = r.just_hit[ids]
        cls._update_rally_result(s)
        r.update_rally_result(pos.clone(), origins)
        _assert_same(s, r, f"divergence after injection, seed {seed} step {step}")
        if step % 4 == 3:
            s.just_hit.zero_()
            r.just_hit.zero_()
    assert injected == n
    # Not vacuous: rallies were decided both ways after the injection, including receiver hits.
    assert bool(r.rally_resolved.float().mean() > 0.9)
    assert set(r.rally_winner.tolist()) >= {0, 1}
