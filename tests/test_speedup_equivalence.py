"""The sync-free predict_flight must give the same results as the original implementation."""

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "serve_rules", ROOT / "source/badmintonlab_game/badmintonlab_game/tasks/manager_based/badmintonlab_game/serve/rules.py"
)
rules = importlib.util.module_from_spec(_spec)
sys.modules["serve_rules"] = rules  # dataclasses look the module up while the class is created
_spec.loader.exec_module(rules)


def predict_flight_ref(pos, vel, drag_over_mass, gravity=9.81, dt=0.01, max_steps=400, net_x=0.0, floor=0.03):
    """Original implementation (P3 package 9c3d6c5): early exit and boolean indexing every step."""
    p, v = pos.clone(), vel.clone()
    net_z = torch.full((p.shape[0],), -torch.inf, device=p.device, dtype=p.dtype)
    active = torch.ones(p.shape[0], dtype=torch.bool, device=p.device)
    g = torch.tensor((0.0, 0.0, -gravity), device=p.device, dtype=p.dtype)
    for _ in range(max_steps):
        if not active.any():
            break
        speed = torch.linalg.norm(v, dim=-1, keepdim=True)
        vm = v + 0.5 * dt * (g - drag_over_mass * speed * v)
        speed_m = torch.linalg.norm(vm, dim=-1, keepdim=True)
        vn = v + dt * (g - drag_over_mass * speed_m * vm)
        pn = torch.where(active[:, None], p + dt * vm, p)
        v = torch.where(active[:, None], vn, v)
        crossing = active & (p[:, 0] < net_x) & (pn[:, 0] >= net_x)
        if crossing.any():
            a = (net_x - p[crossing, 0]) / (pn[crossing, 0] - p[crossing, 0])
            net_z[crossing] = p[crossing, 2] + a * (pn[crossing, 2] - p[crossing, 2])
        p = pn
        active = active & (p[:, 2] > floor)
    return net_z, p[:, :2]


def _random_serves(n, seed):
    g = torch.Generator().manual_seed(seed)
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(n, generator=g)  # noqa: E731
    pos = torch.stack((u(-4.0, -1.0), u(-1.5, 1.5), u(0.3, 1.6)), dim=-1)
    speed, elev, head = u(3.0, 25.0), torch.deg2rad(u(-10.0, 80.0)), torch.deg2rad(u(-200.0, 200.0))
    vel = torch.stack((speed * torch.cos(elev) * torch.cos(head), speed * torch.cos(elev) * torch.sin(head),
                       speed * torch.sin(elev)), dim=-1)
    return pos, vel


def _check(pos, vel, **kw):
    a_net, a_land = predict_flight_ref(pos, vel, 0.17, **kw)
    b_net, b_land = rules.predict_flight(pos, vel, 0.17, **kw)
    assert torch.equal(torch.isinf(a_net), torch.isinf(b_net))
    fin = torch.isfinite(a_net)
    assert torch.allclose(a_net[fin], b_net[fin], atol=1e-4, rtol=0)
    assert torch.allclose(a_land, b_land, atol=1e-4, rtol=0)
    return fin


def test_predict_flight_matches_reference_random_serves():
    pos, vel = _random_serves(1000, 0)
    fin = _check(pos, vel, dt=0.02, max_steps=200)
    # The sample must cover both outcomes (crossed the net / never crossed forward).
    assert 50 < int(fin.sum()) < 950


def test_predict_flight_matches_reference_edge_cases():
    pos = torch.tensor([[-2.0, 0.0, 1.0], [-2.0, 0.0, 0.02], [-0.01, 0.0, 1.0], [-2.0, 0.0, 1.0]])
    vel = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.5, 0.0, 0.0], [-8.0, 0.0, 3.0]])
    _check(pos, vel, dt=0.02, max_steps=200)
    for check_every in (1, 3, 7, 1000):
        a = rules.predict_flight(*_random_serves(300, 1), 0.17, dt=0.02, max_steps=200, check_every=check_every)
        b = predict_flight_ref(*_random_serves(300, 1), 0.17, dt=0.02, max_steps=200)
        assert torch.equal(torch.isinf(a[0]), torch.isinf(b[0]))
        assert torch.allclose(a[1], b[1], atol=1e-4, rtol=0)


def test_predict_flight_truncated_flight_matches():
    """Rows still in the air at max_steps must also agree (no early exit happens)."""
    pos, vel = _random_serves(200, 2)
    _check(pos, vel, dt=0.02, max_steps=int(0.3 / 0.02))


def test_predict_flight_single_row():
    pos, vel = _random_serves(5, 3)
    for i in range(5):
        _check(pos[i : i + 1], vel[i : i + 1], dt=0.02, max_steps=200)

