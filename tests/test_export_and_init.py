"""CPU tests for export, actor initialisation, runner checkpointing and the submission files."""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
SERVE = ROOT / "source/badmintonlab_game/badmintonlab_game/tasks/manager_based/badmintonlab_game/serve"
PLAY_MODEL = Path("D:/rl-badminton-eval/BadmintonLab-Game-Play/model_test.pt")
sys.path.insert(0, str(ROOT / "scripts"))

import export_torchscript as ex  # noqa: E402
from rsl_rl.modules import ActorCritic  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _policy(obs_dim=95, critic_dim=12, noise=0.3):
    obs = TensorDict({"policy": torch.randn(64, obs_dim), "critic": torch.randn(64, critic_dim)}, batch_size=[64])
    pol = ActorCritic(
        obs,
        {"policy": ["policy"], "critic": ["policy", "critic"]},
        23,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=noise,
        noise_std_type="log",
    )
    pol.train()
    for _ in range(5):
        pol.update_normalization(TensorDict({"policy": torch.randn(256, obs_dim) * 3 + 1, "critic": torch.randn(256, critic_dim)}, batch_size=[256]))
    return pol, obs


def test_export_matches_act_inference(tmp_path):
    pol, _ = _policy()
    ckpt = tmp_path / "model.pt"
    torch.save({"model_state_dict": pol.state_dict(), "iter": 7, "infos": {"samples": 123}}, ckpt)
    obs = torch.randn(1024, 95) * 3 + 1
    torch.save(obs, tmp_path / "obs.pt")
    meta = ex.export(ckpt, tmp_path / "out", tmp_path / "obs.pt")
    assert meta["max_abs_err"] < 1e-5 and meta["obs_dim"] == 95 and meta["checked_on"] == "rollout"
    model = torch.jit.load(str(tmp_path / "out/policy.pt"))
    pol.eval()
    with torch.inference_mode():
        ref = pol.act_inference(TensorDict({"policy": obs, "critic": torch.zeros(1024, 12)}, batch_size=[1024]))
        assert (model(obs) - ref).abs().max() < 1e-5
    assert (tmp_path / "out/player_cfg.py").exists()


@pytest.mark.skipif(not PLAY_MODEL.exists(), reason="play checkout not found")
def test_exported_structure_matches_course_model(tmp_path):
    """Same module types and normaliser formula as the course's model_test.pt."""
    pol, _ = _policy(obs_dim=120)
    torch.save({"model_state_dict": pol.state_dict()}, tmp_path / "m.pt")
    ex.export(tmp_path / "m.pt", tmp_path / "o")
    ours = torch.jit.load(str(tmp_path / "o/policy.pt"))
    ref = torch.jit.load(str(PLAY_MODEL))
    assert [type(m).__name__ for m in ours.children()] == [type(m).__name__ for m in ref.children()]
    assert ours.code.strip() == ref.code.strip()
    assert getattr(ours, "0").code.strip() == getattr(ref, "0").code.strip()


def test_course_model_is_reproduced_exactly():
    """Rebuilding model_test.pt's own weights through our exporter gives identical actions."""
    if not PLAY_MODEL.exists():
        pytest.skip("play checkout not found")
    ref = torch.jit.load(str(PLAY_MODEL))
    norm, mlp = getattr(ref, "0"), getattr(ref, "1")
    state = {f"actor_obs_normalizer.{k}": v for k, v in norm.state_dict().items()}
    state.update({f"actor.{k}": v for k, v in mlp.state_dict().items()})
    ours = ex.build_policy(state)
    x = torch.randn(512, 120) * 2
    with torch.inference_mode():
        assert (ours(x) - ref(x)).abs().max() < 1e-6


def test_actor_output_init_holds_pose():
    wrapper = _load_wrapper()
    pol, _ = _policy()
    bias = torch.linspace(-0.4, 0.6, 23)
    wrapper.init_actor_output(pol, bias)
    pol.eval()
    with torch.inference_mode():
        out = pol.act_inference(TensorDict({"policy": torch.randn(2048, 95), "critic": torch.zeros(2048, 12)}, batch_size=[2048]))
    # Normalised inputs are O(1); the scaled weights keep the output within a few hundredths of the bias.
    assert (out - bias).abs().mean() < 0.05
    # The course config uses noise_std_type="log".
    assert pol.log_std.exp().allclose(torch.full((23,), 0.3), atol=1e-6)


def test_hold_action_inverts_joint_action():
    wrapper = _load_wrapper()
    offset, scale = torch.randn(23), torch.rand(23) + 0.5
    q = torch.randn(23)
    raw = wrapper.hold_action(q, offset, scale)
    assert torch.allclose(raw * scale + offset, q, atol=1e-6)


def _load_wrapper():
    """wrapper.py imports the parent wrapper relatively; give it a minimal package."""
    pkg = types.ModuleType("_serve_pkg")
    pkg.__path__ = [str(SERVE.parent)]
    sys.modules["_serve_pkg"] = pkg
    parent = types.ModuleType("_serve_pkg.shared_policy_wrapper")
    parent.SharedPolicyVecEnvWrapper = object
    sys.modules["_serve_pkg.shared_policy_wrapper"] = parent
    sub = types.ModuleType("_serve_pkg.serve")
    sub.__path__ = [str(SERVE)]
    sys.modules["_serve_pkg.serve"] = sub
    return _load("_serve_pkg.serve.wrapper", SERVE / "wrapper.py")


def test_submission_player_cfg_is_self_contained():
    """play loads player_cfg.py from its own package: only Isaac Lab and court imports are allowed."""
    tree = ast.parse((SERVE / "player_cfg.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative imports would break inside play"
            assert node.module.split(".")[0] in {"__future__", "isaaclab", "typing", "torch", "math"}, node.module
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in {"torch", "math"}, alias.name
    names = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert {"make_obs_cfg", "make_events_cfg"} <= names
    assert "ObsCfg" in (SERVE / "player_cfg.py").read_text(encoding="utf-8")
