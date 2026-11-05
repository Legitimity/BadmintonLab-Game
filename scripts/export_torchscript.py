"""Export an RSL-RL checkpoint to the TorchScript model that ``play`` loads (design doc 6.5).

CPU only, no Isaac. The model is ``Sequential(EmpiricalNormalization, MLP)``, the same
structure as ``model_test.pt``: ``(x - mean) / (std + eps)`` then the actor, no clipping.

    python scripts/export_torchscript.py --checkpoint logs/.../model_1000.pt --out exports/serve_1000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from rsl_rl.networks import MLP, EmpiricalNormalization

ROOT = Path(__file__).resolve().parents[1]
PLAYER_CFG = ROOT / "source/badmintonlab_game/badmintonlab_game/tasks/manager_based/badmintonlab_game/serve/player_cfg.py"
ACTION_DIM = 23


def build_policy(state: dict, hidden=(512, 256, 128), activation="elu") -> torch.nn.Sequential:
    """Rebuild normaliser + actor from an ``ActorCritic`` state dict."""
    obs_dim = state["actor.0.weight"].shape[1]
    norm = EmpiricalNormalization(obs_dim)
    if "actor_obs_normalizer._mean" in state:
        norm.load_state_dict({k.split(".", 1)[1]: v for k, v in state.items() if k.startswith("actor_obs_normalizer.")})
    actor = MLP(obs_dim, ACTION_DIM, list(hidden), activation)
    actor.load_state_dict({k.split(".", 1)[1]: v for k, v in state.items() if k.startswith("actor.")})
    return torch.nn.Sequential(norm, actor).eval()


def reference_actions(state: dict, obs: torch.Tensor, hidden=(512, 256, 128)) -> torch.Tensor:
    """Independent re-implementation of ``ActorCritic.act_inference`` for the equality check."""
    mean = state.get("actor_obs_normalizer._mean", torch.zeros(1, obs.shape[1]))
    std = state.get("actor_obs_normalizer._std", torch.ones(1, obs.shape[1]))
    x = (obs - mean) / (std + 1e-2)
    n_layers = len(hidden) + 1
    for i in range(n_layers):
        w, b = state[f"actor.{2 * i}.weight"], state[f"actor.{2 * i}.bias"]
        x = x @ w.T + b
        if i < n_layers - 1:
            x = torch.nn.functional.elu(x)
    return x


def self_check(model: torch.jit.ScriptModule, state: dict, obs_sample: torch.Tensor | None) -> dict:
    dim = state["actor.0.weight"].shape[1]
    report = {"obs_dim": dim}
    with torch.inference_mode():
        for n in (1, 7, 4096):
            out = model(torch.randn(n, dim))
            assert out.shape == (n, ACTION_DIM), out.shape
            assert torch.isfinite(out).all()
        std = state.get("actor_obs_normalizer._std", torch.ones(1, dim))
        mean = state.get("actor_obs_normalizer._mean", torch.zeros(1, dim))
        for extreme in (torch.zeros(4, dim), mean + 10 * std, mean - 10 * std):
            assert torch.isfinite(model(extreme)).all()
        obs = obs_sample if obs_sample is not None else torch.randn(1024, dim) * std + mean
        err = (model(obs) - reference_actions(state, obs)).abs().max().item()
        report.update(max_abs_err=err, checked_on="rollout" if obs_sample is not None else "synthetic")
        assert err < 1e-5, err
    return report


def export(checkpoint: Path, out_dir: Path, obs_sample: Path | None = None, player_cfg: Path = PLAYER_CFG) -> dict:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    model = torch.jit.script(build_policy(state))
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "policy.pt"
    model.save(str(model_path))
    loaded = torch.jit.load(str(model_path), map_location="cpu").eval()
    sample = torch.load(obs_sample, map_location="cpu") if obs_sample else None
    report = self_check(loaded, state, sample)
    shutil.copy(player_cfg, out_dir / "player_cfg.py")
    meta = dict(
        checkpoint=str(checkpoint),
        iteration=ckpt.get("iter"),
        samples=(ckpt.get("infos") or {}).get("samples"),
        sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        **report,
    )
    (out_dir / "export_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--obs_sample", type=Path, default=None, help="Tensor of real rollout observations, (N, obs_dim).")
    ap.add_argument("--player_cfg", type=Path, default=PLAYER_CFG)
    a = ap.parse_args()
    print(json.dumps(export(a.checkpoint, a.out, a.obs_sample, a.player_cfg), indent=1))
