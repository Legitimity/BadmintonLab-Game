"""P5_RECEIVE_DESIGN_V2 3.2: the A / A' fixes stay in the V2 branch, one regression test each.

| fix | test |
|---|---|
| warm start keeps P3's optimizer state and learning rate | ``test_p5_curriculum::test_padded_optimizer_load_keeps_moments_and_lr`` |
| advantages weighted by ``learn_w`` | ``test_p5_ppo::test_masked_samples_have_no_gradient``, ``test_learn_w_zero_samples_get_no_advantage`` below |
| intercept target from the resting racket, reach 0.3 m | ``test_p5_feeder_receive::test_stand_point_puts_resting_racket_on_target``, ``test_racket_reach`` below |
| ``recv_launch_dir`` times cos^2(heading) | ``test_p5_feeder_receive::test_launch_dir_penalises_sideways_return`` |
| ``eval_serve.py`` ``.pt`` opponent (``e4c7ab4``) | ``test_eval_serve_pt_opponent`` below |
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from _serve_pkg import ROOT, load

m = load("serve_reward", "rules", "feeder", "receive", "ppo_ext", "ppo_multicritic")


def _eval_serve():
    spec = importlib.util.spec_from_file_location("eval_serve", ROOT / "scripts" / "eval_serve.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_eval_serve_pt_opponent(tmp_path: Path):
    es = _eval_serve()
    ours = tmp_path / "ours"
    ours.mkdir()
    (ours / "player_cfg.py").write_text("SIDE = 0\nX = 1\n")
    play = tmp_path / "play"
    cfg, model = es.resolve_opponent(str(tmp_path / "baseline.pt"), play, ours, 1)
    assert cfg is None and model == str((tmp_path / "baseline.pt").resolve())
    cfg, model = es.resolve_opponent("zero", play, ours, 1)
    assert model == "zero" and "SIDE = 1\n" in cfg
    cfg, model = es.resolve_opponent("model_test", play, ours, 0)
    assert cfg is None and model.endswith("model_test.pt")
    bdir = tmp_path / "baseline_dir"
    bdir.mkdir()
    cfg, model = es.resolve_opponent(str(bdir), play, ours, 1)  # export without player_cfg: play's default config
    assert cfg is None and model.endswith("policy.pt")
    cfg, model = es.resolve_opponent(str(ours), play, ours, 1)
    assert "SIDE = 1\n" in cfg


def test_racket_reach():
    fd = m["feeder"]
    assert fd.FeederParams().reach == fd.RACKET_REACH == 0.3
    assert tuple(fd.FeederParams().home) == tuple(fd.RACKET_HOME)


def test_learn_w_zero_samples_get_no_advantage():
    st = m["ppo_multicritic"].MultiCriticStorage.__new__(m["ppo_multicritic"].MultiCriticStorage)
    T, N, G = 4, 6, 3
    st.num_transitions_per_env, st.num_envs, st.num_groups = T, N, G
    g = torch.Generator().manual_seed(0)
    st.rewards = torch.randn(T, N, G, generator=g)
    st.values = torch.randn(T, N, G, generator=g)
    st.returns = torch.zeros(T, N, G)
    st.dones = torch.zeros(T, N, 1)
    st.group_mask = torch.ones(T, N, G)
    learn = torch.ones(T, N, 1)
    learn[:, :2] = 0.0
    st.compute_group_returns(torch.zeros(N, G), 0.99, 0.95, (1.0, 1.0, 0.5), learn)
    assert torch.count_nonzero(st.advantages[:, :2]) == 0
    a = st.group_adv[:, 2:, 0]
    assert abs(float(a.mean())) < 1e-5
