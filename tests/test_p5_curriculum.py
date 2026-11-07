"""CPU tests: P5 curriculum (levels, EMA gates, R1 -> R2 blend, natural share) and the P3 -> P5 warm-start padding."""

import torch

from _serve_pkg import load

m = load("serve_reward", "receive", "p5_curriculum")
cur_mod, recv = m["p5_curriculum"], m["receive"]


def _rates(touch, crossed, n=100.0):
    return {"f_rallies": n, "f_touch": touch, "f_crossed": crossed}


def test_level_needs_min_iters_and_gate():
    c = cur_mod.P5Curriculum(ema_iters=10)
    c.levels = tuple(cur_mod.LevelSpec(min_iters=20, max_iters=100, gate=s.gate) for s in c.levels)
    for it in range(19):
        c.update(it, _rates(0.9, 0.9))
        assert c.level == 0
    c.update(19, _rates(0.9, 0.9))
    assert c.level == 1 and c.events == ["passed_L0"] and c.iters_in_level == 0


def test_gate_not_met_alerts_at_max_once_and_stays():
    c = cur_mod.P5Curriculum(ema_iters=5)
    c.levels = tuple(cur_mod.LevelSpec(min_iters=5, max_iters=30, gate=s.gate) for s in c.levels)
    events = []
    for it in range(80):
        c.update(it, _rates(0.9, 0.1))  # crossed below 0.3
        events += c.events
    assert c.level == 0 and events == ["max_L0"]


def test_max_level_cap_and_l3_no_gate():
    c = cur_mod.P5Curriculum(ema_iters=2, max_level=1)
    c.levels = tuple(cur_mod.LevelSpec(min_iters=2, max_iters=100, gate=s.gate) for s in c.levels)
    for it in range(50):
        c.update(it, _rates(1.0, 1.0))
    assert c.level == 1
    c2 = cur_mod.P5Curriculum(ema_iters=2, level=3)
    for it in range(50):
        c2.update(it, _rates(1.0, 1.0))
    assert c2.level == 3 and not c2.alerted


def test_few_rallies_do_not_move_ema():
    c = cur_mod.P5Curriculum(min_rallies=32)
    c.update(0, _rates(0.9, 0.9, n=10))
    assert c.ema == {} and c.ema_count == 0
    c.update(1, _rates(0.5, 0.2, n=64))
    assert abs(c.ema["f_touch"] - 0.5) < 1e-9


def test_reward_blend():
    c = cur_mod.P5Curriculum(r1_iters=10, blend_iters=10)
    r1, r2 = recv.RECV_STAGE_WEIGHTS["R1"], recv.RECV_STAGE_WEIGHTS["R2"]
    assert c.weights() == {k: float(r1[k]) for k in recv.RECV_TERMS}
    for it in range(15):
        w = c.update(it, {})
    assert c.stage() == "R1>R2"
    for k in recv.RECV_TERMS:
        assert abs(w[k] - 0.5 * (r1[k] + r2[k])) < 1e-9
    for it in range(15, 30):
        w = c.update(it, {})
    assert c.stage() == "R2" and all(abs(w[k] - r2[k]) < 1e-9 for k in recv.RECV_TERMS)


def test_natural_share_clamp_and_state_roundtrip():
    c = cur_mod.P5Curriculum(natural_share=0.9)
    assert c.natural_share == 0.7
    assert c.set_natural_share(0.1) and c.natural_share == 0.3
    c0 = cur_mod.P5Curriculum(natural_min=0.0, natural_share=0.0)
    assert c0.natural_share == 0.0
    for it in range(5):
        c.update(it, _rates(0.4, 0.2))
    d = c.state_dict()
    c2 = cur_mod.P5Curriculum()
    c2.load_state_dict(d)
    assert c2.state_dict() == d


def test_pad_state_dict_critic_and_normaliser():
    from rsl_rl.modules import ActorCritic
    from tensordict import TensorDict

    pad = load("runner", "ppo_ext", "p5_runner")["p5_runner"].pad_state_dict
    kw = dict(actor_hidden_dims=[16], critic_hidden_dims=[16], activation="elu", actor_obs_normalization=True,
              critic_obs_normalization=True, init_noise_std=0.3, noise_std_type="log")
    obs_a = TensorDict({"policy": torch.randn(8, 5), "critic": torch.randn(8, 3)}, batch_size=[8])
    obs_b = TensorDict({"policy": torch.randn(8, 5), "critic": torch.randn(8, 3), "p5c": torch.randn(8, 4)}, batch_size=[8])
    a = ActorCritic(obs_a, {"policy": ["policy"], "critic": ["policy", "critic"]}, 2, **kw)
    b = ActorCritic(obs_b, {"policy": ["policy"], "critic": ["policy", "critic", "p5c"]}, 2, **kw)
    a.critic_obs_normalizer.train()
    a.update_normalization(obs_a)
    src = a.state_dict()
    out, rep = pad(src, b.state_dict())
    b.load_state_dict(out)
    assert rep["padded"] and not rep["missing"]
    w = b.state_dict()["critic.0.weight"]
    assert torch.equal(w[:, :8], src["critic.0.weight"]) and torch.count_nonzero(w[:, 8:]) == 0
    mean = b.state_dict()["critic_obs_normalizer._mean"]
    assert torch.equal(mean[:, :8], src["critic_obs_normalizer._mean"]) and torch.count_nonzero(mean[:, 8:]) == 0
    # Same value on the old inputs whatever the new columns hold (zero weights).
    a.eval(), b.eval()
    x = obs_b.clone()
    v1 = b.evaluate(x)
    x["p5c"] = torch.randn(8, 4) * 10
    assert torch.allclose(v1, b.evaluate(x))
    assert torch.allclose(v1, a.evaluate(obs_a.update({"policy": x["policy"], "critic": x["critic"]})), atol=1e-6)
    assert torch.equal(b.act_inference(x), a.act_inference(x))
