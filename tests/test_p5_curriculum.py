"""CPU tests: P5 V2 curriculum (R0 -> R1 -> R2 gates, blend, face tolerance, feeder level, natural share) and the
P3 -> P5 warm-start padding (V2 3.2: P3 optimizer state and learning rate kept)."""

import math

import torch

from _serve_pkg import load

m = load("serve_reward", "receive", "p5_curriculum")
cur_mod, recv = m["p5_curriculum"], m["receive"]

R0_OK = {"rallies": 100.0, "st_err_p50": 0.2, "tf_err_p50": 20.0, "fell": 0.01, "touch": 0.1}


def _small(**kw):
    c = cur_mod.P5Curriculum(ema_iters=5, **kw)
    c.stages = (cur_mod.StageSpec(10, 40, c.stages[0].gate), cur_mod.StageSpec(10, 50, c.stages[1].gate),
                cur_mod.StageSpec(10, None, None))
    return c


def test_r0_gate_needs_all_three_and_min_iters():
    c = _small()
    assert c.stage_name == "R0" and c.level == 2
    for it in range(9):
        c.update(it, R0_OK)
        assert c.stage == 0
    c.update(9, R0_OK)
    assert c.stage_name == "R1" and c.events == ["passed_R0"] and c.iters_in_stage == 0 and c.level == 2
    for bad in ({"st_err_p50": 0.5}, {"tf_err_p50": 50.0}, {"fell": 0.05}):
        c2 = _small()
        for it in range(30):
            c2.update(it, {**R0_OK, **bad})
        assert c2.stage == 0, bad


def test_max_iters_promotes_with_alert_and_r2_opens_l3():
    c = _small()
    events = []
    for it in range(40):
        c.update(it, {"rallies": 100.0, "st_err_p50": 1.0, "tf_err_p50": 90.0, "fell": 0.0, "touch": 0.0})
        events += c.events
    assert events == ["max_R0"] and c.stage_name == "R1" and "max_R0" in c.alerted
    for it in range(40, 60):
        c.update(it, {"rallies": 100.0, "touch": 0.6})
        events += c.events
    assert c.stage_name == "R2" and events[-1] == "passed_R1" and c.level == 3
    for it in range(60, 200):
        c.update(it, {"rallies": 100.0, "touch": 0.9})
    assert c.stage_name == "R2"  # no gate after R2


def test_nan_gate_metric_does_not_pass():
    c = _small()
    for it in range(30):
        c.update(it, {**R0_OK, "st_err_p50": float("nan")})
    assert c.stage == 0 and "st_err_p50" not in c.ema


def test_blend_and_face_tolerance_count_separately():
    c = _small(blend_iters=10, face_anneal_iters=20)
    r0, r1 = recv.RECV_STAGE_WEIGHTS["R0"], recv.RECV_STAGE_WEIGHTS["R1"]
    assert c.weights() == {k: float(r0[k]) for k in recv.RECV_TERMS} and c.face_tolerance() == 90.0
    for it in range(10):
        c.update(it, R0_OK)
    assert c.stage_name == "R1"
    for it in range(10, 15):
        w = c.update(it, {"rallies": 100.0, "touch": 0.0})
    assert c.blend() == 0.5
    for k in recv.RECV_TERMS:
        assert abs(w[k] - 0.5 * (r0[k] + r1[k])) < 1e-9
    assert abs(c.face_tolerance() - (90.0 - 45.0 * 5 / 20)) < 1e-9
    for it in range(15, 25):
        w = c.update(it, {"rallies": 100.0, "touch": 0.0})
    assert c.blend() == 1.0 and all(abs(w[k] - r1[k]) < 1e-9 for k in recv.RECV_TERMS)
    assert abs(c.face_tolerance() - (90.0 - 45.0 * 15 / 20)) < 1e-9
    for it in range(25, 40):
        c.update(it, {"rallies": 100.0, "touch": 0.0})
    assert c.face_tolerance() == 45.0


def test_group_weights_per_stage_blend_with_receive_weights():
    """V2 2.5: R0 1.0/1.0/0.8/0.2, R1 = R2 1.0/1.5/0.5/0.2 (serve, recv, fall, body), switched with the same blend."""
    assert cur_mod.DEFAULT_GROUP_WEIGHTS == {"R0": (1.0, 1.0, 0.8, 0.2), "R1": (1.0, 1.5, 0.5, 0.2),
                                             "R2": (1.0, 1.5, 0.5, 0.2)}
    c = _small(blend_iters=10)
    assert c.group_weights() == (1.0, 1.0, 0.8, 0.2)
    for it in range(10):
        c.update(it, R0_OK)
    for it in range(10, 15):
        c.update(it, {"rallies": 100.0, "touch": 0.0})
    assert c.blend() == 0.5
    assert all(abs(a - b) < 1e-9 for a, b in zip(c.group_weights(), (1.0, 1.25, 0.65, 0.2)))
    assert abs(c.log_values()["P5Curr/w_fall"] - 0.65) < 1e-9
    for it in range(15, 25):
        c.update(it, {"rallies": 100.0, "touch": 0.0})
    assert c.group_weights() == (1.0, 1.5, 0.5, 0.2)
    try:
        cur_mod.P5Curriculum(group_weight_table={"R0": (1.0, 1.0, 0.5)})
        raise AssertionError("3 weights accepted")
    except ValueError:
        pass


def test_stage_weights_table():
    w = recv.RECV_STAGE_WEIGHTS
    assert set(w) == {"R0", "R1", "R2"} and all(set(v) == set(recv.RECV_TERMS) for v in w.values())
    col = lambda k: tuple(w[s][k] for s in ("R0", "R1", "R2"))  # noqa: E731
    assert col("recv_approach") == (2.0, 1.0, 0.3) and col("recv_station") == (1.0, 0.5, 0.2)
    assert col("recv_torso_face") == (1.0, 0.5, 0.2) and col("recv_racket_near") == (0.5, 1.0, 0.5)
    assert col("recv_contact") == (0.0, 2.0, 1.0) and col("recv_face") == (0.0, 0.5, 1.0)
    assert col("recv_swing") == (0.0, 0.3, 0.5) and col("recv_outcome_pred") == (0.0, 1.0, 4.0)
    assert col("recv_launch_dir") == (0.0, 0.5, 1.0) and col("recv_result") == (0.0, 5.0, 10.0)
    assert col("recv_miss") == (-1.0, -2.0, -3.0) and col("recv_tunnel") == (0.0, -1.0, -2.0)


def test_few_rallies_do_not_move_ema():
    c = cur_mod.P5Curriculum(min_rallies=32)
    c.update(0, {"rallies": 10.0, "touch": 0.9})
    assert c.ema == {} and c.ema_count == 0
    c.update(1, {"rallies": 64.0, "touch": 0.5})
    assert abs(c.ema["touch"] - 0.5) < 1e-9


def test_natural_share_clamp_and_state_roundtrip():
    c = cur_mod.P5Curriculum(natural_share=0.9)
    assert c.natural_share == 0.6
    assert c.set_natural_share(0.1) and c.natural_share == 0.4
    for it in range(5):
        c.update(it, R0_OK)
    d = c.state_dict()
    c2 = cur_mod.P5Curriculum()
    c2.load_state_dict(d)
    assert c2.state_dict() == d
    c2.load_state_dict({**d, "level": 1, "iters_in_level": 7})  # a V1 checkpoint's keys are ignored
    assert c2.level == 2 and math.isfinite(c2.face_tolerance())


def test_padded_optimizer_load_keeps_moments_and_lr():
    from types import SimpleNamespace

    runner_cls = load("runner", "ppo_ext", "p5_runner")["p5_runner"].P5Runner
    a = [torch.nn.Parameter(torch.randn(4, 3)), torch.nn.Parameter(torch.randn(4))]
    b = [torch.nn.Parameter(torch.randn(4, 5)), torch.nn.Parameter(torch.randn(4))]
    oa = torch.optim.Adam(a, lr=1e-5)
    (a[0].sum() + a[1].pow(2).sum()).backward()
    oa.step()
    ob = torch.optim.Adam(b, lr=1e-3)
    fake = SimpleNamespace(alg=SimpleNamespace(optimizer=ob, learning_rate=1e-3), device="cpu")
    msg = runner_cls._load_padded_optimizer(fake, oa.state_dict())
    assert msg.startswith("loaded") and fake.alg.learning_rate == 1e-5 and ob.param_groups[0]["lr"] == 1e-5
    m = ob.state[b[0]]["exp_avg"]
    assert m.shape == (4, 5) and torch.equal(m[:, :3], oa.state[a[0]]["exp_avg"]) and torch.count_nonzero(m[:, 3:]) == 0
    assert torch.equal(ob.state[b[1]]["exp_avg_sq"], oa.state[a[1]]["exp_avg_sq"])


def test_padded_optimizer_value_head_rows():
    """1 -> 3 value head: P3's moments in row 0, zeros for the new heads."""
    from types import SimpleNamespace

    runner_cls = load("runner", "ppo_ext", "p5_runner")["p5_runner"].P5Runner
    a = [torch.nn.Parameter(torch.randn(1, 6)), torch.nn.Parameter(torch.randn(1))]
    b = [torch.nn.Parameter(torch.randn(3, 6)), torch.nn.Parameter(torch.randn(3))]
    oa = torch.optim.Adam(a, lr=2e-5)
    (a[0].pow(2).sum() + a[1].sum()).backward()
    oa.step()
    ob = torch.optim.Adam(b, lr=1e-3)
    fake = SimpleNamespace(alg=SimpleNamespace(optimizer=ob, learning_rate=1e-3), device="cpu")
    assert runner_cls._load_padded_optimizer(fake, oa.state_dict()).startswith("loaded")
    m = ob.state[b[0]]["exp_avg"]
    assert torch.equal(m[:1], oa.state[a[0]]["exp_avg"]) and torch.count_nonzero(m[1:]) == 0


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
