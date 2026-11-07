"""CPU tests for the V2 multi-critic PPO (P5_RECEIVE_DESIGN_V2 2.4). No Isaac import."""

from __future__ import annotations

import copy

import pytest
import torch

rsl_rl = pytest.importorskip("rsl_rl")
from rsl_rl.algorithms import PPO  # noqa: E402
from rsl_rl.modules import ActorCritic  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from _serve_pkg import load  # noqa: E402

M = load("ppo_ext", "ppo_multicritic", "serve_reward", "receive")
MC, RC = M["ppo_multicritic"], M["receive"]
N_ENV, T, A, G = 16, 8, 3, 3
GROUPS = {"policy": ["policy"], "critic": ["policy", "critic"]}
KW = dict(num_learning_epochs=2, num_mini_batches=2, learning_rate=1e-3, schedule="adaptive", desired_kl=0.01,
          gamma=0.99, lam=0.95, entropy_coef=0.0, device="cpu")
NET = dict(actor_hidden_dims=[32, 32], critic_hidden_dims=[32, 32], init_noise_std=0.3, noise_std_type="log",
           actor_obs_normalization=True, critic_obs_normalization=True)


def _obs(g, aux=None):
    if aux is None:
        aux = torch.cat((torch.ones(N_ENV, 1), torch.zeros(N_ENV, 1), torch.ones(N_ENV, G)), -1)
    return TensorDict({"policy": torch.randn(N_ENV, 8, generator=g), "critic": torch.randn(N_ENV, 4, generator=g),
                       "aux": aux}, batch_size=[N_ENV])


def _mc_policy(num_values=G, freeze=False):
    torch.manual_seed(0)
    return MC.MultiCriticActorCritic(_obs(torch.Generator().manual_seed(0)), GROUPS, A, num_values=num_values,
                                     freeze_actor_normalizer=freeze, **NET)


def _alg(policy, mode="multi", **kw):
    args = dict(std_min=1e-3, std_max=10.0, critic_only_iters=0, group_mode=mode)
    args.update(kw)
    return MC.MultiCriticPPO(policy, **args, **KW)


def _fill(alg, seed=1, aux=None, group_rewards=None, time_outs=None, dones=None):
    g = torch.Generator().manual_seed(seed)
    obs = _obs(g, aux)
    alg.init_storage("rl", N_ENV, T, obs, (A,))
    for t in range(T):
        alg.act(obs)
        obs = _obs(g, aux)
        gr = torch.randn(N_ENV, G, generator=g) if group_rewards is None else group_rewards[t]
        d = (torch.rand(N_ENV, generator=g) < 0.1).long() if dones is None else dones[t]
        ex = {"group_rewards": gr}
        if time_outs is not None:
            ex["time_outs"] = time_outs[t]
        alg.process_env_step(obs, gr.sum(-1), d, ex)
    alg.compute_returns(obs)
    return obs


def _clone(policy):
    dist, policy.distribution = policy.distribution, None
    try:
        return copy.deepcopy(policy)
    finally:
        policy.distribution = dist


# ---------------------------------------------------------------------- equivalence with RSL-RL PPO
def test_single_group_matches_rsl_rl_ppo():
    """One group, w = 1, all masks 1: same advantages / returns as RSL-RL, same parameters after an update."""
    torch.manual_seed(0)
    base_pol = ActorCritic(_obs(torch.Generator().manual_seed(0)), GROUPS, A, **NET)
    pol = _mc_policy(num_values=1)
    pol.load_state_dict(base_pol.state_dict())
    base = PPO(base_pol, **KW)
    alg = _alg(pol, mode="single")
    for a in (base, alg):
        g = torch.Generator().manual_seed(1)
        obs = _obs(g)
        a.init_storage("rl", N_ENV, T, obs, (A,))
        for _ in range(T):
            torch.manual_seed(3)
            a.act(obs)
            obs = _obs(g)
            gr = torch.randn(N_ENV, G, generator=g)
            d = (torch.rand(N_ENV, generator=g) < 0.1).long()
            to = (torch.rand(N_ENV, generator=g) < 0.2).float()
            a.process_env_step(obs, gr.sum(-1), d, {"group_rewards": gr, "time_outs": to})
        a.compute_returns(obs)
    assert torch.allclose(base.storage.returns, alg.storage.returns, atol=1e-5)
    assert torch.allclose(base.storage.advantages, alg.storage.advantages, atol=1e-4)
    torch.manual_seed(7)
    lb = base.update()
    torch.manual_seed(7)
    la = alg.update()
    for (n, pa), pb in zip(base.policy.named_parameters(), alg.policy.parameters()):
        assert torch.allclose(pa, pb, atol=1e-5), n
    assert la["value_function"] == pytest.approx(lb["value_function"], rel=1e-4)
    assert la["surrogate"] == pytest.approx(lb["surrogate"], rel=1e-3, abs=1e-6)


# ---------------------------------------------------------------------- per-group GAE and normalisation
def test_per_group_normalisation_and_empty_group():
    aux = torch.cat((torch.ones(N_ENV, 1), torch.zeros(N_ENV, 1), torch.ones(N_ENV, G)), -1)
    aux[: N_ENV // 2, 2] = 0.0  # serve group on half the samples
    aux[:, 3] = 0.0  # no receive sample at all
    aux[-2:, 0] = 0.0  # two feeder samples: learn_w 0
    alg = _alg(_mc_policy())
    _fill(alg, aux=aux)
    st = alg.storage
    m = st.group_mask * st.observations["aux"][..., 0:1]
    for gi in (0, 2):
        a = st.group_adv[..., gi][m[..., gi] > 0]
        assert abs(float(a.mean())) < 1e-5 and float(a.std()) == pytest.approx(1.0, rel=1e-4)
        assert torch.count_nonzero(st.group_adv[..., gi][m[..., gi] == 0]) == 0
    assert torch.count_nonzero(st.group_adv[..., 1]) == 0 and torch.isfinite(st.advantages).all()
    want = st.group_adv[..., 0] * 1.0 + st.group_adv[..., 1] * 1.0 + st.group_adv[..., 2] * 0.5
    assert torch.allclose(st.advantages[..., 0], want)
    assert alg._rollout_stats["Critic/adv_std_recv"] == 0.0 and alg._rollout_stats["Critic/share_recv"] == 0.0


def test_gae_is_per_group_with_shared_dones():
    """Rewards only in group 1: group 0 / 2 returns are pure value bootstraps, group 1 follows its own rewards."""
    alg = _alg(_mc_policy())
    gr = torch.zeros(T, N_ENV, G)
    gr[..., 1] = 1.0
    dones = torch.zeros(T, N_ENV, dtype=torch.long)
    dones[3] = 1
    with torch.no_grad():
        for p in alg.policy.critic.parameters():
            p.zero_()
    _fill(alg, group_rewards=gr, dones=dones)
    st = alg.storage
    assert torch.count_nonzero(st.returns[..., 0]) == 0 and torch.count_nonzero(st.returns[..., 2]) == 0
    # Values 0: return_t = sum_k (gamma lam)^k r ... cut at the done of step 3.
    gl = 0.99 * 0.95
    assert float(st.returns[3, 0, 1]) == pytest.approx(1.0)
    assert float(st.returns[0, 0, 1]) == pytest.approx(1 + gl + gl ** 2 + gl ** 3, rel=1e-5)


def test_time_out_bootstrap_is_per_group():
    alg = _alg(_mc_policy())
    g = torch.Generator().manual_seed(1)
    obs = _obs(g)
    alg.init_storage("rl", N_ENV, T, obs, (A,))
    alg.act(obs)
    v = alg.transition.values.clone()
    gr = torch.zeros(N_ENV, G)
    to = torch.zeros(N_ENV)
    to[0] = 1.0
    alg.process_env_step(_obs(g), gr.sum(-1), torch.zeros(N_ENV, dtype=torch.long), {"group_rewards": gr, "time_outs": to})
    r = alg.storage.rewards[0]
    assert torch.allclose(r[0], 0.99 * v[0]) and torch.count_nonzero(r[1:]) == 0


def test_value_loss_is_masked_per_group():
    alg = _alg(_mc_policy())
    B = 6
    v = torch.zeros(B, G)
    ret = torch.ones(B, G)
    mask = torch.ones(B, G)
    mask[:3, 0] = 0.0
    ret[:3, 0] = 100.0  # outside the serve group: must not count
    lw = torch.ones(B)
    lw[-1] = 0.0
    ret[-1] = 50.0  # learn_w 0: must not count anywhere
    alg.use_clipped_value_loss = False
    loss = alg.value_loss(v, v, ret, lw, mask)
    assert float(loss) == pytest.approx(3.0)
    assert alg._batch_stats["Critic/value_loss_serve"] == pytest.approx(1.0)


# ---------------------------------------------------------------------- heads, warm-up, normaliser
def test_value_head_padding_reproduces_p3_critic():
    pad = load("runner", "ppo_ext", "ppo_multicritic", "p5_runner")["p5_runner"].pad_state_dict
    torch.manual_seed(0)
    p3 = ActorCritic(_obs(torch.Generator().manual_seed(0)), GROUPS, A, **NET)
    p3.critic_obs_normalizer.train()
    p3.update_normalization(_obs(torch.Generator().manual_seed(5)))
    mc = _mc_policy()
    out, rep = pad(p3.state_dict(), mc.state_dict())
    mc.load_state_dict(out)
    assert any("critic.4.weight" in s for s in rep["padded"]) and not rep["missing"]
    p3.eval(), mc.eval()
    x = _obs(torch.Generator().manual_seed(9))
    v3, vm = p3.evaluate(x), mc.evaluate(x)
    assert vm.shape == (N_ENV, G)
    assert torch.allclose(vm[:, 0], v3[:, 0], atol=1e-6)
    assert torch.count_nonzero(vm[:, 1:]) == 0


def test_critic_only_moves_no_actor_parameter_even_with_adam_moments():
    alg = _alg(_mc_policy(), critic_only_iters=2)
    _fill(alg)
    alg.updates = 2
    alg.update()  # one normal update: Adam moments for the actor exist
    alg.updates = 0
    actor0 = [p.clone() for p in alg.actor_parameters()]
    critic0 = [p.clone() for p in alg.policy.critic.parameters()]
    for _ in range(2):
        _fill(alg)
        alg.update()
        assert alg.last_stats["Policy/critic_only"] == 1.0
    assert all(torch.equal(a, b) for a, b in zip(actor0, alg.actor_parameters()))
    assert not all(torch.equal(a, b) for a, b in zip(critic0, alg.policy.critic.parameters()))
    assert {"Critic/value_loss_serve", "Critic/explained_var_recv", "Critic/adv_std_body"} <= set(alg.last_stats)


def test_frozen_actor_normaliser():
    pol = _mc_policy(freeze=True)
    pol.actor_obs_normalizer.train()
    pol.critic_obs_normalizer.train()
    a0 = pol.actor_obs_normalizer._mean.clone()
    c0 = pol.critic_obs_normalizer._mean.clone()
    pol.update_normalization(_obs(torch.Generator().manual_seed(3)))
    assert torch.equal(pol.actor_obs_normalizer._mean, a0)
    assert not torch.equal(pol.critic_obs_normalizer._mean, c0)


def test_head_count_must_match_mode():
    with pytest.raises(ValueError):
        _alg(_mc_policy(num_values=1), mode="multi")
    with pytest.raises(ValueError):
        _alg(_mc_policy(num_values=G), mode="single")


# ---------------------------------------------------------------------- reward groups
def test_reward_group_mapping():
    serve = ("serve_legal_contact", "serve_swing_speed", "serve_face_quality", "serve_outcome_pred", "serve_launch_dir",
             "serve_serve_result", "serve_serve_fault", "serve_no_hit", "serve_tunnel")
    body = ("serve_hold_ready", "serve_post_stand", "serve_fall", "anti_fall", "root_orientation", "racket_ground",
            "joint_limit", "action_rate", "foot_slip", "recv_post_stand")
    assert all(RC.reward_group(t) == 0 for t in serve)
    assert all(RC.reward_group(t) == 2 for t in body)
    assert all(RC.reward_group(t) == 1 for t in RC.RECV_TERMS if t != "recv_post_stand")


def test_group_mask_by_role():
    server = torch.tensor([0, 1, 0, 1])
    natural = torch.tensor([True, True, False, False])
    m = RC.group_mask(server, natural)
    # natural, server 0: side 0 serve+body, side 1 recv+body
    assert m[0, 0].tolist() == [1, 0, 1] and m[0, 1].tolist() == [0, 1, 1]
    assert m[1, 1].tolist() == [1, 0, 1] and m[1, 0].tolist() == [0, 1, 1]
    # feeder rally: the feeder (server) has nothing, the receiver recv + body
    assert m[2, 0].tolist() == [0, 0, 0] and m[2, 1].tolist() == [0, 1, 1]
    assert m[3, 1].tolist() == [0, 0, 0] and m[3, 0].tolist() == [0, 1, 1]


def test_every_serve_term_has_a_group():
    sr = M["serve_reward"]
    names = {f"serve_{t}" for t in sr.TERMS}
    assert set(RC.SERVE_GROUP_TERMS) <= names
    assert names - set(RC.SERVE_GROUP_TERMS) == {"serve_hold_ready", "serve_fall", "serve_post_stand"}


def test_save_and_restore_keeps_the_three_heads_and_statistics():
    """V2 2.4: state dict round trip; the restored policy gives the same three values, returns and group stats."""
    alg = _alg(_mc_policy(freeze=True))
    obs = _fill(alg)
    alg.update()
    state = copy.deepcopy(alg.policy.state_dict())
    opt = copy.deepcopy(alg.optimizer.state_dict())
    alg2 = _alg(_mc_policy(freeze=True))
    alg2.policy.load_state_dict(state)
    alg2.optimizer.load_state_dict(opt)
    alg.policy.eval(), alg2.policy.eval()
    assert torch.equal(alg.policy.evaluate(obs), alg2.policy.evaluate(obs))
    for a in (alg, alg2):
        a.policy.train()
        _fill(a, seed=5)
    assert torch.allclose(alg.storage.returns, alg2.storage.returns)
    for k, v in alg._rollout_stats.items():
        assert v == pytest.approx(alg2._rollout_stats[k], rel=1e-6, abs=1e-9), k
    assert {"Norm/over5_serve", "Norm/zmax_recv"} <= set(alg._rollout_stats)
