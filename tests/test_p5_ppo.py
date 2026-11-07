"""CPU tests for the P5 PPO extension (P5_RECEIVE_DESIGN 4.3). No Isaac import."""

from __future__ import annotations

import copy
import math

import pytest
import torch

rsl_rl = pytest.importorskip("rsl_rl")
from rsl_rl.algorithms import PPO  # noqa: E402
from rsl_rl.modules import ActorCritic  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from _serve_pkg import load  # noqa: E402

X = load("ppo_ext")["ppo_ext"]
N_ENV, T, A = 16, 8, 3
GROUPS = {"policy": ["policy"], "critic": ["policy", "critic"]}
KW = dict(num_learning_epochs=2, num_mini_batches=2, learning_rate=1e-3, schedule="adaptive", desired_kl=0.01,
          gamma=0.99, lam=0.95, entropy_coef=0.001, device="cpu")


def _obs(g, learn=None, teach=None):
    aux = torch.stack((torch.ones(N_ENV) if learn is None else learn, torch.zeros(N_ENV) if teach is None else teach), -1)
    return TensorDict({"policy": torch.randn(N_ENV, 8, generator=g), "critic": torch.randn(N_ENV, 4, generator=g),
                       "aux": aux}, batch_size=[N_ENV])


def _policy():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    return ActorCritic(_obs(g), GROUPS, A, actor_hidden_dims=[32, 32], critic_hidden_dims=[32, 32], init_noise_std=0.3,
                       noise_std_type="log", actor_obs_normalization=True, critic_obs_normalization=True)


def _fill(alg, seed=1, learn=None, teach=None):
    g = torch.Generator().manual_seed(seed)
    obs = _obs(g, learn, teach)
    alg.init_storage("rl", N_ENV, T, obs, (A,))
    for _ in range(T):
        alg.act(obs)
        obs = _obs(g, learn, teach)
        alg.process_env_step(obs, torch.randn(N_ENV, generator=g), (torch.rand(N_ENV, generator=g) < 0.1).long(), {})
    alg.compute_returns(obs)


def _clone(policy):
    """deepcopy without the last action distribution (non-leaf tensors)."""
    dist, policy.distribution = policy.distribution, None
    try:
        return copy.deepcopy(policy)
    finally:
        policy.distribution = dist


def _ext(policy, **kw):
    args = dict(std_min=1e-3, std_max=10.0, critic_only_iters=0)
    args.update(kw)
    return X.PPOExt(policy, **args, **KW)


def test_beta_zero_matches_rsl_rl_ppo_elementwise():
    base = PPO(_policy(), **KW)
    _fill(base)
    ext = _ext(_clone(base.policy))
    ext.storage = copy.deepcopy(base.storage)
    assert ext.beta() == 0.0
    torch.manual_seed(7)
    lb = base.update()
    torch.manual_seed(7)
    le = ext.update()
    for (na, pa), (nb, pb) in zip(base.policy.named_parameters(), ext.policy.named_parameters()):
        assert torch.equal(pa, pb), na
    assert base.learning_rate == ext.learning_rate
    for k in ("value_function", "surrogate", "entropy"):
        assert le[k] == pytest.approx(lb[k], rel=1e-6, abs=1e-9)


def _batch(alg):
    return next(alg.storage.mini_batch_generator(1, 1))


def test_masked_samples_have_no_gradient():
    learn = (torch.arange(N_ENV) % 2).float()
    ext = _ext(_policy())
    _fill(ext, learn=learn)
    b = list(_batch(ext))
    obs = b[0]
    masked = obs["aux"][:, 0] == 0
    assert 0 < int(masked.sum()) < masked.numel()

    def grads(batch):
        ext.policy.zero_grad()
        loss, _, _, _ = ext.batch_loss(*batch[:8], adapt_lr=False)
        loss.backward()
        return loss.detach(), [p.grad.clone() for p in ext.policy.parameters()]

    l0, g0 = grads(b)
    pert = [x.clone() if torch.is_tensor(x) else x for x in b]
    pert[0] = obs.clone()
    for i in (1, 2, 3, 4, 5, 6, 7):  # actions, values, advantages, returns, log-probs, mu, sigma
        pert[i][masked] = pert[i][masked] + 3.0 + torch.randn_like(pert[i][masked])
    pert[0]["policy"][masked] += 5.0
    pert[0]["critic"][masked] -= 5.0
    l1, g1 = grads(pert)
    assert torch.allclose(l0, l1, atol=1e-6)
    for a, c in zip(g0, g1):
        assert torch.allclose(a, c, atol=1e-6)


def _frozen_teacher(policy):
    frozen = _clone(policy).eval()

    def teacher(x):
        return frozen.act_inference(TensorDict({"policy": x}, batch_size=[x.shape[0]]))

    return teacher


def test_teacher_equal_to_student_has_zero_loss_and_mse():
    teach = torch.ones(N_ENV)
    pol = _policy()
    # Warm start: the actor is frozen for the critic-only updates, so the student stays the teacher.
    ext = _ext(pol, teacher=_frozen_teacher(pol), beta_start=1.0, beta_end=0.1, beta_iters=2000, critic_only_iters=1)
    _fill(ext, teach=teach)
    # The normaliser of the student moved during the rollout; the teacher keeps its own (a copy before it).
    ext.teacher = _frozen_teacher(ext.policy)
    _, parts, mse, n = ext.batch_loss(*_batch(ext)[:8], adapt_lr=False)
    assert n == N_ENV * T
    assert float(parts["teach"]) == 0.0 and mse == 0.0
    ext.update()
    assert ext.last_stats["Teach/mse"] < 1e-6
    assert ext.last_stats["Teach/beta"] == 1.0 and ext.beta() == pytest.approx(1.0 - 0.9 / 2000)  # used, next


def test_teacher_pulls_the_student_back():
    pol = _policy()
    teacher = _frozen_teacher(pol)
    ext = _ext(pol, teacher=teacher, beta_start=1.0, beta_end=1.0, beta_iters=0)
    with torch.no_grad():
        ext.policy.actor[-1].bias.add_(0.5)
    _fill(ext, teach=torch.ones(N_ENV))
    _, parts, mse, _ = ext.batch_loss(*_batch(ext)[:8], adapt_lr=False)
    assert mse == pytest.approx((0.5 / 0.3) ** 2, rel=0.05) and float(parts["teach"]) == pytest.approx(A * mse, rel=1e-5)


def test_beta_schedule_keeps_the_floor():
    ext = _ext(_policy(), teacher=lambda x: x, beta_start=1.0, beta_end=0.1, beta_iters=2000)
    for u, b in ((0, 1.0), (1000, 0.55), (2000, 0.1), (10000, 0.1)):
        ext.updates = u
        assert ext.beta() == pytest.approx(b)
    assert _ext(_policy()).beta() == 0.0  # no teacher


def test_std_is_clamped_and_reset():
    ext = _ext(_policy(), std_min=0.15, std_max=0.40)
    with torch.no_grad():
        ext.policy.log_std.fill_(math.log(1.0))
        ext.policy.log_std[0] = math.log(0.01)
    _fill(ext)
    ext.update()
    std = ext.policy.log_std.exp()
    assert float(std.max()) <= 0.40 + 1e-6 and float(std.min()) >= 0.15 - 1e-6
    ext.reset_std(0.3)
    assert torch.allclose(ext.policy.log_std.exp(), torch.full((A,), 0.3))
    assert ext.std_mean() == pytest.approx(0.3)


def test_critic_only_warmup_freezes_actor_and_lr():
    ext = _ext(_policy(), critic_only_iters=2)
    actor0 = [p.clone() for p in ext.actor_parameters()]
    critic0 = [p.clone() for p in ext.policy.critic.parameters()]
    lr0 = ext.learning_rate
    for _ in range(2):
        _fill(ext)
        ext.update()
        assert ext.last_stats["Policy/critic_only"] == 1.0
    assert all(torch.equal(a, b) for a, b in zip(actor0, ext.actor_parameters()))
    assert not all(torch.equal(a, b) for a, b in zip(critic0, ext.policy.critic.parameters()))
    assert ext.learning_rate == lr0
    _fill(ext)
    ext.update()
    assert ext.last_stats["Policy/critic_only"] == 0.0
    assert not all(torch.equal(a, b) for a, b in zip(actor0, ext.actor_parameters()))
    assert ext.state_dict_extra() == {"updates": 3}
