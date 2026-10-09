import dataclasses
import math

import pytest
import torch

from conftest import make_saddle, make_start
from rcn import minimize
from rcn.reflect import negative_indices

PROBLEMS = [
    # (n, n_neg, cubic, c, start scale)
    (6, 2, 1.0, 1.0, 1e-6),
    (10, 3, 0.3, 1.0, 1e-6),
    (10, 3, 1.0, 1.0, 0.0),  # exact saddle: g = 0, hard case, Delta_ref = 0
]


def run(problem, seed, **kw):
    n, n_neg, cubic, c, scale = problem
    f = make_saddle(n, n_neg, cubic, c, seed)
    x0 = make_start(n, scale, seed)
    return f, minimize(f, x0, **kw)


def comparable(history):
    return [{k: v for k, v in dataclasses.asdict(h).items() if k not in ("time_sec", "time_extra")} for h in history]


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("mode", ["joint", "greedy"])
@pytest.mark.parametrize("seed", range(3))
def test_monotone_against_cn_step(problem, mode, seed):
    _, res = run(problem, seed, method="rcn", M=3.0, mode=mode, eta=math.inf)
    hist = res.history
    for k in range(len(hist) - 1):
        assert hist[k + 1].f <= hist[k].f_cn
        if hist[k].reflect_taken:
            assert hist[k + 1].f < hist[k].f_cn


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("seed", range(3))
def test_eta_zero_matches_cn_bitwise(problem, seed):
    _, cn = run(problem, seed, method="cn", M=3.0)
    _, rcn = run(problem, seed, method="rcn", M=3.0, eta=0.0)
    assert torch.equal(cn.x, rcn.x)
    assert comparable(cn.history) == comparable(rcn.history)
    assert not any(h.reflect_tried for h in rcn.history)


@pytest.mark.parametrize("mode", ["joint", "greedy"])
def test_convex_quadratic_never_reflects(mode):
    gen = torch.Generator().manual_seed(0)
    B = torch.randn(8, 8, generator=gen, dtype=torch.float64)
    A = B @ B.T + 0.1 * torch.eye(8, dtype=torch.float64)
    b = torch.randn(8, generator=gen, dtype=torch.float64)

    def f(x):
        return 0.5 * x @ A @ x - b @ x

    res = minimize(f, torch.zeros(8, dtype=torch.float64), method="rcn", M=1.0, mode=mode, eta=math.inf)
    assert res.status == "sosp"
    assert not any(h.reflect_tried for h in res.history)


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("seed", range(3))
def test_function_evaluation_counters(problem, seed):
    _, cn = run(problem, seed, method="cn", M=3.0)
    _, joint = run(problem, seed, method="rcn", M=3.0, mode="joint", eta=math.inf)
    _, greedy = run(problem, seed, method="rcn", M=3.0, mode="greedy", eta=math.inf)
    n_neg = problem[1]

    for res, per_iter_max in ((cn, 1), (joint, 2), (greedy, 1 + n_neg)):
        hist = res.history
        assert hist[0].n_f - 1 <= per_iter_max  # 1 = f(x0)
        for prev, cur in zip(hist, hist[1:-1]):
            assert cur.n_f - prev.n_f == 1 + cur.extra_f_evals
            assert cur.n_f - prev.n_f <= per_iter_max
        assert all(b.n_grad - a.n_grad == 1 and b.n_hess - a.n_hess == 1 for a, b in zip(hist, hist[1:]))
        assert res.n_f == hist[-1].n_f
        assert res.n_grad == res.n_hess == len(hist)

    assert all(h.extra_f_evals == 0 for h in cn.history)
    assert sum(h.extra_f_evals for h in joint.history) > 0


@pytest.mark.parametrize("method,mode", [("cn", "joint"), ("rcn", "joint"), ("rcn", "greedy")])
def test_same_seed_same_history(method, mode):
    _, a = run(PROBLEMS[1], 7, method=method, M=3.0, mode=mode)
    _, b = run(PROBLEMS[1], 7, method=method, M=3.0, mode=mode)
    assert torch.equal(a.x, b.x)
    assert comparable(a.history) == comparable(b.history)


@pytest.mark.parametrize("seed", range(5))
def test_exact_saddle_first_step_picks_better_sign(seed):
    # At g = 0 the CN step is +r q1 (hard case); RCN must pick the better of +-.
    f, cn = run(PROBLEMS[2], seed, method="cn", M=3.0)
    _, rcn = run(PROBLEMS[2], seed, method="rcn", M=3.0)
    h0 = rcn.history[0]
    assert h0.hard_case and h0.reflect_tried and h0.delta_ref == 0.0
    assert rcn.history[1].f <= cn.history[1].f

    n = PROBLEMS[2][0]
    H = torch.func.hessian(f)(torch.zeros(n, dtype=torch.float64))
    lam, Q = torch.linalg.eigh(0.5 * (H + H.T))
    r = 2 * abs(lam[0].item()) / 3.0
    f_plus, f_minus = f(r * Q[:, 0]).item(), f(-r * Q[:, 0]).item()
    assert cn.history[1].f == pytest.approx(f_plus, rel=1e-12)
    assert rcn.history[1].f == pytest.approx(min(f_plus, f_minus), rel=1e-12)


@pytest.mark.parametrize("method", ["cn", "rcn"])
@pytest.mark.parametrize("problem", PROBLEMS)
def test_reaches_second_order_point(method, problem):
    _, res = run(problem, 0, method=method, M=3.0)
    assert res.status == "sosp"
    last = res.history[-1]
    assert last.grad_norm <= 1e-8
    assert last.lam_min >= -math.sqrt(3.0 * 1e-8)


def test_api_validation():
    f = lambda x: (x**2).sum()  # noqa: E731
    x0 = torch.ones(3, dtype=torch.float64)
    with pytest.raises(ValueError):
        minimize(f, x0, method="bogus")
    with pytest.raises(ValueError):
        minimize(f, x0, M=0.0)
    with pytest.raises(ValueError):
        minimize(f, x0, mode="bogus")


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("M0", [0.1, 3.0, 100.0])
def test_arc_monotone_and_adapts(problem, M0):
    _, res = run(problem, 0, method="arc", M=M0, max_iter=500)
    assert res.status == "sosp"
    hist = res.history
    for a, b in zip(hist, hist[1:]):
        assert b.f <= a.f  # accepted steps decrease f, rejected ones keep x
        if not a.step_accepted:
            assert b.f == a.f and b.n_hess == a.n_hess  # no new Hessian after a rejection
            assert b.M_k == 2 * a.M_k
    assert res.n_hess == 1 + sum(h.step_accepted for h in hist[:-1])
    if M0 == 0.1:
        assert any(not h.step_accepted for h in hist)  # too small: must grow


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("M0", [0.3, 3.0])
def test_rarc_eta_zero_matches_arc_bitwise(problem, M0):
    _, arc = run(problem, 1, method="arc", M=M0)
    _, rarc = run(problem, 1, method="rarc", M=M0, eta=0.0)
    assert torch.equal(arc.x, rarc.x)
    assert comparable(arc.history) == comparable(rarc.history)


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("mode", ["joint", "greedy"])
def test_rarc_monotone_and_never_worse_than_arc_trial(problem, mode):
    _, res = run(problem, 2, method="rarc", M=3.0, mode=mode, eta=math.inf)
    assert res.status == "sosp"
    hist = res.history
    for a, b in zip(hist, hist[1:]):
        assert b.f <= a.f
        if a.step_accepted:
            assert b.f <= a.f_cn
    assert any(h.reflect_tried for h in hist)


def test_callback_sees_every_iterate():
    f = make_saddle(6, 2, 1.0, 1.0, 0)
    x0 = make_start(6, 1e-6, 0)
    seen = []
    res = minimize(f, x0, method="rcn", M=3.0, callback=lambda k, x: seen.append((k, x.clone())))
    assert [k for k, _ in seen] == list(range(len(res.history)))
    assert torch.equal(seen[0][1], x0) and torch.equal(seen[-1][1], res.x)
    assert all(f(x).item() == h.f for (_, x), h in zip(seen, res.history))


def test_default_eta_is_0_1():
    import inspect

    assert inspect.signature(minimize).parameters["eta"].default == 0.1


@pytest.mark.parametrize("problem", PROBLEMS)
def test_nc_pm(problem):
    f, res = run(problem, 0, method="nc_pm", M=3.0, max_iter=2000)
    hist = res.history
    for a, b in zip(hist[:-1], hist[1:-1]):
        if b.extra_f_evals:  # negative-curvature step: exactly two evaluations
            assert b.n_f - a.n_f == 2
    for a, b in zip(hist, hist[1:]):
        assert b.f <= a.f + 1e-12 * max(1.0, abs(a.f))
    assert hist[0].extra_f_evals == 1  # at the saddle the NC step wins
    assert res.status == "sosp"
    assert all(h.reflect_tried is False for h in hist)
