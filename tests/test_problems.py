import sys
from pathlib import Path

import pytest
import torch
from torch.func import grad, hessian

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from problems import (  # noqa: E402
    REGISTRY,
    choose_radius,
    estimate_hessian_lipschitz,
    flatten_params,
    saddle_tensor,
    unflatten_params,
)
from problems._common import third_directional  # noqa: E402


@pytest.mark.parametrize("n,n_neg", [(5, 1), (20, 3), (12, 12)])
def test_saddle_tensor_structure(n, n_neg):
    p = saddle_tensor.make(3, n=n, n_neg=n_neg, cubic=1.0)
    zero = torch.zeros(n, dtype=torch.float64)
    assert torch.linalg.vector_norm(grad(p.f)(zero)).item() == 0.0
    lam = torch.linalg.eigvalsh(hessian(p.f)(zero))
    neg, pos = lam[:n_neg], lam[n_neg:]
    assert ((neg > -1.5 - 1e-12) & (neg < -0.5 + 1e-12)).all()
    assert ((pos > 0.5 - 1e-12) & (pos < 2.0 + 1e-12)).all()
    assert torch.linalg.vector_norm(p.x0).item() < 1e-4


def test_saddle_tensor_cubic_term_is_symmetric_and_scaled():
    n = 6
    p1 = saddle_tensor.make(0, n=n, cubic=1.0, c=0.0)
    p3 = saddle_tensor.make(0, n=n, cubic=0.3, c=0.0)
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(n, generator=gen, dtype=torch.float64)
    # f(x) - f(-x) = T[x,x,x] / 3 when c = 0
    odd1 = (p1.f(x) - p1.f(-x)).item()
    odd3 = (p3.f(x) - p3.f(-x)).item()
    assert odd3 == pytest.approx(0.3 * odd1, rel=1e-12)
    # third derivative tensor is symmetric
    T = torch.func.jacfwd(hessian(p1.f))(x)
    for perm in [(1, 0, 2), (0, 2, 1), (2, 1, 0)]:
        assert torch.allclose(T, T.permute(*perm), atol=1e-12)


def test_saddle_tensor_deterministic():
    a, b = saddle_tensor.make(42), saddle_tensor.make(42)
    x = torch.linspace(-1, 1, 20, dtype=torch.float64)
    assert torch.equal(a.x0, b.x0)
    assert a.f(x).item() == b.f(x).item()


@pytest.mark.parametrize(
    "name,kw,origin_saddle",
    [
        ("matrix_factorization", {}, True),
        ("phase_retrieval", {}, True),
        ("small_mlp", {}, False),
        ("test_functions", {"fn": "himmelblau"}, False),
        ("test_functions", {"fn": "cubic_saddle"}, True),
    ],
)
def test_new_problems_start_next_to_a_saddle(name, kw, origin_saddle):
    p = REGISTRY[name](1, **kw)
    assert p.x0.dtype == torch.float64 and p.x0.ndim == 1
    lam = torch.linalg.eigvalsh(hessian(p.f)(p.x0))
    assert lam[0].item() < -1e-3  # negative curvature at the start
    if origin_saddle:
        zero = torch.zeros_like(p.x0)
        assert torch.linalg.vector_norm(grad(p.f)(zero)).item() < 1e-12
    a, b = REGISTRY[name](1, **kw), REGISTRY[name](1, **kw)
    assert torch.equal(a.x0, b.x0) and a.f(a.x0).item() == b.f(b.x0).item()


@pytest.mark.parametrize("seed", range(4))
def test_himmelblau_saddles_are_refined(seed):
    p = REGISTRY["test_functions"](seed, fn="himmelblau", start_scale=0.0)
    assert torch.linalg.vector_norm(grad(p.f)(p.x0)).item() < 1e-10
    lam = torch.linalg.eigvalsh(hessian(p.f)(p.x0))
    assert lam[0].item() < 0 < lam[1].item()


def test_flatten_roundtrip():
    params = {"W": torch.arange(6.0).reshape(2, 3), "b": torch.tensor([7.0])}
    x = flatten_params(params)
    back = unflatten_params(x, {"W": (2, 3), "b": (1,)})
    assert all(torch.equal(params[k], back[k]) for k in params)
    with pytest.raises(ValueError):
        unflatten_params(x, {"W": (2, 3)})


def test_small_mlp_size_in_spec_range():
    assert 100 <= REGISTRY["small_mlp"](0).x0.numel() <= 300


def test_choose_radius_is_fixed_point():
    p = REGISTRY["matrix_factorization"](0)
    R = choose_radius(p.f, p.x0, seed=0)
    lam1 = torch.linalg.eigvalsh(hessian(p.f)(p.x0))[0].item()
    L = estimate_hessian_lipschitz(p.f, p.x0, R, n_pairs=50, seed=0)
    assert R == pytest.approx(2 * abs(lam1) / L, rel=0.1)


def test_third_directional_exact():
    # f = (a^T x)^3 / 6 + |x|^4: third derivative along v at x is (a^T v)^3 + 24 (x.v)|v|^2
    a = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)

    def f(x):
        return (a @ x) ** 3 / 6 + (x @ x) ** 2

    x = torch.tensor([0.3, 0.1, -0.2], dtype=torch.float64)
    v = torch.tensor([0.6, 0.0, 0.8], dtype=torch.float64)
    expected = (a @ v).item() ** 3 + 24 * (x @ v).item() * (v @ v).item()
    assert third_directional(f, x, v) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("name", ["matrix_factorization", "phase_retrieval", "small_mlp"])
def test_symmetric_problems_have_no_cubic_asymmetry_at_origin(name):
    p = REGISTRY[name](0)
    zero = torch.zeros_like(p.x0)
    lam, Q = torch.linalg.eigh(hessian(p.f)(zero))
    for i in (lam < -1e-10).nonzero().flatten()[:5]:
        assert abs(third_directional(p.f, zero, Q[:, i])) < 1e-12


def test_lipschitz_estimate_on_pure_cubic():
    # f = (a^T x)^3 / 6 has H(x) = (a^T x) a a^T, so L = ||a||^3 exactly.
    a = torch.tensor([1.0, 2.0, -2.0], dtype=torch.float64)

    def f(x):
        return (a @ x) ** 3 / 6

    L = estimate_hessian_lipschitz(f, torch.zeros(3, dtype=torch.float64), R=1.0, n_pairs=200)
    assert L <= 27.0 + 1e-9
    assert L >= 0.5 * 27.0
