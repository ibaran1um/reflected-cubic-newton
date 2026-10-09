import numpy as np
import pytest
import scipy.optimize
import torch

from rcn.cubic_subproblem import cubic_model, solve_cubic_eig

TOL = 1e-8


def random_orthogonal(n: int, gen: torch.Generator) -> torch.Tensor:
    A = torch.randn(n, n, generator=gen, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    return Q * torch.sign(torch.diagonal(R))


def random_problem(n: int, kind: str, gen: torch.Generator):
    """Random (g, H) with a prescribed curvature pattern."""
    lam = torch.randn(n, generator=gen, dtype=torch.float64) * 3
    if kind == "pd":
        lam = lam.abs() + 0.1
    elif kind == "indefinite":
        lam = lam.abs() + 0.1
        k = max(1, n // 3)
        lam[:k] = -lam[:k]
    elif kind == "psd_singular":
        lam = lam.abs()
        lam[0] = 0.0
    lam, _ = torch.sort(lam)
    Q = random_orthogonal(n, gen)
    H = Q @ torch.diag(lam) @ Q.T
    H = 0.5 * (H + H.T)
    g = torch.randn(n, generator=gen, dtype=torch.float64)
    return g, H


def solve(g: torch.Tensor, H: torch.Tensor, M: float):
    lam, Q = torch.linalg.eigh(H)
    st, mu, hard = solve_cubic_eig(lam, Q.T @ g, M)
    return Q @ st, mu, hard, lam, Q


def model(g, H, s, M):
    return (g @ s + 0.5 * s @ H @ s + M / 6 * torch.linalg.vector_norm(s) ** 3).item()


def assert_optimality(g, H, s, mu, M):
    n = g.numel()
    scale = max(1.0, torch.linalg.vector_norm(g).item(), torch.linalg.matrix_norm(H, ord=2).item())
    resid = g + (H + mu * torch.eye(n, dtype=H.dtype)) @ s
    assert torch.linalg.vector_norm(resid).item() <= TOL * scale * max(1.0, torch.linalg.vector_norm(s).item())
    assert torch.linalg.eigvalsh(H + mu * torch.eye(n, dtype=H.dtype))[0].item() >= -TOL * scale
    assert abs(mu - M / 2 * torch.linalg.vector_norm(s).item()) <= TOL * max(1.0, mu)


CASES = [
    (n, kind, M, seed)
    for n in (1, 2, 5, 20, 50)
    for kind in ("pd", "indefinite", "psd_singular")
    for M in (0.1, 1.0, 10.0)
    for seed in range(3)
]


@pytest.mark.parametrize("n,kind,M,seed", CASES)
def test_optimality_conditions(n, kind, M, seed):
    gen = torch.Generator().manual_seed(1000 * n + seed)
    g, H = random_problem(n, kind, gen)
    s, mu, hard, _, _ = solve(g, H, M)
    assert_optimality(g, H, s, mu, M)


@pytest.mark.parametrize("n", [2, 5, 10])
@pytest.mark.parametrize("kind", ["pd", "indefinite", "psd_singular"])
@pytest.mark.parametrize("M", [0.3, 3.0])
@pytest.mark.parametrize("seed", range(3))
def test_not_worse_than_scipy_multistart(n, kind, M, seed):
    gen = torch.Generator().manual_seed(7000 + 100 * n + seed)
    g, H = random_problem(n, kind, gen)
    s, _, _, _, _ = solve(g, H, M)
    ours = model(g, H, s, M)

    gn, Hn = g.numpy(), H.numpy()

    def fun(x):
        r = np.linalg.norm(x)
        return gn @ x + 0.5 * x @ Hn @ x + M / 6 * r**3

    def jac(x):
        return gn + Hn @ x + M / 2 * np.linalg.norm(x) * x

    rng = np.random.default_rng(seed)
    radius = 2 * (np.linalg.norm(gn) + np.abs(np.linalg.eigvalsh(Hn)).max()) / M + 1
    best = np.inf
    for _ in range(20):
        x0 = rng.standard_normal(n)
        x0 *= radius * rng.uniform() / np.linalg.norm(x0)
        res = scipy.optimize.minimize(fun, x0, jac=jac, method="BFGS", options={"gtol": 1e-12})
        best = min(best, res.fun)
    assert ours <= best + TOL * max(1.0, abs(best))


@pytest.mark.parametrize("n", [2, 5, 20])
@pytest.mark.parametrize("M", [0.5, 3.0])
@pytest.mark.parametrize("seed", range(3))
def test_hard_case_zero_gradient(n, M, seed):
    gen = torch.Generator().manual_seed(300 + 10 * n + seed)
    _, H = random_problem(n, "indefinite", gen)
    g = torch.zeros(n, dtype=torch.float64)
    s, mu, hard, lam, Q = solve(g, H, M)

    assert hard
    lam1 = lam[0].item()
    assert lam1 < 0
    assert abs(torch.linalg.vector_norm(s).item() - 2 * abs(lam1) / M) <= TOL * max(1.0, abs(lam1))
    cos = (Q[:, 0] @ s).item() / torch.linalg.vector_norm(s).item()
    assert abs(abs(cos) - 1) <= TOL
    # sign + is the default in eigen coordinates
    assert cos > 0
    assert_optimality(g, H, s, mu, M)


def test_hard_case_with_gradient_off_the_eigenspace():
    # g has no component on q1 and is small enough that phi(mu_lo) <= 0.
    gen = torch.Generator().manual_seed(11)
    n, M = 6, 2.0
    _, H = random_problem(n, "indefinite", gen)
    lam, Q = torch.linalg.eigh(H)
    gt = torch.zeros(n, dtype=torch.float64)
    gt[1:] = 1e-3
    g = Q @ gt
    s, mu, hard, _, _ = solve(g, H, M)
    assert hard
    assert mu == pytest.approx(-lam[0].item(), abs=0)
    assert_optimality(g, H, s, mu, M)


def test_not_hard_when_gradient_off_eigenspace_is_large():
    gen = torch.Generator().manual_seed(12)
    n, M = 6, 2.0
    _, H = random_problem(n, "indefinite", gen)
    lam, Q = torch.linalg.eigh(H)
    gt = torch.zeros(n, dtype=torch.float64)
    gt[1:] = 100.0
    g = Q @ gt
    s, mu, hard, _, _ = solve(g, H, M)
    assert not hard
    assert mu > -lam[0].item()
    assert_optimality(g, H, s, mu, M)


@pytest.mark.parametrize("eps", [1e-6, 1e-10, 1e-14, 1e-20, 1e-40])
def test_nearly_hard_case(eps):
    # Tiny gradient component along q1: the root sits extremely close to mu_lo.
    lam = torch.tensor([-2.0, -0.5, 1.0, 3.0], dtype=torch.float64)
    gt = torch.tensor([eps, 0.0, 0.0, 0.0], dtype=torch.float64)
    M = 1.0
    st, mu, hard = solve_cubic_eig(lam, gt, M)
    assert not hard
    assert abs(torch.linalg.vector_norm(st).item() - 2 * mu / M) <= TOL * max(1.0, mu)
    # The step goes downhill along -g on the q1 component.
    assert st[0].item() < 0
    assert abs(abs(st[0].item()) - 2 * 2.0 / M) <= 1e-6
    assert (st[1:] == 0).all()


def test_zero_gradient_psd_returns_zero_step():
    lam = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    st, mu, hard = solve_cubic_eig(lam, torch.zeros(3, dtype=torch.float64), 1.0)
    assert not hard
    assert mu == 0.0
    assert (st == 0).all()


def test_model_value_in_eigen_coordinates_matches_full():
    gen = torch.Generator().manual_seed(5)
    g, H = random_problem(8, "indefinite", gen)
    s, _, _, lam, Q = solve(g, H, 1.5)
    assert cubic_model(lam, Q.T @ g, Q.T @ s, 1.5).item() == pytest.approx(model(g, H, s, 1.5), rel=1e-12)


def test_rejects_unsorted_eigenvalues():
    with pytest.raises(ValueError):
        solve_cubic_eig(torch.tensor([1.0, 0.0], dtype=torch.float64), torch.ones(2, dtype=torch.float64), 1.0)
