import math

import pytest
import torch

from rcn.cubic_subproblem import cubic_model, solve_cubic_eig
from rcn.reflect import gate, negative_indices, reflect, reflection_gap


def random_eig_problem(n, n_neg, seed):
    gen = torch.Generator().manual_seed(seed)
    lam = torch.rand(n, generator=gen, dtype=torch.float64) * 3 + 0.1
    lam[:n_neg] = -lam[:n_neg]
    lam, _ = torch.sort(lam)
    gt = torch.randn(n, generator=gen, dtype=torch.float64)
    return lam, gt


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("n,n_neg", [(3, 1), (8, 3), (20, 5)])
@pytest.mark.parametrize("M", [0.5, 3.0])
def test_model_change_equals_delta_ref(seed, n, n_neg, M):
    lam, gt = random_eig_problem(n, n_neg, seed)
    st, _, _ = solve_cubic_eig(lam, gt, M)
    N = negative_indices(lam)
    assert N.tolist() == list(range(n_neg))
    m_s = cubic_model(lam, gt, st, M).item()
    m_ref = cubic_model(lam, gt, reflect(st, N), M).item()
    d_ref = reflection_gap(gt, st, N)
    assert d_ref > 0
    assert m_ref - m_s == pytest.approx(d_ref, rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("M", [0.5, 3.0])
def test_model_unchanged_when_negative_gradient_vanishes(seed, M):
    lam, gt = random_eig_problem(10, 3, seed)
    gt[:3] = 0.0
    st, _, _ = solve_cubic_eig(lam, gt, M)
    N = negative_indices(lam)
    m_s = cubic_model(lam, gt, st, M).item()
    m_ref = cubic_model(lam, gt, reflect(st, N), M).item()
    assert reflection_gap(gt, st, N) == 0.0
    assert m_ref == pytest.approx(m_s, rel=1e-12, abs=1e-14)


def test_negative_indices_threshold():
    lam = torch.tensor([-1.0, -1e-12, 0.0, 2.0], dtype=torch.float64)
    assert negative_indices(lam).tolist() == [0]


def test_gate():
    assert not gate(0.0, 0.0, 1.0)  # eta = 0 disables even a free reflection
    assert gate(0.5, 0.5, 1.0)
    assert not gate(0.5, 0.51, 1.0)
    assert gate(math.inf, 1e9, 0.0)
