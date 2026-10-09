"""Saddle with a random symmetric cubic term (the preliminary-experiment family).

    f(x) = 1/2 x^T A x + 1/6 T[x, x, x] + c/24 ||x||^4

- A = Q diag(lam) Q^T, Q from the QR factorization of a Gaussian matrix.
  n_neg eigenvalues are -U(0.5, 1.5), the rest U(0.5, 2.0).
- T has i.i.d. standard normal entries, symmetrized by averaging the six index
  permutations, then multiplied by `cubic` (no normalization by ||T||).
- x0 = start_scale * N(0, I) around the strict saddle at the origin.
"""

from __future__ import annotations

import itertools

import torch

from ._common import Problem

DEFAULT_R = 1.0


def make(
    seed: int,
    n: int = 20,
    n_neg: int = 3,
    cubic: float = 1.0,
    c: float = 1.0,
    start_scale: float = 1e-6,
    R: float = DEFAULT_R,
) -> Problem:
    if not 0 <= n_neg <= n:
        raise ValueError("need 0 <= n_neg <= n")
    gen = torch.Generator().manual_seed(seed)
    dt = torch.float64

    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=dt))
    lam = torch.empty(n, dtype=dt)
    lam[:n_neg] = -(0.5 + torch.rand(n_neg, generator=gen, dtype=dt))
    lam[n_neg:] = 0.5 + 1.5 * torch.rand(n - n_neg, generator=gen, dtype=dt)
    A = Q @ torch.diag(lam) @ Q.T
    A = 0.5 * (A + A.T)

    G = torch.randn(n, n, n, generator=gen, dtype=dt)
    T = sum(G.permute(*p) for p in itertools.permutations(range(3))) / 6.0
    T = cubic * T

    x0 = start_scale * torch.randn(n, generator=gen, dtype=dt)

    def f(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x @ (A @ x) + ((T @ x) @ x) @ x / 6.0 + c / 24.0 * (x @ x) ** 2

    params = dict(n=n, n_neg=n_neg, cubic=cubic, c=c, start_scale=start_scale)
    return Problem(name="saddle_tensor", f=f, x0=x0, params=params, R=R)
