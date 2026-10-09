"""Symmetric low-rank matrix factorization.

    f(U) = 1/4 ||U U^T - Y||_F^2,   U in R^{d x r},   Y = V V^T of rank r

U = 0 is a strict saddle (every direction of Y's range is negative
curvature). x = vec(U) has n = d r entries; x0 = start_scale * N(0, I).
"""

from __future__ import annotations

import torch

from ._common import Problem


def make(seed: int, d: int = 20, r: int = 2, start_scale: float = 1e-3, R: float = 1.0) -> Problem:
    gen = torch.Generator().manual_seed(seed)
    dt = torch.float64
    V = torch.randn(d, r, generator=gen, dtype=dt)
    Y = V @ V.T
    x0 = start_scale * torch.randn(d * r, generator=gen, dtype=dt)

    def f(x: torch.Tensor) -> torch.Tensor:
        U = x.reshape(d, r)
        E = U @ U.T - Y
        return 0.25 * (E * E).sum()

    return Problem(
        name="matrix_factorization", f=f, x0=x0, params=dict(d=d, r=r, n=d * r, start_scale=start_scale), R=R
    )
