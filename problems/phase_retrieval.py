"""Real phase retrieval with the intensity loss.

    f(x) = 1/(4m) sum_i ((a_i^T x)^2 - y_i)^2,   y_i = (a_i^T x*)^2

a_i ~ N(0, I); x* ~ N(0, I) normalized to ||x*|| = 1.
The origin is a critical point with negative-definite Hessian; x0 is a small
random perturbation of it.
"""

from __future__ import annotations

import torch

from ._common import Problem


def make(seed: int, n: int = 20, m: int = 100, start_scale: float = 1e-3, R: float = 1.0) -> Problem:
    gen = torch.Generator().manual_seed(seed)
    dt = torch.float64
    A = torch.randn(m, n, generator=gen, dtype=dt)
    x_star = torch.randn(n, generator=gen, dtype=dt)
    x_star = x_star / torch.linalg.vector_norm(x_star)
    y = (A @ x_star) ** 2
    x0 = start_scale * torch.randn(n, generator=gen, dtype=dt)

    def f(x: torch.Tensor) -> torch.Tensor:
        r = (A @ x) ** 2 - y
        return (r * r).sum() / (4.0 * m)

    return Problem(name="phase_retrieval", f=f, x0=x0, params=dict(n=n, m=m, start_scale=start_scale), R=R)
