"""Two-dimensional test functions with saddle points (E2, E7).

- himmelblau: (x^2 + y - 11)^2 + (x + y^2 - 7)^2, four saddle points.
- cubic_saddle: 1/2 x^2 - 1/2 y^2 + a/6 y^3 + 1/24 (x^2 + y^2)^2, a strict
  saddle at 0 whose escape direction y is asymmetric (deeper valley at y < 0
  for a > 0). Built to show the reflection on a contour plot.

The seed picks one saddle (himmelblau) and the random start offset.
"""

from __future__ import annotations

import torch
from torch.func import grad, hessian

from ._common import Problem

# Approximate saddle locations of Himmelblau's function, refined by Newton below.
_HIMMELBLAU_SADDLES = [
    (0.086678, 2.884255),
    (-3.073026, -0.081353),
    (3.385154, 0.073852),
    (-0.127961, -1.953715),
]


def himmelblau(x: torch.Tensor) -> torch.Tensor:
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def cubic_saddle_fn(a: float):
    def f(x: torch.Tensor) -> torch.Tensor:
        r2 = x[0] ** 2 + x[1] ** 2
        return 0.5 * x[0] ** 2 - 0.5 * x[1] ** 2 + a / 6.0 * x[1] ** 3 + r2**2 / 24.0

    return f


def _refine(f, x: torch.Tensor, steps: int = 30) -> torch.Tensor:
    """Newton on grad f = 0 (converges to the nearby saddle)."""
    g, H = grad(f), hessian(f)
    for _ in range(steps):
        x = x - torch.linalg.solve(H(x), g(x))
    return x


def make(seed: int, fn: str = "himmelblau", a: float = 1.0, start_scale: float = 1e-6, R: float = 1.0) -> Problem:
    gen = torch.Generator().manual_seed(seed)
    dt = torch.float64
    if fn == "himmelblau":
        f = himmelblau
        saddle = _refine(f, torch.tensor(_HIMMELBLAU_SADDLES[seed % 4], dtype=dt))
        params = dict(fn=fn, n=2, saddle=seed % 4, start_scale=start_scale)
    elif fn == "cubic_saddle":
        f = cubic_saddle_fn(a)
        saddle = torch.zeros(2, dtype=dt)
        params = dict(fn=fn, n=2, a=a, start_scale=start_scale)
    else:
        raise ValueError(f"unknown test function {fn!r}")
    x0 = saddle + start_scale * torch.randn(2, generator=gen, dtype=dt)
    return Problem(name="test_functions", f=f, x0=x0, params=params, R=R)
