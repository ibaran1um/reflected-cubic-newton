"""Shared problem container and the Hessian-Lipschitz estimate of SPEC 4."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch.func import grad, hessian, vmap


@dataclass
class Problem:
    name: str
    f: Callable[[torch.Tensor], torch.Tensor]
    x0: torch.Tensor
    params: dict = field(default_factory=dict)
    R: float = 1.0  # radius of the ball around x0 used for the L-hat estimate


def flatten_params(params: dict[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate a parameter dict into one 1-D tensor (insertion order)."""
    return torch.cat([p.reshape(-1) for p in params.values()])


def unflatten_params(x: torch.Tensor, shapes: dict[str, tuple[int, ...]]) -> dict[str, torch.Tensor]:
    """Inverse of flatten_params for the given ordered shapes (torch.func friendly)."""
    out, i = {}, 0
    for k, s in shapes.items():
        size = 1
        for d in s:
            size *= d
        out[k] = x[i : i + size].reshape(s)
        i += size
    if i != x.numel():
        raise ValueError(f"expected {i} parameters, got {x.numel()}")
    return out


def third_directional(f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, v: torch.Tensor) -> float:
    """grad^3 f(x)[v, v, v] = d^3/dt^3 f(x + t v) at t = 0 (exact, by autodiff)."""

    def line(t: torch.Tensor) -> torch.Tensor:
        return f(x + t * v)

    t0 = torch.zeros((), dtype=x.dtype)
    return grad(grad(grad(line)))(t0).item()


def choose_radius(
    f: Callable[[torch.Tensor], torch.Tensor],
    center: torch.Tensor,
    seed: int = 0,
    n_pairs: int = 50,
    lo: float = 1e-3,
    hi: float = 1e3,
    steps: int = 30,
) -> float:
    """R such that the ball covers the first CN step at M = L-hat(R).

    Solves R = 2 |lam_1(center)| / L-hat(R) by bisection in log R (the left
    side increases and the right side does not increase with R, since L-hat(R)
    is non-decreasing up to sampling noise). Returns 1.0 when there is no
    negative curvature at the center.
    """
    lam1 = torch.linalg.eigvalsh(hessian(f)(center))[0].item()
    if lam1 >= 0:
        return 1.0

    def gap(R: float) -> float:
        return R - 2.0 * abs(lam1) / max(estimate_hessian_lipschitz(f, center, R, n_pairs, seed), 1e-300)

    if gap(lo) >= 0:
        return lo
    if gap(hi) <= 0:
        return hi
    a, b = math.log(lo), math.log(hi)
    for _ in range(steps):
        mid = 0.5 * (a + b)
        if gap(math.exp(mid)) < 0:
            a = mid
        else:
            b = mid
        if b - a < 0.02:
            break
    return math.exp(0.5 * (a + b))


def estimate_hessian_lipschitz(
    f: Callable[[torch.Tensor], torch.Tensor],
    center: torch.Tensor,
    R: float,
    n_pairs: int = 200,
    seed: int = 0,
    chunk: int = 25,
) -> float:
    """L-hat = max over random pairs in B(center, R) of ||H(x) - H(y)||_2 / ||x - y||.

    Points are uniform in the ball, drawn from a seeded generator.
    """
    gen = torch.Generator().manual_seed(seed)
    n = center.numel()

    def sample(k: int) -> torch.Tensor:
        d = torch.randn(k, n, generator=gen, dtype=torch.float64)
        d = d / torch.linalg.vector_norm(d, dim=1, keepdim=True)
        r = R * torch.rand(k, 1, generator=gen, dtype=torch.float64) ** (1.0 / n)
        return center + r * d

    X, Y = sample(n_pairs), sample(n_pairs)
    hess = vmap(hessian(f))
    best = 0.0
    for i in range(0, n_pairs, chunk):
        x, y = X[i : i + chunk], Y[i : i + chunk]
        diff = hess(x) - hess(y)
        diff = 0.5 * (diff + diff.transpose(1, 2))
        num = torch.linalg.matrix_norm(diff, ord=2)
        den = torch.linalg.vector_norm(x - y, dim=1)
        best = max(best, (num / den).max().item())
    return best
