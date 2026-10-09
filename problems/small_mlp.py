"""Two-layer tanh MLP regression on synthetic data.

    model(z) = W2 tanh(W1 z + b1) + b2,   loss = 1/(2N) sum (model(z_i) - t_i)^2

Targets come from a random teacher MLP of the same form (hidden width
`teacher_hidden`) plus Gaussian noise. With small initial weights the
student starts near the saddle at zero weights. The parameter dict is
flattened to one vector with flatten_params / unflatten_params.
"""

from __future__ import annotations

import math

import torch

from ._common import Problem, flatten_params, unflatten_params


def _mlp(p: dict[str, torch.Tensor], Z: torch.Tensor) -> torch.Tensor:
    return torch.tanh(Z @ p["W1"].T + p["b1"]) @ p["W2"].T + p["b2"]


def make(
    seed: int,
    d_in: int = 5,
    hidden: int = 20,
    n_data: int = 200,
    teacher_hidden: int = 5,
    noise: float = 0.1,
    start_scale: float = 1e-3,
    R: float = 1.0,
) -> Problem:
    gen = torch.Generator().manual_seed(seed)
    dt = torch.float64
    Z = torch.randn(n_data, d_in, generator=gen, dtype=dt)
    teacher = {
        "W1": torch.randn(teacher_hidden, d_in, generator=gen, dtype=dt) / math.sqrt(d_in),
        "b1": 0.1 * torch.randn(teacher_hidden, generator=gen, dtype=dt),
        "W2": torch.randn(1, teacher_hidden, generator=gen, dtype=dt) / math.sqrt(teacher_hidden),
        "b2": torch.zeros(1, dtype=dt),
    }
    t = _mlp(teacher, Z) + noise * torch.randn(n_data, 1, generator=gen, dtype=dt)

    shapes = {"W1": (hidden, d_in), "b1": (hidden,), "W2": (1, hidden), "b2": (1,)}
    init = {k: start_scale * torch.randn(*s, generator=gen, dtype=dt) for k, s in shapes.items()}
    x0 = flatten_params(init)

    def f(x: torch.Tensor) -> torch.Tensor:
        p = unflatten_params(x, shapes)
        r = _mlp(p, Z) - t
        return 0.5 * (r * r).mean()

    n = x0.numel()
    params = dict(d_in=d_in, hidden=hidden, n_data=n_data, n=n, start_scale=start_scale)
    return Problem(name="small_mlp", f=f, x0=x0, params=params, R=R)
