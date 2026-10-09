"""Reflected candidates, activation rule and acceptance (SPEC 2.2-2.4).

Flipping the sign of st_i leaves the quadratic and cubic terms of the model
unchanged and only changes g~_i st_i. Since g~_i st_i <= 0 at the CN minimizer,
the model loss of flipping a set S is 2 * sum_{i in S} |g~_i st_i|.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch


def negative_indices(lam: torch.Tensor) -> torch.Tensor:
    """N = {i : lam_i < -delta_lam}, delta_lam = 1e-10 * max(1, |lam|_max)."""
    delta = 1e-10 * max(1.0, lam.abs().max().item())
    return torch.nonzero(lam < -delta).flatten()


def reflection_gap(gt: torch.Tensor, st: torch.Tensor, idx: torch.Tensor) -> float:
    """Delta_ref = m(s') - m(s) when the components idx are flipped."""
    return 2.0 * (gt[idx] * st[idx]).abs().sum().item()


def reflect(st: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    out = st.clone()
    out[idx] = -out[idx]
    return out


def gate(eta: float, d_ref: float, model_decrease: float) -> bool:
    """Activation rule of SPEC 2.3 for a candidate whose model loss is d_ref.

    eta = 0 disables reflection before anything else, so the run matches
    plain CN even when d_ref = 0 (exact saddle, hard case).
    """
    if eta <= 0:
        return False
    if math.isinf(eta):
        return True
    return d_ref <= eta * model_decrease


@dataclass
class Outcome:
    x: torch.Tensor
    f: float
    tried: bool = False
    taken: bool = False
    extra_f_evals: int = 0


def try_reflection(
    f: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    Q: torch.Tensor,
    lam: torch.Tensor,
    gt: torch.Tensor,
    st: torch.Tensor,
    x_cn: torch.Tensor,
    f_cn: float,
    model_decrease: float,
    mode: str,
    eta: float,
) -> Outcome:
    """Compare the CN trial point with reflected candidates; return the best.

    `f` must be the counted objective: every call here is an extra evaluation.
    The returned f value is never larger than f_cn.
    """
    best = Outcome(x=x_cn, f=f_cn)
    N = negative_indices(lam)
    if eta <= 0 or N.numel() == 0:
        return best

    if mode == "joint":
        if not gate(eta, reflection_gap(gt, st, N), model_decrease):
            return best
        cand = x + Q @ reflect(st, N)
        f_cand = f(cand).item()
        best.tried, best.extra_f_evals = True, 1
        if f_cand < best.f:
            best.x, best.f, best.taken = cand, f_cand, True
        return best

    if mode == "greedy":
        # Largest |lam_i| first; the activation rule is applied to the
        # accumulated model loss of the flips accepted so far plus this one.
        order = N[torch.argsort(lam[N].abs(), descending=True, stable=True)]
        st_cur, accepted = st, 0.0
        for i in order:
            idx = i.reshape(1)
            d_i = reflection_gap(gt, st, idx)
            if not gate(eta, accepted + d_i, model_decrease):
                break
            st_try = reflect(st_cur, idx)
            cand = x + Q @ st_try
            f_cand = f(cand).item()
            best.tried = True
            best.extra_f_evals += 1
            if f_cand < best.f:
                best.x, best.f, best.taken = cand, f_cand, True
                st_cur, accepted = st_try, accepted + d_i
        return best

    raise ValueError(f"unknown mode {mode!r}")
