"""Exact solver for the cubic-regularized Newton subproblem (SPEC 2.1).

    m(s) = g^T s + 1/2 s^T H s + M/6 ||s||^3

With H = Q diag(lam) Q^T and gt = Q^T g, the global minimizer in eigen
coordinates is st_i = -gt_i / (lam_i + mu) with mu = (M/2)||s|| and
mu >= max(0, -lam_1).

Internally we bisect on t = mu - mu_lo instead of mu itself. The shifted
eigenvalues d_i = lam_i + mu_lo are exactly 0 on the lam_1 eigenspace, so
st_1 = -gt_1 / t stays resolvable even when t is far below the float spacing
of mu (the "nearly hard" case close to a saddle).
"""

from __future__ import annotations

import math

import torch

_PHI_RTOL = 1e-12
_MAX_BISECT = 200


def cubic_model(lam: torch.Tensor, gt: torch.Tensor, st: torch.Tensor, M: float) -> torch.Tensor:
    """m(s) evaluated in eigen coordinates."""
    return gt @ st + 0.5 * (lam * st**2).sum() + M / 6.0 * torch.linalg.vector_norm(st) ** 3


def solve_cubic_eig(
    lam: torch.Tensor, gt: torch.Tensor, M: float
) -> tuple[torch.Tensor, float, bool]:
    """Globally minimize the cubic model given in eigen coordinates.

    Args:
        lam: eigenvalues of H in ascending order (as returned by torch.linalg.eigh).
        gt: Q^T g.
        M: regularization parameter, M > 0.

    Returns:
        (st, mu, hard_case). st is the minimizer in eigen coordinates. In the
        hard case the free component along the first eigenvector is set with
        sign +; hard_case is True only when that component is nonzero
        (mu_lo > 0), so g = 0 with H positive semidefinite returns s = 0 and
        hard_case = False.
    """
    if M <= 0:
        raise ValueError(f"M must be positive, got {M}")
    if lam.ndim != 1 or lam.shape != gt.shape:
        raise ValueError("lam and gt must be 1-D tensors of the same shape")
    if lam.numel() > 1 and bool((lam[1:] < lam[:-1]).any()):
        raise ValueError("lam must be sorted in ascending order")

    mu_lo = max(0.0, -lam[0].item())
    # Exactly 0 on the lam_1 eigenspace when mu_lo = -lam_1, and >= 0 everywhere.
    d = lam + mu_lo
    zero = d == 0
    # Components of gt on the lam_1 eigenspace below the rounding error of
    # Q^T g are treated as exact zeros, so hard_case and its default sign
    # do not depend on roundoff.
    g_tol = 4.0 * gt.numel() * torch.finfo(gt.dtype).eps * torch.linalg.vector_norm(gt).item()

    if bool((gt[zero].abs() <= g_tol).all()):
        # phi(mu_lo) is finite: check for the hard case.
        st = torch.zeros_like(gt)
        st[~zero] = -gt[~zero] / d[~zero]
        rest = torch.linalg.vector_norm(st).item()
        target = 2.0 * mu_lo / M
        if rest <= target:
            if mu_lo > 0:
                st[int(zero.nonzero()[0])] = math.sqrt(max(target**2 - rest**2, 0.0))
            return st, mu_lo, mu_lo > 0

    t = _bisect(d, gt, mu_lo, M)
    return -gt / (d + t), mu_lo + t, False


def _bisect(d: torch.Tensor, gt: torch.Tensor, mu_lo: float, M: float) -> float:
    """Root of phi(t) = ||gt / (d + t)|| - 2 (mu_lo + t) / M on t > 0."""

    def phi(t: float) -> tuple[float, float]:
        s_norm = torch.linalg.vector_norm(gt / (d + t)).item()
        return s_norm - 2.0 * (mu_lo + t) / M, s_norm

    # phi is strictly decreasing on t > 0 and phi(0+) > 0 here.
    hi = 1.0
    while phi(hi)[0] > 0:
        hi *= 2.0
    lo = 0.0
    best_t, best_abs = hi, abs(phi(hi)[0])
    for _ in range(_MAX_BISECT):
        if lo == 0.0:
            # Step down geometrically first: the root can be orders of magnitude
            # below hi when gt has a tiny component on the lam_1 eigenspace.
            mid = hi * 1e-4 if hi > 1e-300 else 0.5 * hi
        elif hi > 4.0 * lo:
            mid = math.sqrt(lo * hi)
        else:
            mid = 0.5 * (lo + hi)
        if not lo < mid < hi:
            break
        val, s_norm = phi(mid)
        if abs(val) < best_abs:
            best_t, best_abs = mid, abs(val)
        if abs(val) <= _PHI_RTOL * s_norm:
            return mid
        if val > 0:
            lo = mid
        else:
            hi = mid
    return best_t
