"""minimize() and the individual methods (SPEC 2.5, 2.6, 3, 4).

All methods share one loop: at each iterate we take g and H (only when x has
moved), check the second-order stopping rule, and let the method propose the
next point. Iteration k of the history is one subproblem / step attempt, so a
rejected ARC step is an iteration that does not move x.
"""

from __future__ import annotations

import math
import time
from typing import Callable

import torch
from torch.func import grad, hessian

from .cubic_subproblem import cubic_model, solve_cubic_eig
from .history import IterLog, Result
from .reflect import negative_indices, reflection_gap, try_reflection

METHODS = ("cn", "rcn", "arc", "rarc", "nc_pm")

# ARC (Cartis-Gould-Toint) parameters, SPEC 4. M here is 2 * sigma in CGT's
# m(s) = g^T s + 1/2 s^T H s + sigma/3 ||s||^3 convention.
ARC_RHO_ACCEPT, ARC_RHO_VERY = 0.1, 0.9
ARC_UP, ARC_DOWN, ARC_M_MIN = 2.0, 0.5, 1e-8

# nc_pm gradient step: Armijo backtracking.
ARMIJO_C, ARMIJO_MAX_HALVINGS = 1e-4, 60


class CountedFunction:
    """Wraps f and counts explicit evaluations at trial points.

    Gradients and Hessians are taken from the raw f, so autodiff's internal
    calls are not counted here.
    """

    def __init__(self, f: Callable[[torch.Tensor], torch.Tensor]):
        self.f = f
        self.n = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.n += 1
        return self.f(x)


def minimize(
    f: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    method: str = "rcn",
    M: float = 3.0,
    mode: str = "joint",
    eta: float = 0.1,
    eps_g: float = 1e-8,
    max_iter: int = 200,
    lam_tol: float | None = None,
    callback: Callable[[int, torch.Tensor], None] | None = None,
) -> Result:
    """Minimize f from x0 until a second-order stationary point or max_iter.

    f maps a 1-D float64 tensor to a scalar tensor and must be traceable by
    torch.func. Stops when ||g|| <= eps_g and lam_min >= -lam_tol, where
    lam_tol defaults to sqrt(M * eps_g) (SPEC 2.5; for arc, M is the initial
    value). An explicit lam_tol is only for reproducing the preliminary
    experiment (lam_tol = 1e-4).

    method "rarc" is ARC with the reflected candidates of RCN added to every
    step (mode and eta apply as for "rcn"). callback(k, x_k) is called for
    every iterate, including x_0 and the final point (used for trajectory plots).
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if mode not in ("joint", "greedy"):
        raise ValueError(f"unknown mode {mode!r}")
    if M <= 0:
        raise ValueError(f"M must be positive, got {M}")
    if x0.ndim != 1:
        raise ValueError("x0 must be a 1-D tensor")
    if lam_tol is None:
        lam_tol = math.sqrt(M * eps_g)

    if method in ("cn", "rcn"):
        step = _CubicStep(M=M, mode=mode, eta=0.0 if method == "cn" else eta)
    elif method in ("arc", "rarc"):
        step = _ArcStep(M=M, mode=mode, eta=eta if method == "rarc" else 0.0)
    else:
        step = _NcPmStep(M=M)
    return _run(f, x0, step, eps_g=eps_g, max_iter=max_iter, lam_tol=lam_tol, callback=callback)


# --------------------------------------------------------------------------- loop


def _run(f, x0, step, *, eps_g, max_iter, lam_tol, callback=None) -> Result:
    fc = CountedFunction(f)
    grad_f, hess_f = grad(f), hessian(f)
    n_grad = n_hess = 0
    t_start = time.perf_counter()

    x = x0.detach().to(torch.float64).clone()
    f_x = fc(x).item()  # the only f evaluation not at a trial point
    history: list[IterLog] = []
    moved = True

    for k in range(max_iter + 1):
        if callback is not None:
            callback(k, x)
        if moved:
            g = grad_f(x)
            H = hess_f(x)
            n_grad += 1
            n_hess += 1
            H = 0.5 * (H + H.T)
            lam, Q = torch.linalg.eigh(H)
            g_norm = torch.linalg.vector_norm(g).item()
        log = IterLog(k=k, f=f_x, grad_norm=g_norm, lam_min=lam[0].item())

        done = g_norm <= eps_g and log.lam_min >= -lam_tol
        if done or k == max_iter:
            log.time_sec = time.perf_counter() - t_start
            log.n_f, log.n_grad, log.n_hess = fc.n, n_grad, n_hess
            history.append(log)
            status = "sosp" if done else "max_iter"
            break

        x_new, f_new = step(fc, x, f_x, g, lam, Q, log)
        moved = x_new is not x
        log.n_f, log.n_grad, log.n_hess = fc.n, n_grad, n_hess
        log.time_sec = time.perf_counter() - t_start
        history.append(log)
        x, f_x = x_new, f_new

    return Result(x=x, f=f_x, status=status, history=history, n_f=fc.n, n_grad=n_grad, n_hess=n_hess)


# --------------------------------------------------------------------------- methods


class _CubicStep:
    """CN with fixed M; with eta > 0 the reflected candidates are tried (RCN)."""

    def __init__(self, M, mode, eta):
        self.M, self.mode, self.eta = M, mode, eta

    def __call__(self, fc, x, f_x, g, lam, Q, log):
        gt = Q.T @ g
        st, mu, hard = solve_cubic_eig(lam, gt, self.M)
        model_decrease = -cubic_model(lam, gt, st, self.M).item()
        x_cn = x + Q @ st
        f_cn = fc(x_cn).item()

        t0 = time.perf_counter()
        out = try_reflection(fc, x, Q, lam, gt, st, x_cn, f_cn, model_decrease, self.mode, self.eta)
        log.time_extra = time.perf_counter() - t0

        N = negative_indices(lam)
        log.mu = mu
        log.hard_case = hard
        log.reflect_tried = out.tried
        log.reflect_taken = out.taken
        log.extra_f_evals = out.extra_f_evals
        log.f_cn = f_cn
        log.model_decrease = model_decrease
        log.delta_ref = reflection_gap(gt, st, N) if N.numel() else 0.0
        log.M_k = self.M
        return out.x, out.f


class _ArcStep:
    """ARC with adaptive M (= 2 sigma); M passed to minimize() is the initial value.

    With eta > 0 ("rarc") the reflected candidates of RCN are tried at every
    step with the same activation rule and acceptance, and rho uses the best
    candidate's f over the CN model decrease. Since that f is never above the
    CN trial value, rho is never below plain ARC's on the same step.
    """

    def __init__(self, M, mode="joint", eta=0.0):
        self.M, self.mode, self.eta = M, mode, eta

    def __call__(self, fc, x, f_x, g, lam, Q, log):
        gt = Q.T @ g
        st, mu, hard = solve_cubic_eig(lam, gt, self.M)
        model_decrease = -cubic_model(lam, gt, st, self.M).item()
        x_cn = x + Q @ st
        f_cn = fc(x_cn).item()

        t0 = time.perf_counter()
        out = try_reflection(fc, x, Q, lam, gt, st, x_cn, f_cn, model_decrease, self.mode, self.eta)
        log.time_extra = time.perf_counter() - t0
        log.reflect_tried, log.reflect_taken, log.extra_f_evals = out.tried, out.taken, out.extra_f_evals
        x_try, f_try = out.x, out.f

        log.mu, log.hard_case, log.f_cn, log.model_decrease, log.M_k = mu, hard, f_cn, model_decrease, self.M
        rho = (f_x - f_try) / model_decrease if model_decrease > 0 else -math.inf
        accept = rho >= ARC_RHO_ACCEPT
        if rho >= ARC_RHO_VERY:
            self.M = max(ARC_DOWN * self.M, ARC_M_MIN)
        elif not accept:
            self.M = ARC_UP * self.M
        log.step_accepted = accept
        return (x_try, f_try) if accept else (x, f_x)


class _NcPmStep:
    """Simplified negative-curvature method (SPEC 4, nc_pm).

    Negative-curvature step: length 2|lam_1|/M along +-q1, sign chosen by two
    f evaluations (the better one is taken). Gradient step: Armijo
    backtracking (initial step 1, halving, c = 1e-4).

    Deviation from SPEC 4: the NC step is used only when lam_1 < -delta AND its
    predicted decrease 2|lam_1|^3 / (3 M^2) is at least that of a gradient step,
    ||g||^2 / (2 ||H||). With the literal rule (NC whenever lam_1 < -delta) the
    method stalls once lam_1 is slightly negative while ||g|| is still large,
    taking steps of length ~|lam_1|/M forever.
    """

    def __init__(self, M):
        self.M = M

    def __call__(self, fc, x, f_x, g, lam, Q, log):
        log.M_k = self.M
        lam1 = lam[0].item()
        h_norm = max(abs(lam1), abs(lam[-1].item()), 1e-300)
        nc_gain = 2.0 * abs(lam1) ** 3 / (3.0 * self.M**2)
        g_gain = (g @ g).item() / (2.0 * h_norm)
        if negative_indices(lam).numel() and nc_gain >= g_gain:
            v = Q[:, 0] * (2.0 * abs(lam[0].item()) / self.M)
            x_p, x_m = x + v, x - v
            f_p, f_m = fc(x_p).item(), fc(x_m).item()
            log.extra_f_evals = 1
            return (x_p, f_p) if f_p <= f_m else (x_m, f_m)

        g2 = (g @ g).item()
        # Near a minimizer the Armijo decrease c t ||g||^2 falls below the
        # rounding error of f and the test can never pass. In that regime take
        # a step of length 1/||H|| without the f test: it contracts the
        # gradient, and its effect on f, O(||g||^2 / ||H||), is at rounding level.
        if ARMIJO_C * g2 <= 10.0 * torch.finfo(torch.float64).eps * abs(f_x):
            x_try = x - g / h_norm
            return x_try, fc(x_try).item()
        t = 1.0
        for _ in range(ARMIJO_MAX_HALVINGS):
            x_try = x - t * g
            f_try = fc(x_try).item()
            if f_try <= f_x - ARMIJO_C * t * g2:
                return x_try, f_try
            t *= 0.5
        log.step_accepted = False
        return x, f_x
