"""Per-iteration log and run result."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import torch


@dataclass
class IterLog:
    """One iteration: the iterate x_k and the step taken from it.

    The last entry of a run describes the final point; its step fields are
    NaN / False because no step is taken from it.
    """

    k: int
    f: float  # f(x_k)
    grad_norm: float
    lam_min: float
    mu: float = math.nan
    hard_case: bool = False
    reflect_tried: bool = False
    reflect_taken: bool = False
    extra_f_evals: int = 0  # f evaluations beyond the CN trial point
    time_sec: float = 0.0  # wall time since the start of the run
    # Additions to SPEC 3: needed for the monotonicity test and for E3/E4.
    f_cn: float = math.nan  # f(x_k + s_k) of the plain CN step
    model_decrease: float = math.nan  # -m(s_k)
    delta_ref: float = math.nan  # model loss of the reflected candidate (joint: all of N)
    n_f: int = 0  # cumulative counters at the end of this entry
    n_grad: int = 0
    n_hess: int = 0
    # Stage 5 additions: E3 timing and the arc / nc_pm methods.
    time_extra: float = 0.0  # wall time spent on reflected candidates in this step
    M_k: float = math.nan  # regularization used in this step (adaptive for arc)
    step_accepted: bool = True  # False for a rejected arc step or a failed line search

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class Result:
    x: torch.Tensor
    f: float
    status: str  # "sosp" | "max_iter"
    history: list[IterLog] = field(default_factory=list)
    n_f: int = 0
    n_grad: int = 0
    n_hess: int = 0

    @property
    def n_iter(self) -> int:
        """Number of steps taken."""
        return len(self.history) - 1
