"""Benchmark problems. Each module exposes make(seed, **params) -> Problem."""

from . import matrix_factorization, phase_retrieval, saddle_tensor, small_mlp, test_functions
from ._common import Problem, choose_radius, estimate_hessian_lipschitz, flatten_params, unflatten_params

REGISTRY = {
    "saddle_tensor": saddle_tensor.make,
    "matrix_factorization": matrix_factorization.make,
    "phase_retrieval": phase_retrieval.make,
    "small_mlp": small_mlp.make,
    "test_functions": test_functions.make,
}

__all__ = [
    "Problem",
    "REGISTRY",
    "choose_radius",
    "estimate_hessian_lipschitz",
    "flatten_params",
    "unflatten_params",
]
