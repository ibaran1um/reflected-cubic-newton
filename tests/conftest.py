import torch


def make_saddle(n: int, n_neg: int, cubic: float, c: float, seed: int):
    """Small saddle_tensor-style test function with a strict saddle at 0.

    f(x) = 1/2 x^T A x + 1/6 sum_r w_r (a_r^T x)^3 + c/24 ||x||^4
    """
    gen = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64))
    lam = torch.rand(n, generator=gen, dtype=torch.float64) + 0.5
    lam[:n_neg] = -lam[:n_neg]
    A = Q @ torch.diag(lam) @ Q.T
    A = 0.5 * (A + A.T)
    R = 2 * n
    a = torch.randn(R, n, generator=gen, dtype=torch.float64)
    a = a / torch.linalg.vector_norm(a, dim=1, keepdim=True)
    w = cubic * torch.randn(R, generator=gen, dtype=torch.float64)

    def f(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x @ A @ x + (w * (a @ x) ** 3).sum() / 6 + c / 24 * (x @ x) ** 2

    return f


def make_start(n: int, scale: float, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(10_000 + seed)
    return scale * torch.randn(n, generator=gen, dtype=torch.float64)
