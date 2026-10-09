"""E7: README figure, CN vs RCN trajectories on 2-D test functions.

Left: cubic_saddle (a = 1), built to show the mechanism: the start sits on
the shallow side of the saddle, CN follows the gradient's sign into the
shallow valley, RCN reflects the negative-curvature component and reaches the
deep one. Right: Himmelblau from a start where the two methods part.

    python benchmarks/e7.py
writes results/e7_trajectories.png.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from plot import GRID, INK, INK_2, SERIES, SURFACE, style  # noqa: E402
from problems import estimate_hessian_lipschitz  # noqa: E402
from problems.test_functions import cubic_saddle_fn, himmelblau, make  # noqa: E402
from rcn import minimize  # noqa: E402

RESULTS = ROOT / "results"


def path(f, x0, method, M):
    xs = []
    res = minimize(f, x0, method=method, M=M, callback=lambda k, x: xs.append(x.detach().numpy().copy()))
    return np.array(xs), res


def panel(ax, f, x0, M, box, title, legend_loc):
    xs = np.linspace(box[0], box[1], 400)
    ys = np.linspace(box[2], box[3], 400)
    X, Y = np.meshgrid(xs, ys)
    pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], 1), dtype=torch.float64)
    Z = torch.vmap(f)(pts).numpy().reshape(X.shape)
    levels = np.unique(np.quantile(Z, np.linspace(0.0, 0.97, 28)))  # even coverage of the visible range
    ax.contour(X, Y, Z, levels=levels, colors=GRID, linewidths=0.9, linestyles="solid")
    ax.contourf(X, Y, Z, levels=levels, cmap="Greys_r", alpha=0.35, extend="max")
    for method, color, name in (("cn", SERIES[0], "CN"), ("rcn", SERIES[1], "RCN")):
        P, res = path(f, x0, method, M)
        ax.plot(P[:, 0], P[:, 1], "-o", color=color, lw=2, ms=4, mec=SURFACE, mew=0.8,
                label=f"{name}: {res.n_iter} steps, f = {res.f:.2f}")
        ax.plot(P[-1, 0], P[-1, 1], "*", color=color, ms=14, mec=SURFACE, mew=1)
    ax.plot(x0[0], x0[1], "o", color=INK, ms=6, label="start (next to the saddle)")
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_title(title, loc="left")
    ax.legend(loc=legend_loc, fontsize=8, frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=0.95)


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))

    # cubic_saddle: deep valley at y < 0 for a > 0; start just above the saddle.
    f1 = cubic_saddle_fn(1.0)
    x1 = torch.tensor([2e-3, 1e-3], dtype=torch.float64)
    M1 = 3.0 * estimate_hessian_lipschitz(f1, x1, 1.0)
    panel(axes[0], f1, x1, M1, (-3.5, 3.5, -7.2, 2.5),
          f"Constructed cubic saddle (illustration), M = {M1:.1f}", "lower left")

    # Himmelblau: first seed (from the E2 starts) where CN and RCN end at different minima.
    for seed in range(200):
        p = make(seed, fn="himmelblau")
        M = 3.0 * estimate_hessian_lipschitz(p.f, p.x0, 1.0, seed=seed)
        a = minimize(p.f, p.x0, method="cn", M=M)
        b = minimize(p.f, p.x0, method="rcn", M=M)
        if torch.linalg.vector_norm(a.x - b.x).item() > 1e-3:
            break
    panel(axes[1], himmelblau, p.x0, M, (-5.0, 5.0, -5.0, 5.0),
          f"Himmelblau, saddle #{seed % 4} (seed {seed}), M = {M:.0f}", "lower right")
    fig.suptitle("CN vs RCN from the same start next to a saddle (RCN: joint, η = 0.1)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    out = RESULTS / "e7_trajectories.png"
    fig.savefig(out, dpi=150)
    print(f"himmelblau seed {seed}; wrote {out}")


if __name__ == "__main__":
    main()
