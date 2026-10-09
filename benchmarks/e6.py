"""E6: numerical check of Theorem 2 (SPEC 4).

saddle_tensor with c = 0 (no quartic term), one step from the exact saddle
x = 0 (g = 0, hard case). With v the lam_1 eigenvector, tau = grad^3 f[v,v,v]
and a = (2/3)|tau|/M, the decreases are exactly

    RCN: 2|lam_1|^3/M^2 (1 + a)
    CN : 2|lam_1|^3/M^2 (1 - sign(tau) a)     (hard-case sign + along v)

so on trials where CN picks the worse sign, RCN / CN = (1 + a) / (1 - a).

    python benchmarks/e6.py [--seeds 100]
writes results/e6.csv and results/e6_theorem2.png.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.func import hessian  # noqa: E402

from problems import estimate_hessian_lipschitz, saddle_tensor  # noqa: E402
from problems._common import third_directional  # noqa: E402
from rcn import minimize  # noqa: E402

RESULTS = ROOT / "results"


def run(seeds: int) -> list[dict]:
    rows = []
    for n in (20, 50):
        for cubic in (0.3, 1.0):
            for seed in range(seeds):
                p = saddle_tensor.make(seed, n=n, n_neg=3, cubic=cubic, c=0.0, start_scale=0.0)
                x0 = p.x0
                assert torch.count_nonzero(x0) == 0
                H = hessian(p.f)(x0)
                lam, Q = torch.linalg.eigh(0.5 * (H + H.T))
                lam1, v = lam[0].item(), Q[:, 0]
                tau = third_directional(p.f, x0, v)
                L = estimate_hessian_lipschitz(p.f, x0, 1.0, seed=seed)  # c = 0: L does not depend on R
                for fac in (1.0, 3.0, 10.0):
                    M = fac * L
                    a = (2.0 / 3.0) * abs(tau) / M
                    base = 2.0 * abs(lam1) ** 3 / M**2
                    f0 = p.f(x0).item()
                    d_cn = f0 - minimize(p.f, x0, method="cn", M=M, max_iter=1).history[1].f
                    d_rcn = f0 - minimize(p.f, x0, method="rcn", M=M, max_iter=1).history[1].f
                    pred_cn = base * (1.0 - math.copysign(a, tau))
                    pred_rcn = base * (1.0 + a)
                    rows.append(
                        dict(
                            n=n, cubic=cubic, seed=seed, M_factor=fac, M=M, L_hat=L, lam1=lam1,
                            lam_gap=lam[1].item() - lam1, tau=tau, a=a, base=base,
                            d_cn=d_cn, d_rcn=d_rcn, pred_cn=pred_cn, pred_rcn=pred_rcn,
                            relerr_cn=abs(d_cn - pred_cn) / abs(pred_cn),
                            relerr_rcn=abs(d_rcn - pred_rcn) / abs(pred_rcn),
                            cn_wrong_sign=tau > 0,
                            ratio=d_rcn / d_cn if d_cn > 0 else math.nan,
                            pred_ratio=(1 + a) / (1 - a) if tau > 0 and a < 1 else (1.0 if tau <= 0 else math.nan),
                        )
                    )
    return rows


def plot(rows: list[dict]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from plot import INK, INK_2, SERIES, SURFACE, style

    style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    for (cubic, color) in ((0.3, SERIES[0]), (1.0, SERIES[1])):
        sel = [r for r in rows if r["cubic"] == cubic and r["cn_wrong_sign"] and not math.isnan(r["ratio"])]
        ax.plot([r["pred_ratio"] for r in sel], [r["ratio"] for r in sel], "o", ms=5, color=color, mec=SURFACE,
                mew=0.8, alpha=0.85, label=f"cubic size {cubic:g}")
    lim = max(max(r["pred_ratio"] for r in rows if r["cn_wrong_sign"] and not math.isnan(r["pred_ratio"])), 1.0) * 1.05
    ax.plot([1, lim], [1, lim], color=INK_2, lw=1, zorder=0, label="y = x")
    ax.set_xlabel("predicted (1 + a) / (1 − a)")
    ax.set_ylabel("measured RCN / CN first-step decrease")
    ax.set_title("Trials where CN takes the worse sign", loc="left")
    ax.legend(loc="upper left")

    ax = axes[1]
    errs = sorted(max(r["relerr_cn"], r["relerr_rcn"]) for r in rows)
    ax.plot(errs, [i / len(errs) for i in range(1, len(errs) + 1)], color=SERIES[0])
    ax.set_xscale("log")
    ax.set_xlabel("relative error vs theory (max of CN, RCN)")
    ax.set_ylabel("share of trials")
    ax.set_title("Agreement with the closed form", loc="left")
    fig.suptitle(f"E6 · Theorem 2 on saddle_tensor with c = 0, one step from the exact saddle ({len(rows)} trials)",
                 x=0.01, ha="left", color=INK, fontsize=10)
    fig.tight_layout()
    p = RESULTS / "e6_theorem2.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    rows = run(args.seeds)
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "e6.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    wrong = [r for r in rows if r["cn_wrong_sign"]]
    print(f"trials {len(rows)}, CN took the worse sign in {len(wrong)}")
    print(f"max relative error: CN {max(r['relerr_cn'] for r in rows):.2e}, RCN {max(r['relerr_rcn'] for r in rows):.2e}")
    print(f"a range: {min(r['a'] for r in rows):.3f} .. {max(r['a'] for r in rows):.3f}; "
          f"min eigen-gap lam2 - lam1: {min(r['lam_gap'] for r in rows):.2e}")
    ok = [r for r in wrong if not math.isnan(r["pred_ratio"])]
    if ok:
        print("max |measured ratio - predicted| / predicted on wrong-sign trials: "
              f"{max(abs(r['ratio'] - r['pred_ratio']) / r['pred_ratio'] for r in ok):.2e}")
    print(f"wrote {RESULTS / 'e6.csv'} and {plot(rows)}")


if __name__ == "__main__":
    main()
