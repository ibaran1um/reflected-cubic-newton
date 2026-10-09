"""Make figures from results/*.csv.

    python benchmarks/plot.py e1
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Reference categorical slots 1-3 (light surface), text and grid inks.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# SPEC section 1 table (NumPy prototype, 100 problems, greedy).
TABLE = {  # (cubic, M): (step-1 ratio, iters cn, iters rcn)
    (0.3, 10.0): (1.02, 20.2, 18.3),
    (0.3, 3.0): (1.07, 11.8, 10.8),
    (1.0, 10.0): (1.07, 19.0, 18.0),
    (1.0, 3.0): (1.32, 13.0, 12.2),
}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2,
            "font.size": 9,
        }
    )


def read(name: str) -> list[dict]:
    with open(RESULTS / f"{name}.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def _errbar(ax, x, y, e, color, marker="o", label=None, filled=True):
    ax.errorbar(
        x, y, yerr=e, fmt=marker, ms=6, color=color, mfc=color if filled else SURFACE,
        mec=color, mew=1.5, elinewidth=1.5, capsize=0, label=label,
    )


def plot_e1() -> list[Path]:
    rows = read("e1")
    out = []

    # --- figure 1: comparison with the preliminary table
    cells = [(0.3, 10.0), (0.3, 3.0), (1.0, 10.0), (1.0, 3.0)]
    xt = [f"cubic {c:g}\nM = {M:g}" for c, M in cells]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    variants = [
        ("table_repro", "rcn_greedy", "greedy, 100 problems", SERIES[0], True),
        ("table_repro_1000", "rcn_greedy", "greedy, 1000 problems", SERIES[0], False),
        ("table_repro_1000", "rcn_joint", "joint, 1000 problems", SERIES[1], False),
    ]
    offs = [-0.18, 0.0, 0.18]
    for (grp, lab, name, color, filled), dx in zip(variants, offs):
        sel = {(r["cubic"], r["M_factor"]): r for r in rows if r["group"] == grp and r["label"] == lab}
        xs = [i + dx for i in range(len(cells))]
        _errbar(axes[0], xs, [sel[c]["step1_ratio_mean"] for c in cells],
                [sel[c]["step1_ratio_se"] for c in cells], color, label=name, filled=filled)
        _errbar(axes[1], xs, [sel[c]["iter_diff_mean"] for c in cells],
                [sel[c]["iter_diff_se"] for c in cells], color, label=name, filled=filled)
    tx = [i + 0.36 for i in range(len(cells))]
    axes[0].plot(tx, [TABLE[c][0] for c in cells], "D", ms=6, color=INK, label="SPEC table (NumPy)")
    axes[1].plot(tx, [TABLE[c][2] - TABLE[c][1] for c in cells], "D", ms=6, color=INK, label="SPEC table (NumPy)")
    axes[0].axhline(1.0, color=INK_2, lw=1)
    axes[1].axhline(0.0, color=INK_2, lw=1)
    axes[0].set_title("First-step decrease, RCN / CN (mean ± SE)", loc="left")
    axes[1].set_title("Iterations, RCN − CN (paired mean ± SE)", loc="left")
    for ax in axes:
        ax.set_xticks(range(len(cells)), xt)
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper left")
    fig.suptitle("E1 · saddle_tensor, n = 20, 3 negative eigenvalues, η = ∞, absolute M", x=0.01, ha="left", color=INK, fontsize=10)
    fig.tight_layout()
    p = RESULTS / "e1_table_repro.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)

    # --- figure 2: production sweep, small multiples by n
    main = [r for r in rows if r["group"] == "main" and r["label"] == "rcn_joint"]
    ns = sorted({r["n"] for r in main})
    facs = sorted({r["M_factor"] for r in main})
    cubics = sorted({r["cubic"] for r in main})
    fig, axes = plt.subplots(2, len(ns), figsize=(3.4 * len(ns), 5.6), sharey="row", sharex=True)
    for j, n in enumerate(ns):
        for k, cubic in enumerate(cubics):
            for i, metric in enumerate(("step1_ratio", "iter_diff")):
                agg = defaultdict(list)
                for r in main:
                    if r["n"] == n and r["cubic"] == cubic:
                        agg[r["M_factor"]].append((r[f"{metric}_mean"], r[f"{metric}_se"]))
                # mean of the n_neg cells; SE of that mean from the cell SEs (independent seeds)
                ys = [sum(m for m, _ in agg[f]) / len(agg[f]) for f in facs]
                es = [math.sqrt(sum(s * s for _, s in agg[f])) / len(agg[f]) for f in facs]
                ax = axes[i, j]
                xs = [f * (1.06 if k else 1 / 1.06) for f in facs]  # dodge the two series
                ax.plot(xs, ys, "-", color=SERIES[k], lw=2, label=f"cubic {cubic:g}")
                ax.errorbar(xs, ys, yerr=es, fmt="o", ms=6, color=SERIES[k], mec=SURFACE,
                            mew=1.5, elinewidth=1.5, capsize=0)
        axes[0, j].set_title(f"n = {n:g}", loc="left")
        axes[0, j].axhline(1.0, color=INK_2, lw=1, zorder=0)
        axes[1, j].axhline(0.0, color=INK_2, lw=1, zorder=0)
        axes[1, j].set_xscale("log")
        axes[1, j].set_xticks(facs, [f"{f:g}×L̂" for f in facs])
        axes[1, j].minorticks_off()
        axes[1, j].set_xlabel("M")
    axes[0, 0].set_ylabel("first-step decrease\nRCN / CN (± SE)")
    axes[1, 0].set_ylabel("iterations\nRCN − CN (± SE)")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("E1 · RCN (joint, η = 0.5) vs CN, averaged over n_neg ∈ {1, 3, 5}, 50 seeds each",
                 x=0.01, ha="left", color=INK, fontsize=10)
    fig.tight_layout()
    p = RESULTS / "e1_main.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)
    return out


METHOD_COLOR = {"rcn_joint": SERIES[0], "rcn_greedy": SERIES[1], "arc": SERIES[2], "nc_pm": "#e87ba4"}
METHOD_NAME = {"rcn_joint": "RCN joint", "rcn_greedy": "RCN greedy", "arc": "ARC", "nc_pm": "nc_pm"}


def problem_label(r: dict) -> str:
    p = r["problem"]
    if p == "saddle_tensor":
        return f"tensor\n{r['n']:g} · {r['cubic']:g}"
    if p == "test_functions":
        return {"himmelblau": "Himmelblau", "cubic_saddle": "cubic\nsaddle 2D"}[r["fn"]]
    return {"matrix_factorization": "matrix\nfact.", "phase_retrieval": "phase\nretrieval", "small_mlp": "small\nMLP"}[p]


def _cells(rows: list[dict]) -> list[str]:
    seen = []
    for r in rows:
        lab = problem_label(r)
        if lab not in seen:
            seen.append(lab)
    return seen


def _dodge(i: int, j: int, m: int, width: float = 0.6) -> float:
    return i + (j - (m - 1) / 2) * width / max(1, m)


def plot_e2() -> list[Path]:
    rows = read("e2")
    cells = _cells(rows)
    facs = sorted({r["M_factor"] for r in rows})
    labels = ["rcn_joint", "rcn_greedy"]
    out = []

    # D(k) ratio: mean (filled) and median (hollow) with bootstrap 95% CI
    fig, axes = plt.subplots(len(facs), 3, figsize=(16, 3.2 * len(facs)), sharex=True)
    for a, fac in enumerate(facs):
        for b, k in enumerate((1, 3, 5)):
            ax = axes[a, b]
            for j, lab in enumerate(labels):
                sel = {problem_label(r): r for r in rows if r["label"] == lab and r["M_factor"] == fac}
                for stat, filled, shift in (("mean", True, -0.08), ("median", False, 0.08)):
                    xs, ys, lo, hi = [], [], [], []
                    for i, c in enumerate(cells):
                        r = sel.get(c)
                        if r is None or math.isnan(r[f"d{k}_ratio_{stat}"]):
                            continue
                        xs.append(_dodge(i, j, len(labels), 0.5) + shift)
                        ys.append(r[f"d{k}_ratio_{stat}"])
                        lo.append(r[f"d{k}_ratio_{stat}"] - r[f"d{k}_ratio_{stat}_lo"])
                        hi.append(r[f"d{k}_ratio_{stat}_hi"] - r[f"d{k}_ratio_{stat}"])
                    _errbar(ax, xs, ys, [lo, hi], METHOD_COLOR[lab], filled=filled,
                            label=f"{METHOD_NAME[lab]} {stat}" if (a, b) == (0, 0) else None)
            ax.axhline(1.0, color=INK_2, lw=1, zorder=0)
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            ax.set_title(f"D({k}) RCN / CN · M = {fac:g}×L̂", loc="left")
            ax.grid(axis="x", visible=False)
            ax.set_xticks(range(len(cells)), cells, fontsize=7)
    axes[0, 0].legend(loc="upper left", ncol=2)
    fig.suptitle("E2 · cumulative decrease after k iterations relative to CN (95% bootstrap CI, log scale), "
                 "100 seeds · tensor labels: n · cubic size · small MLP at M = 1×L̂ excludes runs where CN "
                 "itself increased f", x=0.01, ha="left", color=INK, fontsize=10)
    fig.tight_layout()
    p = RESULTS / "e2_dk.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)

    # share of problems improved / worsened by >= 1% at k = 3
    fig, axes = plt.subplots(1, len(facs), figsize=(5.2 * len(facs), 3.6), sharey=True)
    for a, fac in enumerate(facs):
        ax = axes[a]
        for j, lab in enumerate(labels):
            sel = {problem_label(r): r for r in rows if r["label"] == lab and r["M_factor"] == fac}
            xs = [_dodge(i, j, len(labels), 0.5) for i, c in enumerate(cells) if c in sel]
            up = [sel[c]["d3_frac_up"] for c in cells if c in sel]
            down = [-sel[c]["d3_frac_down"] for c in cells if c in sel]
            ax.bar(xs, up, width=0.22, color=METHOD_COLOR[lab], label=f"{METHOD_NAME[lab]}" if a == 0 else None)
            ax.bar(xs, down, width=0.22, color=METHOD_COLOR[lab], alpha=0.45)
        ax.axhline(0, color=INK_2, lw=1)
        ax.set_title(f"M = {fac:g}×L̂", loc="left")
        ax.set_xticks(range(len(cells)), cells, fontsize=7)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("share of problems\n↑ D(3) ≥ +1%   ↓ D(3) ≤ −1%")
    axes[0].legend(loc="upper left")
    fig.suptitle("E2 · how often RCN changes the 3-iteration decrease by at least 1% (vs CN)",
                 x=0.01, ha="left", color=INK, fontsize=10)
    fig.tight_layout()
    p = RESULTS / "e2_share.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)

    # secondary: iterations at M = 3 L-hat, 500 seeds
    rows_it = read("e2_iters")
    cells_it = _cells(rows_it)
    labs = ["rcn_joint", "rcn_greedy", "arc", "nc_pm"]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for j, lab in enumerate(labs):
        sel = {problem_label(r): r for r in rows_it if r["label"] == lab}
        xs = [_dodge(i, j, len(labs)) for i, c in enumerate(cells_it) if c in sel]
        ys = [sel[c]["iter_diff_mean"] for c in cells_it if c in sel]
        lo = [sel[c]["iter_diff_mean"] - sel[c]["iter_diff_ci_lo"] for c in cells_it if c in sel]
        hi = [sel[c]["iter_diff_ci_hi"] - sel[c]["iter_diff_mean"] for c in cells_it if c in sel]
        _errbar(ax, xs, ys, [lo, hi], METHOD_COLOR[lab], label=METHOD_NAME[lab])
    ax.axhline(0, color=INK_2, lw=1, zorder=0)
    ax.set_yscale("symlog", linthresh=2)
    ax.set_xticks(range(len(cells_it)), cells_it, fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("iterations − CN\n(paired mean, 95% CI, symlog)")
    ax.legend(loc="upper left", ncol=4)
    ax.set_title("E2 (secondary) · iterations to a second-order point, M = 3×L̂, 500 seeds", loc="left")
    fig.tight_layout()
    p = RESULTS / "e2_iters.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)
    return out


def plot_e3() -> list[Path]:
    rows = read("e3")
    cells = _cells(rows)
    labs = ["rcn_joint", "rcn_greedy", "rcn_joint_always"]
    colors = {"rcn_joint": SERIES[0], "rcn_greedy": SERIES[1], "rcn_joint_always": SERIES[2]}
    names = {"rcn_joint": "joint, η = 0.5", "rcn_greedy": "greedy, η = 0.5", "rcn_joint_always": "joint, η = ∞"}
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for j, lab in enumerate(labs):
        sel = {problem_label(r): r for r in rows if r["label"] == lab}
        xs = [_dodge(i, j, len(labs)) for i, c in enumerate(cells) if c in sel]
        ax.bar(xs, [100 * sel[c]["time_extra_frac"] for c in cells if c in sel], width=0.18,
               color=colors[lab], label=names[lab])
    ax.set_xticks(range(len(cells)), cells, fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("% of wall time on\nreflected candidates")
    ax.legend(loc="upper right")
    ax.set_title("E3 · cost of reflection (1 worker, 1 thread, M = 3×L̂, 20 seeds)", loc="left")
    fig.tight_layout()
    p = RESULTS / "e3_cost.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return [p]


def plot_e4() -> list[Path]:
    rows = read("e4")
    cells = _cells(rows)
    etas = ["eta_0", "eta_0.1", "eta_0.5", "eta_1", "eta_inf"]
    eta_x = {e: i for i, e in enumerate(etas)}
    facs = sorted({r["M_factor"] for r in rows})
    palette = SERIES + ["#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    fig, axes = plt.subplots(2, len(facs), figsize=(6 * len(facs), 6.4), sharex=True)
    for a, fac in enumerate(facs):
        for i, c in enumerate(cells):
            sel = {r["label"]: r for r in rows if problem_label(r) == c and r["M_factor"] == fac}
            xs = [eta_x[e] for e in etas if e in sel]
            axes[0, a].plot(xs, [sel[e]["d3_ratio_mean"] for e in etas if e in sel], "-o", ms=5,
                            color=palette[i % len(palette)], mec=SURFACE, label=c.replace("\n", " "))
            axes[1, a].plot(xs, [100 * sel[e]["extra_f_frac"] for e in etas if e in sel], "-o", ms=5,
                            color=palette[i % len(palette)], mec=SURFACE)
        axes[0, a].axhline(1.0, color=INK_2, lw=1, zorder=0)
        axes[0, a].set_title(f"M = {fac:g}×L̂", loc="left")
        axes[1, a].set_xticks(range(len(etas)), ["0", "0.1", "0.5", "1", "∞"])
        axes[1, a].set_xlabel("η")
    axes[0, 0].set_ylabel("D(3) RCN / CN (mean)")
    axes[1, 0].set_ylabel("% of f evaluations\nspent on reflection")
    axes[0, 0].legend(loc="upper left", fontsize=7, ncol=2)
    fig.suptitle("E4 · sensitivity to the activation threshold η (joint, 100 seeds)", x=0.01, ha="left",
                 color=INK, fontsize=10)
    fig.tight_layout()
    p = RESULTS / "e4_eta.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return [p]


def plot_e5() -> list[Path]:
    rows = read("e5")
    ks = list(range(1, 11))
    groups = [("table_setting", "M (absolute)"), ("production", "M (× L̂)")]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.4), sharex=True)
    for a, (grp, mlab) in enumerate(groups):
        sub = [r for r in rows if r["group"] == grp]
        conds = sorted({(r["cubic"], r["n_neg"]) for r in sub})
        facs = sorted({r["M_factor"] for r in sub})
        for b, (cubic, n_neg) in enumerate(conds[:4]):
            ax = axes[a, b]
            for j, lab in enumerate(("rcn_joint", "rcn_greedy")):
                for fac, ls in zip(facs, ("-", "--")):
                    r = next((r for r in sub if r["label"] == lab and r["cubic"] == cubic
                              and r["n_neg"] == n_neg and r["M_factor"] == fac), None)
                    if r is None:
                        continue
                    ax.plot(ks, [r[f"dk{k}_mean"] for k in ks], ls, color=METHOD_COLOR[lab], lw=2,
                            label=f"{METHOD_NAME[lab]}, {mlab.split()[0]} = {fac:g}" + ("×L̂" if grp == "production" else ""))
            ax.axhline(1.0, color=INK_2, lw=1, zorder=0)
            ax.set_title(f"{'table' if grp == 'table_setting' else 'production'} · c3 = {cubic:g}, n_neg = {n_neg:g}",
                         loc="left", fontsize=9)
            if a == 1:
                ax.set_xlabel("k")
        axes[a, 0].set_ylabel("D(k) RCN / CN (mean)")
        axes[a, 0].legend(loc="upper right", fontsize=7)
    fig.suptitle("E5 · joint vs greedy over the first iterations (saddle_tensor, n = 20)",
                 x=0.01, ha="left", color=INK, fontsize=10)
    fig.tight_layout()
    p = RESULTS / "e5_joint_greedy.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return [p]


PLOTS = {"e1": plot_e1, "e2": plot_e2, "e3": plot_e3, "e4": plot_e4, "e5": plot_e5}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="+", choices=sorted(PLOTS))
    args = ap.parse_args()
    style()
    for e in args.experiments:
        for p in PLOTS[e]():
            print(f"wrote {p}")


if __name__ == "__main__":
    main()
