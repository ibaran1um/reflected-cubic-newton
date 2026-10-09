"""Print the Markdown result tables used in README.md from results/*.csv.

    python benchmarks/readme_tables.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def read(name: str) -> list[dict]:
    with open(RESULTS / f"{name}.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f(r: dict, k: str) -> float:
    try:
        return float(r[k])
    except (KeyError, ValueError):
        return math.nan


def name(r: dict) -> str:
    p = r["problem"]
    if p == "saddle_tensor":
        return f"saddle_tensor n={f(r, 'n'):g}, cubic {f(r, 'cubic'):g}"
    if p == "test_functions":
        return {"himmelblau": "Himmelblau", "cubic_saddle": "cubic saddle 2-D †"}[r["fn"]]
    return {"matrix_factorization": "matrix factorization ‡", "phase_retrieval": "phase retrieval ‡",
            "small_mlp": "small MLP ‡"}[p]


def ci(r: dict, k: int) -> str:
    return f"{f(r, f'd{k}_ratio_mean'):.3f} [{f(r, f'd{k}_ratio_mean_lo'):.3f}, {f(r, f'd{k}_ratio_mean_hi'):.3f}]"


def e2_table(fac: str) -> str:
    rows = read("e2")
    by = {(name(r), r["label"]): r for r in rows if r["M_factor"] == fac}
    names = list(dict.fromkeys(name(r) for r in rows if r["M_factor"] == fac))
    out = [
        "| Problem | D(1) ratio, mean [95% CI] | D(3) ratio, mean [95% CI] | D(3) median | D(3) ≥1% better / worse "
        "| extra f-evals | reached SOSP, CN / RCN |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for nm in names:
        r, c = by.get((nm, "rcn_joint")), by.get((nm, "cn"))
        if r is None:
            continue
        out.append(
            f"| {nm} | {ci(r, 1)} | {ci(r, 3)} | {f(r, 'd3_ratio_median'):.3f} "
            f"| {100 * f(r, 'd3_frac_up'):.0f}% / {100 * f(r, 'd3_frac_down'):.0f}% "
            f"| {100 * f(r, 'extra_f_frac'):.1f}% | {100 * f(c, 'sosp_rate'):.0f}% / {100 * f(r, 'sosp_rate'):.0f}% |"
        )
    return "\n".join(out)


def iters_table() -> str:
    rows = read("e2_iters")
    out = ["| Problem | CN median | RCN − CN, mean [95% CI] | final f lower / higher than CN |",
           "| --- | --- | --- | --- |"]
    by = {(name(r), r["label"]): r for r in rows}
    for nm in dict.fromkeys(name(r) for r in rows):
        r, c = by[(nm, "rcn_joint")], by[(nm, "cn")]
        out.append(f"| {nm} | {f(c, 'iter_median'):.0f} | {f(r, 'iter_diff_mean'):+.2f} "
                   f"[{f(r, 'iter_diff_ci_lo'):+.2f}, {f(r, 'iter_diff_ci_hi'):+.2f}] "
                   f"| {100 * f(r, 'frac_lower_f_final'):.0f}% / {100 * f(r, 'frac_higher_f_final'):.0f}% |")
    return "\n".join(out)


def e8_table() -> str:
    rows = [r for r in read("e8") if r["label"] == "rarc_joint"]
    out = ["| Problem | M₀ | D(1) ratio vs ARC, mean [95% CI] | D(3) ratio vs ARC, mean [95% CI] | D(3) ≥1% better / worse |",
           "| --- | --- | --- | --- | --- |"]
    for r in rows:
        out.append(f"| {name(r)} | {f(r, 'M_factor'):g}×L̂ | {ci(r, 1)} | {ci(r, 3)} "
                   f"| {100 * f(r, 'd3_frac_up'):.0f}% / {100 * f(r, 'd3_frac_down'):.0f}% |")
    return "\n".join(out)


def e3_table() -> str:
    rows = read("e3")
    out = ["| Problem | time on reflection, joint | greedy | joint, η = ∞ |", "| --- | --- | --- | --- |"]
    by = {(name(r), r["label"]): r for r in rows}
    for nm in dict.fromkeys(name(r) for r in rows):
        cells = [f"{100 * f(by[(nm, lab)], 'time_extra_frac'):.1f}%" for lab in ("rcn_joint", "rcn_greedy", "rcn_joint_always")]
        out.append(f"| {nm} | " + " | ".join(cells) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    for fac in ("1.0", "3.0", "10.0"):
        print(f"### E2, M = {fac} x L-hat\n\n{e2_table(fac)}\n")
    print(f"### E2 iterations\n\n{iters_table()}\n")
    for nm, fn in (("E8", e8_table), ("E3", e3_table)):
        try:
            print(f"### {nm}\n\n{fn()}\n")
        except FileNotFoundError:
            print(f"### {nm}: not run yet\n")
