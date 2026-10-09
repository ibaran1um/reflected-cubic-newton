"""Rebuild results/NAME.csv from results/raw/NAME_runs.csv (after changing the aggregation).

    python benchmarks/reaggregate.py e2
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import RESULTS, aggregate, write_csv  # noqa: E402

INT_COLS = {"seed", "n_iter", "n_f", "n_grad", "n_hess", "n_reflect_tried", "n_reflect_taken", "n_hard",
            "n_rejected", "extra_f_evals"}


def _parse(v: str):
    try:
        x = float(v)
    except ValueError:
        return v
    return x


def main() -> None:
    for name in sys.argv[1:]:
        with open(RESULTS / "raw" / f"{name}_runs.csv", encoding="utf-8") as fh:
            runs = []
            for r in csv.DictReader(fh):
                row = {k: (int(float(v)) if k in INT_COLS else _parse(v)) for k, v in r.items()}
                runs.append(row)
        write_csv(RESULTS / f"{name}.csv", aggregate(runs))
        print(f"rebuilt results/{name}.csv from {len(runs)} runs")


if __name__ == "__main__":
    main()
