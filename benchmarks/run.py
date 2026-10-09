"""Run benchmark experiments and write per-run, per-iteration and aggregate CSVs.

    python benchmarks/run.py --config configs/e2.yaml [--workers 8 --threads 2]
    python benchmarks/run.py --problem saddle_tensor --method cn rcn --M 3 --seeds 10 --out results/adhoc.csv

Outputs for a config named NAME:
    results/NAME.csv              aggregate (committed)
    results/raw/NAME_runs.csv     one row per run
    results/raw/NAME_iters.csv    one row per iteration (IterLog), unless save_iters: false

L-hat (and the ball radius R it is estimated on) is computed once per
(problem, params, seed, R setting) and cached in results/raw/lhat_cache.json,
so all methods, M multiples and experiments reuse it.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import multiprocessing as mp
import statistics
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from problems import REGISTRY, choose_radius, estimate_hessian_lipschitz  # noqa: E402
from rcn import minimize  # noqa: E402

RESULTS = ROOT / "results"
LHAT_CACHE = RESULTS / "raw" / "lhat_cache.json"
K_MAX = 10  # f(x_k) is stored for k = 0..K_MAX
D_KS = (1, 3, 5)  # cumulative-decrease ratios reported in the aggregate
N_BOOT, BOOT_SEED = 2000, 0


# --------------------------------------------------------------------------- config


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        p = Path(__file__).resolve().parent / path
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def adhoc_config(args) -> dict:
    methods = []
    for m in args.method:
        if m in ("cn", "arc", "nc_pm"):  # rcn / rarc take mode and eta
            methods.append({"name": m, "method": m})
        else:
            methods.append({"name": f"{m}_{args.mode}", "method": m, "mode": args.mode, "eta": args.eta})
    return {
        "name": Path(args.out).stem,
        "groups": [
            {
                "label": "adhoc",
                "problem": args.problem,
                "grid": {},
                "M": {"absolute": args.M},
                "methods": methods,
                "solver": {},
                "seeds": {"start": 0, "count": args.seeds},
            }
        ],
    }


def expand_jobs(cfg: dict) -> list[dict]:
    jobs = []
    for g in cfg["groups"]:
        grid = g.get("grid", {})
        keys = list(grid)
        seeds = g["seeds"]
        for combo in itertools.product(*(grid[k] for k in keys)):
            params = dict(g.get("fixed", {}), **dict(zip(keys, combo)))
            for seed in range(seeds.get("start", 0), seeds.get("start", 0) + seeds["count"]):
                jobs.append({"group": g, "params": params, "seed": seed})
    return jobs


# --------------------------------------------------------------------------- L-hat cache


def lhat_key(job: dict) -> str | None:
    g = job["group"]
    if "lipschitz_multiple" not in g["M"]:
        return None
    return json.dumps(
        {"problem": g["problem"], "params": job["params"], "seed": job["seed"], "R": g.get("R", "problem")},
        sort_keys=True,
    )


def compute_lhat(job: dict) -> tuple[str, dict]:
    g, seed = job["group"], job["seed"]
    prob = REGISTRY[g["problem"]](seed, **job["params"])
    r_spec = g.get("R", "problem")
    if r_spec == "auto":
        R = choose_radius(prob.f, prob.x0, seed=seed)
    elif r_spec == "problem":
        R = prob.R
    else:
        R = float(r_spec)
    L = estimate_hessian_lipschitz(prob.f, prob.x0, R, seed=seed)
    return lhat_key(job), {"R": R, "L_hat": L}


def load_lhat_cache() -> dict:
    if LHAT_CACHE.exists():
        with open(LHAT_CACHE, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_lhat_cache(cache: dict) -> None:
    LHAT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(LHAT_CACHE, "w", encoding="utf-8") as fh:
        json.dump(cache, fh)


# --------------------------------------------------------------------------- worker


def _init_worker(threads: int) -> None:
    torch.set_num_threads(threads)
    warnings.filterwarnings("ignore", category=FutureWarning)


def run_job(job: dict) -> tuple[list[dict], list[dict]]:
    g, params, seed = job["group"], job["params"], job["seed"]
    prob = REGISTRY[g["problem"]](seed, **params)

    m_spec = g["M"]
    if "absolute" in m_spec:
        L_hat, R = math.nan, math.nan
        Ms = [("abs", float(v), float(v)) for v in m_spec["absolute"]]
    else:
        L_hat, R = job["lhat"]["L_hat"], job["lhat"]["R"]
        Ms = [("Lmult", float(v), float(v) * L_hat) for v in m_spec["lipschitz_multiple"]]

    # Untimed warm-up: the first torch.func grad / hessian call on a new
    # function costs ~2 s, which would otherwise land on whichever method runs first.
    torch.func.grad(prob.f)(prob.x0)
    torch.func.hessian(prob.f)(prob.x0)

    runs, iters = [], []
    for (m_kind, m_factor, M), meth in itertools.product(Ms, g["methods"]):
        kw = dict(g.get("solver", {}))
        kw.update({k: meth[k] for k in ("mode", "eta") if k in meth})
        res = minimize(prob.f, prob.x0, method=meth["method"], M=M, **kw)
        h = res.history
        key = dict(
            group=g["label"],
            problem=prob.name,
            **prob.params,
            seed=seed,
            M_kind=m_kind,
            M_factor=m_factor,
            M=M,
            L_hat=L_hat,
            R=R,
            label=meth["name"],
            reference=g.get("reference", "cn"),  # baseline label for the paired ratios
        )
        row = dict(
            key,
            method=meth["method"],
            mode=meth.get("mode", ""),
            eta=meth.get("eta", ""),
            status=res.status,
            n_iter=res.n_iter,
            n_f=res.n_f,
            n_grad=res.n_grad,
            n_hess=res.n_hess,
            time_sec=h[-1].time_sec,
            time_extra=sum(e.time_extra for e in h),
            f_final=res.f,
            x_dist=torch.linalg.vector_norm(res.x - prob.x0).item(),
            n_reflect_tried=sum(e.reflect_tried for e in h),
            n_reflect_taken=sum(e.reflect_taken for e in h),
            n_hard=sum(e.hard_case for e in h),
            n_rejected=sum(not e.step_accepted for e in h[:-1]),
            extra_f_evals=sum(e.extra_f_evals for e in h),
        )
        for k in range(K_MAX + 1):
            row[f"f_{k}"] = h[min(k, len(h) - 1)].f
        row["f0"], row["f1"] = row["f_0"], row["f_1"]
        runs.append(row)
        if g.get("save_iters", True):
            iters.extend(dict(key, **e.as_dict()) for e in h)
    return runs, iters


# --------------------------------------------------------------------------- aggregate


def _q(xs, q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return math.nan
    pos = q * (len(xs) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _mean_se(xs) -> tuple[float, float]:
    xs = [x for x in xs if not math.isnan(x)]
    if not xs:
        return math.nan, math.nan
    if len(xs) == 1:
        return xs[0], math.nan
    return statistics.mean(xs), statistics.stdev(xs) / math.sqrt(len(xs))


def _boot(xs) -> dict:
    """Percentile bootstrap 95% CIs for the mean and the median."""
    x = np.asarray([v for v in xs if not math.isnan(v)], dtype=float)
    if x.size == 0:
        return dict(mean=math.nan, mean_lo=math.nan, mean_hi=math.nan, median=math.nan, median_lo=math.nan, median_hi=math.nan)
    rng = np.random.default_rng(BOOT_SEED)
    sample = x[rng.integers(0, x.size, size=(N_BOOT, x.size))]
    means, medians = sample.mean(axis=1), np.median(sample, axis=1)
    return dict(
        mean=float(x.mean()),
        mean_lo=float(np.quantile(means, 0.025)),
        mean_hi=float(np.quantile(means, 0.975)),
        median=float(np.median(x)),
        median_lo=float(np.quantile(medians, 0.025)),
        median_hi=float(np.quantile(medians, 0.975)),
    )


# Per-seed values that must not split an aggregate cell ("saddle" = which
# Himmelblau saddle the seed picked).
PARAM_SKIP = {"seed", "M", "L_hat", "R", "label", "saddle"}
RUN_ONLY = {
    "method", "mode", "eta", "status", "n_iter", "n_f", "n_grad", "n_hess", "time_sec", "time_extra",
    "f0", "f1", "f_final", "x_dist", "n_reflect_tried", "n_reflect_taken", "n_hard", "n_rejected",
    "extra_f_evals", *(f"f_{k}" for k in range(K_MAX + 1)),
}


def aggregate(runs: list[dict]) -> list[dict]:
    """Group by everything except seed; compare each label with the group's
    reference method (default cn) on the same seed and M."""
    by_cell: dict[tuple, dict[str, dict[int, dict]]] = {}
    for r in runs:
        cell = tuple((k, v) for k, v in r.items() if k not in RUN_ONLY and k not in PARAM_SKIP)
        by_cell.setdefault(cell, {}).setdefault(r["label"], {})[r["seed"]] = r

    out = []
    for cell, labels in by_cell.items():
        ref = dict(cell).get("reference") or "cn"
        cn = labels.get(ref, {})
        for label, by_seed in labels.items():
            rs = list(by_seed.values())
            it = [r["n_iter"] for r in rs]
            row = dict(cell, label=label, n_runs=len(rs))
            row["sosp_rate"] = sum(r["status"] == "sosp" for r in rs) / len(rs)
            row["M_mean"] = statistics.mean(r["M"] for r in rs)
            finite_L = [r["L_hat"] for r in rs if not math.isnan(r["L_hat"])]
            row["L_hat_mean"] = statistics.mean(finite_L) if finite_L else math.nan
            finite_R = [r["R"] for r in rs if not math.isnan(r["R"])]
            row["R_mean"] = statistics.mean(finite_R) if finite_R else math.nan
            row["iter_mean"], row["iter_se"] = _mean_se([float(v) for v in it])
            row["iter_median"], row["iter_q1"], row["iter_q3"] = _q(it, 0.5), _q(it, 0.25), _q(it, 0.75)
            for col in ("n_f", "n_hess", "time_sec"):
                vals = [r[col] for r in rs]
                row[f"{col}_median"], row[f"{col}_q1"], row[f"{col}_q3"] = _q(vals, 0.5), _q(vals, 0.25), _q(vals, 0.75)
            row["n_f_mean"] = statistics.mean(r["n_f"] for r in rs)
            row["time_median"] = row["time_sec_median"]
            total_t = sum(r["time_sec"] for r in rs)
            row["time_extra_frac"] = sum(r["time_extra"] for r in rs) / total_t if total_t > 0 else math.nan
            row["extra_f_frac"] = sum(r["extra_f_evals"] for r in rs) / max(1, sum(r["n_f"] for r in rs))
            row["reflect_taken_per_iter"] = sum(r["n_reflect_taken"] for r in rs) / max(1, sum(it))
            row["f_final_mean"] = statistics.mean(r["f_final"] for r in rs)

            paired = [(r, cn[s]) for s, r in by_seed.items() if s in cn]
            # cumulative decrease D(k) = f(x0) - f(x_k), ratio against cn on the same problem
            for k in D_KS:
                ratios, invalid = [], 0
                for r, c in paired:
                    d_c, d_r = c["f_0"] - c[f"f_{k}"], r["f_0"] - r[f"f_{k}"]
                    if d_c > 0:
                        ratios.append(d_r / d_c)
                    else:
                        invalid += 1
                b = _boot(ratios)
                for stat, v in b.items():
                    row[f"d{k}_ratio_{stat}"] = v
                row[f"d{k}_frac_up"] = sum(v >= 1.01 for v in ratios) / len(ratios) if ratios else math.nan
                row[f"d{k}_frac_down"] = sum(v <= 0.99 for v in ratios) / len(ratios) if ratios else math.nan
                row[f"d{k}_n_invalid"] = invalid
                if k == 1:  # E1 names
                    row["step1_ratio_mean"], row["step1_ratio_se"] = _mean_se(ratios)
                    row["step1_ratio_median"] = b["median"]
            # mean / median of the ratio for every k (E5 curves; no CI)
            for k in range(1, K_MAX + 1):
                ratios = [
                    (r["f_0"] - r[f"f_{k}"]) / (c["f_0"] - c[f"f_{k}"]) for r, c in paired if c["f_0"] > c[f"f_{k}"]
                ]
                row[f"dk{k}_mean"] = statistics.mean(ratios) if ratios else math.nan
                row[f"dk{k}_median"] = statistics.median(ratios) if ratios else math.nan

            diff = [float(r["n_iter"] - c["n_iter"]) for r, c in paired]
            row["iter_diff_mean"], row["iter_diff_se"] = _mean_se(diff)
            bd = _boot(diff)
            row["iter_diff_ci_lo"], row["iter_diff_ci_hi"] = bd["mean_lo"], bd["mean_hi"]
            # final f lower / higher than the reference's by at least 1% of the
            # reference's total decrease f0 - f_final. Scale-free, and robust
            # where the minima are at f = 0 (an absolute or f-relative
            # threshold counted floating-point noise as a difference).
            def _tol(c):
                return 0.01 * max(c["f_0"] - c["f_final"], 0.0)

            row["frac_lower_f_final"] = (
                sum(r["f_final"] < c["f_final"] - _tol(c) for r, c in paired) / len(paired) if paired else math.nan
            )
            row["frac_higher_f_final"] = (
                sum(r["f_final"] > c["f_final"] + _tol(c) for r, c in paired) / len(paired) if paired else math.nan
            )
            row["frac_same_iters"] = sum(d == 0 for d in diff) / len(paired) if paired else math.nan
            out.append(row)
    return out


# --------------------------------------------------------------------------- io


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        fields.extend(k for k in r if k not in fields)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _pmap(fn, items, workers: int, threads: int, label: str):
    t0 = time.perf_counter()
    if workers <= 1:
        _init_worker(threads)
        results = map(fn, items)
        pool = None
    else:
        pool = mp.Pool(workers, initializer=_init_worker, initargs=(threads,))
        results = pool.imap(fn, items, chunksize=1)
    step = max(1, len(items) // 20)
    for i, res in enumerate(results, 1):
        yield res
        if i % step == 0 or i == len(items):
            print(f"  {label}: {i}/{len(items)}, {time.perf_counter() - t0:.0f}s", flush=True)
    if pool is not None:
        pool.close()
        pool.join()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--problem", default="saddle_tensor")
    ap.add_argument("--method", nargs="+", default=["cn", "rcn"])
    ap.add_argument("--mode", default="joint")
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--M", type=float, nargs="+", default=[3.0])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="results/adhoc.csv")
    ap.add_argument("--workers", type=int, default=None, help="default: config run.workers, else 8")
    ap.add_argument("--threads", type=int, default=None, help="torch threads per worker (default: config, else 2)")
    ap.add_argument("--max-seeds", type=int, default=None, help="cap seeds per group (smoke tests)")
    ap.add_argument("--name", default=None, help="override the output name")
    args = ap.parse_args(argv)

    cfg = load_config(args.config) if args.config else adhoc_config(args)
    if args.max_seeds is not None:
        for g in cfg["groups"]:
            g["seeds"]["count"] = min(g["seeds"]["count"], args.max_seeds)
    if args.name:
        cfg["name"] = args.name
    run_cfg = cfg.get("run", {})
    workers = args.workers if args.workers is not None else run_cfg.get("workers", 8)
    threads = args.threads if args.threads is not None else run_cfg.get("threads", 2)
    name = cfg["name"]
    out = RESULTS / f"{name}.csv" if args.config else ROOT / args.out
    jobs = expand_jobs(cfg)
    t0 = time.perf_counter()

    # Phase 1: L-hat for every (problem, params, seed) not cached yet. Always
    # run in parallel: it does not affect the timings measured in phase 2.
    cache = load_lhat_cache()
    todo = {}
    for j in jobs:
        key = lhat_key(j)
        if key is not None and key not in cache:
            todo.setdefault(key, j)
    if todo:
        print(f"{name}: L-hat for {len(todo)} instances", flush=True)
        for key, val in _pmap(compute_lhat, list(todo.values()), max(workers, 8), threads, "L-hat"):
            cache[key] = val
        save_lhat_cache(cache)
    for j in jobs:
        key = lhat_key(j)
        j["lhat"] = cache[key] if key is not None else None

    # Phase 2: the runs.
    print(f"{name}: {len(jobs)} jobs on {workers} workers x {threads} threads", flush=True)
    runs, iters = [], []
    for r, it in _pmap(run_job, jobs, workers, threads, "jobs"):
        runs.extend(r)
        iters.extend(it)

    write_csv(RESULTS / "raw" / f"{name}_runs.csv", runs)
    if iters:
        write_csv(RESULTS / "raw" / f"{name}_iters.csv", iters)
    write_csv(out, aggregate(runs))
    print(f"wrote {out} ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
