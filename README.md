# Reflected Cubic Newton

**Cubic Newton with sign-reflected negative-curvature steps.** After the usual cubic-regularized Newton (CN) step, RCN also evaluates the step with its negative-curvature components flipped, and keeps whichever has the lower function value. It costs at most one extra function evaluation per iteration and no extra derivatives.

![CN vs RCN trajectories from the same start next to a saddle](results/e7_trajectories.png)

*Left: a 2-D function built to show the mechanism (an illustration, not evidence). Right: Himmelblau's function. Same start, same M; RCN reflects the first step.*

> Research prototype, October 2026. Small dense problems only: tested up to n = 141; the dense eigendecomposition limits it to a few thousand. Read [Limitations](#limitations) before using the numbers below.

## Idea

The cubic term of the CN model, (M/6)‖s‖³, is *even*: it does not change when s → −s. The true third-order Taylor term, (1/6)∇³f(x)[s, s, s], is *odd*. Near a strict saddle the model therefore rates the steps ±s along negative curvature as almost equally good, and CN chooses between them by a tiny gradient component. RCN repeats that choice using actual function values.

## Algorithm

One iteration, with g = ∇f(x), H = ∇²f(x) = QΛQᵀ and fixed M > 0:

1. Solve the cubic subproblem m(s) = gᵀs + ½sᵀHs + (M/6)‖s‖³ exactly in the eigenbasis (bisection on the shift μ, hard case included).
2. Let N = {i : λᵢ < −δ}. The reflected candidate s′ flips the sign of the components of s in N. Only the linear term of the model changes:

   ```text
   Δ_ref = m(s′) − m(s) = 2 Σ_{i∈N} |g̃ᵢ s̃ᵢ| ≥ 0
   ```

3. Evaluate s′ only if η > 0, N ≠ ∅ and Δ_ref ≤ η·(−m(s)) (default η = 0.1). Far from a saddle the gradient is large, Δ_ref grows and the test skips the extra evaluation.
4. Take whichever of x + s and x + s′ has the lower f (`mode="joint"`), or flip the components of N one at a time (`mode="greedy"`, up to |N| extra evaluations).

The CN step is always a candidate, so every iterate is at least as good as the CN step from the same point.

```python
def rcn_step(f, x, M, eta=0.1):
    g, H = grad(f)(x), hessian(f)(x)
    lam, Q = eigh(H)
    gt = Q.T @ g
    st = solve_cubic(lam, gt, M)                       # exact global minimizer of m
    best = x + Q @ st
    N = lam < -delta(lam)
    if eta > 0 and N.any():
        d_ref = 2 * abs(gt[N] * st[N]).sum()
        if d_ref <= eta * (-model(lam, gt, st, M)):
            st2 = st.clone(); st2[N] *= -1             # reflect the negative-curvature part
            cand = x + Q @ st2
            if f(cand) < f(best):
                best = cand
    return best
```

## What is claimed

- **Monotone, and CN's rate at the CN trial points.** Each iterate is no worse than the CN trial point y_k = x_k + s_k, so every iteration decreases f at least as much as CN would from x_k. With M ≥ L (the Hessian Lipschitz constant), the standard CN argument then gives the O(ε^(−3/2)) bound for the trial points y_k. When a reflection is accepted, the guarantee is at y_k, not at the next iterate x_{k+1} ([Theorem 1](docs/proofs.md#theorem-1-monotonicity-and-complexity-at-the-cn-trial-points)).
- **A larger first step at strict saddles.** From an exact saddle (g = 0) with a simple λ₁ and eigenvector v, the decrease is 2|λ₁|³/M² · (1 + a) for RCN and 2|λ₁|³/M² · (1 ± a) for CN, with a = (2/3)|∇³f[v, v, v]|/M. This is exact for cubic polynomials; in general it holds up to (2/3)L₃λ₁⁴/M⁴, where L₃ is the Lipschitz constant of the third derivative ([Theorem 2](docs/proofs.md#theorem-2-first-step-from-an-exact-strict-saddle)). E6 checks the exact case (see [Results](#results)).

Not claimed:

- Better local minima. On saddle_tensor RCN ends at a lower final value than CN in 14–31% of runs and at a higher one in 21–31% (a difference of at least 1% of CN's total decrease), and at a different point in about half.
- Fewer iterations to a second-order point in general. Where the 95% CI of the paired difference contains 0, no saving is claimed.
- Any effect at saddles that are fixed points of a sign symmetry (f(x) = f(−x) for matrix factorization and phase retrieval; flipping a hidden unit's weights for the tanh MLP). There ∇³f[v, v, v] = 0 along the negative-curvature eigenvectors; reflections occasionally win by tiny margins, but no run's 1-, 3- or 5-iteration decrease moves by more than 0.6%.
- Being faster than adaptive regularization. ARC improves on fixed-M CN by much more than the reflection does; the reflection is an add-on that works with either (E8).
- Large-scale use. Every iteration runs a dense eigendecomposition.

## Results

**Near a saddle with third-order asymmetry, RCN gets more decrease from the first few iterations than CN at the same M, for 3.5–10% extra function evaluations (more on symmetric problems, see the table) and under 2.5% extra wall time. On sign-symmetric problems it changes nothing.** The gain shrinks as M grows and as n grows, and it is about the same when added to ARC.

The main metric is the paired ratio D_RCN(k) / D_CN(k), where D(k) = f(x₀) − f(x_k), on the same problem, start and M (100 seeds per row; RCN joint, η = 0.1). It looks at the iterations before the two runs can head to different minima, which is what Theorem 2 is about.

### E2 at M = 3×L̂

| Problem | D(1) ratio, mean [95% CI] | D(3) ratio, mean [95% CI] | D(3) median | D(3) ≥1% better / worse | extra f-evals | reached SOSP, CN / RCN |
| --- | --- | --- | --- | --- | --- | --- |
| saddle_tensor n=20, cubic 0.3 | 1.033 [1.024, 1.042] | 1.339 [1.226, 1.461] | 1.064 | 52% / 1% | 7.1% | 100% / 100% |
| saddle_tensor n=20, cubic 1 | 1.035 [1.025, 1.045] | 1.434 [1.280, 1.607] | 1.073 | 52% / 1% | 4.4% | 100% / 100% |
| saddle_tensor n=50, cubic 0.3 | 1.026 [1.020, 1.033] | 1.270 [1.196, 1.359] | 1.023 | 52% / 2% | 4.9% | 100% / 100% |
| saddle_tensor n=50, cubic 1 | 1.026 [1.020, 1.034] | 1.291 [1.210, 1.388] | 1.024 | 52% / 2% | 4.0% | 100% / 100% |
| saddle_tensor n=100, cubic 0.3 | 1.018 [1.012, 1.024] | 1.163 [1.104, 1.232] | 1.000 | 42% / 5% | 3.9% | 100% / 100% |
| saddle_tensor n=100, cubic 1 | 1.018 [1.012, 1.024] | 1.169 [1.109, 1.241] | 1.000 | 43% / 5% | 3.5% | 100% / 100% |
| matrix factorization ‡ | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 | 0% / 0% | 38.1% | 100% / 100% |
| phase retrieval ‡ | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 | 0% / 0% | 4.0% | 100% / 100% |
| small MLP ‡ | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 | 0% / 0% | 2.8% | 0% / 0% |
| Himmelblau | 1.002 [1.002, 1.003] | 1.020 [1.015, 1.025] | 1.000 | 49% / 0% | 7.8% | 100% / 100% |
| cubic saddle 2-D † | 1.134 [1.106, 1.161] | 4.744 [3.956, 5.529] | 1.000 | 47% / 0% | 9.6% | 100% / 100% |

† Constructed to show the mechanism (the figure at the top); an illustration, not evidence.
‡ The saddle is a fixed point of a sign symmetry: f(x) = f(−x) for matrix factorization and phase retrieval, and flipping a hidden unit's input weights, bias and output weight together for the tanh MLP. ∇³f(0)[v, v, v] = 0 along the negative-curvature eigenvectors (checked by autodiff). Reflections are still accepted in some runs, winning by tiny margins because the start is slightly off the symmetric point, but in every run D(1), D(3) and D(5) stay within 0.6% of CN's and the final value within 0.5% of CN's total decrease. The MLP does not reach a second-order point within 200 iterations with any method (its landscape is nearly flat), and M = 1×L̂ is left out for it because CN itself increased f there (L̂ was not an upper bound).

At M = 1×L̂ the first-step gain on saddle_tensor is larger (D(1) 1.06–1.13), but 9–15% of problems then have a smaller 3-iteration decrease; at M = 10×L̂ it is 1.005–1.010 with no problem worse. Full tables: [`results/e2.csv`](results/e2.csv), figures [`e2_dk.png`](results/e2_dk.png) and [`e2_share.png`](results/e2_share.png). ARC and nc_pm are in the same files; nc_pm is a simplified baseline written for this comparison and does not represent the method that inspired it.

### Iterations to a second-order point (secondary; M = 3×L̂, 500 seeds)

| Problem | CN median | RCN − CN, mean [95% CI] | final f lower / higher than CN |
| --- | --- | --- | --- |
| saddle_tensor n=20, cubic 1 | 20 | +0.22 [−0.46, +0.90] | 23% / 24% |
| matrix factorization ‡ | 8 | 0 | 0% / 0% |
| phase retrieval ‡ | 18 | 0 | 0% / 0% |
| small MLP ‡ | 200 (cap) | 0 | 0% / 0% |
| Himmelblau | 11 | −0.47 [−0.55, −0.39] | 0% / 0% |
| cubic saddle 2-D † | 8 | +0.66 [+0.60, +0.73] | 51% / 0% |

No saving is claimed where the interval contains 0. "Lower / higher" counts runs whose final value differs from CN's by at least 1% of CN's total decrease f(x₀) − f(x_final). On the cubic saddle RCN needs more iterations because it goes to the deeper, farther valley.

**On top of ARC (E8).** Adding the same reflected candidates to every ARC step (`method="rarc"`) keeps the early gain: D(3) relative to plain ARC is 1.11–1.90 on saddle_tensor (all M₀ and n), 1.000 on the symmetric problems, and 1.00–1.02 on Himmelblau. Iteration counts do not change significantly. Theorem 1 is for fixed M and does not cover `rarc`. [`results/e8.csv`](results/e8.csv)

**Theorem 2 (E6).** On saddle_tensor without the quartic term, one step from the exact saddle matches the closed form to a relative error of 4 × 10⁻¹⁵ in all 1200 trials; in the 630 trials where CN takes the worse sign, RCN / CN equals (1 + a)/(1 − a). [`e6_theorem2.png`](results/e6_theorem2.png)

**Cost (E3, one thread).** Share of wall time spent on reflected candidates: joint 0.1–2.2%, greedy 0.2–10.3% (the larger value on the MLP with about 70 negative eigenvalues). Each iteration is dominated by the Hessian and its eigendecomposition. [`e3_cost.png`](results/e3_cost.png)

**Threshold η (E4).** D(k) is identical for η = 0.1, 0.5, 1 and ∞ on every problem; only the wasted evaluations change, hence the default η = 0.1. [`e4_eta.png`](results/e4_eta.png)

**Preliminary table (E1).** The NumPy prototype's numbers are reproduced within sampling error with 1000 problems. [`e1_table_repro.png`](results/e1_table_repro.png)

## Limitations

- **The effect fades as M grows.** The gain scales with (third-order asymmetry) / M. With M = 10×L̂ the mean first-step gain on saddle_tensor is 0.5–1%.
- **It fades with dimension** at a fixed multiple of L̂: at n = 100 it is about half of that at n = 20.
- **Symmetric saddles get nothing**, and the activation test cannot filter them: with Δ_ref ≈ 0 the reflected candidate is still evaluated. On matrix factorization this wastes about 5 function evaluations per run (2–3% of wall time). A rule that stops after repeated exact ties would not fire there either: the reflected value differs from the CN value by a relative 1e-11 to 5e-2, because the start is not exactly at the symmetric point.
- **Heavy tails.** Means are driven by a minority of problems; medians of the per-problem ratio are 1.00–1.07. Both are reported, together with the share of problems changed by at least 1% either way.
- **Later iterations can be worse.** After the first step RCN and CN follow different paths; at M = 1×L̂, 9–15% of saddle_tensor problems have a smaller 3-iteration decrease under RCN.
- **greedy is expensive on many negative eigenvalues.** On the MLP (about 70 negative eigenvalues near the start) 90% of its function evaluations go to reflections, 10% of the wall time. It beats joint only with η = ∞ and large cubic terms. Use the default `joint`.
- **Dense and small.** O(n³) per iteration from the eigendecomposition; tested up to n = 141.
- **L̂ is an estimate.** M is set as a multiple of a sampled Hessian-Lipschitz estimate; on the MLP, M = 1×L̂ was not an upper bound (CN itself increased f), so that setting is excluded.

## Benchmarks

- **Problems.** `saddle_tensor`: ½xᵀAx + (1/6)T[x, x, x] + (c/24)‖x‖⁴, A with n_neg eigenvalues in −U(0.5, 1.5) and the rest in U(0.5, 2.0), T a standard-normal tensor averaged over the six index permutations and scaled by the "cubic size", c = 1, start 10⁻⁶·N(0, I) from the saddle at 0. `matrix_factorization`: ¼‖UUᵀ − Y‖²_F, d = 20, rank 2. `phase_retrieval`: (1/4m) Σ((aᵢᵀx)² − yᵢ)², n = 20, m = 100. `small_mlp`: 2-layer tanh MLP, 141 parameters, regression on a random teacher. `test_functions`: Himmelblau (start 10⁻⁶ from one of its four saddles) and a constructed 2-D cubic saddle used for illustration only.
- **Regularization.** M ∈ {1, 3, 10} × L̂, where L̂ is the largest ‖∇²f(x) − ∇²f(y)‖₂/‖x − y‖ over 200 random pairs in a ball of radius R around the start. R is chosen per instance so that the ball covers the first CN step at M = L̂, i.e. R = 2|λ₁(x₀)|/L̂(R), solved by bisection; a fixed R = 1 made M far too small on matrix factorization.
- **Fairness.** All methods share the exact subproblem solver; CN and RCN always use the same M, start and seed. Function evaluations count trial points only (gradients and Hessians are counted separately).
- **Baselines.** `cn`: Nesterov–Polyak with fixed M. `arc`: Cartis–Gould–Toint adaptive regularization (ρ thresholds 0.1 / 0.9, M × 2 / × 0.5). `nc_pm`: a simplified negative-curvature method written for this comparison — a step of length 2|λ₁|/M along ±v₁ with the sign chosen by two function values, used when its predicted decrease beats a gradient step's (Armijo backtracking otherwise). **It is a simplified baseline and does not represent the performance of the method in the paper that inspired it.**

## Install

```bash
git clone https://github.com/ibaran1um/reflected-cubic-newton
cd reflected-cubic-newton
python -m venv .venv && . .venv/bin/activate        # Python 3.11+
pip install -e ".[dev]"                             # torch>=2.3, numpy, matplotlib, pyyaml; dev: pytest, scipy
pytest
```

## Usage

```python
import torch
from rcn import minimize

def f(x):                                           # 1-D float64 tensor -> scalar tensor, torch.func-traceable
    return 0.5 * x[0]**2 - 0.5 * x[1]**2 + x[1]**3 / 6 + (x @ x)**2 / 24

res = minimize(f, torch.tensor([2e-3, 1e-3], dtype=torch.float64),
               method="rcn",                        # "cn" | "rcn" | "arc" | "rarc" | "nc_pm"
               M=3.0, mode="joint", eta=0.1, eps_g=1e-8, max_iter=200)
res.x, res.status, res.history                      # final point, "sosp" | "max_iter", list of IterLog
```

`method="rarc"` is ARC with the reflected candidates added to every step (experimental; the guarantees in [docs/proofs.md](docs/proofs.md) are for fixed M).

## Reproduce

```bash
python benchmarks/run.py --config benchmarks/configs/e2.yaml   # also e1, e2_iters, e3, e4, e5, e8
python benchmarks/e6.py                             # Theorem 2 check
python benchmarks/e7.py                             # the figure at the top
python benchmarks/plot.py e1 e2 e3 e4 e5            # figures from results/*.csv
```

Aggregated CSVs and figures are in `results/`. L̂ is cached in `results/raw/lhat_cache.json` after the first run; computing it dominates the run time (about 1.5 h for E2 with the default 8 worker processes × 2 torch threads, on a 16-thread CPU; the runs themselves took another hour).

## Related work

To our knowledge, as of October 2026 the same method has not been published; we found no prior work that reflects part of a cubic-regularized Newton step. The closest idea is choosing ±q along a negative-curvature direction by two function values, which applies to the curvature direction alone rather than to the CN step.

- Y. Nesterov, B. T. Polyak. [Cubic regularization of Newton method and its global performance](https://doi.org/10.1007/s10107-006-0706-8). Mathematical Programming 108(1):177–205, 2006. — the base CN method.
- C. Cartis, N. I. M. Gould, Ph. L. Toint. [Adaptive cubic regularisation methods for unconstrained optimization. Part I: motivation, convergence and numerical results](https://doi.org/10.1007/s10107-009-0286-5). Mathematical Programming 127(2):245–295, 2011. — the `arc` baseline.
- A. S. Berahas, R. Bollapragada, W. Dong. [Negative Curvature Methods with High-Probability Complexity Guarantees for Stochastic Nonconvex Optimization](https://arxiv.org/abs/2603.03625). arXiv:2603.03625, 2026. — choosing ±q by two function values; closest in spirit, and the idea behind the simplified `nc_pm` baseline (not a reproduction).
- S. Gratton, S. Jerad, Ph. L. Toint. [Yet another fast variant of Newton's method for nonconvex optimization](https://arxiv.org/abs/2302.10065). arXiv:2302.10065, 2023. — alternates regularized Newton and negative-curvature steps.
- A. Anandkumar, R. Ge. [Efficient approaches for escaping higher order saddle points in non-convex optimization](https://arxiv.org/abs/1602.05908). arXiv:1602.05908, 2016. — third-order information at saddles.
- Y. Yu, P. Xu, Q. Gu. [Third-order Smoothness Helps: Even Faster Stochastic Optimization Algorithms for Finding Local Minima](https://arxiv.org/abs/1712.06585). arXiv:1712.06585, 2017. — larger negative-curvature steps from third-order smoothness.

Titles, authors and venues checked against the arXiv abstract pages and the publishers' DOI records on 2026-10-09.

## License

MIT © ibaran1um
