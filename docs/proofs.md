# Proofs

This note proves the two statements in the README's "What is claimed". Notation follows the README: $g = \nabla f(x)$, $H = \nabla^2 f(x)$, and the CN model

$$
m_x(s) = \langle g, s\rangle + \tfrac12 \langle Hs, s\rangle + \tfrac{M}{6}\|s\|^3 .
$$

$s_k$ is the exact global minimizer of $m_{x_k}$ (the CN step), $r_k = \|s_k\|$, and $y_k = x_k + s_k$ is the CN trial point. RCN takes $x_{k+1} \in \{y_k\} \cup \{\text{reflected candidates}\}$ with the lowest $f$, so

$$
f(x_{k+1}) \le f(y_k) \qquad \text{for every } k. \tag{0}
$$

This holds for both `mode="joint"` and `mode="greedy"` and for every $\eta$, because the CN step is always a candidate. Everything below uses only (0).

## Assumptions

- **(A1)** $\nabla^2 f$ is $L$-Lipschitz: $\|\nabla^2 f(x) - \nabla^2 f(y)\| \le L\|x - y\|$.
- **(A2)** $f$ is bounded below by $f_{\mathrm{low}}$.
- **(A3)** (Theorem 2 only) $\nabla^3 f$ is $L_3$-Lipschitz.

## Theorem 1 (monotonicity and complexity at the CN trial points)

Assume (A1), (A2) and $M \ge L$. Then

1. $f(x_k) - f(x_{k+1}) \ge \frac{M}{12} r_k^3$ for every $k$; in particular $f(x_k)$ is non-increasing.
2. For $\varepsilon > 0$ and
   $$
   K \;\ge\; \frac{12\,(f(x_0) - f_{\mathrm{low}})}{M}\left(\frac{L+M}{2\varepsilon}\right)^{3/2},
   $$
   some $k < K$ has a CN trial point $y_k$ with
   $$
   \|\nabla f(y_k)\| \le \varepsilon, \qquad \lambda_{\min}\!\left(\nabla^2 f(y_k)\right) \ge -\sqrt{2(L+M)\,\varepsilon}.
   $$

So the iteration count is $O(\varepsilon^{-3/2})$, as for CN. **The guarantee is at $y_k$.** In iterations where no reflection is accepted, $y_k = x_{k+1}$ and it is a statement about the iterates, exactly as for CN; when a reflection is accepted, nothing is claimed about $\nabla f(x_{k+1})$.

### Proof

We use three standard facts about the exact CN step (Nesterov & Polyak, 2006). Write $s = s_k$, $r = r_k$.

**(F1) Optimality.** The global minimizer satisfies
$$
g + Hs + \tfrac{M}{2} r\, s = 0, \qquad H + \tfrac{M}{2} r\, I \succeq 0 .
$$

**(F2) Model decrease.** By (F1), $\langle g, s\rangle = -\langle Hs, s\rangle - \tfrac{M}{2} r^3$, so
$$
m_x(s) = -\tfrac12\langle Hs, s\rangle - \tfrac{M}{3} r^3 \le \tfrac{M}{4} r^3 - \tfrac{M}{3} r^3 = -\tfrac{M}{12} r^3 ,
$$
using $\langle Hs, s\rangle \ge -\tfrac{M}{2} r^3$ from the second part of (F1).

**(F3) Upper model.** By (A1), $f(x+s) \le f(x) + \langle g,s\rangle + \tfrac12\langle Hs,s\rangle + \tfrac{L}{6} r^3 = f(x) + m_x(s) - \tfrac{M-L}{6} r^3$. With $M \ge L$ and (F2),
$$
f(y_k) \le f(x_k) - \tfrac{M}{12} r_k^3 .
$$

Part 1 follows from (F3) and (0).

For part 2, the trial point satisfies, by (A1) and (F1),
$$
\|\nabla f(y_k)\| \le \|\nabla f(y_k) - g - Hs\| + \|g + Hs\| \le \tfrac{L}{2} r^2 + \tfrac{M}{2} r^2 = \tfrac{L+M}{2} r_k^2 ,
$$
$$
\nabla^2 f(y_k) \succeq H - L r I \succeq -\left(L + \tfrac{M}{2}\right) r_k\, I .
$$

Summing part 1 over $k = 0, \dots, K-1$ gives $\sum_k r_k^3 \le 12 (f(x_0) - f_{\mathrm{low}})/M$, so some $k < K$ has
$$
r_k^3 \le \frac{12\,(f(x_0) - f_{\mathrm{low}})}{M K} \le \left(\frac{2\varepsilon}{L+M}\right)^{3/2}
$$
for $K$ as stated. With $r_k \le \rho := \sqrt{2\varepsilon/(L+M)}$, the two bounds give $\|\nabla f(y_k)\| \le \varepsilon$ and $\lambda_{\min}(\nabla^2 f(y_k)) \ge -(L + \tfrac{M}{2})\rho \ge -(L+M)\rho = -\sqrt{2(L+M)\varepsilon}$. $\square$

### Remarks

- **Why not at $x_{k+1}$.** The gradient bound uses (F1) at the point $x_k + s_k$. The reflected point $x_k + s_k'$ does not satisfy (F1), so the bound does not transfer.
- **$M \ge L$ matters.** The benchmarks set $M$ as a multiple of a sampled estimate $\hat L$, which can be below $L$. Then (F3) can fail; this is what happened on the MLP at $M = 1\times\hat L$, where CN itself increased $f$.
- **Not covered.** `method="rarc"` (ARC with reflections) changes $M$ adaptively and computes $\rho$ from the accepted candidate; Theorem 1 does not apply to it as stated.

## Theorem 2 (first step from an exact strict saddle)

Assume (A3). Let $x$ satisfy $g = 0$, and let $\lambda_1 < 0$ be a **simple** smallest eigenvalue of $H$ with unit eigenvector $v$. Put
$$
\tau = \nabla^3 f(x)[v,v,v], \qquad D_0 = \frac{2|\lambda_1|^3}{M^2}, \qquad a = \frac{2|\tau|}{3M}.
$$
Then:

1. The CN steps are exactly $s = \pm r v$ with $r = 2|\lambda_1|/M$, and the reflected candidate is $-s$.
2. The decreases $D_\pm = f(x) - f(x \pm r v)$ satisfy
   $$
   D_\pm = D_0\left(1 \mp \frac{2\tau}{3M}\right) - R_\pm, \qquad |R_\pm| \le \frac{2}{3}\,\frac{L_3 \lambda_1^4}{M^4} .
   $$
3. Hence CN (with either fixed sign convention) gets $D_0(1 \pm a)$ up to $R$, and RCN gets $\max(D_+, D_-) \ge D_0(1 + a) - \tfrac23 L_3\lambda_1^4/M^4$. For a cubic polynomial ($L_3 = 0$) both are exact, and when CN takes the worse sign, $D_{\mathrm{RCN}} / D_{\mathrm{CN}} = (1+a)/(1-a)$.

### Proof

**Step 1.** With $g = 0$, $m_x(s) = \tfrac12\langle Hs, s\rangle + \tfrac{M}{6}\|s\|^3$. For fixed $\|s\| = t$, $\tfrac12\langle Hs,s\rangle \ge \tfrac12\lambda_1 t^2$, with equality only for $s = \pm t v$ because $\lambda_1$ is simple. Minimizing $\tfrac12\lambda_1 t^2 + \tfrac{M}{6} t^3$ over $t \ge 0$ gives $\lambda_1 t + \tfrac{M}{2} t^2 = 0$, i.e. $t = 2|\lambda_1|/M$. So the global minimizers are exactly $\pm r v$. The step has no component outside $v$, so reflecting the negative-curvature components flips only the $v$ component: $s' = -s$. The activation test passes because $\Delta_{\mathrm{ref}} = 0 \le \eta\,(-m_x(s))$ for any $\eta > 0$ (here $-m_x(s) > 0$ by (F2)).

**Step 2.** By (A3), the third-order Taylor expansion has remainder at most $\tfrac{L_3}{24}\|s\|^4$:
$$
f(x \pm r v) - f(x) = \tfrac12 \lambda_1 r^2 \pm \tfrac16 \tau r^3 + R'_\pm, \qquad |R'_\pm| \le \tfrac{L_3}{24} r^4 .
$$
Substitute $r = 2|\lambda_1|/M$:
$$
\tfrac12 \lambda_1 r^2 = -\frac{2|\lambda_1|^3}{M^2} = -D_0, \qquad \tfrac16 \tau r^3 = \frac{4\tau|\lambda_1|^3}{3M^3} = D_0 \cdot \frac{2\tau}{3M}, \qquad \tfrac{L_3}{24} r^4 = \frac{2}{3}\,\frac{L_3\lambda_1^4}{M^4}.
$$
This is part 2 with $R_\pm = R'_\pm$.

**Step 3.** One of $D_\pm$ equals $D_0(1+a) - R$ and the other $D_0(1-a) - R$. CN picks one sign by convention; RCN evaluates both and keeps the larger decrease. With $L_3 = 0$ the ratio in the worse case is $(1+a)/(1-a)$. $\square$

### Remarks

- **$a < 1$ under $M \ge L$.** By (A1), $|\tau| \le L$, so $M \ge L$ gives $a \le 2/3$ and the CN decrease $D_0(1-a)$ stays positive. If $a \ge 1$ (only possible with $M < \tfrac23|\tau|$), CN's wrong-sign step need not decrease $f$ at all.
- **Exact saddles only.** The theorem assumes $g = 0$ exactly. The benchmarks start $10^{-6}$ away from the saddle; there the small gradient selects CN's sign, and the theorem describes the limit. E6 checks the exact case numerically (`results/e6_theorem2.png`).
- **Multiple $\lambda_1$.** If $\lambda_1$ is not simple, the CN step can be any unit vector in its eigenspace scaled by $r$, and $\tau$ depends on that choice; part 1 then fails as stated.

## Reference

Y. Nesterov, B. T. Polyak. Cubic regularization of Newton method and its global performance. *Mathematical Programming* 108, 177–205, 2006.
