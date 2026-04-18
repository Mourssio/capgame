# Mathematical Notes

Derivations for the formulations used in `capgame/`. The conventions here
match the project proposal (Section 5).

## 1. Static Cournot with linear demand

Inverse demand $P(Q) = a - bQ$ with $a, b > 0$. Firm $i$ has constant
marginal cost $c_i$ and capacity $\mathrm{cap}_i$, and chooses $q_i \in
[0, \mathrm{cap}_i]$.

Payoff:
$$
\pi_i(q_i; q_{-i}) = (a - b Q) q_i - c_i q_i.
$$

### Unconstrained best response

Taking $\partial \pi_i / \partial q_i = 0$:
$$
q_i = \frac{a - b Q_{-i} - c_i}{2 b}.
$$

Summing over $i$ and solving for $Q$:
$$
Q = \frac{N a - \sum_j c_j}{b (N + 1)},
\qquad
q_i = \frac{a - N c_i + \sum_{j \neq i} c_j}{b (N + 1)}.
$$

### KKT / MCP form with capacities

$$
0 \le q_i \quad\perp\quad c_i - P(Q) + b q_i + \mu_i \ge 0,
\qquad
0 \le \mu_i \quad\perp\quad \mathrm{cap}_i - q_i \ge 0.
$$

Our `solve_constrained` solves this by Gauss-Seidel best-response iteration:
$$
q_i^{(k+1)} = \operatorname{clip}\!\left(\frac{a - b Q_{-i}^{(k)} - c_i}{2 b},\ 0,\ \mathrm{cap}_i\right).
$$
Convergence follows from the contraction property of the best-response map
on the compact strategy set under strict concavity in own action.

## 2. Nash-Cournot consumer surplus

Linear demand $\Rightarrow$ consumer surplus is a triangle:
$$
\mathrm{CS}(Q) = \tfrac12 b Q^2.
$$

## 3. Reliability options identities

Let $\pi_0$ be the option premium, $K$ the strike, and $x_i$ the committed
capacity. Per-period net profit (energy + option):
$$
\Pi_i = (P(Q) - c_i) q_i + \pi_0 x_i - \max(P(Q) - K, 0) \cdot x_i.
$$

**Limits used in tests.**

- $K \to \infty$: refund term vanishes, mechanism = capacity payment of
  $\pi_0$. Setting $\pi_0 = 0$ recovers energy-only.
- $K = 0$: refund $= P(Q)\,x_i$, so $\Pi_i = (P(Q)-c_i)q_i + (\pi_0 - P(Q))x_i$.

## 4. Forward capacity auction clearing

Procurement curve: $D(\rho) = \max(0, Q^\star - k \rho)$.

Offers $(x_i, r_i)$ are sorted ascending in $r_i$. Accept quantity up to
$D(r_i) - \sum_{j: r_j < r_i}\hat x_j$ until exhausted. Clearing price is
the marginal accepted reservation price, capped at the price that clears
the unfilled residual demand.

## 5. Forced-outage capacity distribution

Unit $i$ has capacity $c_i$ with availability Bernoulli$(1 - f_i)$.
Available capacity $C = \sum_i B_i c_i$. The distribution of $C$ is the
convolution
$$
\Pr[C = x] = \sum_{S : \sum_{i \in S} c_i = x} \prod_{i \in S}(1 - f_i)\prod_{i \notin S} f_i.
$$
For $N \le 20$ we compute this exactly; otherwise Monte Carlo.

## 6. LOLE and EUE

Demand state $s$ has peak load $d_s$ and probability $p_s$.

$$
\mathrm{LOLE} = \sum_s p_s \Pr[C < d_s],
$$
$$
\mathrm{EUE}  = \sum_s p_s \operatorname{E}[\max(d_s - C, 0)].
$$

Both are expressed in "per period" units; multiply by `periods_per_unit`
(e.g. 8760 for hours per year) for reporting.

## 7. Backward induction (MPE skeleton)

Per-firm Bellman equation:
$$
V_i(t, s, x) = \max_{\Delta x_i \ge 0}\left\{
  \pi_i(t, s, x; \Delta x) + \beta\, \mathbb E\!\left[V_i(t+1, s', x') \mid s\right]
\right\},
$$
with transition
$$
x' = (1 - \delta) x + \Delta x_{t - L}.
$$

The MVP uses $L = 0$ and a finite discretization of both $x$ and
$\Delta x$. Future versions will replace grid search with projected
gradient iteration on value-function approximations.
