# 03 Propensity scores

**Sep 22–24, Block 2: core estimation approaches** · [back to the graph](../README.md)

**What is learned:** the propensity score $e(x) = P(T=1 \mid X=x)$, a one-dimensional summary of $X$ that is enough to
remove confounding; and, for doubly robust estimators, the outcome model too.

## Readings

| | Reading | Notes |
| --- | --- | --- |
| L6–L7 | Propensity weighting | balancing property, IPW, IP-weighted regression, doubly robust estimators |
| R | Hernán & Robins, *What If*, §12.2 | IP weighting |
| R | Morgan & Winship, *Counterfactuals and Causal Inference* (2014), §7.3 | doubly robust weighted regression |

## Key ideas

- **The balancing property.** If $\{Y(0), Y(1)\} \perp\!\!\!\perp T \mid X$ then
  $\{Y(0), Y(1)\} \perp\!\!\!\perp T \mid e(X)$. Proof: $P(T=1 \mid Y(t), e(X)) = E[E[T|X] \mid Y(t), e(X)] = e(X)$.
  So matching, stratification and regression can all use the scalar $e(X)$ instead of $X$.
- **Inverse propensity weighting.** $E\big[\mathbb{1}_{T=1} Y / e(X)\big] = E[Y(1)]$, hence
  $\text{ATE} = E\big[TY/e(X) - (1-T)Y/(1-e(X))\big]$. Each treated unit stands for $1/e(x)$ units with the same $x$.
  The same estimator appears in bandits and off-policy RL ([11](../11-reinforcement-learning/)).
- **IP-weighted regression.** Regress $Y$ on $T$ with weights $1/P(T=t|X)$: $\beta_0 = E[Y(0)]$,
  $\beta_0 + \beta_T = E[Y(1)]$ (this is the normalized, or Hajek, version of IPW).
- **Positivity, quantitatively.** Weights blow up where $e(x)$ is near 0 or 1, so the variance of IPW is driven by poor
  overlap; in practice propensities are clipped or trimmed.
- **Doubly robust (AIPW).**
  $\hat\psi = \frac1n\sum_i \hat\mu_1(X_i) - \hat\mu_0(X_i) + \frac{T_i(Y_i - \hat\mu_1(X_i))}{\hat e(X_i)} - \frac{(1-T_i)(Y_i - \hat\mu_0(X_i))}{1 - \hat e(X_i)}$
  is consistent if *either* the outcome models *or* the propensity is correct. Its deeper property, Neyman
  orthogonality, is the subject of [08](../08-orthogonal-learning/).

## Code

No folder of its own. The estimators are in [`12-one-dataset-many-estimators/src/estimators.py`](../12-one-dataset-many-estimators/src/estimators.py)
(`ipw`, `ip_weighted_regression`, `aipw`), compared on the hospital in
[12](../12-one-dataset-many-estimators/). [08 `dml/`](../08-orthogonal-learning/dml/) shows numerically why AIPW
tolerates errors in the nuisance models that break IPW and the outcome plug-in.

## Connections

- ← [02 Regression](../02-regression/): the regression lecture ends where this one starts, with weighted least squares.
- → [05 Meta-learners](../05-meta-learners/): the X-learner blends its two CATE models with $e(x)$.
- → [04 Causal trees](../04-causal-trees/): propensity weighting inside the leaves adapts honest trees to
  observational data.
- → [07 Causal deep learning](../07-causal-deep-learning/) and [08 Orthogonal learning](../08-orthogonal-learning/):
  AIPW with cross-fitted ML nuisances is double/debiased ML; TMLE and Dragonnet reach the same score by targeting.
- → [11 Reinforcement learning](../11-reinforcement-learning/): the doubly robust policy value.
