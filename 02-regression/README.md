# 02 Regression

**Sep 17–22, Block 2: core estimation approaches** · [back to the graph](../README.md)

**What is learned:** a model of $E[Y \mid T, X]$, used to extrapolate where stratification has no data. With a single
coefficient on $T$ it is a *variance-weighted* average of conditional effects, not the ATE.

## Readings

| | Reading | Notes |
| --- | --- | --- |
| L5 | Linear regression for causal effects | FWL theorem, omitted-variable bias, three interpretations |
| L6 | Linear regression and propensity scores | regression as counterfactual prediction, confidence intervals |

## Key ideas

- **Regression on the treatment alone** gives $\beta_T = E[Y|T=1] - E[Y|T=0]$, the naive difference. If the truth
  is $Y = \delta_0 + \delta_T T + \delta_S S + \varepsilon$ and $S$ is omitted,
  $\beta_T = \delta_T + \delta_S\,\text{Cov}(S,T)/\text{Var}(T)$: **omitted-variable (confounding) bias**.
- **Frisch–Waugh–Lovell.** In $Y \sim T + X$, $\beta_T = \text{Cov}(Y, \tilde T)/\text{Var}(\tilde T)$ where $\tilde T$
  is the residual of $T$ regressed on $X$. Partial out $X$ from $T$ (and optionally from $Y$), then run a bivariate
  regression. This two-step view is the template for [DML](../08-orthogonal-learning/).
- **Regression as conditioning, with the wrong weights.** For categorical $X$,
  $\beta_T = \sum_x w_x\,(E[Y|T=1,x] - E[Y|T=0,x])$ with $w_x \propto P(x)\,\text{Var}(T|x)$. Strata where treatment is
  nearly deterministic barely count, strata without both arms count zero (positivity), and under **heterogeneous
  effects** the result is not the ATE. With continuous $X$ the regression also imposes a linear propensity and
  extrapolates outside the overlap region.
- **Regression as counterfactual prediction.** $\beta_0 + \beta_S S$ predicts $Y(0)$ from the controls and
  $\beta_T$ is a (very simple) model of the effect. Richer models of $Y(t)$ given $X$ are the
  [meta-learners](../05-meta-learners/). Adding $T \times X$ interactions and averaging the predicted difference over
  the sample (g-computation) fixes the weighting.
- **Inference vs prediction.** Asymptotic normality of $\hat\beta_T$ gives intervals; they are about the coefficient,
  which is only a causal quantity under the assumptions above.

## Code

No folder of its own. [12 One dataset, many estimators](../12-one-dataset-many-estimators/) §2 makes the implicit
weights $P(x)\text{Var}(T|x)$ visible on the hospital and checks OLS = FWL = variance-weighted average to machine
precision; §3–4 compare `Y ~ T + X` and the interacted regression with every other estimator.

## Connections

- ← [01 Potential outcomes](../01-potential-outcomes/): a model-based version of the conditioning estimator, with the
  same two assumptions (ignorability, positivity).
- → [03 Propensity scores](../03-propensity-scores/): weighted regression with weights $1/P(T=t|X)$ estimates the ATE.
- → [05 Meta-learners](../05-meta-learners/): "regression as counterfactual prediction", with ML models.
- → [08 Orthogonal learning](../08-orthogonal-learning/): FWL with ML first stages is the partially linear DML
  estimator, and it inherits the variance weighting.
