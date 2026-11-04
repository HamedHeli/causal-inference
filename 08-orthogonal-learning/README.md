# 08 Orthogonal learning

**Nov 3, Block 3: causal machine learning** · [back to the graph](../README.md)

**What is learned:** a target function (for example the CATE $\tau(x)$) by minimizing a loss that depends on nuisance
functions estimated in a first stage, where the loss is chosen so that first-stage errors affect the target only at
second order.

## Papers

| | Paper | Venue | Links |
| --- | --- | --- | --- |
| R1 | **[Foster & Syrgkanis 2023] Orthogonal Statistical Learning.** Dylan J. Foster, Vasilis Syrgkanis | Annals of Statistics 2023 | [arXiv:1901.09036](https://arxiv.org/abs/1901.09036) (read up to and including §3) |
| R2 | [Chernozhukov+ 2018] Double/debiased machine learning for treatment and structural parameters | Econometrics Journal 2018 | [arXiv:1608.00060](https://arxiv.org/abs/1608.00060) |
| R3 | [Nie & Wager 2021] Quasi-oracle estimation of heterogeneous treatment effects (the R-learner) | Biometrika 2021 | [arXiv:1712.04912](https://arxiv.org/abs/1712.04912) |

## Key ideas

- **Two-stage learning.** Stage 1 estimates nuisances $\hat g$ (outcome regressions, propensities). Stage 2 minimizes
  a population risk $L_D(\theta, \hat g)$ over the target class. The question is how the excess risk of $\hat\theta$
  depends on the error of $\hat g$.
- **Neyman orthogonality.** If the directional derivative of $\nabla_\theta L_D(\theta^*, g)$ with respect to $g$
  vanishes at $g_0$, nuisance errors enter the target's excess risk only through *second-order* terms (products or
  squares of nuisance errors). Then nuisances converging at $n^{-1/4}$, typical for flexible ML, are enough for the
  target to reach the rate it would have with known nuisances (an "oracle" rate).
- **Sample splitting / cross-fitting** keeps the stage-1 overfitting out of stage 2, so no complexity restriction
  (Donsker class) is needed on the nuisance class.
- **Examples.** The doubly robust score for the ATE (the special case "target = a constant"), the R-learner loss
  $\big((Y - \hat m(X)) - \tau(X)(T - \hat e(X))\big)^2$ for the CATE, and the DR-learner (regress the AIPW
  pseudo-outcome on $X$). The paper also treats IV and policy learning.
- **Partially linear DML** (Chernozhukov+): residualize $Y$ and $T$ on $X$ with ML, regress residuals: FWL with ML.
  Under heterogeneous effects it estimates the overlap-weighted effect $E[e(1-e)\tau]/E[e(1-e)]$, not the ATE.

## Code in this folder

- [`dml/`](dml/): orthogonality made visible. Perturb the true nuisances by $\delta$ and watch the plug-in and IPW
  estimates move like $\delta$ and AIPW like $\delta^2$ (and not at all when only one nuisance is wrong);
  regularization bias with real ML nuisances; why cross-fitting matters (57% vs 97% interval coverage with overfit
  nuisances); 95% coverage of DML intervals; the partially linear estimand; and the R-learner beating the T-learner
  on CATE error.

## Connections

- ← [03 Propensity scores](../03-propensity-scores/) and [07 Causal deep learning](../07-causal-deep-learning/): the
  doubly robust score, now explained.
- ← [02 Regression](../02-regression/): partially linear DML is FWL with ML first stages.
- ← [05 Meta-learners](../05-meta-learners/): the R- and DR-learners are orthogonal meta-learners.
- → [11 Reinforcement learning](../11-reinforcement-learning/): orthogonal off-policy learning of a policy.
- ↔ [12 One dataset, many estimators](../12-one-dataset-many-estimators/): AIPW, DML-PLR and the R-learner against the
  rest.
