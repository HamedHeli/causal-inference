# 05 Meta-learners

**Oct 1, Block 3: causal machine learning** · [back to the graph](../README.md)

**What is learned:** the CATE $\tau(x)$, assembled from off-the-shelf supervised learners ("base learners") that each
solve an ordinary regression problem.

## Paper

| | Paper | Venue | Links |
| --- | --- | --- | --- |
| R1 | **[Künzel+ 2019] Metalearners for Estimating Heterogeneous Treatment Effects using Machine Learning.** Sören R. Künzel, Jasjeet S. Sekhon, Peter J. Bickel, Bin Yu | PNAS 2019 | [arXiv:1706.03461](https://arxiv.org/abs/1706.03461) |

## Key ideas

- **S-learner** ("single"): fit $\hat\mu(x, t)$ with $T$ as one more feature; $\hat\tau(x) = \hat\mu(x,1) - \hat\mu(x,0)$.
  Regularization can shrink the treatment's role to zero, biasing $\hat\tau$ toward 0.
- **T-learner** ("two"): fit $\hat\mu_1$ on the treated and $\hat\mu_0$ on the controls; $\hat\tau = \hat\mu_1 - \hat\mu_0$.
  Each model is regularized for its own outcome surface, not for their difference, so a simple $\tau$ between two
  complicated surfaces is estimated poorly, especially when one arm is small.
- **X-learner.** (1) Fit $\hat\mu_0, \hat\mu_1$. (2) Impute individual effects with the *other* arm's model:
  $D_i^1 = Y_i - \hat\mu_0(X_i)$ for the treated, $D_i^0 = \hat\mu_1(X_i) - Y_i$ for the controls. (3) Regress them to
  get $\hat\tau_1, \hat\tau_0$. (4) Blend: $\hat\tau(x) = g(x)\hat\tau_0(x) + (1-g(x))\hat\tau_1(x)$, typically
  $g = \hat e$. The model learned from the large arm does the imputing for the small arm, which is where the X-learner
  wins (unbalanced designs, smooth $\tau$).
- **Convergence rates.** The paper shows the X-learner can reach the rate of the CATE's own complexity when one arm is
  much larger, while the T-learner is stuck at the rate of the harder response surface.
- **A failure mode they share.** All three are plug-ins: errors in the outcome models pass to $\hat\tau$ at first
  order. The orthogonal R- and DR-learners of [08](../08-orthogonal-learning/) fix this.

## Code

No folder of its own. `s_learner`, `t_learner`, `x_learner` are in
[`12-one-dataset-many-estimators/src/estimators.py`](../12-one-dataset-many-estimators/src/estimators.py) (kernel
ridge base learners), compared in [12](../12-one-dataset-many-estimators/) §5 on CATE error, next to the R-learner,
the DR-learner and the causal tree.

## Connections

- ← [02 Regression](../02-regression/): "regression as counterfactual prediction" with flexible models.
- ← [03 Propensity scores](../03-propensity-scores/): the X-learner's blending weight is the propensity.
- ← [04 Causal trees](../04-causal-trees/): honest trees and forests as base learners.
- → [08 Orthogonal learning](../08-orthogonal-learning/): the R-learner and DR-learner are meta-learners with
  orthogonal losses.
- → [07 Causal deep learning](../07-causal-deep-learning/): Dragonnet is a neural S/T-learner with a shared
  representation and a targeting step.
