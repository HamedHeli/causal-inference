# 04 Causal trees

**Sep 29, Block 3: causal machine learning** · [back to the graph](../README.md)

**What is learned:** a partition of covariate space and one treatment effect per leaf, $\hat\tau(x) = \hat\tau_{\ell(x)}$,
with confidence intervals that stay valid although the partition was chosen from the data.

## Paper

| | Paper | Venue | Links |
| --- | --- | --- | --- |
| R1 | **[Athey & Imbens 2016] Recursive Partitioning for Heterogeneous Causal Effects.** Susan Athey, Guido W. Imbens | PNAS 2016 | [arXiv:1504.01132](https://arxiv.org/abs/1504.01132) |

## Key ideas

- **The target is never observed.** CART splits to reduce $\sum_i (Y_i - \hat\mu(X_i))^2$; for effects we would need
  $\sum_i(\tau_i - \hat\tau(X_i))^2$, and $\tau_i$ is never seen (the "fundamental problem of causal inference"), which
  also breaks ordinary cross-validation.
- **Transformed outcome.** In a trial, $Y^* = Y\,(T - p)/(p(1-p))$ has $E[Y^*|X] = \tau(X)$, so a regression tree on
  $Y^*$ estimates the CATE. It is unbiased but very noisy, and it is the trick that makes cross-validation possible:
  $\text{mean}(Y^* - \hat\tau)^2$ is the MSE of $\hat\tau$ up to a constant.
- **Honesty.** Split the sample: one half builds the partition, the other estimates the leaf effects. The leaf
  estimates are then unbiased given the partition and the usual difference-in-means intervals are valid. "Adaptive"
  trees (same data for both) produce over-extreme leaves and undercovering intervals, the price of searching.
- **An honest splitting criterion.** Anticipating honest estimation, an unbiased estimate of minus the expected MSE of
  the effects is
  $\frac{1}{N^{tr}}\sum_i \hat\tau^2(X_i) - \big(\frac{1}{N^{tr}} + \frac{1}{N^{est}}\big)\sum_\ell \big(\frac{S^2_{\ell,1}}{p} + \frac{S^2_{\ell,0}}{1-p}\big)$:
  reward splits that separate effects, penalize leaves whose effect estimate will be noisy.
- **Observational data**: reweight by inverse propensity inside each leaf. And the idea scales: averaging many honest
  trees gives causal forests (Wager & Athey 2018).

## Code in this folder

- [`causal-tree/`](causal-tree/): an honest causal tree in NumPy (adaptive, honest and transformed-outcome criteria,
  cost-complexity penalty chosen by cross-validation on $Y^*$, propensity-weighted leaves). The notebook reproduces the
  paper's main finding on the hospital: adaptive leaf intervals cover about 80%, honest ones about 95%.

## Connections

- ← [01 Potential outcomes](../01-potential-outcomes/): a leaf effect is a stratified difference in means, with the
  strata chosen by the data.
- → [05 Meta-learners](../05-meta-learners/): the meta-learner paper uses honest trees and forests as base learners,
  and the transformed outcome is the simplest "pseudo-outcome" learner.
- → [07 Causal deep learning](../07-causal-deep-learning/): the lecture lists optimization bias and the need for honest
  estimation among the reasons naive S/T-learners can fail.
- ↔ [12 One dataset, many estimators](../12-one-dataset-many-estimators/): the tree's CATE against the meta-learners and
  the R-learner on the same data.
