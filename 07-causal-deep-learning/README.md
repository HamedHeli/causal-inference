# 07 Causal deep learning

**Oct 27–29 (and Nov 17), Block 3: causal machine learning** · [back to the graph](../README.md)

**What is learned:** depends on the role deep learning plays. Either a *nuisance* (outcome or propensity model) inside a
classical estimator, a flexible *CATE or potential-outcome model*, a *latent confounder* or a *causal representation*,
or the *graph itself*.

## Readings (from the lecture)

| | Paper | Venue | Links |
| --- | --- | --- | --- |
| R1 | **[Chernozhukov+ 2018] Double/debiased machine learning for treatment and structural parameters** | Econometrics Journal 2018 | [arXiv:1608.00060](https://arxiv.org/abs/1608.00060) |
| R2 | **[van der Laan & Rose 2011] Targeted Learning** | Springer 2011 | book |
| R3 | **[Shi, Blei & Veitch 2019] Adapting neural networks for the estimation of treatment effects** (Dragonnet) | NeurIPS 2019 | [arXiv:1906.02120](https://arxiv.org/abs/1906.02120) |
| R4 | **[Wang & Blei 2019] The blessings of multiple causes** (the deconfounder) and [D'Amour 2019] counterexamples | JASA 2019 | [arXiv:1805.06826](https://arxiv.org/abs/1805.06826), [arXiv:1902.10286](https://arxiv.org/abs/1902.10286) |
| R5 | **[Wang & Jordan 2024] Desiderata for representation learning: a causal perspective** | JMLR 2024 | [arXiv:2109.03795](https://arxiv.org/abs/2109.03795) |
| R6 | **[Veitch & Zaveri 2020] Sense and sensitivity analysis** (Austen plots) | NeurIPS 2020 | [arXiv:2003.01747](https://arxiv.org/abs/2003.01747) |
| R7 | **[Zheng+ 2018] DAGs with NO TEARS** | NeurIPS 2018 | [arXiv:1803.01422](https://arxiv.org/abs/1803.01422) |
| R8 | [Schölkopf+ 2021] Toward causal representation learning · [Glymour, Zhang & Spirtes 2019] Review of causal discovery · [Spirtes & Zhang 2016] | Proc. IEEE 2021 · Front. Genetics 2019 | |

## Key ideas

The lecture's map has four parts.

1. **Deep learning to improve causal estimators.**
   - *Nuisance models.* With $\mu(x, t) = E[Y|X,T]$ and $\pi(x) = P(T=1|X)$ fitted by ML, the doubly robust score
     $\Phi = \hat\mu(X,1) - \hat\mu(X,0) + \frac{T - \hat\pi}{\hat\pi(1-\hat\pi)}(Y - \hat\mu(X,T))$ has mean equal to the
     ATE if either nuisance is right. **DML**: fit nuisances, average $\Phi$ on held-out data (cross-fitting); the score
     is Neyman orthogonal, so nuisance errors enter only at second order.
   - *TMLE* debiases $\hat\mu$ differently: regress its residuals on the "clever covariate"
     $H = \frac{T - \hat\pi}{\hat\pi(1 - \hat\pi)}$, add $\hat\epsilon H$ to $\hat\mu$, then plug in. *Dragonnet* learns
     $\hat Q(t, x)$ and $\hat g(x)$ with a shared representation and adds the targeting term to the loss, end to end.
   - *Extrapolation.* Predict the missing counterfactuals (X-learner style, time-series counterfactuals), or predict at
     a threshold ([RDD](../10-instruments-and-discontinuities/)).
2. **Flexible CATE models.** An S/T-learner converges to the right thing in the limit but suffers extrapolation,
   regularization and optimization bias. Better: orthogonal losses ([08](../08-orthogonal-learning/)), cocycles
   ([09](../09-cocycles/)), and instruments as a training signal ([10](../10-instruments-and-discontinuities/)).
3. **Expanding causal inference.**
   - *Deconfounding.* With many causes, fit a factor model $Z_i \to A_{i,1..m}$ and adjust for the inferred
     $\hat z_i$. Useful but with positivity problems (D'Amour 2019).
   - *Causal representation learning.* Features that are non-spurious, efficient and independently intervenable.
     Wang & Jordan maximize a lower bound on each feature's PNS (supervised) or regularize toward independent support
     (unsupervised, disentanglement).
   - *Sensitivity analysis.* How strong would an unobserved confounder need to be to change the conclusion? Veitch &
     Zaveri's Austen plots work with any outcome and propensity models: the bias is closed-form in the confounder's
     influence on treatment ($\alpha$) and outcome (partial $R^2$), compared with the strength of observed covariates.
4. **Causal discovery.** Impossible in general; possible under assumptions. Conditional-independence tests recover
   only a Markov equivalence class and are fragile; continuous optimization (NOTEARS: least squares plus a smooth
   acyclicity constraint on $W$); and asymmetries of noise in non-Gaussian additive noise models to find the
   direction.

## Code

No folder of its own. The DML / AIPW part is worked out in [08 `dml/`](../08-orthogonal-learning/dml/) (orthogonality
measured, cross-fitting, coverage); RDD-as-prediction in [10 `iv-rdd/`](../10-instruments-and-discontinuities/iv-rdd/).

## Connections

- ← [03 Propensity scores](../03-propensity-scores/): the doubly robust score is the same object.
- → [08 Orthogonal learning](../08-orthogonal-learning/): from an orthogonal score for a number to an orthogonal loss
  for a function.
- ← [06 Structural causal models](../06-structural-causal-models/): PNS, d-separation, and Markov equivalence classes.
- → [09 Cocycles](../09-cocycles/) and [10 Instruments](../10-instruments-and-discontinuities/): the other two flexible
  CATE approaches of the lecture.
