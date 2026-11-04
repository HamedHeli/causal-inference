# Orthogonal learning, by hand: why the doubly robust score works

A notebook that *measures* Neyman orthogonality instead of asserting it, and shows what cross-fitting buys, on the
nonlinear hospital with ML nuisance models written in NumPy.

## The problem

Estimate the ATE (and the CATE) of surgery when confounding by severity and age is nonlinear, so the outcome and
propensity must be learned with flexible, regularized models. How much of their error leaks into the estimate?

## Notes: double/debiased ML in five lines

1. Nuisances: $\mu_t(x) = E[Y|T=t,X=x]$, $e(x) = P(T=1|X=x)$, fitted by ML.
2. Score: $\phi = \hat\mu_1 - \hat\mu_0 + \frac{T(Y - \hat\mu_1)}{\hat e} - \frac{(1-T)(Y - \hat\mu_0)}{1 - \hat e}$.
3. Orthogonal: $\partial_\delta E[\phi]$ at the true nuisances is 0, so the bias is a *product* of nuisance errors.
4. Cross-fit: predict each unit's nuisances with models trained on the other folds.
5. Then $\bar\phi \pm 1.96\,\text{sd}(\phi)/\sqrt n$ is a valid 95% interval when nuisances converge at $n^{-1/4}$.

## Start here

[`notebooks/orthogonal_learning_explained.ipynb`](notebooks/orthogonal_learning_explained.ipynb):

1. perturb the true nuisances by $\delta$: plug-in and IPW errors grow like $\delta$, AIPW like $\delta^2$, and AIPW
   does not move when only one nuisance is wrong (log-log plot);
2. real ML nuisances at increasing regularization: who drifts, and when even AIPW fails (both nuisances bad);
3. overfit nuisances without cross-fitting: interval coverage 57% vs 97% with cross-fitting;
4. 60 DML intervals and their coverage;
5. the partially linear model estimates the overlap-weighted effect, not the ATE;
6. the R-learner's orthogonal CATE loss vs the T-learner and DR-learner as $n$ grows.

## Layout

```
dml/
├── src/
│   ├── hospital.py      the simulator (and its true nuisance functions)
│   ├── learners.py      NumPy kernel ridge (random Fourier features) and logistic regression
│   └── estimators.py    cross_fit_nuisances, ipw, aipw, dml_plr, r_learner, dr_learner, ...
├── notebooks/
│   └── orthogonal_learning_explained.ipynb
└── requirements.txt
```

## Run

```bash
cd 08-orthogonal-learning/dml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/orthogonal_learning_explained.ipynb
```
