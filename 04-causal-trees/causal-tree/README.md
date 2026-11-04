# Causal trees, from scratch: who benefits from surgery?

A NumPy implementation of **honest causal trees** (Athey & Imbens 2016) and a notebook that reproduces the paper's
central claim: leaf confidence intervals from an "adaptive" tree (same data to choose splits and estimate effects)
undercover, while honest ones have nominal coverage.

## The problem

A randomized trial of surgery vs drug on 1000–4000 patients with six covariates: severity and age (which drive the
effect) and four pure-noise variables. Find the subgroups with different effects, and say how sure we are about each.

## Notes: a causal tree in five lines

1. A tree partitions $X$; in leaf $\ell$, $\hat\tau_\ell = \bar Y_{\ell,1} - \bar Y_{\ell,0}$.
2. $\tau_i$ is never observed, so split on an unbiased estimate of $-\text{MSE}(\hat\tau)$:
   $\frac{1}{N^{tr}}\sum_i\hat\tau^2(X_i) - \big(\frac{1}{N^{tr}} + \frac{1}{N^{est}}\big)\sum_\ell\big(\frac{S^2_{\ell,1}}{p} + \frac{S^2_{\ell,0}}{1-p}\big)$.
3. **Honesty**: one half of the data builds the tree, the other half estimates each leaf.
4. Prune with a per-leaf cost chosen by cross-validation on the transformed outcome $Y^* = Y(T-p)/(p(1-p))$, since
   $E[Y^*|X] = \tau(X)$.
5. Observational data: weight units by inverse propensity inside each leaf.

## Start here

[`notebooks/causal_tree_explained.ipynb`](notebooks/causal_tree_explained.ipynb):

1. why a tree on $Y$ is not a tree on $\tau$, and the noisy transformed outcome;
2. an honest causal tree with CV pruning: the printed tree and its CATE map vs the truth;
3. the winner's curse: choosing a split inflates the reported difference;
4. **coverage of leaf intervals over 60 replications: adaptive ≈ 79%, honest ≈ 95%**;
5. transformed-outcome vs adaptive vs honest criteria as $n$ grows;
6. observational data with propensity-weighted leaves;
7. **slider:** minimum leaf size.

## Layout

```
causal-tree/
├── src/
│   ├── hospital.py   the simulator
│   └── tree.py       CausalTree (criteria "adaptive", "honest", "tot"), cv_alpha, leaf_effect
├── notebooks/
│   └── causal_tree_explained.ipynb
└── requirements.txt
```

## Run

```bash
cd 04-causal-trees/causal-tree
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/causal_tree_explained.ipynb
```
