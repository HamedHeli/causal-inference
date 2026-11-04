# Potential outcomes, by hand: drug or surgery?

A worked, inspectable version of the first three weeks of the course on a simulator where **both** potential outcomes
of every patient are known, so every quantity the lectures define can be computed exactly and compared with what an
analyst would estimate.

## The problem

A hospital treats patients with a drug ($T=0$) or surgery ($T=1$). Severity $S \sim U(0,2)$ drives everything:

```
S ~ Uniform(0, 2)
T ~ Bernoulli(sigmoid(3 (S - 1)))          sicker patients get surgery
Y(0) = 0.9 - 0.35 S + noise                sicker patients do worse
Y(1) = Y(0) + tau(S),  tau(S) = -0.15 + 0.25 S
```

The true ATE is 0.10: surgery helps on average. The pooled difference in means is about −0.07: surgery *looks*
harmful. Within every severity bucket the comparison flips back (Simpson's paradox).

## Notes: identification in five lines

1. Potential outcomes $Y_i(0), Y_i(1)$; we see $Y_i = T_i Y_i(1) + (1-T_i) Y_i(0)$.
2. $E[Y|T=1] - E[Y|T=0] = \text{ATE} + \text{baseline bias} + (1-\pi)\,\text{differential effect bias}$.
3. Ignorability $\{Y(0),Y(1)\} \perp\!\!\!\perp T$ kills both biases; randomization delivers it.
4. Conditional ignorability given $X$ plus positivity: $\text{ATE} = E\big[E[Y|T=1,X] - E[Y|T=0,X]\big]$.
5. Matching and stratification estimate this; neither can check the ignorability assumption.

## Start here

[`notebooks/potential_outcomes_explained.ipynb`](notebooks/potential_outcomes_explained.ipynb):

1. Simpson's paradox: pooled vs per-severity comparisons;
2. the table we want (both potential outcomes) vs the table we have;
3. the bias decomposition, checked to machine precision, as a waterfall;
4. 300 replications: observational vs randomized studies;
5. conditional randomization and the conditioning estimator;
6. matching, and which population (ATT, ATC, ATE) it estimates;
7. stratification: residual confounding with coarse strata, and positivity failures;
8. CLT vs Hoeffding confidence intervals, coverage and width;
9. **slider:** propensity slope (confounding strength) vs naive and stratified estimates.

The slider needs a running Jupyter kernel. The saved outputs (with static versions of the slider) render on GitHub.

## Layout

```
simpson/
├── src/
│   ├── hospital.py      the simulator (both potential outcomes known)
│   ├── estimators.py    naive, stratified, matching (and the rest of the course's estimators)
│   └── learners.py      NumPy nuisance models used by estimators.py
├── notebooks/
│   └── potential_outcomes_explained.ipynb
└── requirements.txt
```

`hospital.py`, `estimators.py` and `learners.py` are the same files as in
[`12-one-dataset-many-estimators/src/`](../../12-one-dataset-many-estimators/src/), copied so the folder is
self-contained.

## Run

```bash
cd 01-potential-outcomes/simpson
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/potential_outcomes_explained.ipynb
```
