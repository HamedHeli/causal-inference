# 12 One dataset, many estimators

**Cross-topic** · [back to the graph](../README.md)

Every topic of the course is a different answer to the same question: *what would happen to these patients under
surgery instead of the drug?* This folder takes **one** simulated hospital and runs each estimator from topics 01–08
on it, so the differences you see are about the methods and not about the data.

## The data: the hospital, made hard

The course's running example ([`src/hospital.py`](src/hospital.py)) with `nonlinear=True, n_noise=3`:

- covariates: severity $S \sim U(0,2)$, age $\sim U(0,1)$, and three standard-normal noise variables;
- propensity $e(x) = \sigma\big(3(S-1) + \cos(3\,\text{age}) - 0.3\big)$: sicker patients get surgery, and age matters
  too, so **both** confound;
- baseline $E[Y(0)|x] = 0.9 - 0.35 S - 0.15\sin(2.5 S) - 0.1(\text{age} - 0.5)^2$;
- effect $\tau(x) = -0.15 + 0.25 S + 0.1\tanh(4(\text{age}-0.5))\,\mathbb{1}[S > 1]$: surgery harms mild patients,
  helps severe ones, and age modifies the effect for severe patients;
- true ATE $\approx 0.10$, ATT $\approx 0.18$, ATC $\approx 0.03$. 2000 patients per dataset.

## Start here

[`notebooks/one_dataset_many_estimators.ipynb`](notebooks/one_dataset_many_estimators.ipynb), a few minutes on a laptop CPU.

| § | Content |
| --- | --- |
| 1 | the data: propensity, true CATE, overlap |
| 2 | regression's hidden weights: OLS = FWL = $P(x)\text{Var}(T\mid x)$-weighted average, made visible |
| 3 | all 14 ATE estimators on one dataset, with intervals and timings |
| 4 | the same over 40 datasets: bias, RMSE, coverage |
| 5 | CATE: S-, T-, X-, R-, DR-learners, interacted OLS and the causal tree, scored on 20,000 new patients |
| 6 | linear vs nonlinear world |

| Estimator | Topic | Module |
| --- | --- | --- |
| naive difference, stratification, nearest-neighbour matching | [01](../01-potential-outcomes/) | [`estimators.py`](src/estimators.py) |
| OLS `Y ~ T + X`, FWL, interacted OLS / g-computation | [02](../02-regression/) | [`estimators.py`](src/estimators.py) |
| IPW (Horvitz–Thompson, Hajek), IP-weighted regression, AIPW | [03](../03-propensity-scores/) | [`estimators.py`](src/estimators.py) |
| honest causal tree with propensity-weighted leaves | [04](../04-causal-trees/) | [`tree.py`](src/tree.py) |
| S-, T-, X-learners | [05](../05-meta-learners/) | [`estimators.py`](src/estimators.py) |
| cross-fitted AIPW (DML), partially linear DML, R-learner, DR-learner | [08](../08-orthogonal-learning/) | [`estimators.py`](src/estimators.py) |
| kernel ridge (random Fourier features) and logistic regression as nuisance models | – | [`learners.py`](src/learners.py) |

## Results

ATE error over 40 datasets of 2000 patients (true ATE ≈ 0.10). Coverage is of the nominal 95% interval, where the
estimator provides one.

| Estimator | bias | RMSE | CI coverage |
| --- | --- | --- | --- |
| naive difference | −0.046 | 0.047 | 0% |
| stratified, 10 severity bins | −0.007 | 0.009 | 85% |
| 1-NN matching on X | −0.014 | 0.015 | 25% |
| OLS `Y ~ T + X` | +0.037 | 0.038 | 0% |
| interacted OLS (g-computation) | +0.013 | 0.014 | – |
| IPW, logistic propensity | −0.007 | 0.026 | 100% |
| IPW (Hajek), flexible propensity | +0.002 | 0.008 | – |
| **AIPW / DML (cross-fitted)** | **−0.000** | **0.008** | **100%** |
| S-learner | +0.002 | 0.005 | – |
| T-learner | −0.004 | 0.006 | – |
| X-learner | +0.002 | 0.006 | – |
| DML partially linear | +0.025 | 0.026 | 8% |
| R-learner | +0.009 | 0.010 | – |
| causal tree (honest, IPW leaves) | −0.003 | 0.011 | – |

CATE error (RMSE of $\hat\tau(x)$ on new patients, 10 datasets; predicting the ATE everywhere scores 0.153):
X-learner 0.039, R-learner 0.038, T-learner 0.048, DR-learner 0.051, S-learner 0.052, causal tree 0.060,
interacted OLS 0.083.

What the numbers say:

- Confounding (naive), misspecification (OLS) and the wrong weighting (OLS, partially linear DML) are each worth a
  bias of 0.02–0.05, a quarter to a half of the effect.
- With flexible nuisances, outcome-model methods (S/T/X-learners) and AIPW are all close; **AIPW is the one with an
  interval that means something**, which is the practical case for orthogonality.
- Intervals of estimators that ignore their own selection or modelling step (matching, stratification with residual
  confounding) undercover badly.
- For the CATE, the methods that target $\tau$ directly (X-learner, R-learner) beat the plug-ins (S, T) and the
  piecewise-constant tree.

These are numbers from one simulator and one seed range; the notebook regenerates them.

## Layout

```
12-one-dataset-many-estimators/
├── src/
│   ├── hospital.py      the simulator (both potential outcomes known)
│   ├── learners.py      NumPy kernel ridge, logistic regression, k-fold splits
│   ├── estimators.py    every ATE / CATE estimator of topics 01-08
│   └── tree.py          honest causal tree (Athey & Imbens)
├── notebooks/
│   └── one_dataset_many_estimators.ipynb
└── requirements.txt
```

## Run

```bash
cd 12-one-dataset-many-estimators
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/one_dataset_many_estimators.ipynb
```
