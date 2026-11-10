# Offline RL is causal inference: learning a treatment policy from logged data

Off-policy evaluation and offline policy learning, written as the causal-inference problems they are, on the course's
drug-vs-surgery example: first as a single decision (a contextual bandit), then as a five-visit treatment regime.

## The problem

A hospital has logs of what its doctors did and how patients fared. It wants to (1) estimate how well a *new* treatment
policy would do, and (2) find a good policy, without experimenting on patients. The value of a policy is the mean
potential outcome of a dynamic treatment regime, $V(\pi) = E[Y(\pi)]$, and the logs are an observational study
collected under the doctors' behavior policy.

The multi-visit simulator ([`src/regime.py`](src/regime.py)): severity $s \in \{0..4\}$, drug or surgery at each of 5
visits, reward = health after the visit. Surgery helps severe patients and harms mild ones (as in the one-step
hospital). A hidden per-visit **frailty** reduces the benefit of surgery; with `confounding > 0` the doctors see it and
avoid operating on frail patients, which the logs never record. Tabular, so every policy's true value is exact.

## Notes: the dictionary in five lines

1. Behavior policy $\pi_b(a|s)$ = propensity score; coverage = positivity; no hidden confounders of action and outcome
   at every step = sequential ignorability.
2. Importance sampling = IPW with a product of ratios; weighted IS = Hajek; per-decision IS uses the causal fact that
   later actions cannot affect earlier rewards.
3. Model-based evaluation / FQE = g-computation (Robins' g-formula).
4. Doubly robust OPE (Jiang & Li 2016) = AIPW, applied recursively backwards in time.
5. Policy learning maximizes an estimated value; the max finds where the estimate is wrong (positivity → pessimism) and
   over-states the winner (optimizer's curse → honest splitting).

## Start here

[`notebooks/offline_rl_explained.ipynb`](notebooks/offline_rl_explained.ipynb), under a minute on a laptop:

1. one decision: evaluate and learn treatment rules on the hospital with doubly robust scores (the contextual bandit;
   a learned severity threshold matches the oracle CATE rule);
2. the five-visit regime, the doctors' policy and the best of the 32 deterministic policies;
3. off-policy evaluation of the best policy: IS, WIS, PDIS, WPDIS, model-based, DR (bias, sd, RMSE), and the curse of
   horizon;
4. positivity becomes coverage: evaluating "always operate" when the doctors almost never operate on mild patients;
5. offline policy learning: behavior cloning vs fitted Q iteration vs pessimistic FQI as data grow;
6. the optimizer's curse: reported value of the selected policy +0.25 too high on the same data, unbiased with an
   honest split;
7. hidden confounding: every severity-only estimator is biased, by up to 0.25, and more data does not help.

## Layout

```
offline-rl/
├── src/
│   ├── regime.py            multi-visit treatment regime: simulator, behavior policy, exact policy values
│   ├── ope.py               IS, WIS, PDIS, WPDIS, model-based (g-formula), doubly robust
│   ├── policy_learning.py   behavior cloning, (pessimistic) fitted Q iteration, OPE-argmax with honest splitting
│   ├── hospital.py          the one-step hospital (copied from 12)
│   ├── estimators.py        cross-fitted nuisances, R-learner (copied from 12)
│   └── learners.py          NumPy nuisance models (copied from 12)
├── notebooks/
│   └── offline_rl_explained.ipynb
└── requirements.txt
```

## Run

```bash
cd 11-reinforcement-learning/offline-rl
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/offline_rl_explained.ipynb
```
