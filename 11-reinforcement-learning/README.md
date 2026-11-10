# 11 Reinforcement learning and offline RL

**Nov 24, Block 3: causal machine learning** · [back to the graph](../README.md)

**What is learned:** a *policy* $\pi(a \mid x)$ that picks treatments, evaluated and improved from logged data with the
same doubly robust corrections used for the ATE. Learning it from a fixed log, without new experiments, is **offline
RL**, and its value is identified exactly when a causal effect is: no unobserved confounders and positivity, now at
every decision.

## Readings (from the lecture)

| | Paper | Venue | Links |
| --- | --- | --- | --- |
| R1 | **[Jiang & Li 2016] Doubly robust off-policy value evaluation for reinforcement learning** | ICML 2016 | [arXiv:1511.03722](https://arxiv.org/abs/1511.03722) |
| R2 | **[Chernozhukov, Newey, Singh & Syrgkanis 2022] Automatic debiased machine learning for dynamic treatment effects and general nested functionals** | arXiv 2022 | [arXiv:2203.13887](https://arxiv.org/abs/2203.13887) |
| R3 | **[Schulte & Poupart 2025] When should reinforcement learning use causal reasoning?** | TMLR 2025 | |

## Further reading: offline RL as causal inference

| | Paper | Venue | Links |
| --- | --- | --- | --- |
| F1 | [Robins 1986] A new approach to causal inference in mortality studies with a sustained exposure period (the g-formula) | Math. Modelling 1986 | |
| F2 | [Murphy 2003] Optimal dynamic treatment regimes | JRSS-B 2003 | |
| F3 | [Precup, Sutton & Singh 2000] Eligibility traces for off-policy policy evaluation (per-decision IS) | ICML 2000 | |
| F4 | [Thomas & Brunskill 2016] Data-efficient off-policy policy evaluation for RL (weighted DR, MAGIC) | ICML 2016 | [arXiv:1604.00923](https://arxiv.org/abs/1604.00923) |
| F5 | [Athey & Wager 2021] Policy learning with observational data | Econometrica 2021 | [arXiv:1702.02896](https://arxiv.org/abs/1702.02896) |
| F6 | [Kallus & Uehara 2020] Double reinforcement learning for efficient off-policy evaluation in MDPs | JMLR 2020 | |
| F7 | [Levine, Kumar, Tucker & Fu 2020] Offline reinforcement learning: tutorial, review, and perspectives | arXiv 2020 | [arXiv:2005.01643](https://arxiv.org/abs/2005.01643) |
| F8 | [Fujimoto, Meger & Precup 2019] Off-policy deep RL without exploration (BCQ) · [Kumar+ 2020] Conservative Q-learning (CQL) | ICML 2019 · NeurIPS 2020 | [arXiv:1812.02900](https://arxiv.org/abs/1812.02900) · [arXiv:2006.04779](https://arxiv.org/abs/2006.04779) |
| F9 | [Tennenholtz, Mannor & Shalit 2020] Off-policy evaluation in partially observable environments · [Kallus & Zhou 2020] Confounding-robust policy evaluation in infinite-horizon RL | AAAI 2020 · NeurIPS 2020 | |

## Key ideas

- **Bandits are repeated experiments.** Choosing between two treatments to maximize reward is
  $\arg\max_a E[r(a)]$, a maximization over potential outcomes; "exploration" is randomization to measure the ATE.
- **Contextual bandits** learn $\pi(x)$. The doubly robust value of action $a$ at context $x$,
  $\beta(x)_a = \hat r(a, x) + \mathbb{1}\{A = a\}\frac{R - \hat r(a, x)}{\hat P(a \mid x)}$, is unbiased for
  $E[R(a) \mid X = x]$. Maximizing $\frac1n\sum_t \sum_a \pi(a \mid x_t)\beta(x_t)_a$ over policies is **off-policy
  learning**; the loss is universally Neyman orthogonal in the nuisances $\hat r$ and $\hat P(a|x)$
  ([08](../08-orthogonal-learning/)).
- **Positivity becomes exploration.** Off-policy learning needs $P(a \mid x) \ge \eta > 0$. With online data collection
  the logging distribution drifts, and the orthogonal-learning guarantees need modification.
- **RL** adds states and transitions: data $(x_t, a_t, r_t, x_{t+1})$, values $V^\pi_t$ and $Q^\pi_t$. The same
  correction, applied recursively along the episode, gives doubly robust off-policy evaluation (Jiang & Li) and policy
  optimization (Chernozhukov+ 2022).

- **Offline RL is off-policy learning with a fixed log.** No new data can be collected, so exploration cannot repair
  missing actions afterwards. The dictionary:

  | causal inference | offline RL |
  | --- | --- |
  | potential outcome under a dynamic treatment regime, $E[Y(\pi)]$ | policy value $V(\pi)$ |
  | observational study, propensity score | logged dataset, behavior policy $\pi_b(a \mid s)$ |
  | sequential conditional ignorability | no unobserved confounders of action and next state / reward |
  | positivity | coverage / concentrability of $\pi$ by $\pi_b$ |
  | IPW, Hajek | importance sampling (trajectory-wise or per-decision), weighted IS |
  | g-computation, Robins' g-formula | model-based evaluation, fitted Q evaluation (FQE) |
  | AIPW, DML | doubly robust OPE, double RL (Kallus & Uehara) |
  | CATE-based treatment rule, policy learning (Athey & Wager) | greedy policy from a learned $Q$ |
  | extrapolation outside the overlap region | distribution shift, over-estimated $Q$ on unseen actions |
  | winner's curse, honest estimation | optimizer's curse in policy selection, held-out evaluation |
  | unobserved confounding: IV, proxies, sensitivity bounds | confounded / partially observable offline RL |

- **What the sequential case adds.** Products of ratios make IS variance grow exponentially with the horizon (per-decision
  weighting and DR curb it); today's action changes tomorrow's state, so a per-step regression of reward on action
  misses delayed effects (the g-formula / Bellman backup handles them); and the max over actions in policy learning
  turns thin coverage into optimism, the reason offline RL methods are *pessimistic* (BCQ, CQL, lower-confidence-bound
  value iteration): stay where the behavior data have support.
- **Hidden confounding** is the failure no estimator can detect from the log: if the doctors acted on something not
  recorded, every severity-only estimator is biased, at every sample size. The fixes are the causal ones: instruments,
  proxies of the confounder, or sensitivity bounds.

## Code in this folder

- [`offline-rl/`](offline-rl/): the hospital as a contextual bandit (DR policy evaluation and threshold-rule learning)
  and as a five-visit treatment regime, with every off-policy estimator next to its causal name (IS = IPW, model-based =
  g-formula, DR = AIPW), the curse of horizon, positivity failures, behavior cloning vs fitted Q iteration vs pessimistic
  FQI, honest policy selection, and the bias created by a hidden confounder.

## Connections

- ← [03 Propensity scores](../03-propensity-scores/): IPW and doubly robust estimators, per action.
- ← [08 Orthogonal learning](../08-orthogonal-learning/): orthogonal losses for learning a policy instead of a number.
- ← [01 Potential outcomes](../01-potential-outcomes/): exploration is randomization; coverage is positivity.
- ← [04 Causal trees](../04-causal-trees/): honest splitting, reused for policy selection.
- ← [06 Structural causal models](../06-structural-causal-models/): hidden confounding separates logged and
  interventional distributions; the g-formula is the sequential backdoor adjustment.
- ← [10 Instruments & discontinuities](../10-instruments-and-discontinuities/): remedies when the log is confounded.
