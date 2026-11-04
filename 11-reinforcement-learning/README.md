# 11 Reinforcement learning

**Nov 24, Block 3: causal machine learning** · [back to the graph](../README.md)

**What is learned:** a *policy* $\pi(a \mid x)$ that picks treatments, evaluated and improved from logged data with the
same doubly robust corrections used for the ATE.

## Readings (from the lecture)

| | Paper | Venue | Links |
| --- | --- | --- | --- |
| R1 | **[Jiang & Li 2016] Doubly robust off-policy value evaluation for reinforcement learning** | ICML 2016 | [arXiv:1511.03722](https://arxiv.org/abs/1511.03722) |
| R2 | **[Chernozhukov, Newey, Singh & Syrgkanis 2022] Automatic debiased machine learning for dynamic treatment effects and general nested functionals** | arXiv 2022 | [arXiv:2203.13887](https://arxiv.org/abs/2203.13887) |
| R3 | **[Schulte & Poupart 2025] When should reinforcement learning use causal reasoning?** | TMLR 2025 | |

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

## Code

None yet. The doubly robust value is the AIPW score of [03](../03-propensity-scores/) /
[08](../08-orthogonal-learning/) with one term per action.

## Connections

- ← [03 Propensity scores](../03-propensity-scores/): IPW and doubly robust estimators, per action.
- ← [08 Orthogonal learning](../08-orthogonal-learning/): orthogonal losses for learning a policy instead of a number.
- ← [01 Potential outcomes](../01-potential-outcomes/): exploration is randomization.
