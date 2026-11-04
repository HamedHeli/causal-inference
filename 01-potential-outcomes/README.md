# 01 Potential outcomes

**Sep 3–15, Block 1: defining and identifying causal effects** · [back to the graph](../README.md)

**What is identified:** population averages of potential outcomes, $E[Y(1)]$ and $E[Y(0)]$, from observable
conditional means, once treatment is (conditionally) ignorable and every stratum contains both arms.

## Readings

| | Reading | Notes |
| --- | --- | --- |
| L1 | Introduction: why ML needs causal inference | pricing and drug-vs-surgery examples, Simpson's paradox |
| L2 | The potential outcomes framework: definitions | $Y_i(0), Y_i(1)$, SUTVA, ATE / ATT / ATC / CATE |
| L3 | Identifiability, ignorability, randomization | the bias decomposition, (conditional) randomization |
| L4 | Identification and estimation without randomization | matching, conditioning, positivity, confidence intervals |
| R | Hernán & Robins, *Causal Inference: What If* (2020) | Ch. 1–3 |

## Key ideas

- **Potential outcomes.** Each unit $i$ has one outcome per treatment value; we observe only
  $Y_i = T_i Y_i(1) + (1 - T_i) Y_i(0)$. The individual effect $Y_i(1) - Y_i(0)$ is never observed, so targets are
  averages: $\text{ATE} = E[Y(1) - Y(0)]$, the ATT and ATC on the treated and controls, and the CATE
  $E[Y(1) - Y(0) \mid X]$.
- **SUTVA.** One version of treatment, consistency $Y_i(t) = Y_i$ when $T_i = t$, and no interference between units.
  Vaccines and peer effects break it, and then potential outcomes must be indexed by everybody's treatment.
- **Why association is not causation, as an identity.**
  $E[Y|T=1] - E[Y|T=0] = \text{ATE} + \underbrace{E[Y(0)|T=1] - E[Y(0)|T=0]}_{\text{baseline bias}} + (1-\pi)\underbrace{(\text{ATT} - \text{ATC})}_{\text{differential effect}}$.
- **Ignorability** $\{Y(0), Y(1)\} \perp\!\!\!\perp T$ removes both biases, so the ATE is **identified** (written in
  observable terms). Randomization gives it by design; conditional randomization gives conditional ignorability.
- **Without randomization** we assume conditional ignorability given $X$ and need **positivity**
  $0 < P(T=1|X) < 1$. Then $\text{ATE} = E\big[E[Y|T=1,X] - E[Y|T=0,X]\big]$ (conditioning), or build comparable
  groups explicitly (matching). Neither assumption is free: ignorability is untestable, positivity is testable.
- **Statistical inference** is a separate layer: law of large numbers, CLT intervals, Hoeffding bounds for bounded
  outcomes.

## Code in this folder

- [`simpson/`](simpson/): the drug-vs-surgery hospital with both potential outcomes known. Simpson's paradox, the bias
  decomposition checked to machine precision, randomization vs observation, conditional randomization, matching (ATT,
  ATC, ATE), stratification with residual confounding and positivity failures, CLT vs Hoeffding coverage, and a slider
  for confounding strength.

## Connections

- → [02 Regression](../02-regression/): when $X$ is continuous or high-dimensional, model $E[Y|T,X]$ instead of
  stratifying.
- → [03 Propensity scores](../03-propensity-scores/): condition on the one-dimensional $e(X)$ instead of $X$.
- → [06 Structural causal models](../06-structural-causal-models/): $Y(t)$ is $Y$ under $do(T=t)$, and the backdoor
  criterion says *which* $X$ makes $T$ ignorable.
- → [04 Causal trees](../04-causal-trees/): from the ATE to the CATE $\tau(x)$.
- ↔ [12 One dataset, many estimators](../12-one-dataset-many-estimators/): the same hospital with nonlinear confounding,
  estimated by every method of the course.
