# 10 Instruments and discontinuities

**Nov 19, Block 2: core estimation approaches** · [back to the graph](../README.md)

**What is identified:** a causal effect *without* conditional ignorability, from a source of variation in the treatment
that is as good as random: an instrument (identifies the structural function $h(p)$, or a weighted average of its
slope) or a threshold rule (identifies the effect at the threshold only).

## Papers

| | Paper | Venue | Links |
| --- | --- | --- | --- |
| R1 | **[Hartford+ 2017] Deep IV: A Flexible Approach for Counterfactual Prediction.** Jason Hartford, Greg Lewis, Kevin Leyton-Brown, Matt Taddy | ICML 2017 | [PMLR](https://proceedings.mlr.press/v70/hartford17a/hartford17a.pdf) |
| R2 | **[Bennett+ 2019] Deep Generalized Method of Moments for Instrumental Variable Analysis.** Andrew Bennett, Nathan Kallus, Tobias Schnabel | NeurIPS 2019 | [arXiv:1905.12495](https://arxiv.org/abs/1905.12495) |
| R3 | [Imbens & Lemieux 2008] Regression discontinuity designs: a guide to practice | J. Econometrics 2008 | bandwidth by cross-validation |
| R4 | [Lee & Card 2008] RD inference with specification error · [Lee & Lemieux 2010] RD designs in economics | J. Econometrics 2008 · JEL 2010 | |

## Key ideas

- **Instruments.** $Z$ is an instrument for $T \to Y$ if it is relevant ($Z$ moves $T$), excluded ($Z$ affects $Y$ only
  through $T$) and independent of the unobserved confounders. In graph terms: an edge $Z \to T$, and no other path from
  $Z$ to $Y$.
- **Linear IV.** $\beta_{IV} = \text{Cov}(Y, Z)/\text{Cov}(T, Z)$, or two-stage least squares: regress $T$ on $Z$, then
  $Y$ on the fitted $\hat T$. With a nonlinear $h$, it is a weighted average of $h'$ over where the instrument moves
  $T$.
- **Nonparametric IV.** $E[Y \mid Z] = \int h(t)\, dF(t \mid Z)$, an integral equation (ill-posed: small errors in the
  left side can mean large errors in $h$). Deep IV: stage 1 is a mixture density network for $F(t \mid z, x)$; stage 2
  fits a network $h$ by minimizing $(Y - \int h\, d\hat F)^2$ with sampled treatments. Deep GMM: an adversarial game
  over moment conditions $E[(Y - h(T)) f(Z)] = 0$, no density model needed.
- **Failure modes.** Weak instruments (erratic, OLS-biased estimates) and exclusion violations (bias that no amount of
  data removes). Neither is visible in the data without assumptions.
- **Regression discontinuity.** With $T = \mathbb{1}[A \ge a]$ and $E[Y(t)|A]$ continuous at $a$,
  $\tau(a) = \lim_{A \downarrow a} E[Y|A] - \lim_{A \uparrow a} E[Y|A]$. No ignorability needed, but the effect is local
  to the cutoff.
- **RDD is a prediction problem at a boundary.** Local linear fits with a bandwidth; the bandwidth trades bias for
  variance and can be chosen by cross-validation on points away from the threshold (Imbens & Lemieux). Global
  high-order polynomials are fragile (Lee & Card). Manipulation of the running variable (claiming to be older to get
  treated) breaks the design (Lee & Lemieux).

## Code in this folder

- [`iv-rdd/`](iv-rdd/): the hotel pricing example from the first lecture, with unobserved demand. OLS gets the demand
  curve wrong; Wald / 2SLS recover the slope; weak instruments and exclusion violations; sieve 2SLS (the linear-basis
  version of what Deep IV and Deep GMM do) recovers the whole curve and the revenue-maximizing price. Then the
  "treat iff age ≥ 6" RDD: local linear estimate, bandwidth bias/variance, Imbens–Lemieux cross-validation, global
  polynomials and placebo cutoffs, with a bandwidth slider.

## Connections

- ← [06 Structural causal models](../06-structural-causal-models/): when no backdoor set exists (unobserved confounder),
  exclusion restrictions and threshold rules are other routes to identification.
- ↔ [07 Causal deep learning](../07-causal-deep-learning/): instruments as a training signal for flexible CATE models;
  RDD as "extrapolation for causal inference".
- ← [01 Potential outcomes](../01-potential-outcomes/): the hotel-pricing example of the first lecture, solved.
