# 09 Cocycles

**Nov 5 and Nov 17, Block 3: causal machine learning** · [back to the graph](../README.md)

**What is learned:** the causal effect as a *transformation of outcomes*: a map that sends a unit's outcome under one
treatment to its outcome under another. Under additional assumptions this reaches individual-level counterfactuals,
not only conditional means.

## Paper

| | Paper | Venue | Links |
| --- | --- | --- | --- |
| R1 | **[Dance & Bloem-Reddy 2024] Causal Inference with Cocycles.** Hugh Dance, Benjamin Bloem-Reddy | arXiv 2024 | [arXiv:2405.13844](https://arxiv.org/abs/2405.13844) |
| R2 | [Xi, Dance, Orbanz & Bloem-Reddy 2025] Distinguishing cause from effect with causal velocity models | ICML 2025 | cocycle ideas for causal discovery (cited in the deep-learning lecture) |

## Key ideas

These notes stay at the level of the lecture slides ("learns causal effects as data transformations, works for full
counterfactuals under more assumptions"); see the paper for the formal statements.

- **Effects as maps.** Instead of modelling $E[Y(t) \mid X]$ for each $t$ and subtracting, model a family of maps
  $c(t', t)$ with $Y(t') = c(t', t)\big(Y(t)\big)$ (given covariates). A **cocycle** is such a family that is
  consistent under composition: $c(t, t) = \text{id}$ and $c(t'', t') \circ c(t', t) = c(t'', t)$, the way a change of
  coordinates composes.
- **Why it helps.** Conditional means (ATE, CATE) only need the two marginal distributions of $Y(0)$ and $Y(1)$.
  Questions like "how much would *this* patient have improved" or PNS ([06](../06-structural-causal-models/)) need the
  coupling between them, which observations and experiments do not identify. Restricting the counterfactual map to a
  structured class (for example invertible, monotone transformations) is one way to make such quantities identifiable.
- **Under conditional ignorability**, the maps can be learned from the observed conditional distributions of $Y$ given
  $(X, T)$, with flexible (e.g. neural, flow-like) parametrizations.
- **Connection to discovery.** The same "dynamics of how outcomes change" view underlies the causal velocity models
  of Xi et al. for deciding the direction of causation.

## Code

None yet.

## Connections

- ← [08 Orthogonal learning](../08-orthogonal-learning/): the lecture lists cocycles next to orthogonal learning as the
  two ways to fit flexible effect models under conditional ignorability; orthogonal learning targets the mean
  effect, cocycles the transformation.
- ← [06 Structural causal models](../06-structural-causal-models/): the third rung of the ladder (counterfactuals) is
  what the extra structure buys.
- ↔ [07 Causal deep learning](../07-causal-deep-learning/): flexible CATE models and causal discovery.
