# 06 Structural causal models

**Oct 6–22, Block 1: defining and identifying causal effects** · [back to the graph](../README.md)

**What is identified:** whatever the graph allows. A DAG states the causal assumptions (mostly as *missing* edges); the
backdoor criterion and do-calculus turn them into formulas for $P(Y \mid do(T))$ in terms of observational
distributions, or show that no such formula exists.

## Readings

| | Reading | Notes |
| --- | --- | --- |
| L8–L9 | DAGs and SCMs | structural equations, Markov factorization, the do operator, three building blocks |
| L10 | DAG structures, paths, d-separation | open and blocked paths, collider bias |
| L11 | Backdoor criterion, do-calculus | three rules, front-door identification |
| L12 | Counterfactuals, ladder of causation, PN / PS / PNS | separation of the three rungs, bounds |
| R | Judea Pearl, *Causality* (2009), Ch. 3, 7, 9 | |
| R | Huang & Valtorta 2006; Shpitser & Pearl 2006 | completeness of do-calculus, identification algorithms |
| R | Shpitser, VanderWeele & Robins 2010 | validity of covariate adjustment |

## Key ideas

- **SCMs.** Each node is a function of its parents and an independent noise term, $V := f_V(\text{pa}(V), U_V)$.
  The equations are not reversible: they encode mechanisms. The induced distribution factorizes along the DAG, and each
  node is independent of its non-descendants given its parents (Markov property).
- **The do operator** replaces $f_T$ by a constant (deletes the arrows into $T$). $P(Y \mid do(T=t))$ is the
  distribution of $Y(t)$; a unit is a draw of the noise variables. ATE, ATT, ATC are rewritten with $do$.
- **Three building blocks.** Mediator $T \to M \to Y$ and confounder $T \leftarrow S \to Y$: conditioning on the middle
  node blocks the path. Collider $T \to C \leftarrow Y$: conditioning on $C$ or a descendant *opens* it.
- **d-separation**: all paths blocked implies conditional independence.
- **Backdoor criterion.** $Z$ blocks every path into $T$ and contains no descendant of $T$ ⇒
  $P(Y|do(t)) = \sum_z P(Y|t, z)P(z)$. This is conditional ignorability, made checkable.
- **Do-calculus.** Three rules (insert/delete observations, exchange action and observation, insert/delete actions),
  each a d-separation test in a mutilated graph. Complete: if an effect is identifiable, the rules find it. Example:
  the front-door formula through a mediator when the confounder is unobserved.
- **The ladder.** Observational, interventional and counterfactual distributions are strictly separated: two SCMs can
  agree on all observations and differ under intervention, or agree on all interventions and differ on counterfactuals
  like the **probability of necessity and sufficiency** $\text{PNS} = P(Y(1)=1, Y(0)=0)$, which experiments only bound:
  $\max(0, P(y|do(1)) - P(y|do(0))) \le \text{PNS} \le \min(P(y|do(1)), P(y'|do(0)))$.

## Code in this folder

- [`dags/`](dags/): DAGs in plain Python (paths, blocking rules, d-separation, backdoor-set enumeration, graph
  surgery), the three rules of do-calculus as functions, and an SCM class with `do` and counterfactual evaluation. The
  notebook checks every claim by simulation: collider bias, observational vs interventional means, all 256 adjustment
  sets on a 10-node graph (valid ones hit the truth, invalid ones miss), the front-door formula, and PNS for two SCMs
  with identical experiments.

## Connections

- ↔ [01 Potential outcomes](../01-potential-outcomes/): $Y(t) \sim P(Y|do(t))$; the backdoor criterion says which $X$
  gives conditional ignorability.
- → [10 Instruments & discontinuities](../10-instruments-and-discontinuities/): when no backdoor set exists, an
  instrument (exclusion = a missing edge) is another route.
- → [07 Causal deep learning](../07-causal-deep-learning/): causal discovery tries to learn the graph; PNS becomes a
  training objective for causal representations.
- → [09 Cocycles](../09-cocycles/): full counterfactuals, the third rung, under structural assumptions.
