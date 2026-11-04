# Instruments and discontinuities: unobserved confounding

Two designs that identify causal effects when a confounder is **not** observed, each on an example from the lectures:
instrumental variables on hotel pricing under unobserved demand, and a regression discontinuity on "treat iff age ≥ 6".

## The problems

**Pricing.** Demand $D$ (unobserved) raises both the price $P$ and the rooms sold $Y$; the hotel's cost $C$ shifts the
price but not demand:

```
D ~ N(0, 1)                     unobserved demand
C ~ N(0, 1)                     cost: the instrument
P = 5 + 2 C + 1.5 D + noise
Y = h(P) + 8 D + noise,         h(p) = 20 sigmoid(-0.9 (p - 5))   (the demand curve)
```

**Age cutoff.** Severity is unobserved, but doctors treat if and only if age ≥ 6. The potential outcomes are smooth in
age, so the jump in $E[Y | \text{age}]$ at 6 is the effect at 6.

## Notes in five lines

1. IV: relevance, exclusion, independence ⇒ $\beta = \text{Cov}(Y,C)/\text{Cov}(P,C)$ (2SLS for several instruments).
2. Nonparametric IV solves $E[Y|C] = E[h(P)|C]$; sieve 2SLS does it in a spline basis, Deep IV and Deep GMM with
   networks.
3. Weak instruments make the ratio erratic; exclusion violations bias it for any sample size.
4. RDD: $\tau(a) = \lim_{\downarrow a}E[Y|A] - \lim_{\uparrow a}E[Y|A]$, under continuity of $E[Y(t)|A]$.
5. The RDD estimate is two boundary predictions; choose the bandwidth by cross-validation.

## Start here

[`notebooks/iv_rdd_explained.ipynb`](notebooks/iv_rdd_explained.ipynb):

1. OLS gets the demand curve wrong (coloured by the unobserved demand);
2. Wald and 2SLS slopes;
3. weak instruments and exclusion violations;
4. sieve IV recovers the demand curve; revenue-maximizing price from each curve;
5. RDD on the age cutoff: potential vs observed outcomes, the jump;
6. bandwidth: CV criterion vs true bias/variance over 60 datasets;
7. global polynomials by degree, placebo cutoffs;
8. **slider:** bandwidth.

## Layout

```
iv-rdd/
├── src/
│   ├── iv.py     pricing simulator, naive OLS, Wald, 2SLS, sieve IV
│   └── rdd.py    age-cutoff simulator, local linear RDD, global polynomials, Imbens-Lemieux CV
├── notebooks/
│   └── iv_rdd_explained.ipynb
└── requirements.txt
```

## Run

```bash
cd 10-instruments-and-discontinuities/iv-rdd
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/iv_rdd_explained.ipynb
```
