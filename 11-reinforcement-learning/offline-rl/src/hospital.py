"""The running example of the course: drug vs surgery, confounded by severity.

Every estimator in this repository is tested on data from this simulator, because
here (unlike in real data) both potential outcomes of every patient are known, so the
true ATE, ATT, ATC and CATE can be printed next to each estimate.

The story
---------
A hospital treats patients with either a drug (``T = 0``, the control) or surgery
(``T = 1``). The outcome ``Y`` is a remission score (higher is better).

* ``severity`` ``S`` in ``[0, 2]`` drives everything. Sicker patients
  (1) have worse outcomes whatever they receive, and (2) are more often sent to
  surgery. That makes ``S`` a **confounder** (``T <- S -> Y``).
* Surgery helps severe patients and slightly hurts mild ones, so the effect is
  **heterogeneous**: ``tau(S) = -0.15 + 0.25 S``.
* With ``n_noise > 0`` the patient also has an age and extra covariates. Only
  ``severity`` and ``age`` matter; the others are noise that an estimator must learn to
  ignore.

Structural equations (``nonlinear=False``)::

    S ~ Uniform(0, 2)
    T ~ Bernoulli(e(S)),          e(S) = sigmoid(3 (S - 1))
    Y(0) = 0.9 - 0.35 S + eps,    eps ~ N(0, 0.1^2)
    Y(1) = Y(0) + tau(S),         tau(S) = -0.15 + 0.25 S
    Y = T Y(1) + (1 - T) Y(0)

The true ATE is ``E[tau(S)] = 0.10`` (surgery helps on average), but sicker patients
get surgery, so the naive difference of means is *negative*: Simpson's paradox.

With ``nonlinear=True`` the propensity, baseline and effect become nonlinear in
``(S, age)``, which is the setting where flexible ML nuisance models are needed.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class Data(NamedTuple):
    """One simulated dataset. Columns of ``X``: severity, then age, then noise."""

    X: np.ndarray        # (n, d) covariates
    T: np.ndarray        # (n,) treatment in {0, 1}
    Y: np.ndarray        # (n,) observed outcome
    Y0: np.ndarray       # (n,) potential outcome under the drug   (unobservable)
    Y1: np.ndarray       # (n,) potential outcome under surgery    (unobservable)
    e: np.ndarray        # (n,) true propensity P(T = 1 | X)      (unknown in practice)
    tau: np.ndarray      # (n,) true CATE E[Y(1) - Y(0) | X]      (unknown in practice)
    names: tuple

    @property
    def S(self):
        return self.X[:, 0]

    @property
    def ate(self):
        return float(self.tau.mean())

    @property
    def att(self):
        return float(self.tau[self.T == 1].mean())

    @property
    def atc(self):
        return float(self.tau[self.T == 0].mean())


def baseline(S, age, nonlinear):
    """E[Y(0) | S, age]."""
    if not nonlinear:
        return 0.9 - 0.35 * S
    return 0.9 - 0.35 * S - 0.15 * np.sin(2.5 * S) - 0.10 * (age - 0.5) ** 2


def cate(S, age, nonlinear):
    """tau(S, age) = E[Y(1) - Y(0) | S, age]."""
    if not nonlinear:
        return -0.15 + 0.25 * S
    return -0.15 + 0.25 * S + 0.10 * np.tanh(4 * (age - 0.5)) * (S > 1)


def propensity(S, age, nonlinear, strength=3.0):
    """e(S, age) = P(T = 1 | S, age)."""
    if not nonlinear:
        return sigmoid(strength * (S - 1))
    return sigmoid(strength * (S - 1) + 1.0 * np.cos(3 * age) - 0.3)


def make_hospital(n=2000, seed=0, n_noise=0, nonlinear=False, randomized=False,
                  strength=3.0, noise=0.1, clip=0.02):
    """Simulate ``n`` patients.

    Args:
        n: Number of patients.
        seed: RNG seed.
        n_noise: Number of irrelevant covariates (age is added when ``n_noise > 0`` or
            ``nonlinear``).
        nonlinear: Use the nonlinear propensity, baseline and CATE.
        randomized: Assign surgery by a fair coin instead of by severity (an RCT).
        strength: Slope of the propensity in severity; larger means more confounding
            and worse overlap.
        noise: Outcome noise standard deviation.
        clip: Propensities are clipped to ``[clip, 1 - clip]`` (positivity).
    """
    rng = np.random.default_rng(seed)
    S = rng.uniform(0, 2, n)
    use_age = nonlinear or n_noise > 0
    age = rng.uniform(0, 1, n) if use_age else np.full(n, 0.5)
    cols, names = [S], ["severity"]
    if use_age:
        cols.append(age); names.append("age")
    for j in range(n_noise):
        cols.append(rng.normal(size=n)); names.append(f"noise{j}")
    X = np.column_stack(cols)

    e = np.full(n, 0.5) if randomized else propensity(S, age, nonlinear, strength)
    e = np.clip(e, clip, 1 - clip)
    T = (rng.uniform(size=n) < e).astype(float)
    Y0 = baseline(S, age, nonlinear) + noise * rng.normal(size=n)
    tau = cate(S, age, nonlinear)
    Y1 = Y0 + tau
    Y = T * Y1 + (1 - T) * Y0
    return Data(X, T, Y, Y0, Y1, e, tau, tuple(names))


def severity_bucket(S):
    """The physician's three categories from the lecture: 0 low, 1 medium, 2 high."""
    return np.digitize(S, [2 / 3, 4 / 3])
