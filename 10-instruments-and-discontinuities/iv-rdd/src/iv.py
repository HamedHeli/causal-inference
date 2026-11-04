"""Instrumental variables on the pricing example from the first lecture.

The story
---------
A hotel sets its nightly price ``P`` and observes rooms sold ``Y``. Demand ``D`` (a
holiday, a conference in town) is *unobserved* and pushes both up: the manager raises
prices when demand is high, and more rooms sell. Regressing ``Y`` on ``P`` therefore
finds that higher prices "cause" more sales, the opposite of the truth.

An **instrument** is a variable ``C`` that (1) moves the price (relevance), (2) has no
path to sales except through the price (exclusion), and (3) is independent of the
unobserved demand (as good as random). Here ``C`` is the hotel's cost (energy prices,
wages): the manager passes cost onto price, and guests neither know nor care about it.

Structural equations::

    D ~ N(0, 1)                                 unobserved demand  (confounder)
    C ~ N(0, 1)                                 cost shifter       (instrument)
    P = 5 + 2.0 C + 1.5 D + 0.3 eps_P           price
    Y = h(P) + 8 D + eps_Y                      sales
    h(p) = 20 sigmoid(-0.9 (p - 5))             the true (structural) demand curve

Estimators
----------
* :func:`naive_ols`: regress Y on P. Biased by the open path ``P <- D -> Y``.
* :func:`wald`: ``Cov(Y, C) / Cov(P, C)``, the IV estimate of a linear slope.
* :func:`tsls`: two-stage least squares with any number of instruments / controls.
* :func:`sieve_iv`: nonparametric IV with a basis expansion ``h(p) = phi(p)^T beta``.
  Stage 1 predicts every basis function of P from (a basis of) C; stage 2 regresses Y
  on those predictions. DeepIV (Hartford+ 2017) replaces stage 1 by a mixture density
  network for ``P | C`` and stage 2 by a network ``h`` trained on
  ``E[h(P) | C]``; DeepGMM (Bennett+ 2019) instead plays an adversarial game over
  moment conditions ``E[(Y - h(P)) f(C)] = 0``. All three solve the same integral
  equation ``E[Y | C] = E[h(P) | C]``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


def h_true(p):
    return 20.0 / (1.0 + np.exp(0.9 * (p - 5.0)))


class IVData(NamedTuple):
    C: np.ndarray
    P: np.ndarray
    Y: np.ndarray
    D: np.ndarray


def make_pricing(n=10000, seed=0, instrument_strength=2.0, confounding=1.5, demand_effect=8.0):
    rng = np.random.default_rng(seed)
    D = rng.normal(size=n)
    C = rng.normal(size=n)
    P = 5 + instrument_strength * C + confounding * D + 0.3 * rng.normal(size=n)
    Y = h_true(P) + demand_effect * D + rng.normal(size=n)
    return IVData(C, P, Y, D)


def _lstsq(F, y):
    return np.linalg.lstsq(F, y, rcond=None)[0]


def naive_ols(P, Y):
    F = np.column_stack([np.ones_like(P), P])
    return _lstsq(F, Y)  # (intercept, slope)


def wald(C, P, Y):
    """Linear IV slope ``Cov(Y, C) / Cov(P, C)``."""
    return np.cov(Y, C)[0, 1] / np.cov(P, C)[0, 1]


def tsls(Y, T, Z, X=None):
    """Two-stage least squares. ``T`` endogenous (n, k), ``Z`` instruments (n, m >= k),
    ``X`` exogenous controls. Returns coefficients ``[intercept, T..., X...]``."""
    T = T.reshape(len(Y), -1); Z = Z.reshape(len(Y), -1)
    X = np.empty((len(Y), 0)) if X is None else X.reshape(len(Y), -1)
    W = np.column_stack([np.ones(len(Y)), Z, X])           # all exogenous variables
    T_hat = W @ _lstsq(W, T)                               # stage 1
    F_hat = np.column_stack([np.ones(len(Y)), T_hat, X])
    return _lstsq(F_hat, Y)                                 # stage 2


def bspline_basis(x, knots, degree=3):
    """Truncated-power cubic spline basis (simple and good enough here)."""
    cols = [x ** d for d in range(1, degree + 1)]
    cols += [np.maximum(x - k, 0) ** degree for k in knots]
    return np.column_stack(cols)


def sieve_iv(C, P, Y, n_knots=3, ridge=1e-3, p_grid=None):
    """Nonparametric IV (sieve 2SLS). Returns ``h_hat`` evaluated on ``p_grid`` (up to
    an additive constant, which IV cannot pin down without a normalization; we fix it
    by matching the mean of Y)."""
    kp = np.quantile(P, np.linspace(0, 1, n_knots + 2)[1:-1])
    kc = np.quantile(C, np.linspace(0, 1, n_knots + 4)[1:-1])
    phiP = bspline_basis(P, kp)
    mu, sd = phiP.mean(0), phiP.std(0) + 1e-12
    phiP = (phiP - mu) / sd
    psiC = bspline_basis(C, kc)
    psiC = np.column_stack([np.ones(len(C)), (psiC - psiC.mean(0)) / (psiC.std(0) + 1e-12)])
    # stage 1: E[phi(P) | C] by regression on a richer basis of C
    G = psiC.T @ psiC + ridge * np.eye(psiC.shape[1])
    phi_hat = psiC @ np.linalg.solve(G, psiC.T @ phiP)
    # stage 2: regress Y on the predicted basis
    F = np.column_stack([np.ones(len(Y)), phi_hat])
    beta = np.linalg.solve(F.T @ F + ridge * np.eye(F.shape[1]), F.T @ Y)
    p_grid = np.linspace(np.quantile(P, 0.02), np.quantile(P, 0.98), 200) if p_grid is None else p_grid
    Fg = np.column_stack([np.ones(len(p_grid)), (bspline_basis(p_grid, kp) - mu) / sd])
    return p_grid, Fg @ beta


def sieve_regression(P, Y, n_knots=3, p_grid=None):
    """The same spline fitted by plain least squares: the confounded curve E[Y | P]."""
    kp = np.quantile(P, np.linspace(0, 1, n_knots + 2)[1:-1])
    F = np.column_stack([np.ones(len(P)), bspline_basis(P, kp)])
    mu, sd = F[:, 1:].mean(0), F[:, 1:].std(0) + 1e-12
    F[:, 1:] = (F[:, 1:] - mu) / sd
    beta = _lstsq(F, Y)
    p_grid = np.linspace(np.quantile(P, 0.02), np.quantile(P, 0.98), 200) if p_grid is None else p_grid
    Fg = np.column_stack([np.ones(len(p_grid)), (bspline_basis(p_grid, kp) - mu) / sd])
    return p_grid, Fg @ beta
