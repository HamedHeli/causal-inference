"""Regression discontinuity designs (RDD), as in the lecture's age-cutoff example.

The story
---------
Doctors treat a patient **if and only if** ``Age >= a`` (here ``a = 6``). Severity is
unobserved, so conditional ignorability fails. But if the potential outcomes
``E[Y(t) | Age]`` are *continuous* in age, then just left of the cutoff everybody is a
control and just right everybody is treated, and

    tau(a) = E[Y(1) - Y(0) | Age = a] = lim_{age -> a+} E[Y | age] - lim_{age -> a-} E[Y | age].

The estimate is a pair of **predictions at the boundary**, one from each side: an ML
problem, with the twist that we extrapolate to the edge of each side's support.

* :func:`local_linear` fits a weighted (triangular-kernel) line on each side within a
  bandwidth ``h`` and takes the jump at the cutoff.
* :func:`cv_bandwidth` picks ``h`` by the Imbens-Lemieux (2008) cross-validation:
  predict each point near the cutoff using only data on its own side *and further from
  the cutoff*, which mimics the boundary prediction the estimator actually makes.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class RDDData(NamedTuple):
    age: np.ndarray
    T: np.ndarray
    Y: np.ndarray
    Y0: np.ndarray
    Y1: np.ndarray


def mu0(age):
    return 0.35 + 0.06 * age - 0.012 * (age - 6) ** 2 + 0.05 * np.sin(age)


def mu1(age):
    return mu0(age) + 0.30 - 0.04 * (age - 6)


def make_rdd(n=1500, cutoff=6.0, seed=0, noise=0.08):
    rng = np.random.default_rng(seed)
    age = rng.uniform(0, 12, n)
    T = (age >= cutoff).astype(float)
    Y0 = mu0(age) + noise * rng.normal(size=n)
    Y1 = mu1(age) + noise * rng.normal(size=n)
    return RDDData(age, T, T * Y1 + (1 - T) * Y0, Y0, Y1)


def true_effect(cutoff=6.0):
    return float(mu1(cutoff) - mu0(cutoff))


def _wls_line(x, y, w):
    F = np.column_stack([np.ones_like(x), x])
    Fw = F * w[:, None]
    return np.linalg.solve(Fw.T @ F + 1e-12 * np.eye(2), Fw.T @ y)


def boundary_fit(x, y, at, h, degree=1):
    """Weighted polynomial fit with a triangular kernel, evaluated at ``at``."""
    w = np.clip(1 - np.abs(x - at) / h, 0, None)
    m = w > 0
    if m.sum() < degree + 2:
        return np.nan, np.nan
    F = np.vander(x[m] - at, degree + 1, increasing=True)
    Fw = F * w[m, None]
    A = Fw.T @ F + 1e-12 * np.eye(degree + 1)
    beta = np.linalg.solve(A, Fw.T @ y[m])
    r = y[m] - F @ beta
    # sandwich variance of the intercept
    Ainv = np.linalg.inv(A)
    meat = (Fw * r[:, None] ** 2).T @ (F * w[m, None])
    var = (Ainv @ meat @ Ainv)[0, 0]
    return beta[0], np.sqrt(var)


def local_linear(age, Y, cutoff, h, degree=1):
    """RDD estimate (jump at the cutoff) and its standard error."""
    L, R = age < cutoff, age >= cutoff
    yl, sl = boundary_fit(age[L], Y[L], cutoff, h, degree)
    yr, sr = boundary_fit(age[R], Y[R], cutoff, h, degree)
    return yr - yl, np.sqrt(sl ** 2 + sr ** 2), (yl, yr)


def global_polynomial(age, Y, cutoff, degree):
    """A global polynomial on each side (the misspecification-prone alternative)."""
    out = []
    for m in (age < cutoff, age >= cutoff):
        F = np.vander(age[m] - cutoff, degree + 1, increasing=True)
        out.append(np.linalg.lstsq(F, Y[m], rcond=None)[0][0])
    return out[1] - out[0]


def cv_bandwidth(age, Y, cutoff, hs, window=0.5, degree=1):
    """Imbens-Lemieux cross-validation criterion for each bandwidth in ``hs``.

    For each point within the ``window`` fraction of data closest to the cutoff on its
    side, fit only on points *beyond* it (away from the cutoff, same side) within h and
    predict it as a boundary point.
    """
    scores = []
    for h in hs:
        err = []
        for side in (age < cutoff, age >= cutoff):
            a, y = age[side], Y[side]
            dist = np.abs(a - cutoff)
            near = dist <= np.quantile(dist, window)
            for i in np.where(near)[0]:
                beyond = dist > dist[i]
                pred, _ = boundary_fit(a[beyond], y[beyond], a[i], h, degree)
                if np.isfinite(pred):
                    err.append((y[i] - pred) ** 2)
        scores.append(np.mean(err))
    return np.array(scores)
