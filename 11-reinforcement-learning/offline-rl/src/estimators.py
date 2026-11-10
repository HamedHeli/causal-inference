"""Every ATE / CATE estimator of the course, on the same ``(X, T, Y)`` interface.

Each function takes covariates ``X`` (n, d), a binary treatment ``T`` (n,) and an outcome
``Y`` (n,) and returns an :class:`Estimate`: a point estimate of the ATE, an optional
standard error, and (for CATE methods) a function ``x -> tau_hat(x)``.

The sections follow the course order:

1. potential outcomes: naive difference, stratification (conditioning), matching
2. regression: ``Y ~ T + X`` (variance-weighted!) and the interacted / g-computation fix
3. propensity scores: IPW (Horvitz-Thompson and Hajek), IP-weighted regression, AIPW
4. meta-learners: S-, T- and X-learners
5. orthogonal learning: double/debiased ML (partially linear) and the R-learner

The causal tree has its own module, :mod:`src.tree`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .learners import KernelRidge, Logistic, Ridge, add_intercept, kfold, weighted_lstsq


@dataclass
class Estimate:
    ate: float
    se: Optional[float] = None
    cate: Optional[Callable] = None
    info: Optional[dict] = None

    def ci(self, z=1.96):
        return (self.ate - z * self.se, self.ate + z * self.se) if self.se is not None else None


def default_outcome_model():
    return KernelRidge(n_features=200, lengthscale=1.5, lam=1.0)


def default_cate_model():
    """Second-stage model for pseudo-outcomes (R- and DR-learner). Pseudo-outcomes are
    much noisier than outcomes, so it is regularized more."""
    return KernelRidge(n_features=200, lengthscale=1.5, lam=10.0)


def default_propensity_model():
    return Logistic(lam=1e-3, n_features=100, lengthscale=1.5)


# ---------------------------------------------------------------------------
# 1. Potential outcomes: naive, conditioning, matching
# ---------------------------------------------------------------------------

def naive_difference(X, T, Y):
    """``E[Y | T=1] - E[Y | T=0]``: ATE + baseline bias + differential-effect bias."""
    y1, y0 = Y[T == 1], Y[T == 0]
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    return Estimate(y1.mean() - y0.mean(), se)


def stratified(strata, T, Y):
    """Conditioning estimator ``sum_x P(X=x) (E[Y|T=1,X=x] - E[Y|T=0,X=x])``.

    ``strata`` is a discrete variable (e.g. severity buckets). Strata without both
    treated and control units violate positivity and are dropped (reported in info).
    """
    ate, var, kept, dropped = 0.0, 0.0, 0.0, []
    n = len(Y)
    for s in np.unique(strata):
        m = strata == s
        y1, y0 = Y[m & (T == 1)], Y[m & (T == 0)]
        if len(y1) < 2 or len(y0) < 2:
            dropped.append(s)
            continue
        p = m.mean()
        ate += p * (y1.mean() - y0.mean())
        var += p ** 2 * (y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
        kept += p
    return Estimate(ate / kept, np.sqrt(var) / kept, info=dict(dropped=dropped, mass_kept=kept))


def matching(X, T, Y, k=1, target="ate"):
    """Nearest-neighbour matching (with replacement, on standardized X).

    Each unit's missing potential outcome is imputed by the mean outcome of its ``k``
    nearest neighbours in the other arm. ``target`` in {"ate", "att", "atc"}.
    """
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    imputed = np.empty(len(Y))
    for t in (0, 1):
        src = np.where(T == 1 - t)[0]          # donors from the other arm
        dst = np.where(T == t)[0]
        d2 = ((Z[dst, None, :] - Z[None, src, :]) ** 2).sum(-1)
        nn = np.argsort(d2, axis=1)[:, :k]
        imputed[dst] = Y[src][nn].mean(1)
    y1 = np.where(T == 1, Y, imputed)
    y0 = np.where(T == 0, Y, imputed)
    keep = {"ate": np.ones_like(T, bool), "att": T == 1, "atc": T == 0}[target]
    d = (y1 - y0)[keep]
    return Estimate(d.mean(), d.std(ddof=1) / np.sqrt(len(d)))


# ---------------------------------------------------------------------------
# 2. Regression
# ---------------------------------------------------------------------------

def ols(F, y):
    """OLS with classical standard errors. ``F`` already contains the intercept."""
    beta = np.linalg.lstsq(F, y, rcond=None)[0]
    r = y - F @ beta
    s2 = r @ r / (len(y) - F.shape[1])
    cov = s2 * np.linalg.inv(F.T @ F)
    return beta, np.sqrt(np.diag(cov))


def regression_adjustment(X, T, Y):
    """``Y ~ 1 + T + X``; the coefficient of T.

    By Frisch-Waugh-Lovell this is ``Cov(Y, T~) / Var(T~)``, a **variance-weighted**
    average of conditional effects: strata where ``Var(T|X)`` is large count more. It is
    the ATE only if the effect is homogeneous or the weights happen to match ``P(X)``.
    """
    beta, se = ols(np.column_stack([np.ones(len(Y)), T, X]), Y)
    return Estimate(beta[1], se[1])


def fwl(X, T, Y):
    """The same coefficient, computed in the two FWL steps (residualize T, then regress)."""
    F = add_intercept(X)
    T_res = T - F @ np.linalg.lstsq(F, T, rcond=None)[0]
    return float(np.cov(Y, T_res)[0, 1] / T_res.var(ddof=1)), T_res


def g_computation(X, T, Y, model=None):
    """Regression as counterfactual prediction (a.k.a. g-computation / interacted OLS).

    Fit ``mu(x, t)`` and average ``mu(x, 1) - mu(x, 0)`` over the whole sample. With a
    linear model including ``T * X`` interactions this fixes the variance weighting.
    """
    if model is None:
        F = lambda X, T: np.column_stack([np.ones(len(T)), T, X, T[:, None] * X])
        beta = np.linalg.lstsq(F(X, T), Y, rcond=None)[0]
        mu = lambda X, t: F(X, np.full(len(X), float(t))) @ beta
    else:
        m = model.fit(np.column_stack([X, T]), Y)
        mu = lambda X, t: m.predict(np.column_stack([X, np.full(len(X), float(t))]))
    tau = lambda x: mu(x, 1) - mu(x, 0)
    return Estimate(float(tau(X).mean()), cate=tau)


# ---------------------------------------------------------------------------
# 3. Propensity scores
# ---------------------------------------------------------------------------

def fit_propensity(X, T, model=None, clip=0.01):
    model = model or default_propensity_model()
    return np.clip(model.fit(X, T).predict(X), clip, 1 - clip)


def ipw(X, T, Y, e=None, normalized=False):
    """Inverse propensity weighting.

    Horvitz-Thompson: ``mean(T Y / e - (1 - T) Y / (1 - e))`` (claim 2 of the lecture).
    Hajek (``normalized=True``): weights are normalized to sum to one in each arm, which
    equals the IP-weighted regression ``Y ~ 1 + T`` with weights ``1 / P(T=t | X)``.
    """
    e = fit_propensity(X, T) if e is None else e
    if not normalized:
        phi = T * Y / e - (1 - T) * Y / (1 - e)
        return Estimate(phi.mean(), phi.std(ddof=1) / np.sqrt(len(Y)), info=dict(e=e))
    w1, w0 = T / e, (1 - T) / (1 - e)
    ate = (w1 * Y).sum() / w1.sum() - (w0 * Y).sum() / w0.sum()
    return Estimate(ate, info=dict(e=e))


def ip_weighted_regression(X, T, Y, e=None):
    """``Y ~ beta0 + betaT T`` with weights ``1/P(T=t|X)``: ``beta0 = E[Y(0)]``,
    ``beta0 + betaT = E[Y(1)]``."""
    e = fit_propensity(X, T) if e is None else e
    w = T / e + (1 - T) / (1 - e)
    beta = weighted_lstsq(np.column_stack([np.ones(len(T)), T]), Y, w)
    return Estimate(beta[1], info=dict(beta0=beta[0], w=w))


def cross_fit_nuisances(X, T, Y, outcome_model=default_outcome_model,
                        propensity_model=default_propensity_model, k=5, seed=0, clip=0.01):
    """Out-of-fold predictions ``mu0(x), mu1(x), m(x) = E[Y|x], e(x)`` for every unit."""
    n = len(Y)
    mu0, mu1, m, e = (np.empty(n) for _ in range(4))
    for tr, te in kfold(n, k, seed):
        Xt, Tt, Yt = X[tr], T[tr], Y[tr]
        mu0[te] = outcome_model().fit(Xt[Tt == 0], Yt[Tt == 0]).predict(X[te])
        mu1[te] = outcome_model().fit(Xt[Tt == 1], Yt[Tt == 1]).predict(X[te])
        m[te] = outcome_model().fit(Xt, Yt).predict(X[te])
        e[te] = propensity_model().fit(Xt, Tt).predict(X[te])
    return dict(mu0=mu0, mu1=mu1, m=m, e=np.clip(e, clip, 1 - clip))


def aipw(X, T, Y, nuis=None, **kw):
    """Doubly robust (augmented IPW) estimator with cross-fitting.

    ``phi = mu1 - mu0 + T (Y - mu1) / e - (1 - T) (Y - mu0) / (1 - e)``. Unbiased if
    *either* the outcome models or the propensity are right; with cross-fitting this is
    the DML estimator of the ATE, and ``std(phi)/sqrt(n)`` is a valid standard error.
    """
    nu = nuis or cross_fit_nuisances(X, T, Y, **kw)
    mu0, mu1, e = nu["mu0"], nu["mu1"], nu["e"]
    phi = mu1 - mu0 + T * (Y - mu1) / e - (1 - T) * (Y - mu0) / (1 - e)
    return Estimate(phi.mean(), phi.std(ddof=1) / np.sqrt(len(Y)), info=dict(phi=phi, **nu))


# ---------------------------------------------------------------------------
# 4. Meta-learners (Kunzel, Sekhon, Bickel & Yu 2019)
# ---------------------------------------------------------------------------

def s_learner(X, T, Y, model=default_outcome_model):
    """One model ``mu(x, t)``, treatment as just another feature."""
    m = model().fit(np.column_stack([X, T]), Y)
    tau = lambda x: (m.predict(np.column_stack([x, np.ones(len(x))]))
                     - m.predict(np.column_stack([x, np.zeros(len(x))])))
    return Estimate(float(tau(X).mean()), cate=tau)


def t_learner(X, T, Y, model=default_outcome_model):
    """Two models, one per arm: ``tau(x) = mu1(x) - mu0(x)``."""
    m1 = model().fit(X[T == 1], Y[T == 1])
    m0 = model().fit(X[T == 0], Y[T == 0])
    tau = lambda x: m1.predict(x) - m0.predict(x)
    return Estimate(float(tau(X).mean()), cate=tau, info=dict(m0=m0, m1=m1))


def x_learner(X, T, Y, model=default_outcome_model, propensity_model=default_propensity_model):
    """X-learner: impute individual effects with the *other* arm's model, regress them,
    and blend the two CATE models with the propensity score ``g(x) = e(x)``.

    D1 = Y - mu0(X) on the treated, D0 = mu1(X) - Y on the controls;
    tau(x) = e(x) tau0(x) + (1 - e(x)) tau1(x).
    """
    t = t_learner(X, T, Y, model)
    m0, m1 = t.info["m0"], t.info["m1"]
    D1 = Y[T == 1] - m0.predict(X[T == 1])
    D0 = m1.predict(X[T == 0]) - Y[T == 0]
    tau1 = model().fit(X[T == 1], D1)
    tau0 = model().fit(X[T == 0], D0)
    pm = propensity_model().fit(X, T)
    tau = lambda x: (lambda g: g * tau0.predict(x) + (1 - g) * tau1.predict(x))(
        np.clip(pm.predict(x), 0.01, 0.99))
    return Estimate(float(tau(X).mean()), cate=tau)


# ---------------------------------------------------------------------------
# 5. Orthogonal learning
# ---------------------------------------------------------------------------

def dml_plr(X, T, Y, nuis=None, **kw):
    """Double/debiased ML for the partially linear model ``Y = theta T + g(X) + eps``.

    Residualize both Y and T on X with cross-fitted ML models (the FWL theorem with
    flexible first stages), then regress residual on residual. Under heterogeneous
    effects ``theta`` is again a variance-weighted average, like OLS.
    """
    nu = nuis or cross_fit_nuisances(X, T, Y, **kw)
    Yr, Tr = Y - nu["m"], T - nu["e"]
    theta = (Tr @ Yr) / (Tr @ Tr)
    psi = (Yr - theta * Tr) * Tr
    se = np.sqrt(np.mean(psi ** 2)) / np.mean(Tr ** 2) / np.sqrt(len(Y))
    return Estimate(theta, se)


def r_learner(X, T, Y, nuis=None, model=default_cate_model, **kw):
    """R-learner (Nie & Wager; the orthogonal CATE loss of Foster & Syrgkanis).

    minimize ``sum_i ((Y_i - m(X_i)) - tau(X_i) (T_i - e(X_i)))^2``, which equals a
    weighted regression of the pseudo-outcome ``(Y - m) / (T - e)`` on X with weights
    ``(T - e)^2``.
    """
    nu = nuis or cross_fit_nuisances(X, T, Y, **kw)
    Yr, Tr = Y - nu["m"], T - nu["e"]
    Tr = np.where(np.abs(Tr) < 1e-3, 1e-3 * np.sign(Tr + 1e-12), Tr)
    tm = model().fit(X, Yr / Tr, w=Tr ** 2)
    tau = tm.predict
    return Estimate(float(tau(X).mean()), cate=tau)


def dr_learner(X, T, Y, nuis=None, model=default_cate_model, **kw):
    """DR-learner: regress the AIPW pseudo-outcome ``phi`` on X to get a CATE model."""
    est = aipw(X, T, Y, nuis=nuis, **kw)
    tm = model().fit(X, est.info["phi"])
    return Estimate(est.ate, est.se, cate=tm.predict)
