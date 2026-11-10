"""Off-policy evaluation (OPE) for the finite-horizon treatment regime, and its
causal-inference names.

Given ``n`` logged episodes from the doctors' (behavior) policy ``pi_b``, estimate the
value ``V(pi_e) = E[sum_t r_t]`` of a different (evaluation) policy ``pi_e`` *without
running it*. In causal language: ``V(pi_e)`` is the mean potential outcome of a
**dynamic treatment regime**, and the logged episodes are an observational study.

=====================================  =========================================
estimator here                          causal-inference counterpart
=====================================  =========================================
:func:`importance_sampling` (IS)        IPW with the product of propensities
:func:`per_decision_is` (PDIS)          IPW, one weight per time step
``weighted=True`` variants (WIS)        Hajek (normalized) IPW
:func:`model_based`                     g-computation / Robins' g-formula
:func:`doubly_robust` (Jiang & Li)      AIPW, applied recursively
=====================================  =========================================

Behavior propensities: by default ``P(a | s)`` is *estimated* from the logged
(state, action) counts, like a fitted propensity score. Passing ``pb=data.pb`` uses the
true logging probabilities instead (which, under hidden confounding, depend on a
variable the estimator never sees).
"""

from __future__ import annotations

import numpy as np

from .regime import N_A, N_S


# ---------------------------------------------------------------------------
# nuisances: behavior policy (propensity) and the outcome model (Q-function)
# ---------------------------------------------------------------------------

def estimate_behavior(data, prior=0.5):
    """``pi_b_hat[s, a]`` from counts, with a small symmetric prior."""
    c = np.full((N_S, N_A), prior)
    np.add.at(c, (data.s[:, :-1].ravel(), data.a.ravel()), 1)
    return c / c.sum(1, keepdims=True)


def logged_propensities(data, pi_b_hat=None):
    """``P_b(a_t | s_t)`` for every logged step, from the estimated behavior policy."""
    pi_b_hat = estimate_behavior(data) if pi_b_hat is None else pi_b_hat
    return pi_b_hat[data.s[:, :-1], data.a]


def ratios(data, pi_e, pb=None):
    """Per-step importance ratios ``rho_t = pi_e(a_t|s_t) / pi_b(a_t|s_t)``."""
    pb = logged_propensities(data) if pb is None else pb
    return pi_e[data.s[:, :-1], data.a] / pb


def fit_model(data, unseen="stay"):
    """Empirical MDP: ``P_hat[a, s, s']``, ``R_hat[a, s]``, and visit counts ``N[a, s]``.

    Unseen (s, a) pairs have no data: ``unseen="stay"`` guesses "nothing happens and the
    reward is today's health", the kind of arbitrary extrapolation a model must make
    where positivity fails.
    """
    N = np.zeros((N_A, N_S)); T = np.zeros((N_A, N_S, N_S)); Rs = np.zeros((N_A, N_S))
    s, a, s2, r = data.s[:, :-1].ravel(), data.a.ravel(), data.s[:, 1:].ravel(), data.r.ravel()
    np.add.at(N, (a, s), 1); np.add.at(T, (a, s, s2), 1); np.add.at(Rs, (a, s), r)
    P = np.where(N[..., None] > 0, T / np.maximum(N, 1)[..., None], 0.0)
    R = np.where(N > 0, Rs / np.maximum(N, 1), 0.0)
    for a_ in range(N_A):
        for s_ in range(N_S):
            if N[a_, s_] == 0:
                P[a_, s_, s_] = 1.0
                R[a_, s_] = 1 - s_ / (N_S - 1)
    return P, R, N


def q_functions(P, R, pi_e, H):
    """Backward induction: ``Q[t, a, s]`` and ``V[t, s]`` of ``pi_e`` in the model."""
    Q = np.zeros((H, N_A, N_S)); V = np.zeros((H + 1, N_S))
    for t in reversed(range(H)):
        Q[t] = R + np.einsum("asn,n->as", P, V[t + 1])
        V[t] = np.einsum("sa,as->s", pi_e, Q[t])
    return Q, V


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------

def importance_sampling(data, pi_e, pb=None, weighted=False):
    """Trajectory-wise IS: weight each episode's return by ``prod_t rho_t``.

    The causal reading: each logged episode is a "unit"; the product of ratios is the
    inverse of the probability that the doctors would have followed ``pi_e`` along it.
    Unbiased (``weighted=False``) but the variance grows exponentially with the horizon.
    """
    w = np.prod(ratios(data, pi_e, pb), axis=1)
    G = data.r.sum(1)
    if weighted:
        return float((w * G).sum() / w.sum()) if w.sum() > 0 else 0.0
    return float(np.mean(w * G))


def per_decision_is(data, pi_e, pb=None, weighted=False):
    """Per-decision IS (Precup et al. 2000): reward ``r_t`` only needs the ratios up to
    ``t``, because later actions cannot have caused it."""
    w = np.cumprod(ratios(data, pi_e, pb), axis=1)        # (n, H)
    if weighted:
        ws = w.sum(0)
        return float(np.sum(np.where(ws > 0, (w * data.r).sum(0) / np.where(ws > 0, ws, 1), 0.0)))
    return float(np.mean((w * data.r).sum(1)))


def model_based(data, pi_e, H=None):
    """Plug-in: fit the MDP, evaluate ``pi_e`` in it (the g-formula with a saturated
    tabular model). With a tabular model this coincides with fitted Q evaluation (FQE),
    the standard deep-RL baseline that regresses ``r_t + V(s_{t+1})`` on ``(s_t, a_t)``."""
    H = data.a.shape[1] if H is None else H
    P, R, _ = fit_model(data)
    _, V = q_functions(P, R, pi_e, H)
    d0 = np.bincount(data.s[:, 0], minlength=N_S) / len(data.s)
    return float(d0 @ V[0])


def doubly_robust(data, pi_e, pb=None, model=None):
    """Doubly robust OPE (Jiang & Li 2016), finite horizon.

    ``V_DR^H = 0``;  ``V_DR^t = V_hat_t(s_t) + rho_t (r_t + V_DR^{t+1} - Q_hat_t(s_t, a_t))``.

    This is AIPW applied at every step: the model's prediction plus an importance-weighted
    correction of its residual. Unbiased if the propensities are right, and low-variance if
    the Q-model is good.
    """
    H = data.a.shape[1]
    P, R, _ = fit_model(data) if model is None else model
    Q, V = q_functions(P, R, pi_e, H)
    rho = ratios(data, pi_e, pb)
    n = len(data.s)
    v = np.zeros(n)
    for t in reversed(range(H)):
        s, a = data.s[:, t], data.a[:, t]
        v = V[t][s] + rho[:, t] * (data.r[:, t] + v - Q[t][a, s])
    return float(v.mean())


ESTIMATORS = {
    "IS": importance_sampling,
    "WIS": lambda d, p, pb=None: importance_sampling(d, p, pb, weighted=True),
    "PDIS": per_decision_is,
    "WPDIS": lambda d, p, pb=None: per_decision_is(d, p, pb, weighted=True),
    "model-based (g-formula)": lambda d, p, pb=None: model_based(d, p),
    "doubly robust": doubly_robust,
}
