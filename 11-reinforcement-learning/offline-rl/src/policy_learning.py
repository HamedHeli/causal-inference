"""Offline policy learning: choose a treatment policy from logged data alone.

Three families, each with a causal-inference reading:

* :func:`behavior_cloning`: imitate the doctors. No causal question is asked, so it
  inherits whatever the doctors do, good or bad.
* :func:`fitted_q_iteration`: optimize in the fitted model (certainty equivalence).
  Unbiased where the data are dense, but the ``max`` over actions seeks out exactly the
  (state, action) pairs where the model is most wrong: a **positivity** problem.
  ``pessimism > 0`` subtracts ``pessimism / sqrt(N(s, a))`` from the reward (a lower
  confidence bound), the idea behind CQL, BCQ-style constraints and LCB algorithms.
* :func:`ope_argmax`: evaluate every candidate policy with an off-policy estimator and
  pick the best. Selecting the maximum of noisy estimates over-states the winner's value
  (the optimizer's curse, the same "winner's curse" honest causal trees fix);
  ``honest=True`` selects on one half of the episodes and estimates on the other.
"""

from __future__ import annotations

import numpy as np

from .ope import doubly_robust, estimate_behavior, fit_model, q_functions
from .regime import N_A, N_S, all_deterministic_policies


def behavior_cloning(data):
    return estimate_behavior(data, prior=1e-3)


def greedy(Q):
    """Deterministic policy from ``Q[a, s]``."""
    pi = np.zeros((N_S, N_A))
    pi[np.arange(N_S), Q.argmax(0)] = 1.0
    return pi


def fitted_q_iteration(data, pessimism=0.0, unseen_penalty=1.0):
    """Value iteration in the fitted model, then the greedy policy of the first step.

    With ``pessimism`` the reward of (s, a) becomes ``R_hat - pessimism / sqrt(N)``, and
    unseen pairs get ``-unseen_penalty``.
    """
    H = data.a.shape[1]
    P, R, N = fit_model(data)
    if pessimism > 0:
        R = np.where(N > 0, R - pessimism / np.sqrt(np.maximum(N, 1)), -unseen_penalty)
    V = np.zeros(N_S)
    for _ in range(H):
        Q = R + np.einsum("asn,n->as", P, V)
        V = Q.max(0)
    return greedy(Q)


def ope_argmax(data, estimator=doubly_robust, honest=False, seed=0):
    """Pick the deterministic policy with the highest estimated value.

    Returns ``(policy, reported_value)``. With ``honest=True`` the choice uses one half of
    the episodes and the reported value is re-estimated on the other half.
    """
    cands = all_deterministic_policies()
    n = len(data.s)
    if honest:
        idx = np.random.default_rng(seed).permutation(n)
        A, B = idx[: n // 2], idx[n // 2:]
        sub = lambda I: type(data)(*(x[I] for x in data))
        da, db = sub(A), sub(B)
        vals = [estimator(da, pi) for pi in cands]
        k = int(np.argmax(vals))
        return cands[k], estimator(db, cands[k])
    vals = [estimator(data, pi) for pi in cands]
    k = int(np.argmax(vals))
    return cands[k], vals[k]
