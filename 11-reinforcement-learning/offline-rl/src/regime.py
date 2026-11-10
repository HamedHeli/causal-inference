"""A multi-visit treatment regime: the hospital example as a small finite-horizon MDP.

This is the sequential version of :mod:`src.hospital`. A patient visits the clinic
``H`` times. At each visit the doctor sees the current severity ``s`` in
``{0, 1, 2, 3, 4}`` (0 = healthy, 4 = critical) and chooses the drug (``a = 0``) or
surgery (``a = 1``). The reward of a visit is the patient's health after it,
``r = 1 - s'/4``, minus a small cost for surgery. The value of a policy is the expected
total health over the ``H`` visits.

As in the one-step hospital, surgery helps severe patients and harms mild ones. At
each visit the patient also has a hidden **frailty** ``u_t`` in ``{0, 1}`` (how the
patient looks that day, drawn afresh at each visit): frail patients recover less from
surgery. Frailty is never written into the dataset. If the doctors' choices depend on
it (``confounding > 0``: doctors avoid operating on patients who look frail), the logged
data are confounded in a way no method that only sees severity can detect. With
``confounding = 0`` the behavior policy depends on the recorded severity only, which is
the sequential version of conditional ignorability ("sequential randomization").

Because ``u_t`` is fresh at each visit, the recorded severity is a Markov state for any
policy that does not look at ``u``: averaging over frailty gives an ordinary MDP with
transitions ``(1 - p) P_0 + p P_1``.

Everything is tabular, so the true value of any policy is computed exactly by dynamic
programming (:func:`true_value`), and every estimator can be compared with the truth.

Policies are arrays ``pi[s, a] = P(a | s)`` (stationary, the same at every visit).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

N_S, N_A = 5, 2
SURGERY_COST = 0.05


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def transition_matrices(u):
    """``P[a, s, s']`` for a patient with frailty ``u``.

    Drug: severity drifts up with probability 0.35, down with 0.25 (the disease
    progresses slowly). Surgery: for severe patients a large chance of improving by one
    or two levels; for mild patients a risk of complications (worsening). Frailty cuts
    the benefit of surgery.
    """
    P = np.zeros((N_A, N_S, N_S))
    for s in range(N_S):
        up, down = 0.35, 0.25
        P[0, s, min(s + 1, N_S - 1)] += up
        P[0, s, max(s - 1, 0)] += down
        P[0, s, s] += 1 - up - down
        # surgery: success probability grows with severity, shrinks with frailty
        success = np.clip(0.15 + 0.17 * s - 0.35 * u, 0.02, 0.95)
        worse = np.clip(0.30 - 0.07 * s + 0.15 * u, 0.02, 0.9)
        stay = 1 - success - worse
        P[1, s, max(s - 2, 0)] += success * 0.5
        P[1, s, max(s - 1, 0)] += success * 0.5
        P[1, s, min(s + 1, N_S - 1)] += worse
        P[1, s, s] += stay
    return P


def reward(s_next, a):
    return 1.0 - s_next / (N_S - 1) - SURGERY_COST * a


def expected_reward(P):
    """``R[a, s] = E[r | s, a]``."""
    health = 1.0 - np.arange(N_S) / (N_S - 1)
    return np.einsum("asn,n->as", P, health) - SURGERY_COST * np.arange(N_A)[:, None]


def behavior_policy(slope=1.5, confounding=0.0, clip=0.03):
    """Doctors' policy ``pi_b[u, s, a]``: operate more on severe patients, and (with
    ``confounding > 0``) less on frail ones."""
    s = np.arange(N_S)
    out = np.zeros((2, N_S, N_A))
    for u in (0, 1):
        p1 = sigmoid(slope * (s - 2) - confounding * u)
        p1 = np.clip(p1, clip, 1 - clip)
        out[u, :, 1], out[u, :, 0] = p1, 1 - p1
    return out


class Regime(NamedTuple):
    H: int = 5
    p_frail: float = 0.4
    d0: np.ndarray = np.array([0.15, 0.25, 0.25, 0.2, 0.15])

    @property
    def P(self):
        return np.stack([transition_matrices(u) for u in (0, 1)])  # [u, a, s, s']

    @property
    def P_bar(self):
        """Transitions of the observed MDP for policies that do not see frailty."""
        return (1 - self.p_frail) * transition_matrices(0) + self.p_frail * transition_matrices(1)


def true_value(pi, regime=Regime()):
    """Exact value of a state policy ``pi[s, a]`` by backward induction on the observed
    MDP (frailty averaged out)."""
    P = regime.P_bar
    R = expected_reward(P)
    V = np.zeros(N_S)
    for _ in range(regime.H):
        Q = R + np.einsum("asn,n->as", P, V)          # [a, s]
        V = np.einsum("sa,as->s", pi, Q)
    return float(regime.d0 @ V)


def optimal_policy(regime=Regime()):
    """The best deterministic stationary policy (searched over all 2^5)."""
    vals = [true_value(pi, regime) for pi in all_deterministic_policies()]
    k = int(np.argmax(vals))
    return all_deterministic_policies()[k], vals[k]


def deterministic_policy(actions):
    pi = np.zeros((N_S, N_A))
    pi[np.arange(N_S), np.asarray(actions)] = 1.0
    return pi


def all_deterministic_policies():
    return [deterministic_policy([(k >> s) & 1 for s in range(N_S)]) for k in range(2 ** N_S)]


def describe(pi):
    return "".join("S" if pi[s, 1] > 0.5 else "d" for s in range(N_S))


class Dataset(NamedTuple):
    """``n`` logged episodes. Arrays have shape ``(n, H)``; ``s`` has ``H + 1`` columns.
    ``pb`` is the true behavior probability of each logged action (it depends on the
    hidden frailty when there is confounding); real logs rarely contain it, so the
    estimators estimate the propensity from the recorded states by default. ``u`` is
    kept only for diagnostics."""

    s: np.ndarray
    a: np.ndarray
    r: np.ndarray
    pb: np.ndarray
    u: np.ndarray


def simulate(n, regime=Regime(), pi_b=None, seed=0):
    """Roll out ``n`` episodes of the doctors' policy (``pi_b[u, s, a]``)."""
    rng = np.random.default_rng(seed)
    pi_b = behavior_policy() if pi_b is None else pi_b
    P = regime.P
    H = regime.H
    S = np.zeros((n, H + 1), int); A = np.zeros((n, H), int)
    R = np.zeros((n, H)); PB = np.zeros((n, H))
    U = (rng.uniform(size=(n, H)) < regime.p_frail).astype(int)
    S[:, 0] = rng.choice(N_S, size=n, p=regime.d0)
    for t in range(H):
        u = U[:, t]
        p1 = pi_b[u, S[:, t], 1]
        A[:, t] = (rng.uniform(size=n) < p1).astype(int)
        PB[:, t] = np.where(A[:, t] == 1, p1, 1 - p1)
        cum = np.cumsum(P[u, A[:, t], S[:, t]], axis=1)
        S[:, t + 1] = (rng.uniform(size=(n, 1)) > cum).sum(1)
        R[:, t] = reward(S[:, t + 1], A[:, t])
    return Dataset(S, A, R, PB, U)
