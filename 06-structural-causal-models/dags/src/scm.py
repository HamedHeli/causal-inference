"""Structural causal models you can sample, intervene on, and query counterfactually.

An :class:`SCM` is an ordered list of structural equations ``V := f_V(parents, U_V)``
with independent exogenous noise ``U_V``. The three rungs of Pearl's ladder are three
ways of running the same equations:

* **observational** (:meth:`SCM.sample`): draw the noise, evaluate the equations;
* **interventional** (:meth:`SCM.sample` with ``do=``): replace ``f_T`` by a constant,
  keep everything else (the noise is redrawn, so we only learn the *distribution*);
* **counterfactual** (:meth:`SCM.counterfactual`): draw the noise *once* per unit
  and evaluate the model under several interventions with that same noise. A unit
  *is* its noise vector, so ``Y_i(1)`` and ``Y_i(0)`` are both defined for every unit,
  which is how joint quantities like ``P(Y(1) = 1, Y(0) = 0)`` get a meaning.
"""

from __future__ import annotations

import numpy as np


class SCM:
    def __init__(self, equations, noise):
        """``equations``: dict ``name -> f(values: dict, u: array) -> array``, in
        topological order. ``noise``: dict ``name -> sampler(rng, n) -> array``."""
        self.equations, self.noise = equations, noise

    def draw_noise(self, n, seed=0):
        rng = np.random.default_rng(seed)
        return {v: self.noise[v](rng, n) for v in self.equations}

    def evaluate(self, U, do=None):
        do = do or {}
        n = len(next(iter(U.values())))
        vals = {}
        for v, f in self.equations.items():
            vals[v] = np.full(n, do[v], dtype=float) if v in do else np.asarray(f(vals, U[v]), float)
        return vals

    def sample(self, n, do=None, seed=0):
        return self.evaluate(self.draw_noise(n, seed), do)

    def counterfactual(self, n, interventions, seed=0):
        """Evaluate each intervention in ``interventions`` (a dict name -> do-dict) on
        the same units. Also returns the factual world under key ``"factual"``."""
        U = self.draw_noise(n, seed)
        out = {"factual": self.evaluate(U)}
        for k, do in interventions.items():
            out[k] = self.evaluate(U, do)
        return out


def bernoulli(p):
    return lambda rng, n: (rng.uniform(size=n) < p).astype(float)


def uniform():
    return lambda rng, n: rng.uniform(size=n)


def probabilities_of_causation(Y1, Y0, T, Y):
    """PN, PS, PNS from per-unit counterfactuals (only possible in a simulator)."""
    Y1, Y0, T, Y = (np.asarray(a).astype(bool) for a in (Y1, Y0, T, Y))
    pns = np.mean(Y1 & ~Y0)
    pn = np.mean(~Y0[T & Y]) if (T & Y).any() else np.nan
    ps = np.mean(Y1[~T & ~Y]) if (~T & ~Y).any() else np.nan
    return dict(PN=pn, PS=ps, PNS=pns)


def pns_bounds(p_y1_do1, p_y1_do0):
    """Tian-Pearl bounds on PNS from experimental data alone:
    ``max(0, P(y|do(1)) - P(y|do(0))) <= PNS <= min(P(y|do(1)), P(y'|do(0)))``."""
    return max(0.0, p_y1_do1 - p_y1_do0), min(p_y1_do1, 1 - p_y1_do0)
