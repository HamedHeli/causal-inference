"""Causal trees (Athey & Imbens 2016, "Recursive partitioning for heterogeneous causal
effects"), written from scratch in NumPy.

A causal tree partitions covariate space into leaves and estimates one treatment effect
per leaf, ``tau_hat(leaf) = mean(Y | T=1, leaf) - mean(Y | T=0, leaf)``. It differs from
a CART regression tree in two places:

1. **What a split optimizes.** We never observe ``tau_i``, so the split criterion
   cannot be "MSE of tau". Athey & Imbens show that, up to a constant that does not
   depend on the partition, minus the MSE of the leaf effects is
   ``E[tau_hat(X)^2]`` (reward splits that separate large from small effects) minus the
   variance of the leaf estimates (penalize leaves with few treated or few controls).

2. **Honesty.** One half of the data chooses the splits, the other half estimates the
   leaf effects. The partition is then independent of the estimation sample, so the
   leaf estimates are unbiased and their textbook confidence intervals are valid.

Criteria implemented (``criterion=``):

* ``"adaptive"`` (CT-A): ``(1/N) sum_i tau_hat(X_i)^2``, the "in-sample" fit. Splits
  always improve it, so depth / leaf size must be capped.
* ``"honest"`` (CT-H): the adaptive term minus
  ``(1/N_tr + 1/N_est) sum_leaves (S2_treat / p + S2_control / (1 - p))``. It stops on
  its own once a split's gain is smaller than the extra variance it creates.
* ``"tot"``: CART on the transformed outcome ``Y* = Y (T - p) / (p (1 - p))``, whose
  conditional mean is ``tau(x)``. A simple baseline that ignores the structure.

For observational data, pass propensities ``e``: units are then weighted by inverse
propensity inside each leaf (Section 6 of the paper).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Node:
    depth: int
    idx_tr: np.ndarray
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    tau: float = np.nan            # leaf effect (from the estimation sample if honest)
    se: float = np.nan
    n_est: int = 0
    leaf_id: int = -1

    @property
    def is_leaf(self):
        return self.left is None


def _arm_stats(Y, T, w):
    """Weighted means and variances of Y in each arm."""
    out = []
    for t in (1, 0):
        m = T == t
        ww, yy = w[m], Y[m]
        sw = ww.sum()
        if sw <= 0 or m.sum() < 2:
            return None
        mu = (ww * yy).sum() / sw
        var = (ww * (yy - mu) ** 2).sum() / sw
        out += [mu, var, m.sum()]
    return out  # mu1, var1, n1, mu0, var0, n0


def leaf_effect(Y, T, w=None):
    """Difference in (weighted) means and its standard error within a leaf."""
    w = np.ones(len(Y)) if w is None else w
    s = _arm_stats(Y, T, w)
    if s is None:
        return np.nan, np.nan
    mu1, v1, n1, mu0, v0, n0 = s
    return mu1 - mu0, np.sqrt(v1 / n1 + v0 / n0)


class CausalTree:
    def __init__(self, criterion="honest", max_depth=6, min_leaf=25, n_thresholds=32,
                 honest=True, est_fraction=0.5, alpha=0.0, seed=0):
        """``alpha`` is a cost per extra leaf (cost-complexity penalty): a split is kept
        only if it improves the criterion by more than ``alpha``. Choose it with
        :func:`cv_alpha`."""
        self.alpha = alpha
        self.criterion, self.max_depth, self.min_leaf = criterion, max_depth, min_leaf
        self.n_thresholds, self.honest, self.est_fraction, self.seed = (
            n_thresholds, honest, est_fraction, seed)

    # ---- the split criterion ------------------------------------------------
    def _leaf_score(self, idx):
        """Contribution of one leaf to the criterion (additive over leaves)."""
        Y, T, w = self.Y[idx], self.T[idx], self.w[idx]
        if self.criterion == "tot":
            ys = self.Ystar[idx]
            return len(idx) / self.n_tr * ys.mean() ** 2
        s = _arm_stats(Y, T, w)
        if s is None or min(s[2], s[5]) < self.min_arm:
            return -np.inf
        mu1, v1, n1, mu0, v0, n0 = s
        fit = len(idx) / self.n_tr * (mu1 - mu0) ** 2
        if self.criterion == "adaptive":
            return fit
        p = self.p
        return fit - (1 / self.n_tr + 1 / self.n_est) * (v1 / p + v0 / (1 - p))

    def _best_split(self, node):
        idx = node.idx_tr
        parent = self._leaf_score(idx)
        best = (self.alpha, None, None)
        for j in range(self.X.shape[1]):
            x = self.X[idx, j]
            qs = np.unique(np.quantile(x, np.linspace(0, 1, self.n_thresholds + 2)[1:-1]))
            for c in qs:
                L, R = idx[x <= c], idx[x > c]
                if len(L) < self.min_leaf or len(R) < self.min_leaf:
                    continue
                gain = self._leaf_score(L) + self._leaf_score(R) - parent
                if gain > best[0] + 1e-12:
                    best = (gain, j, c)
        return best

    def _grow(self, node):
        if node.depth >= self.max_depth:
            return
        gain, j, c = self._best_split(node)
        if j is None:
            return
        x = self.X[node.idx_tr, j]
        node.feature, node.threshold = j, c
        node.left = Node(node.depth + 1, node.idx_tr[x <= c])
        node.right = Node(node.depth + 1, node.idx_tr[x > c])
        self._grow(node.left)
        self._grow(node.right)

    # ---- public API -----------------------------------------------------------
    def fit(self, X, T, Y, e=None):
        n = len(Y)
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)
        if self.honest:
            n_est = int(self.est_fraction * n)
            est, tr = perm[:n_est], perm[n_est:]
        else:
            est = tr = perm
        e = np.full(n, T.mean()) if e is None else e
        # inverse-propensity weights make each leaf's arms comparable (Section 6)
        w = np.where(T == 1, T.mean() / e, (1 - T.mean()) / (1 - e))
        self.X, self.T, self.Y, self.w = X[tr], T[tr], Y[tr], w[tr]
        self.p = T.mean()
        self.Ystar = self.Y * (self.T - self.p) / (self.p * (1 - self.p)) * self.w
        self.n_tr, self.n_est = len(tr), len(est)
        self.min_arm = max(2, self.min_leaf // 10)
        self.root = Node(0, np.arange(len(tr)))
        self._grow(self.root)

        # estimation: drop the estimation sample down the tree
        self.leaves = []
        leaf_of = self.apply(X[est])
        for k, leaf in enumerate(self._iter_leaves(self.root)):
            leaf.leaf_id = k
            m = leaf_of == k
            if m.sum() == 0:
                leaf.tau, leaf.se = np.nan, np.nan
            else:
                leaf.tau, leaf.se = leaf_effect(Y[est][m], T[est][m], w[est][m])
            leaf.n_est = int(m.sum())
            self.leaves.append(leaf)
        # fall back to the parent-free overall effect for empty arms
        overall, _ = leaf_effect(Y[est], T[est], w[est])
        for leaf in self.leaves:
            if not np.isfinite(leaf.tau):
                leaf.tau, leaf.se = overall, np.inf
        return self

    def _iter_leaves(self, node):
        if node.is_leaf:
            yield node
        else:
            yield from self._iter_leaves(node.left)
            yield from self._iter_leaves(node.right)

    def apply(self, X):
        """Leaf index of each row of X (leaves numbered left to right)."""
        ids = {id(l): k for k, l in enumerate(self._iter_leaves(self.root))}
        out = np.empty(len(X), int)
        for i, x in enumerate(X):
            node = self.root
            while not node.is_leaf:
                node = node.left if x[node.feature] <= node.threshold else node.right
            out[i] = ids[id(node)]
        return out

    def predict(self, X):
        taus = np.array([l.tau for l in self.leaves])
        return taus[self.apply(X)]

    def predict_se(self, X):
        ses = np.array([l.se for l in self.leaves])
        return ses[self.apply(X)]

    @property
    def n_leaves(self):
        return sum(1 for _ in self._iter_leaves(self.root))

    def split_features(self, node=None):
        node = node or self.root
        if node.is_leaf:
            return []
        return [node.feature] + self.split_features(node.left) + self.split_features(node.right)

    def describe(self, names=None, node=None, indent=""):
        """A text rendering of the tree."""
        node = node or self.root
        names = names or [f"x{j}" for j in range(self.X.shape[1])]
        if node.is_leaf:
            return f"{indent}leaf {node.leaf_id}: tau = {node.tau:+.3f} ± {1.96 * node.se:.3f}  (n_est = {node.n_est})\n"
        s = f"{indent}{names[node.feature]} <= {node.threshold:.2f}\n"
        s += self.describe(names, node.left, indent + "  ")
        s += f"{indent}{names[node.feature]} >  {node.threshold:.2f}\n"
        s += self.describe(names, node.right, indent + "  ")
        return s


def transformed_outcome(T, Y, e):
    """``Y* = Y (T - e) / (e (1 - e))``, with ``E[Y* | X] = tau(X)`` under unconfoundedness."""
    return Y * (T - e) / (e * (1 - e))


def cv_alpha(X, T, Y, alphas, e=None, k=5, seed=0, **tree_kw):
    """Pick the per-leaf penalty by k-fold cross-validation.

    The held-out loss is ``mean((Y*_i - tau_hat(X_i))^2)``. Because ``E[Y* | X] = tau(X)``,
    this equals the MSE of ``tau_hat`` plus a constant that does not depend on the tree,
    which is how the paper gets around never observing ``tau_i`` in the validation fold.
    Returns ``(best_alpha, mean_losses)``.
    """
    n = len(Y)
    e = np.full(n, T.mean()) if e is None else e
    ystar = transformed_outcome(T, Y, e)
    folds = np.array_split(np.random.default_rng(seed).permutation(n), k)
    losses = np.zeros(len(alphas))
    for j in range(k):
        te = folds[j]; tr = np.concatenate([folds[i] for i in range(k) if i != j])
        for a_i, a in enumerate(alphas):
            t = CausalTree(alpha=a, seed=seed, **tree_kw).fit(X[tr], T[tr], Y[tr], e=e[tr])
            losses[a_i] += np.mean((ystar[te] - t.predict(X[te])) ** 2) / k
    return alphas[int(np.argmin(losses))], losses
