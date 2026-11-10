"""Small NumPy learners used as "nuisance models" by the causal estimators.

Causal ML methods (IPW, doubly robust, meta-learners, DML, ...) are recipes that call an
off-the-shelf regressor or classifier as a sub-routine. To keep every line visible this
repository writes those sub-routines in NumPy instead of importing scikit-learn:

* :class:`Ridge`: linear ridge regression on raw features (with an intercept),
* :class:`KernelRidge`: ridge on random Fourier features, i.e. an approximate RBF
  kernel regression, flexible enough to fit the nonlinear hospital,
* :class:`Logistic`: L2-penalised logistic regression fitted by Newton's method,
  optionally on random Fourier features too.

All three follow the same ``fit(X, y, w=None) -> self`` / ``predict(X)`` interface, and
accept sample weights (needed by the X-learner, the R-learner and weighted regression).
"""

from __future__ import annotations

import numpy as np


def add_intercept(X):
    return np.column_stack([np.ones(len(X)), X])


def weighted_lstsq(F, y, w=None, lam=0.0, penalize_first=False):
    """Solve ``argmin_b sum_i w_i (y_i - F_i b)^2 + lam |b|^2``.

    ``(F^T W F + lam I)^{-1} F^T W y``: the weighted normal equations from the
    propensity lecture.
    """
    w = np.ones(len(y)) if w is None else w
    FW = F * w[:, None]
    P = lam * np.eye(F.shape[1])
    if not penalize_first:
        P[0, 0] = 0.0
    return np.linalg.solve(FW.T @ F + P + 1e-10 * np.eye(F.shape[1]), FW.T @ y)


class Standardizer:
    def fit(self, X):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        return self

    def __call__(self, X):
        return (X - self.mu) / self.sd


class Ridge:
    """Linear regression ``y ~ 1 + X`` with an optional ridge penalty."""

    def __init__(self, lam=0.0):
        self.lam = lam

    def fit(self, X, y, w=None):
        self.std = Standardizer().fit(X)
        self.coef = weighted_lstsq(add_intercept(self.std(X)), y, w, self.lam)
        return self

    def predict(self, X):
        return add_intercept(self.std(X)) @ self.coef


class RFF:
    """Random Fourier features ``sqrt(2/D) cos(X W / ell + b)`` for an RBF kernel."""

    def __init__(self, d, n_features=200, lengthscale=1.0, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(size=(d, n_features)) / lengthscale
        self.b = rng.uniform(0, 2 * np.pi, n_features)
        self.scale = np.sqrt(2.0 / n_features)

    def __call__(self, X):
        return self.scale * np.cos(X @ self.W + self.b)


class KernelRidge:
    """Approximate RBF kernel ridge regression (ridge on random Fourier features).

    The raw standardized features are kept next to the Fourier features, so the model
    can represent linear trends exactly and nonlinear ones approximately.
    """

    def __init__(self, n_features=200, lengthscale=1.0, lam=1e-2, seed=0):
        self.n_features, self.lengthscale, self.lam, self.seed = n_features, lengthscale, lam, seed

    def _features(self, X):
        Z = self.std(X)
        return np.column_stack([np.ones(len(X)), Z, self.rff(Z)])

    def fit(self, X, y, w=None):
        self.std = Standardizer().fit(X)
        self.rff = RFF(X.shape[1], self.n_features, self.lengthscale, self.seed)
        self.coef = weighted_lstsq(self._features(X), y, w, self.lam)
        return self

    def predict(self, X):
        return self._features(X) @ self.coef


class Logistic:
    """L2-penalised logistic regression, fitted with Newton's method (IRLS).

    With ``n_features > 0`` it runs on random Fourier features, which gives a flexible
    propensity model ``e(x) = P(T = 1 | X = x)``.
    """

    def __init__(self, lam=1e-3, n_features=0, lengthscale=1.0, seed=0, n_iter=50):
        self.lam, self.n_features, self.lengthscale, self.seed, self.n_iter = (
            lam, n_features, lengthscale, seed, n_iter)

    def _features(self, X):
        Z = self.std(X)
        cols = [np.ones(len(X)), Z]
        if self.n_features:
            cols.append(self.rff(Z))
        return np.column_stack(cols)

    def fit(self, X, y, w=None):
        self.std = Standardizer().fit(X)
        if self.n_features:
            self.rff = RFF(X.shape[1], self.n_features, self.lengthscale, self.seed)
        F = self._features(X)
        w = np.ones(len(y)) if w is None else w
        b = np.zeros(F.shape[1])
        P = self.lam * len(y) * np.eye(F.shape[1]); P[0, 0] = 0
        for _ in range(self.n_iter):
            p = 1 / (1 + np.exp(-np.clip(F @ b, -30, 30)))
            g = F.T @ (w * (p - y)) + P @ b
            H = (F * (w * p * (1 - p))[:, None]).T @ F + P + 1e-8 * np.eye(len(b))
            step = np.linalg.solve(H, g)
            b -= step
            if np.abs(step).max() < 1e-8:
                break
        self.coef = b
        return self

    def predict(self, X):
        return 1 / (1 + np.exp(-np.clip(self._features(X) @ self.coef, -30, 30)))


def kfold(n, k=5, seed=0):
    """Yield ``(train_idx, test_idx)`` for k-fold cross-fitting."""
    idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, k)
    for j in range(k):
        yield np.concatenate([folds[i] for i in range(k) if i != j]), folds[j]
