"""The three rules of do-calculus as graphical tests on a :class:`~src.dag.DAG`.

Notation (Pearl 2009, Ch. 3): ``G_{bar X}`` deletes edges into X, ``G_{underline T}``
deletes edges out of T. Each rule says when a term of ``P(Y | do(X), ...)`` can be
rewritten, and each test is a d-separation in a mutilated graph:

* **Rule 1** (insert / delete an observation T):
  ``P(Y | do(X), T, Z) = P(Y | do(X), Z)``  if  ``Y _||_ T | X, Z``  in ``G_{bar X}``.
* **Rule 2** (exchange action and observation):
  ``P(Y | do(X), do(T), Z) = P(Y | do(X), T, Z)``  if  ``Y _||_ T | X, Z``  in
  ``G_{bar X, underline T}``.  (With X empty: the backdoor criterion.)
* **Rule 3** (insert / delete an action):
  ``P(Y | do(X), do(T), Z) = P(Y | do(X), Z)``  if  ``Y _||_ T | X, Z``  in
  ``G_{bar X, bar T(Z)}``, where ``T(Z)`` are the T-nodes that are not ancestors of any
  Z-node in ``G_{bar X}``.
"""

from __future__ import annotations


def _s(v):
    return {v} if isinstance(v, str) else set(v)


def rule1(G, Y, T, X=(), Z=()):
    X, Z = _s(X), _s(Z)
    return G.do(X).d_separated(_s(Y), _s(T), X | Z)


def rule2(G, Y, T, X=(), Z=()):
    X, Z = _s(X), _s(Z)
    return G.do(X).cut_outgoing(_s(T)).d_separated(_s(Y), _s(T), X | Z)


def rule3(G, Y, T, X=(), Z=()):
    X, Z, T = _s(X), _s(Z), _s(T)
    GX = G.do(X)
    anc_Z = set().union(*(GX.ancestors(z) for z in Z)) if Z else set()
    TZ = {t for t in T if t not in anc_Z}
    return GX.do(TZ).d_separated(_s(Y), T, X | Z)
