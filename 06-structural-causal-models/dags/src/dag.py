"""Causal DAGs as plain Python: paths, d-separation, the backdoor criterion and the
graph surgeries used by the do-calculus.

A :class:`DAG` is a set of named nodes and directed edges ``parent -> child``. Nothing
here is probabilistic: these are the graphical tests the lectures run by eye, written as
code so they can be checked on graphs too large to eyeball.

Main entry points:

* :meth:`DAG.paths` lists every undirected path between two nodes, and
  :meth:`DAG.path_is_open` applies the blocking rules (mediator / confounder
  conditioned on, or collider with no conditioned descendant).
* :meth:`DAG.d_separated` is "all paths are blocked".
* :meth:`DAG.backdoor_paths`, :meth:`DAG.satisfies_backdoor` and
  :meth:`DAG.backdoor_sets` implement the backdoor criterion.
* :meth:`DAG.do` (remove edges *into* a set, ``G_{\\bar X}``) and
  :meth:`DAG.cut_outgoing` (remove edges *out of* a set, ``G_{\\underline X}``) are the
  two mutilations the three rules of do-calculus are stated on; :mod:`src.docalc` uses
  them.
"""

from __future__ import annotations

from itertools import chain, combinations


class DAG:
    def __init__(self, edges, nodes=(), latent=()):
        """``edges``: iterable of ``(parent, child)``; ``latent``: unobserved nodes."""
        self.edges = set(edges)
        self.nodes = set(nodes) | {a for e in self.edges for a in e}
        self.latent = set(latent)
        self._check_acyclic()

    # ---- basic relations ----------------------------------------------------
    def parents(self, v):
        return {a for a, b in self.edges if b == v}

    def children(self, v):
        return {b for a, b in self.edges if a == v}

    def descendants(self, v):
        out, stack = set(), [v]
        while stack:
            for c in self.children(stack.pop()):
                if c not in out:
                    out.add(c); stack.append(c)
        return out

    def ancestors(self, v):
        out, stack = set(), [v]
        while stack:
            for p in self.parents(stack.pop()):
                if p not in out:
                    out.add(p); stack.append(p)
        return out

    def _check_acyclic(self):
        for v in self.nodes:
            if v in self.descendants(v):
                raise ValueError(f"cycle through {v}")

    def topological_order(self):
        order, seen = [], set()

        def visit(v):
            if v in seen:
                return
            seen.add(v)
            for p in sorted(self.parents(v)):
                visit(p)
            order.append(v)

        for v in sorted(self.nodes):
            visit(v)
        return order

    # ---- paths and d-separation --------------------------------------------
    def paths(self, a, b):
        """All simple undirected paths from ``a`` to ``b``, as node lists."""
        nbrs = {v: self.parents(v) | self.children(v) for v in self.nodes}
        out, stack = [], [[a]]
        while stack:
            p = stack.pop()
            if p[-1] == b:
                out.append(p); continue
            for n in sorted(nbrs[p[-1]]):
                if n not in p:
                    stack.append(p + [n])
        return sorted(out, key=len)

    def node_role(self, path, i):
        """Role of ``path[i]`` on the path: 'chain', 'fork' or 'collider'."""
        prev, v, nxt = path[i - 1], path[i], path[i + 1]
        into_from_prev = (prev, v) in self.edges
        into_from_next = (nxt, v) in self.edges
        if into_from_prev and into_from_next:
            return "collider"
        if not into_from_prev and not into_from_next:
            return "fork"
        return "chain"

    def path_is_open(self, path, Z=()):
        """Blocking rules from the lecture.

        Blocked if a non-collider on the path is in Z, or a collider on the path has
        neither itself nor any descendant in Z.
        """
        Z = set(Z)
        for i in range(1, len(path) - 1):
            v, role = path[i], self.node_role(path, i)
            if role == "collider":
                if v not in Z and not (self.descendants(v) & Z):
                    return False
            elif v in Z:
                return False
        return True

    def d_separated(self, A, B, Z=()):
        A = {A} if isinstance(A, str) else set(A)
        B = {B} if isinstance(B, str) else set(B)
        return not any(self.path_is_open(p, Z) for a in A for b in B for p in self.paths(a, b))

    def explain(self, a, b, Z=()):
        """Human-readable list of paths and whether each is open given Z."""
        lines = []
        for p in self.paths(a, b):
            s = p[0]
            for u, v in zip(p, p[1:]):
                s += (" -> " if (u, v) in self.edges else " <- ") + v
            lines.append(f"{'OPEN   ' if self.path_is_open(p, Z) else 'blocked'}  {s}")
        return "\n".join(lines)

    # ---- backdoor criterion --------------------------------------------------
    def backdoor_paths(self, t, y):
        """Paths from t to y that start with an arrow *into* t."""
        return [p for p in self.paths(t, y) if (p[1], t) in self.edges]

    def satisfies_backdoor(self, t, y, Z):
        Z = set(Z)
        if Z & self.descendants(t):
            return False
        if Z & self.latent:
            return False
        return not any(self.path_is_open(p, Z) for p in self.backdoor_paths(t, y))

    def backdoor_sets(self, t, y, max_size=None, minimal=False):
        """Every observed adjustment set satisfying the backdoor criterion."""
        cand = sorted(self.nodes - {t, y} - self.latent - self.descendants(t))
        max_size = len(cand) if max_size is None else max_size
        sets = [set(s) for s in chain.from_iterable(combinations(cand, k) for k in range(max_size + 1))
                if self.satisfies_backdoor(t, y, s)]
        if minimal:
            sets = [s for s in sets if not any(o < s for o in sets)]
        return sets

    # ---- graph surgery -------------------------------------------------------
    def do(self, X):
        """``G_{\\bar X}``: delete every edge pointing *into* X (the do-operator)."""
        X = {X} if isinstance(X, str) else set(X)
        return DAG({(a, b) for a, b in self.edges if b not in X}, self.nodes, self.latent)

    def cut_outgoing(self, X):
        """``G_{\\underline X}``: delete every edge pointing *out of* X."""
        X = {X} if isinstance(X, str) else set(X)
        return DAG({(a, b) for a, b in self.edges if a not in X}, self.nodes, self.latent)

    # ---- drawing ---------------------------------------------------------------
    def draw(self, ax, pos, Z=(), do=(), title=None, highlight_path=None):
        """Matplotlib drawing: conditioned nodes grey, intervened nodes with a thick box,
        latent nodes dashed."""
        import matplotlib.patches as mp

        Z, do = set(Z), set(do)
        hl = set(zip(highlight_path, highlight_path[1:])) if highlight_path else set()
        hl |= {(b, a) for a, b in hl}
        for a, b in self.edges:
            (x0, y0), (x1, y1) = pos[a], pos[b]
            col = "tab:red" if (a, b) in hl else "0.3"
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2 if (a, b) in hl else 1.4,
                                        shrinkA=16, shrinkB=16,
                                        linestyle="--" if a in self.latent else "-"))
        for v, (x, y) in pos.items():
            if v not in self.nodes:
                continue
            fc = "0.75" if v in Z else "white"
            ax.add_patch(mp.Circle((x, y), 0.32, fc=fc, ec="black", lw=3 if v in do else 1.2,
                                   ls="--" if v in self.latent else "-", zorder=3))
            ax.text(x, y, v, ha="center", va="center", fontsize=10, zorder=4)
        xs, ys = zip(*pos.values())
        ax.set_xlim(min(xs) - 0.7, max(xs) + 0.7); ax.set_ylim(min(ys) - 0.7, max(ys) + 0.7)
        ax.set_aspect("equal"); ax.axis("off")
        if title:
            ax.set_title(title, fontsize=10)
