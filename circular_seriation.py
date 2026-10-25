"""
Strict circular seriation with Q/PQ-trees.

Implements the Recursive Seriation algorithm of
    S. Armstrong, C. Guzman, C. A. Sing Long,
    "An Optimal Algorithm for Strict Circular Seriation",
    SIAM J. Math. Data Sci. 3(4), 2021; arXiv:2106.05944, Section 5,
and applies it to the BROOQS F-matrix (a similarity matrix).

Input convention of the paper: a dissimilarity D. A similarity F is converted by
D_ij = c - F_ij (i != j), D_ii = 0, with c = 2 * max_{i!=j} F_ij + 1; only comparisons
of D-values are used, so any c > max F gives the same result. For an NMSC F-matrix,
c = 2*binom(n-2, 2) reproduces the Kalmanson dissimilarity D of the BROOQS notes.

Definitions used (paper, Sec. 3):
  * circular order: indices around a cycle, up to rotation and reflection (Dih_n);
  * D is strict circular Robinson (strict CR) in an order pi if for every i the
    sequence j -> D(pi_i, pi_{i+j}), j = 1..n-1, is strictly unimodal (Prop. 3.7):
    strictly increasing up to a mode and strictly decreasing after it, where the mode
    may be a pair of two equal, adjacent values (the paper's m1 < m2 case, p. 1229).
    For a similarity F this reads: each row of F strictly decreases, then strictly
    increases, around the cycle, with at most two equal adjacent minima.

Algorithm (paper, Algorithms 5.1-5.7):
  * Trees are Q-trees (all internal nodes are Q-nodes). A Q-node's children may be
    reversed as a block. A node is "fixed" into its parent when its orientation is
    decided: its children are spliced into the parent's child list. A node whose
    orientation cannot be decided is kept as a genuine Q-node of the solution tree.
  * B(T) (border candidates, Alg. 5.3): leaves that can be extreme in T;
    B(T) = B(T^L) U B(T^R), where T^L / T^R is the first / last child subtree.
  * Each level: (1) for every ordered pair of trees, d_min over B(T) x B(T') and its
    minimizers, then External Orientation (Alg. 5.7) of each tree towards its
    minimizer, ending with Complete Internal Orientation of that tree (Alg. 5.7,
    line 18); (2) Arc Partition: connected components of the nearest-neighbour
    graph of trees under d_min (paths, or one cycle); (3) one component left ->
    Final Orientation (Alg. 5.6, root treated as a ring) and stop; otherwise each
    component (in path order) becomes the children of a new Q-node, Complete Internal
    Orientation (Alg. 5.5) orients its inner children with Consecutive Orientation
    (Alg. 5.4) and Border Candidates Orientation (Alg. 5.1), and we recurse.
  * Output (Thm 5.17): the solution set is Dih_n o S(T) for the final tree T.

Points where the paper's text is ambiguous, and the reading used here:
  * Alg. 5.5 prints the triple (T_{i-1}, T_i, T_{i-1}); we use (T_{i-1}, T_i, T_{i+1}).
  * Arc Partition (App. B) is implemented as the DFS of the nearest-neighbour graph
    from a degree-1 tree (or any tree when the component is a cycle).
  * Final Orientation with two children (T_0, T_1): Consecutive Orientation of T_0
    against the two ends T_1^R, T_1^L of T_1, then T_1 is fixed arbitrarily (a
    reflection of the whole ring). If either child is a single leaf, the orientation
    of the other is immaterial on a ring and it is fixed arbitrarily.
  * Q-nodes nested inside a genuine (non-orientable) Q-node are never visited by
    Algs. 5.5-5.6, which treat the genuine node as fixed; orient_remaining() orients them
    afterwards with Alg. 5.1 against their left/right context (an addition).

Two modes (an addition; the paper only treats strict CR input):
  * tolerant (default, strict=False): never raises; always returns an order. Where an
    assumption of the paper fails, a fallback rule is used and recorded in info["warnings"]:
      - nearest-neighbour graph not a union of paths -> greedy path cover of its edges,
        shortest d_min first (identical to the paper's Arc Partition on strict CR input);
      - contradictory orientation evidence in Alg. 5.1 -> majority vote over z;
      - a minimizer that is no longer a border candidate -> that External Orientation step
        is skipped;
      - several represented orders, none strict CR -> the one with the largest tour score.
    info["strict_circular_robinson"] says whether the returned order is strict CR for the
    input; when it is True, the paper's guarantees apply.
  * strict (strict=True): the behaviour of the first version. Contradictory evidence, a
    nearest-neighbour graph of degree > 2, or an output that is not strict CR raise
    NotStrictCircularRobinson.

BROOQS helpers (not in the paper): an NMSC F-matrix can be strict circular Robinson in
the true order and in one arc-reversed order; brooqs_step2() keeps the candidate with the
largest Kalmanson margin, which is the true order for a population F. order_cycle() falls
back to the spectral order when the input is not strict circular Robinson (e.g. noisy F-hat).

Quick use:
    from circular_seriation import brooqs_step2
    order, info = brooqs_step2(F)               # always returns an order
    info["strict_circular_robinson"]            # True -> exact guarantee applies
    info["warnings"]                            # which assumptions failed, with counts
    order, info = brooqs_step2(F, strict=True)  # raise NotStrictCircularRobinson instead
"""
from __future__ import annotations

from collections import Counter
from itertools import product
from math import comb
import numpy as np

__all__ = [
    "NotStrictCircularRobinson", "QNode", "similarity_to_dissimilarity",
    "recursive_seriation", "seriate_dissimilarity", "seriate_F",
    "is_strict_circular_robinson", "is_circular_robinson", "canonical_cycle",
    "tree_to_string", "kalmanson_margin", "tour_score", "brooqs_step2",
    "spectral_order", "order_cycle",
]


class NotStrictCircularRobinson(Exception):
    """Raised when the input is detected not to be strict circular Robinson."""


class QNode:
    """A Q-node: ordered children (ints = leaves, or QNodes). `genuine` marks a node
    whose orientation is undecidable; it stays a Q-node in the solution PQ-tree."""
    __slots__ = ("children", "genuine")

    def __init__(self, children):
        self.children = list(children)
        self.genuine = False

    def __repr__(self):
        return tree_to_string(self)


# ----------------------------------------------------------------------------- helpers
def _violation(check, log, kind: str) -> None:
    """Strict mode: raise. Tolerant mode: record the violated assumption and continue."""
    if check:
        raise NotStrictCircularRobinson(kind)
    if log is not None:
        log[kind] += 1


def _is_leaf(t) -> bool:
    return not isinstance(t, QNode)


def _free(t) -> bool:
    """An internal node whose orientation is still undecided."""
    return isinstance(t, QNode) and not t.genuine


def border(t) -> set:
    """B(T): border candidates (Algorithm 5.3)."""
    if _is_leaf(t):
        return {t}
    return border(t.children[0]) | border(t.children[-1])


def frontier(t) -> list:
    """Leaves of T in the current configuration."""
    if _is_leaf(t):
        return [t]
    out = []
    for c in t.children:
        out.extend(frontier(c))
    return out


def _fix_child(parent: QNode, idx: int, reverse: bool) -> None:
    """Fix child `idx` into `parent` (optionally reversing it first): splice its children."""
    c = parent.children[idx]
    kids = c.children[::-1] if reverse else list(c.children)
    parent.children[idx:idx + 1] = kids


def tree_to_string(t) -> str:
    """Q-nodes as [..], genuine (non-orientable) Q-nodes as {..}."""
    if _is_leaf(t):
        return str(t)
    inner = " ".join(tree_to_string(c) for c in t.children)
    return "{" + inner + "}" if t.genuine else "[" + inner + "]"


# ----------------------------------------------------------- Algorithm 5.1 (BCO)
def border_candidates_orientation(D, X, Ap, A, B, Bp, check=True, log=None) -> str:
    """Border Candidates Orientation (Algorithm 5.1).

    A, B are border candidates of the two ends of an arc I (A on the side currently
    facing Ap); Ap, Bp are border candidates of the two neighbouring parts of I^c.
    Returns 'correct', 'reverse' or 'not orientable'. On strict CR input all z that give
    a verdict agree. If they do not: strict mode raises, tolerant mode takes the majority
    (a tie gives 'not orientable')."""
    votes = Counter()
    for z in X:
        row = D[z]
        mnA, mxA = min(row[s] for s in A), max(row[s] for s in A)
        mnB, mxB = min(row[s] for s in B), max(row[s] for s in B)
        mnAp, mxAp = min(row[s] for s in Ap), max(row[s] for s in Ap)
        mnBp, mxBp = min(row[s] for s in Bp), max(row[s] for s in Bp)
        o1 = max(mnA, mnAp) < min(mxB, mxBp)
        o2 = max(mnB, mnBp) < min(mxA, mxAp)
        if o1 or o2:
            v = "correct"
        else:
            o3 = max(mnA, mnBp) < min(mxAp, mxB)
            o4 = max(mnB, mnAp) < min(mxBp, mxA)
            v = "reverse" if (o3 or o4) else None
        if v is not None:
            votes[v] += 1
    if not votes:
        return "not orientable"
    if len(votes) == 1:
        return next(iter(votes))
    _violation(check, log, "contradictory orientation evidence (majority vote used)")
    if votes["correct"] == votes["reverse"]:
        return "not orientable"
    return "correct" if votes["correct"] > votes["reverse"] else "reverse"


# -------------------------------------------------- Algorithm 5.4 (Consecutive Or.)
def consecutive_orientation(D, X, parent: QNode, i: int, prev, nxt, check=True, log=None) -> str:
    """Orient child i of `parent`, whose neighbours (in the parent's current order) are
    `prev` and `nxt`: A' = B(prev), B' = B(nxt), A = B^L(T_i), B = B^R(T_i)."""
    c = parent.children[i]
    res = border_candidates_orientation(
        D, X, border(prev), border(c.children[0]), border(c.children[-1]), border(nxt), check, log)
    if res == "correct":
        _fix_child(parent, i, reverse=False)
    elif res == "reverse":
        _fix_child(parent, i, reverse=True)
    else:
        c.genuine = True
    return res


# ------------------------------------------------- Algorithm 5.7 (External Or.)
def external_orientation(D, X, T, x, check=True, log=None) -> None:
    """External Orientation (Algorithm 5.7) of T towards the minimizer x.

    Lines 2-17: orient the end chain of T that contains the border element x so that x
    becomes an extreme leaf of T (x is a true border of the arc of T by Lemma 5.14).
    Line 18: Complete Internal Orientation of T, which orients the subtrees that fixing
    the chain exposed as inner children of T's root (cf. Figure 5(c) -> 5(d))."""
    if _is_leaf(T):
        return
    if x in border(T.children[0]):
        side = 0
    elif x in border(T.children[-1]):
        side = -1
    else:
        _violation(check, log, "minimizer no longer a border candidate (orientation step skipped)")
        side = None
    while side is not None:
        idx = 0 if side == 0 else len(T.children) - 1
        J = T.children[idx]
        if not _free(J):
            break
        keep = x in border(J.children[0] if side == 0 else J.children[-1])
        if not keep and x not in border(J.children[-1] if side == 0 else J.children[0]):
            _violation(check, log, "minimizer no longer a border candidate (orientation step skipped)")
            break
        _fix_child(T, idx, reverse=not keep)
    complete_internal_orientation(D, X, T, check, log)     # Alg. 5.7, line 18


# ---------------------------------------- Algorithm 5.5 (Complete Internal Or.)
def complete_internal_orientation(D, X, T: QNode, check=True, log=None) -> None:
    """Orient every inner child (positions 1..k-2) of the root of T."""
    while True:
        idx = next((i for i in range(1, len(T.children) - 1) if _free(T.children[i])), None)
        if idx is None:
            return
        consecutive_orientation(D, X, T, idx, T.children[idx - 1], T.children[idx + 1], check, log)


# ------------------------------------------------- Algorithm 5.6 (Final Or.)
def final_orientation(D, X, T: QNode, check=True, log=None) -> None:
    """Orient all children of the root, the root being a ring (cyclic list)."""
    while True:
        k = len(T.children)
        idx = next((i for i in range(k) if _free(T.children[i])), None)
        if idx is None:
            return
        if k == 1:
            _fix_child(T, 0, reverse=False)
        elif k == 2:
            c, o = T.children[idx], T.children[1 - idx]
            if _is_leaf(o):
                _fix_child(T, idx, reverse=False)          # ring of c and one leaf
                continue
            # ring neighbours of c are the two ends of o: before c comes o's last child.
            res = border_candidates_orientation(
                D, X, border(o.children[-1]), border(c.children[0]),
                border(c.children[-1]), border(o.children[0]), check, log)
            if res == "correct":
                _fix_child(T, idx, reverse=False)
            elif res == "reverse":
                _fix_child(T, idx, reverse=True)
            else:
                c.genuine = True
            j = next((i for i in range(len(T.children)) if T.children[i] is o), None)
            if j is not None and _free(o):
                _fix_child(T, j, reverse=False)            # a reflection of the ring
        else:
            consecutive_orientation(D, X, T, idx, T.children[idx - 1],
                                    T.children[(idx + 1) % k], check, log)

def orient_remaining(D, X, root: QNode, check=True, log=None) -> None:
    """Addition to the paper's procedure: Q-nodes nested inside a genuine (non-orientable)
    Q-node are never visited by Algorithms 5.5-5.6, which treat the genuine node as fixed.
    Orient each such free node by Algorithm 5.1 against its left and right context in the
    final tree (genuine ancestors taken in their current orientation)."""
    while True:
        todo = _paths_to_free_nodes(root, [], [])
        if not todo:
            return
        c, path = todo[0]
        parent, i = path[-1]
        prev, nxt = _context(path, -1), _context(path, +1)
        if prev is nxt:                             # ring of two parts
            if _is_leaf(prev):
                _fix_child(parent, i, reverse=False)
                continue
            Ap, Bp = border(prev.children[-1]), border(prev.children[0])
        else:
            Ap, Bp = border(prev), border(nxt)
        res = border_candidates_orientation(
            D, X, Ap, border(c.children[0]), border(c.children[-1]), Bp, check, log)
        if res == "correct":
            _fix_child(parent, i, reverse=False)
        elif res == "reverse":
            _fix_child(parent, i, reverse=True)
        else:
            c.genuine = True


def _paths_to_free_nodes(t, path, out):
    """(node, [(ancestor, index), ...]) for every free non-root node, root first."""
    if _is_leaf(t):
        return out
    for i, c in enumerate(t.children):
        p = path + [(t, i)]
        if _free(c):
            out.append((c, p))
        _paths_to_free_nodes(c, p, out)
    return out


def _context(path, step):
    """Nearest subtree to the left (step=-1) or right (step=+1) of the node at the end
    of `path`, walking up through ancestors (genuine ones in their current orientation);
    the root is a ring."""
    for depth in range(len(path) - 1, -1, -1):
        parent, i = path[depth]
        j = i + step
        if 0 <= j < len(parent.children):
            return parent.children[j]
        if depth == 0:                              # wrap around the ring
            return parent.children[j % len(parent.children)]
    return None


# ------------------------------------------------------- Tree dissimilarity
def tree_dissimilarity(D, T1, T2):
    """d_min(T1, T2) = min over B(T1) x B(T2) of D, and its minimizers."""
    best, arg = None, []
    for x in border(T1):
        row = D[x]
        for y in border(T2):
            v = row[y]
            if best is None or v < best:
                best, arg = v, [(x, y)]
            elif v == best:
                arg.append((x, y))
    return best, arg


# ---------------------------------------------------------- Arc Partition
def _greedy_path_cover(adj, dmin, m):
    """Keep nearest-neighbour edges, shortest d_min first, as long as every tree keeps
    degree <= 2 and no cycle forms. On strict CR input this keeps the whole
    nearest-neighbour graph (minus one edge when it is a single cycle, which is immaterial
    because the final root is a ring)."""
    edges = sorted({(min(a, b), max(a, b)) for a in range(m) for b in adj[a]},
                   key=lambda e: dmin[e[0]][e[1]])
    parent = list(range(m))

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    new_adj = [set() for _ in range(m)]
    for a, b in edges:
        if len(new_adj[a]) < 2 and len(new_adj[b]) < 2 and find(a) != find(b):
            new_adj[a].add(b)
            new_adj[b].add(a)
            parent[find(a)] = find(b)
    return new_adj


def arc_partition(dmin, m: int, check=True, log=None):
    """Connected components of the nearest-neighbour graph of m trees, each as an
    ordered tuple (path order; any start for a cycle)."""
    adj = [set() for _ in range(m)]
    for a in range(m):
        others = [b for b in range(m) if b != a]
        best = min(dmin[a][b] for b in others)
        for b in others:
            if dmin[a][b] == best:
                adj[a].add(b)
                adj[b].add(a)
    if any(len(s) > 2 for s in adj):
        _violation(check, log, "nearest-neighbour graph not a union of paths (greedy path cover used)")
        adj = _greedy_path_cover(adj, dmin, m)
    seen, comps = [False] * m, []
    starts = [a for a in range(m) if len(adj[a]) == 1] + list(range(m))
    for s in starts:
        if seen[s]:
            continue
        comp, prev, cur = [], None, s
        while cur is not None and not seen[cur]:
            seen[cur] = True
            comp.append(cur)
            nxt = [b for b in adj[cur] if b != prev and not seen[b]]
            prev, cur = cur, (nxt[0] if nxt else None)
        comps.append(comp)
    return comps


# ------------------------------------------------- Algorithm 5.2 (Recursive Ser.)
def recursive_seriation(D, check=True, log=None) -> QNode:
    """Run Recursive Seriation on a dissimilarity matrix D (n x n, list of rows or array).
    Returns the final PQ-tree (root = ring). check=True raises NotStrictCircularRobinson
    on a detected violation; check=False records it in `log` (a Counter) and continues."""
    n = len(D)
    if n < 3:
        return QNode(list(range(n)))
    X = list(range(n))
    trees = list(range(n))
    while True:
        m = len(trees)
        dmin = [[None] * m for _ in range(m)]
        for a in range(m):
            for b in range(m):
                if a != b:
                    v, arg = tree_dissimilarity(D, trees[a], trees[b])
                    dmin[a][b] = v
                    for (x, y) in arg:
                        external_orientation(D, X, trees[a], x, check, log)
                        external_orientation(D, X, trees[b], y, check, log)
        comps = arc_partition(dmin, m, check, log)
        if len(comps) == 1:
            root = QNode([trees[a] for a in comps[0]])
            final_orientation(D, X, root, check, log)
            # orient_remaining(D, X, root, check, log)
            return root
        new = []
        for comp in comps:
            if len(comp) == 1:
                new.append(trees[comp[0]])
                continue
            node = QNode([trees[a] for a in comp])
            complete_internal_orientation(D, X, node, check, log)
            new.append(node)
        trees = new


# ------------------------------------------------------------ verification
def _strictly_unimodal(seq) -> bool:
    """Strict unimodality as defined in the paper (p. 1229): strictly increasing up to
    the mode, strictly decreasing after it, where the mode may be two equal, adjacent
    values (m1 < m2 with f_{m1} = f_{m2}). No other ties are allowed."""
    k, L = 0, len(seq)
    while k + 1 < L and seq[k] < seq[k + 1]:
        k += 1
    if k + 1 < L and seq[k] == seq[k + 1]:      # optional second, equal mode
        k += 1
    while k + 1 < L and seq[k] > seq[k + 1]:
        k += 1
    return k == L - 1


def _unimodal(seq) -> bool:
    k, L = 0, len(seq)
    while k + 1 < L and seq[k] <= seq[k + 1]:
        k += 1
    while k + 1 < L and seq[k] >= seq[k + 1]:
        k += 1
    return k == L - 1


def is_strict_circular_robinson(D, order) -> bool:
    """D (dissimilarity) is strict circular Robinson in the cyclic order `order`
    (Prop. 3.7: every row strictly unimodal around the cycle; two equal adjacent modes
    are allowed)."""
    n = len(order)
    for t in range(n):
        i = order[t]
        if not _strictly_unimodal([D[i][order[(t + j) % n]] for j in range(1, n)]):
            return False
    return True


def is_circular_robinson(D, order) -> bool:
    n = len(order)
    for t in range(n):
        i = order[t]
        if not _unimodal([D[i][order[(t + j) % n]] for j in range(1, n)]):
            return False
    return True


def canonical_cycle(order) -> tuple:
    """Representative of a cyclic order modulo rotation and reflection."""
    n = len(order)
    k = list(order).index(min(order))
    r = list(order[k:]) + list(order[:k])
    s = [r[0]] + r[1:][::-1]
    return tuple(min(r, s))


def _free_nodes(t, out, is_root=True):
    if _is_leaf(t):
        return out
    if not is_root:
        out.append(t)
    for c in t.children:
        _free_nodes(c, out, False)
    return out


def _greedy_flips(root, score):
    """Too many undecided Q-nodes to enumerate: flip nodes one at a time while the score
    of the frontier improves (local search)."""
    nodes = _free_nodes(root, [])
    best = score(frontier(root))
    improved = True
    while improved:
        improved = False
        for nd in nodes:
            nd.children.reverse()
            s = score(frontier(root))
            if s > best:
                best, improved = s, True
            else:
                nd.children.reverse()
    return canonical_cycle(frontier(root))


def solution_orders(D, root: QNode, verify=True, max_enum=1 << 12):
    """Orders represented by the final tree, modulo Dih_n (Thm 5.17). Every non-root
    remaining Q-node may be reversed; with verify=True only strict-CR orders are kept."""
    nodes = _free_nodes(root, [])
    if (1 << len(nodes)) > max_enum:
        raise RuntimeError(f"{len(nodes)} undecided Q-nodes; too many to enumerate")
    sols = set()
    for flips in product((False, True), repeat=len(nodes)):
        for nd, f in zip(nodes, flips):
            if f:
                nd.children.reverse()
        order = frontier(root)
        for nd, f in zip(nodes, flips):
            if f:
                nd.children.reverse()
        if (not verify) or is_strict_circular_robinson(D, order):
            sols.add(canonical_cycle(order))
    return sorted(sols)


# ------------------------------------------------------------- front ends
def similarity_to_dissimilarity(F, c=None):
    n = len(F)
    if c is None:
        c = 2 * max(F[i][j] for i in range(n) for j in range(n) if i != j) + 1
    return [[0 if i == j else c - F[i][j] for j in range(n)] for i in range(n)]


def seriate_dissimilarity(D, strict=False, log=None, max_enum=1 << 12):
    """Returns (orders, tree).

    strict=True: all strict-CR circular orders of D modulo Dih_n; raises
    NotStrictCircularRobinson if D is detected not to be strict CR.
    strict=False: never raises. If some order represented by the final tree is strict CR,
    those orders are returned (identical to strict mode on strict CR input); otherwise all
    represented orders are returned, or a single locally optimised one when there are too
    many to enumerate. Violations are counted in `log` (a Counter) if given."""
    log = Counter() if log is None else log
    root = recursive_seriation(D, check=strict, log=log)
    nodes = _free_nodes(root, [])
    if strict or (1 << len(nodes)) <= max_enum:
        verified = solution_orders(D, root, verify=True, max_enum=max_enum)
        if verified:
            return verified, root
        if strict:
            raise NotStrictCircularRobinson("no represented order is strict circular Robinson")
        log["output not strict circular Robinson"] += 1
        return solution_orders(D, root, verify=False, max_enum=max_enum), root
    log["too many undecided Q-nodes (local search over flips)"] += 1
    Fsim = [[-v for v in row] for row in D]
    return [_greedy_flips(root, lambda o: tour_score(Fsim, o))], root


def _to_D(F):
    n = len(F)
    fmax = max(F[i][j] for i in range(n) for j in range(n) if i != j)
    c = 2 * comb(n - 2, 2) if (n >= 4 and fmax <= comb(n - 2, 2)) else 2 * fmax + 1
    return similarity_to_dissimilarity(F, c)


def seriate_F(F, strict=False, log=None):
    """Seriate a BROOQS F-matrix (similarity); see seriate_dissimilarity. For an NMSC
    F-matrix of n components the dissimilarity is D_ij = 2*binom(n-2,2) - F_ij (the
    Kalmanson dissimilarity)."""
    d = len(F)
    D = (d-2)*(d-3)/2 - F
    for i in range(len(F)):
        D[i,i] = 0
    return seriate_dissimilarity(D, strict=strict, log=log)


# ------------------------------------------------------- BROOQS-specific helpers
def kalmanson_margin(F, order):
    """Smallest local circular-split coefficient of F in the cyclic `order`:
    min over non-adjacent gap pairs of F[i,j+1] + F[i+1,j] - F[i,j] - F[i+1,j+1]
    (indices along `order`). For a population NMSC F-matrix it is >= 0 exactly in the
    true order (one coefficient is identically 0 there), and some coefficient is < 0
    in every other order (unique circular split representation)."""
    n = len(order)
    o = list(order)
    best = None
    for a in range(n):
        for b in range(a + 2, n):
            if (b - a) % n in (1, n - 1):
                continue
            i, i1, j, j1 = o[a], o[(a + 1) % n], o[b], o[(b + 1) % n]
            v = F[i][j1] + F[i1][j] - F[i][j] - F[i1][j1]
            best = v if best is None or v < best else best
    return best


def tour_score(F, order):
    """T_F(order) = sum of F over consecutive pairs; the true order is its unique
    maximizer for a population NMSC F-matrix."""
    n = len(order)
    return sum(F[order[t]][order[(t + 1) % n]] for t in range(n))


def brooqs_step2(F, strict=False):
    """Ordering step for one cycle: circular seriation of F with the PQ-tree algorithm.
    If several orders remain, keep the one with the largest Kalmanson margin (ties: largest
    tour score) when they are strict CR (population F), and the one with the largest tour
    score otherwise (noisy F-hat).

    Returns (order, info):
      info["strict_circular_robinson"]  True if F is strict CR in the returned order
      info["warnings"]                  dict: violated assumption -> number of occurrences
      info["candidates"]                number of orders left by the tree
      info["tree"]                      the final PQ-tree ([..] Q-node, {..} genuine Q-node)
    strict=True raises NotStrictCircularRobinson instead of returning a non-CR order."""
    log = Counter()
    orders, root = seriate_F(F, strict=strict, log=log)
    d = len(F)
    D = (d-2)*(d-3)/2 - F
    for i in range(len(F)):
        D[i,i] = 0
    # deg = F.sum(axis=1)
    # D = np.diag(deg) - F
    # D = _to_D(F)
    # print
    is_scr = is_strict_circular_robinson(D, orders[0])
    if len(orders) == 1:
        best = orders[0]
    # elif is_scr:
    #     best = max(orders, key=lambda o: (kalmanson_margin(F, o), tour_score(F, o)))
    else:
        # print("number of orders found: ", len(orders), )
        # print(orders)
        # print([tour_score(F, o) for o in orders])
        # print(F)
        best = max(orders, key=lambda o: tour_score(F, o))
    return list(best), {"strict_circular_robinson": is_scr, "warnings": dict(log),
                        "candidates": len(orders), "tree": tree_to_string(root)}


def spectral_order(F):
    """BROOQS's original Step 2: angular order of the 2nd/3rd eigenvectors of L = rho I - F
    (equivalently the top two non-constant eigenvectors of F). Requires numpy."""
    import numpy as np
    A = np.array([[float(x) for x in r] for r in F])
    E, V = np.linalg.eigh(A)
    idx = np.argsort(E)[::-1]
    P = V[:, idx[1:3]]
    return [int(k) for k in np.argsort(np.arctan2(P[:, 1], P[:, 0]))]


def order_cycle(F, fallback="spectral"):
    """PQ-tree order when F is strict circular Robinson in it; otherwise the spectral order
    (fallback="spectral") or the PQ-tree order anyway (fallback=None).
    Returns (order, method)."""
    order, info = brooqs_step2(F)
    # if info["strict_circular_robinson"] or fallback is None:
    return order, "pq-tree"
    # return spectral_order(F), "spectral (fallback)"
