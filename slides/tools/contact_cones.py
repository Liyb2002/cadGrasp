"""Small 3D contact cones and heuristic contact-set search.

Extracted unchanged from the former demand slide on 2026-09-06. These helpers
belong to support feasibility, independently of the paired-demand figures.
"""
from itertools import combinations

import numpy as np


def in_cone(v, G):
    """Batch membership in a SMALL nonnegative cone, including ranks 0--3.

    Work in the generators' span: conic Caratheodory needs at most `rank(G)`
    generators, and a smaller independent subset can be extended to a basis.
    Full-rank inputs retain the 3x3 enumeration; lines and planes use 1x1 and
    2x2 systems after rejecting queries outside their span. Positive rescaling
    does not change a cone, so normalise first. Floating-point decisions use
    1e-12 for span rank and 1e-9 for residuals and nonnegativity; they are not
    symbolic certificates. The zero vector belongs even to the empty cone.
    """
    v = np.asarray(v, dtype=float).reshape(-1, 3)
    G = np.asarray(G, dtype=float).reshape(-1, 3)
    vn, gn = np.linalg.norm(v, axis=1), np.linalg.norm(G, axis=1)
    ok = vn == 0
    if not len(v) or not np.any(gn > 0):
        return ok
    v = np.divide(v, vn[:, None], out=np.zeros_like(v), where=vn[:, None] > 0)
    G = G[gn > 0] / gn[gn > 0, None]
    _, singular, axes = np.linalg.svd(G, full_matrices=False)
    rank = int(np.count_nonzero(singular > 1e-12 * singular[0]))
    in_span = np.ones(len(v), bool)
    if rank < 3:
        basis = axes[:rank].T
        projected = v @ basis
        in_span = np.linalg.norm(v - projected @ basis.T, axis=1) <= 1e-9
        v, G = projected, G @ basis
    for subset in combinations(range(len(G)), rank):
        A = G[list(subset)].T
        try:
            lam = np.linalg.solve(A, v.T)
        except np.linalg.LinAlgError:
            continue
        residual = np.linalg.norm(A @ lam - v.T, axis=0)
        ok |= in_span & (lam >= -1e-9).all(axis=0) & (residual <= 1e-9)
        if ok.all():
            break
    return ok


def fewest(want, cands, free, beam=200, kmax=6):
    """Find a small contact set; the beam/greedy search is not globally minimal.

    Check `free` and every single candidate first: a line or plane can answer
    a demand lying within it. Then enumerate pairs, grow triples from the best
    `beam` pairs, and greedily extend. The set search avoids stalling when a
    full-dimensional demand receives no coverage from any single candidate.

    Returns chosen indices, demand coverage, and the number of pairs examined
    (the existing page/log statistic, not the total number of membership calls).
    """
    want = np.asarray(want, dtype=float).reshape(-1, 3)
    cands = np.asarray(cands, dtype=float).reshape(-1, 3)
    free = np.asarray(free, dtype=float).reshape(-1, 3)
    covered = in_cone(want, free)
    if covered.all():
        return [], 1.0, 0
    n = len(cands)
    chosen, bf = [], float(covered.mean())
    if not n or kmax < 1:
        return chosen, bf, 0
    singles = np.array([in_cone(want, np.vstack([free, cands[[k]]])).mean()
                        for k in range(n)])
    j = int(np.argmax(singles))
    if singles[j] > bf:
        chosen, bf = [j], float(singles[j])
    if bf >= 1.0 or n < 2 or kmax < 2:
        return chosen, bf, 0
    pairs = np.array(list(combinations(range(n), 2)))
    got = np.array([in_cone(want, np.vstack([free, cands[list(p)]])).mean()
                    for p in pairs])
    j = int(np.argmax(got))
    if got[j] >= 1.0:
        return list(pairs[j]), 1.0, len(pairs)
    if got[j] > bf:
        chosen, bf = list(pairs[j]), float(got[j])
    if n < 3 or kmax < 3:
        return chosen, bf, len(pairs)
    best, triple_fraction = None, -1.0
    for p in pairs[np.argsort(-got)[:beam]]:
        rest = np.setdiff1d(np.arange(n), p)
        f = np.array([in_cone(want, np.vstack([free, cands[list(p)], cands[[k]]])).mean()
                      for k in rest])
        i = int(np.argmax(f))
        if f[i] > triple_fraction:
            best = [int(p[0]), int(p[1]), int(rest[i])]
            triple_fraction = float(f[i])
        if triple_fraction >= 1.0:
            break
    if best is None:
        return chosen, bf, len(pairs)
    growing = list(best)
    if triple_fraction > bf:
        chosen, bf = growing.copy(), triple_fraction
    while triple_fraction < 1.0 and len(growing) < min(kmax, n):
        rest = np.setdiff1d(np.arange(n), growing)
        f = np.array([in_cone(want, np.vstack([free, cands[growing], cands[[k]]])).mean()
                      for k in rest])
        i = int(np.argmax(f))
        if f[i] <= triple_fraction + 1e-12:
            break
        growing.append(int(rest[i]))
        triple_fraction = float(f[i])
        if triple_fraction > bf:
            chosen, bf = growing.copy(), triple_fraction
    return chosen, bf, len(pairs)
