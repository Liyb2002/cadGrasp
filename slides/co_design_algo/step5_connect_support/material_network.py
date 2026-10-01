"""Connected group cover on a shared, node-weighted material graph.

Nodes denote actual material, never virtual connections between foot choices.
Greedy paths and the optional flow MILP use exactly the same graph and groups.
"""
from dataclasses import dataclass
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.spatial import ConvexHull


@dataclass
class Group:
    members: np.ndarray
    kind: str
    pose: int = -1
    direction: object = None
    threshold: float = 0.


def adjacency(n, edges, costs=None):
    edges = np.asarray(edges, int).reshape(-1, 2)
    a, b = np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]
    w = np.ones(len(a)) if costs is None else np.asarray(costs)[b]
    return coo_matrix((w, (a, b)), shape=(n, n)).tocsr()


def connected_selection(selected, edges, n):
    ids = np.flatnonzero(selected)
    return bool(len(ids) and connected_components(adjacency(n, edges)[ids][:, ids], directed=False)[0] == 1)


def covered(selected, groups):
    return np.array([np.any(selected[g.members]) for g in groups])


def group_weights(groups):
    # One total weight for head connectivity and one for each pose. Merely
    # sampling a pose with more directions cannot multiply its total reward.
    keys = [(g.kind, g.pose) for g in groups]
    return np.array([1./keys.count(key) for key in keys])


def greedy(cost, edges, groups, root, seed=None):
    n = len(cost)
    selected = np.zeros(n, bool) if seed is None else np.asarray(seed, bool).copy()
    selected[root] = True
    if not connected_selection(selected, edges, n):
        raise ValueError('A growth seed must be one physically connected network')
    weights = group_weights(groups); history = []
    for iteration in range(len(groups)+1):
        hit = covered(selected, groups)
        if hit.all():
            return selected, history
        remaining_cost = np.where(selected, 0., cost)
        dist, pred, _ = dijkstra(adjacency(n, edges, remaining_cost), directed=True,
            indices=np.flatnonzero(selected), min_only=True, return_predecessors=True)
        candidates = set()
        for group, done in zip(groups, hit):
            if done: continue
            valid = group.members[np.isfinite(dist[group.members])]
            if not len(valid):
                raise RuntimeError('An unmet material group is unreachable from the connected network')
            candidates.update(valid[np.argsort(dist[valid])[:3]].tolist())
        best = None
        for target in sorted(candidates):
            path = []; current = target
            while not selected[current]:
                path.append(current); current = int(pred[current])
                if current < 0 or len(path) > n:
                    raise RuntimeError('Invalid shortest-path predecessor chain')
            proposal = selected.copy(); proposal[path] = True
            gain_mask = covered(proposal, groups) & ~hit
            gain = float(weights[gain_mask].sum())
            added = float(cost[path].sum())
            key = (added/gain, added, target)
            if best is None or key < best[0]:
                best = key, proposal, path, gain_mask, added
        _, selected, path, gain_mask, added = best
        history.append(dict(iteration=iteration, added_nodes=path, added_volume_cm3=added,
                            satisfied_groups=np.flatnonzero(gain_mask).tolist()))
    raise RuntimeError('Connected cover did not converge')


def prune(selected, cost, edges, groups, heads, seed=None):
    """Reverse-delete while preserving every group and actual graph connectivity."""
    selected = selected.copy(); required = set(heads)
    if seed is not None: required.update(np.flatnonzero(seed))
    graph = adjacency(len(cost), edges)
    changed = True
    while changed:
        changed = False
        for v in sorted(np.flatnonzero(selected), key=lambda x: -cost[x]):
            if v in required: continue
            selected[v] = False
            ids = np.flatnonzero(selected)
            if covered(selected, groups).all() and connected_components(graph[ids][:, ids], directed=False)[0] == 1:
                changed = True
            else:
                selected[v] = True
    return selected


def floor_points(selected, floors, pose):
    arrays = [floors[v][pose] for v in np.flatnonzero(selected) if len(floors[v][pose])]
    return np.unique(np.vstack(arrays), axis=0) if arrays else np.empty((0, 2))


def containment(points, demand, tolerance=1e-8):
    if len(points) < 3 or np.linalg.matrix_rank(points-points.mean(0), tol=1e-10) < 2:
        return dict(passed=False, maximum_violation_m=None, normals=[])
    hull = ConvexHull(points)
    errors = demand@hull.equations[:, :2].T+hull.equations[:, 2]
    worst = errors.max(0)
    return dict(passed=bool(worst.max() <= tolerance), maximum_violation_m=float(worst.max()),
                normals=hull.equations[worst > tolerance, :2].tolist())


def add_direction(groups, floors, demands, pose, direction, margin=0.):
    u = np.asarray(direction, float); u /= np.linalg.norm(u)
    target = float((demands[pose]@u).max()+margin)
    members = np.array([i for i, fs in enumerate(floors)
                       if len(fs[pose]) and (fs[pose]@u).max() >= target-1e-9], int)
    if not len(members):
        raise RuntimeError(f'Candidate material cannot cover pose {pose} direction {u.tolist()}')
    for g in groups:
        if g.kind == 'floor' and g.pose == pose and np.array_equal(g.members, members):
            return False
    groups.append(Group(members, 'floor', pose, u.tolist(), target))
    return True


def solve_cover(cost, edges, floors, demands, heads, margin=.002, max_rounds=24, groups=None, seed=None):
    if groups is None:
        groups = [Group(np.array([h]), 'head') for h in heads]
        for pose in (0, 1):
            for angle in np.arange(16)*2*np.pi/16:
                add_direction(groups, floors, demands, pose, [np.cos(angle), np.sin(angle)], margin)
    else:
        groups = list(groups)
    log = []; candidates = []
    for round_id in range(max_rounds):
        candidates = []
        # Once a verified connected candidate exists, continue growing from
        # its entire network. This preserves every existing reaction ray.
        for root in heads if seed is None else heads[:1]:
            chosen, steps = greedy(cost, edges, groups, root, seed)
            chosen = prune(chosen, cost, edges, groups, heads, seed)
            checks = [containment(floor_points(chosen, floors, k), demands[k]) for k in (0, 1)]
            candidates.append((float(cost[chosen].sum()), chosen, root, steps, checks))
        candidates.sort(key=lambda c: c[0])
        complete = [c for c in candidates if all(x['passed'] for x in c[4])]
        log.append(dict(round=round_id, groups=len(groups), costs_cm3=[c[0] for c in candidates],
                        complete_roots=[c[2] for c in complete]))
        if complete:
            return complete, groups, log
        added = False
        for check, pose in zip(candidates[0][4], (0, 1)):
            for normal in check['normals']:
                added |= add_direction(groups, floors, demands, pose, normal, margin)
        if not added:
            raise RuntimeError('Degenerate or unresolved floor coverage; refine the material graph')
    raise RuntimeError('Geometric separation iteration limit reached')


def flow_milp(cost, edges, groups, heads, root, incumbent=None, seconds=20.):
    """Exact formulation on the supplied graph; time limit may leave a gap.

    Each selected non-root node consumes one unit of artificial connectivity
    flow. This is not a physical reaction force and cannot join virtual groups.
    """
    n = len(cost); edge = np.asarray(edges, int).reshape(-1, 2)
    arc = np.vstack([edge, edge[:, ::-1]]); m = len(arc); cap = n-1
    rows = []; cols = []; values = []; lower = []; upper = []
    def row(ids, vals, lo, hi):
        index = len(lower); rows.extend([index]*len(ids)); cols.extend(ids); values.extend(vals)
        lower.append(lo); upper.append(hi)
    for group in groups:
        row(group.members.tolist(), [1.]*len(group.members), 1., np.inf)
    for v in range(n):
        ids = (n+np.flatnonzero(arc[:, 1] == v)).tolist()+(n+np.flatnonzero(arc[:, 0] == v)).tolist()
        vals = [1.]*np.count_nonzero(arc[:, 1] == v)+[-1.]*np.count_nonzero(arc[:, 0] == v)
        if v == root:
            ids += [j for j in range(n) if j != root]; vals += [1.]*(n-1)
        else:
            ids.append(v); vals.append(-1.)
        row(ids, vals, 0., 0.)
    for j, (a, b) in enumerate(arc):
        for v in (a, b): row([n+j, int(v)], [1., -float(cap)], -np.inf, 0.)
    if incumbent is not None:
        row(list(range(n)), cost.tolist(), -np.inf, float(cost[incumbent].sum())+1e-7)
    matrix = coo_matrix((values, (rows, cols)), shape=(len(lower), n+m)).tocsc()
    lb = np.zeros(n+m); ub = np.r_[np.ones(n), np.full(m, cap)]
    lb[heads] = 1
    start = time.monotonic()
    result = milp(np.r_[cost, np.zeros(m)], integrality=np.r_[np.ones(n), np.zeros(m)],
        bounds=Bounds(lb, ub), constraints=LinearConstraint(matrix, lower, upper),
        options=dict(time_limit=seconds, mip_rel_gap=.001))
    record = dict(status=int(result.status), message=result.message, elapsed_s=time.monotonic()-start,
                  nodes=n, arcs=m, solver_converged_on_supplied_graph=bool(result.success),
                  relative_gap_tolerance=.001)
    for name in ('fun', 'mip_gap', 'mip_dual_bound', 'mip_node_count'):
        value = getattr(result, name, None)
        if value is not None and np.isfinite(value): record[name] = float(value)
    selected = None
    if result.x is not None:
        selected = result.x[:n] > .5
        if not covered(selected, groups).all() or not connected_selection(selected, edges, n):
            raise RuntimeError('MILP incumbent failed independent connectivity/group checks')
    return selected, record
