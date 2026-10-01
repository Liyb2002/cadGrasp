"""Shared workpiece/support reactions, retaining ownership of every contact ray.

Forces are in object-weight units, positions in metres. All moments are about
the workpiece COM; rows are conditioned by the same length as Step 3.
"""
from step1.needs import COORD
from fractions import Fraction
import numpy as np
from scipy.linalg import qr
from scipy.optimize import linprog

TOL = 2e-9
OPTIONS = {'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9}


def wrench(points, forces, origin):
    return np.concatenate([forces, np.cross(points-origin, forces)], axis=-1)


def contact_rays(domain, contacts, pivot):
    points = [np.asarray(pivot).reshape(1, 3)]
    normals = [np.array([[0., 0., 1.]])]
    owners = [np.array([-1])]
    for j, contact in enumerate(contacts):
        p = contact['triangles_m'].reshape(-1, 3)
        n = np.repeat(-domain.mesh.face_normals[contact['source_faces']], 3, axis=0)
        # Exact duplicate removal WITHIN a support only, never across owners.
        ids = np.sort(np.unique(np.c_[p, n], axis=0, return_index=True)[1])
        points.append(p[ids]); normals.append(n[ids]); owners.append(np.full(len(ids), j))
    return np.concatenate(points), np.concatenate(normals), np.concatenate(owners)


def relaxed_matrix(points, normals, owners, origin, scale, count):
    """Necessary test allowing any floor moment/tangential reaction, N_j >= 0.

    A failed test rules out EVERY unanchored footprint. Passing only seeds the
    finite-footprint design; it is not an independent-support certificate.
    """
    g = wrench(points, normals, origin)*scale
    matrix = np.zeros((6+count, len(points)+count))
    matrix[:6, :len(points)] = g.T
    for j in range(count):
        ids = np.flatnonzero(owners == j)
        matrix[6+j, ids] = normals[ids, 2]
        matrix[6+j, len(points)+j] = -1.
    return matrix


def grounded_matrix(points, normals, owners, origin, scale, feet, friction):
    """Actual floor vertices with an inscribed four-ray Coulomb cone.

    Each floor ray has strictly positive normal force, so no unloaded support
    can obtain a horizontal force or yaw couple for free. A reported finite mu
    is a sufficient friction requirement, consistent with 'sufficient friction'.
    """
    count = len(feet)
    columns = []
    floor_points, floor_forces, floor_owners = [], [], []
    for point, normal, owner in zip(points, normals, owners):
        g = wrench(point, normal, origin)*scale
        column = np.zeros(6*(count+1)); column[:6] = g
        if owner >= 0:
            column[6*(owner+1):6*(owner+2)] = -g
        columns.append(column)
    rays = np.array([[friction, 0, 1], [-friction, 0, 1],
                     [0, friction, 1], [0, -friction, 1]], float)
    for j, foot in enumerate(feet):
        xy = np.concatenate(foot['pads_xy_m'])
        for point in COORD.lift_floor(xy):
            for force in rays:
                column = np.zeros(6*(count+1))
                column[6*(j+1):6*(j+2)] = wrench(point, force, origin)*scale
                columns.append(column)
                floor_points.append(point); floor_forces.append(force); floor_owners.append(j)
    return np.asarray(columns).T, dict(points_m=np.asarray(floor_points),
        forces=np.asarray(floor_forces), owners=np.asarray(floor_owners, int))


def padded_targets(loads, scale, rows):
    targets = np.zeros((len(loads), rows)); targets[:, :6] = loads*scale
    return targets


def basis_for(matrix, solution):
    """Keep the basic solution and complete it with zero-weight columns."""
    m = matrix.shape[0]
    active = np.flatnonzero(solution > 1e-11)
    if len(active) > m:
        return None
    selected, orthogonal = [], []
    # Prioritise used columns. This avoids changing the force allocation.
    for index in np.r_[active, np.setdiff1d(np.arange(matrix.shape[1]), active)]:
        v = matrix[:, index].copy()
        for q in orthogonal:
            v -= (q@v)*q
        for q in orthogonal:
            v -= (q@v)*q
        norm = np.linalg.norm(v)
        if norm > 1e-10*max(1., np.linalg.norm(matrix[:, index])):
            selected.append(index); orthogonal.append(v/norm)
        if len(selected) == m:
            return np.asarray(selected)
    return None


def membership(matrix, ids, targets, certified=False):
    basis = matrix[:, ids]
    inv = np.linalg.inv(basis)
    weights = targets@inv.T
    residual = np.max(np.abs(weights@basis.T-targets), axis=1)
    if not certified:
        good = (weights.min(axis=1) >= -TOL) & (residual <= TOL)
        return good, weights, residual
    # Neumann-series inverse bound plus explicit matrix/target entry guard.
    # This is the same kind of sufficient coefficient bound used by Step 3,
    # extended to the per-body system. It may decline boundary certificates.
    round_guard = 256*np.finfo(float).eps
    entry_guard = 1e-10
    invnorm = np.linalg.norm(inv, np.inf)*(1+round_guard)
    eta = np.linalg.norm(np.eye(len(ids))-basis@inv, np.inf)
    eta += round_guard*np.linalg.norm(np.abs(basis)@np.abs(inv)+np.eye(len(ids)), np.inf)
    entry_error = entry_guard*(1+np.abs(basis))
    eta += np.linalg.norm(entry_error@np.abs(inv), np.inf)
    if not np.isfinite(eta) or eta >= 1:
        return np.zeros(len(targets), bool), weights, np.full(len(targets), -np.inf)
    inverse_bound = invnorm/(1-eta)*(1+round_guard)
    error = residual+round_guard*np.max(np.abs(weights)@np.abs(basis).T+np.abs(targets), axis=1)
    error += np.max(np.abs(weights)@entry_error.T+entry_guard*(1+np.abs(targets)), axis=1)
    margin = weights.min(axis=1)-inverse_bound*error
    return margin > 0, weights, margin


class BatchSolver:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, float)
        self.bases = []
        self.lp_count = 0

    def solve(self, targets, certified=False):
        targets = np.asarray(targets, float)
        n, m = targets.shape
        assignment = np.full(n, -1, int)
        weights = np.zeros((n, m))
        diagnostics = []
        for k, ids in enumerate(self.bases):
            pending = np.flatnonzero(assignment < 0)
            if not len(pending): break
            good, x, _ = membership(self.matrix, ids, targets[pending], certified)
            assignment[pending[good]] = k; weights[pending[good]] = x[good]
        while np.any(assignment < 0):
            pending = np.flatnonzero(assignment < 0)
            index = int(pending[0])
            self.lp_count += 1
            result = linprog(np.ones(self.matrix.shape[1]), A_eq=self.matrix,
                             b_eq=targets[index], bounds=(0, None),
                             method='highs-ds', options=OPTIONS)
            if not result.success:
                retry = linprog(np.zeros(self.matrix.shape[1]), A_eq=self.matrix,
                                b_eq=targets[index], bounds=(0, None), method='highs-ipm',
                                options=dict(OPTIONS, presolve=False))
                if retry.success:
                    result = retry
                else:
                    diagnostics.append(dict(index=index, status='infeasible_numeric' if result.status == retry.status == 2 else 'solver_unresolved',
                        simplex_status=int(result.status), ipm_status=int(retry.status)))
                    break
            ids = basis_for(self.matrix, result.x)
            if ids is None:
                diagnostics.append(dict(index=index, status='rank_deficient_certificate_unresolved'))
                break
            good, x, margins = membership(self.matrix, ids, targets[pending], certified)
            if certified and not good[0]:
                # The minimum-total-force LP may choose a degenerate basis
                # (e.g. an unused support under pure gravity). Look for a
                # different allocation of the SAME load and fixed feet.
                for seed in range(8):
                    self.lp_count += 1
                    objective = np.random.default_rng(seed).uniform(.5, 1.5, self.matrix.shape[1])
                    alternate = linprog(objective, A_eq=self.matrix, b_eq=targets[index],
                        bounds=(0, None), method='highs-ds', options=OPTIONS)
                    if not alternate.success: continue
                    alternative_ids = basis_for(self.matrix, alternate.x)
                    if alternative_ids is None: continue
                    ok, values, bound = membership(self.matrix, alternative_ids, targets[pending], True)
                    if ok[0]:
                        ids, good, x, margins = alternative_ids, ok, values, bound
                        break
            if not good[0]:
                diagnostics.append(dict(index=index, status='coefficient_certificate_unresolved' if certified else 'basis_unresolved',
                                        minimum_coefficient=float(x[0].min())))
                break
            assignment[pending[good]] = len(self.bases)
            weights[pending[good]] = x[good]
            self.bases.append(ids)
        return dict(passed=bool(np.all(assignment >= 0)), assignment=assignment,
                    weights=weights, diagnostics=diagnostics, lp_count=self.lp_count)


def support_wrenches(solution, solver, points, normals, owners, origin, count):
    raw = wrench(points, normals, origin)
    result = np.zeros((len(solution['assignment']), count, 6))
    for k, basis in enumerate(solver.bases):
        rows = np.flatnonzero(solution['assignment'] == k)
        if not len(rows): continue
        for j in range(count):
            slots = np.flatnonzero((basis < len(points)) & (np.r_[owners, np.full(solver.matrix.shape[1]-len(points), -2)][basis] == j))
            result[rows, j] = solution['weights'][rows][:, slots]@raw[basis[slots]]
    return result


def pressure_centers(wrenches, origin):
    moments = wrenches[..., 3:]+np.cross(origin, wrenches[..., :3])
    normal = wrenches[..., 2]
    points = np.full(wrenches.shape[:-1]+(2,), np.nan)
    loaded = normal > 1e-10
    points[loaded] = np.stack([-moments[..., 1][loaded], moments[..., 0][loaded]], axis=-1)/normal[loaded, None]
    return points, loaded


def exact_relaxed_separator(matrix, target, points, normals, owners, origin, scale, count, physical_target=None):
    """Farkas witness checked on rational point/normal geometry, not rounded G."""
    result = linprog(target, A_ub=-matrix.T, b_ub=np.zeros(matrix.shape[1]),
                     bounds=[(-1, 1)]*matrix.shape[0], method='highs', options=OPTIONS)
    if not result.success or result.fun >= -1e-9:
        return None
    # A tiny interior perturbation of the dual constraints is tried first to
    # avoid treating floating-point zeros as exact signs.
    candidates = [result.x]
    for guard in (1e-11, 1e-10, 1e-9):
        r = linprog(target, A_ub=-matrix.T, b_ub=np.full(matrix.shape[1], -guard),
                    bounds=[(-1, 1)]*matrix.shape[0], method='highs', options=OPTIONS)
        if r.success and r.fun < -1e-9: candidates.insert(0, r.x)
    Q = lambda x: Fraction(float(x))
    origin_q = list(map(Q, origin))
    physical_target = target[:6]/scale if physical_target is None else physical_target
    target_q = list(map(Q, physical_target))+[Fraction(0)]*count
    exact = []
    for point, normal, owner in zip(points, normals, owners):
        n = list(map(Q, normal)); r = [Q(p)-c for p, c in zip(point, origin_q)]
        g = n+[r[1]*n[2]-r[2]*n[1], r[2]*n[0]-r[0]*n[2], r[0]*n[1]-r[1]*n[0]]+[Fraction(0)]*count
        if owner >= 0: g[6+owner] = n[2]
        exact.append(g)
    for j in range(count):
        g = [Fraction(0)]*(6+count); g[6+j] = Fraction(-1); exact.append(g)

    def check(dual):
        scores = [sum(a*b for a, b in zip(dual, g)) for g in exact]
        debt = sum(a*b for a, b in zip(dual, target_q))
        if min(scores) >= 0 and debt < 0:
            return dict(dual_exact=list(map(str, dual)), min_generator_dot_exact=str(min(scores)),
                        target_dot_exact=str(debt), exact_rational_geometry=True,
                        explanation='Every available generator has nonnegative dual dot product; the required load has a strictly negative dot product.')
        return None

    for vector in candidates:
        dual = [Q(vector[i])*Q(scale[i]) for i in range(6)]+list(map(Q, vector[6:]))
        verified = check(dual)
        if verified: return verified
    # Dual optima often lie exactly on facets. Recover that vertex from its
    # active generator/bound constraints using rational geometry, then check
    # EVERY original generator. No tolerance can make a negative sign pass.
    from step3_scheculer.verification import rational_solve
    d = len(target); scaling = list(map(Q, scale))+[Fraction(1)]*count
    scaled_exact = [[a*b for a, b in zip(g, scaling)] for g in exact]
    for threshold in (1e-10, 1e-8, 1e-6):
        active = np.flatnonzero(np.abs(matrix.T@result.x) <= threshold)
        rows = [scaled_exact[i] for i in active]
        rhs = [Fraction(0)]*len(rows)
        for i in range(d):
            if abs(abs(result.x[i])-1) < threshold:
                row = [Fraction(0)]*d; row[i] = Fraction(1)
                rows.append(row); rhs.append(Fraction(1 if result.x[i] > 0 else -1))
        if len(rows) < d: continue
        numeric = np.array(rows, float)
        for seed in range(4):
            order = np.arange(len(rows)) if seed == 0 else np.random.default_rng(seed).permutation(len(rows))
            _, upper, pivots = qr(numeric[order].T, mode='economic', pivoting=True)
            if np.min(np.abs(np.diag(upper[:, :d]))) < 1e-12: continue
            chosen = order[pivots[:d]]
            try:
                solution = rational_solve([rows[i] for i in chosen], [rhs[i] for i in chosen])
                verified = check([a*b for a, b in zip(solution, scaling)])
                if verified: return verified
            except ValueError:
                continue
    return None
