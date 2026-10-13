"""Continuous distance to the existing unbounded contact wrench cone.

Generators are rows, including the existing seventh lifted coordinate. No
contact-force capacities are introduced. Positive ray scaling (for example by
contact area) cannot change an unbounded cone. ``scale`` is an optional fixed
metric: loss = .5 * ||scale * (predicted - target)||**2.
"""
import numpy as np
from scipy.optimize import nnls, lsq_linear


def _rows(generators):
    rays = np.asarray(generators, dtype=float)
    if rays.size == 0:
        rays = np.empty((0, 7))
    if rays.ndim != 2 or rays.shape[1] != 7 or not np.isfinite(rays).all():
        raise ValueError('generators must be finite rows of shape (N, 7)')
    return rays


def cone_projection(generators, target, scale=None):
    """Project a seven-vector onto a nonnegative, unbounded generator cone.

    ``residual`` is prediction minus target in input units. ``dual`` is the
    metric-weighted residual, i.e. gradient with respect to prediction; the
    target gradient is ``-dual``. Coefficients refer to the original rows.
    Exact duplicate normalized rays are merged before the low-dimensional
    solve, with their coefficient assigned to the first original occurrence.
    """
    rays = _rows(generators)
    target = np.asarray(target, dtype=float)
    metric = np.ones(7) if scale is None else np.asarray(scale, dtype=float)
    if target.shape != (7,) or not np.isfinite(target).all():
        raise ValueError('target must be a finite seven-vector')
    if metric.shape != (7,) or not np.isfinite(metric).all() or np.any(metric <= 0):
        raise ValueError('scale must contain seven finite positive values')
    weighted = rays * metric
    norms = np.linalg.norm(weighted, axis=1)
    valid = np.flatnonzero(norms > 0)
    coefficients = np.zeros(len(rays))
    solver = 'empty'
    if len(valid):
        normalized = weighted[valid] / norms[valid, None]
        unique, first = np.unique(normalized, axis=0, return_index=True)
        original = valid[first]
        matrix = unique.T
        try:
            if len(unique) <= 128:
                values, _ = nnls(matrix, target * metric, maxiter=max(100, 10 * len(unique)))
                solver = 'nnls'
            else:
                # scipy 1.14 NNLS builds N x N Gram matrices. Column generation
                # instead solves tiny active cones and scans all seven-D rays.
                selected = []
                values = np.zeros(len(unique))
                tolerance = 1e-10 * max(1., np.linalg.norm(target * metric))
                for _ in range(256):
                    costs = matrix.T @ (matrix @ values - target * metric)
                    worst = int(np.argmin(costs))
                    if costs[worst] >= -tolerance:
                        break
                    if worst in selected:
                        raise RuntimeError('active cone stalled')
                    selected.append(worst)
                    active_values, _ = nnls(matrix[:, selected], target * metric,
                                            maxiter=max(100, 10 * len(selected)))
                    values[:] = 0.
                    values[selected] = active_values
                    selected = [index for index in selected if values[index] > 1e-12]
                else:
                    raise RuntimeError('active cone iteration limit')
                solver = 'column_generation_nnls'
        except (RuntimeError, np.linalg.LinAlgError):
            fallback = lsq_linear(matrix, target * metric, bounds=(0, np.inf),
                                  tol=1e-12, max_iter=1000, lsq_solver='exact')
            if not fallback.success:
                raise RuntimeError('contact cone projection failed: ' + fallback.message)
            values = fallback.x
            solver = 'lsq_linear'
        coefficients[original] = values / norms[original]
    residual = coefficients @ rays - target
    dual = metric**2 * residual
    reduced = rays @ dual
    # KKT: reduced costs nonnegative; active coefficients have zero cost.
    dual_violation = max(0., float(-np.min(reduced))) if len(reduced) else 0.
    active = coefficients > 1e-10
    active_violation = float(np.max(np.abs(reduced[active]))) if active.any() else 0.
    return dict(loss=float(.5 * np.dot(metric * residual, metric * residual)),
                residual=residual, dual=dual, coefficients=coefficients,
                kkt_max_violation=max(dual_violation, active_violation),
                solver=solver)


def missing_values(generators, projection, normalize=False):
    """Positive first-order benefit of adding each missing generator.

    A value > 0 means a small nonnegative force along this ray reduces loss.
    This is a reduced cost, not an area derivative or finite force capacity.
    Optional normalization gives a direction-only comparison in input units.
    """
    rays = _rows(generators)
    values = np.maximum(0., -(rays @ np.asarray(projection['dual'])))
    if normalize:
        norms = np.linalg.norm(rays, axis=1)
        values = np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)
    return values
