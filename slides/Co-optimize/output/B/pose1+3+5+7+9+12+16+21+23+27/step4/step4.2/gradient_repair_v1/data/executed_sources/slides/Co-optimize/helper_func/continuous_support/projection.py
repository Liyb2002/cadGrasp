"""Batched distances to the original unbounded nonnegative reaction cone.

Only the numerical variables are positively rescaled. Every projection is
checked against ALL rays, using primal nonnegativity and the NNLS KKT
conditions. No contact-area capacity, reaction cap or equilibrium LP is added.
"""
import numpy as np
from numba import njit
from whole_search.projection import cone_projection


@njit(cache=True, boundscheck=True)
def _project(rays, targets, tolerance=2e-10):
    count, dimension = targets.shape
    residuals = np.empty_like(targets)
    errors = np.zeros(count)
    iterations = np.zeros(count, np.int64)
    for index in range(count):
        target = targets[index]
        ids = np.empty(32, np.int64)
        coefficients = np.zeros(32)
        size = 0
        residual = target.copy()
        for iteration in range(160):
            reduced = rays @ residual
            entering = int(np.argmax(reduced))
            if reduced[entering] <= tolerance:
                break
            if size >= 31:
                break
            duplicate = False
            for j in range(size):
                if ids[j] == entering:
                    duplicate = True
            if duplicate:
                break
            ids[size] = entering
            coefficients[size] = 0.
            size += 1
            for inner in range(80):
                matrix = np.empty((dimension, size))
                for j in range(size):
                    matrix[:, j] = rays[ids[j]]
                proposed = np.linalg.lstsq(matrix, target, rcond=1e-12)[0]
                if np.min(proposed) > 1e-12:
                    coefficients[:size] = proposed
                    break
                fraction = 1.
                for j in range(size):
                    if proposed[j] <= 1e-12 and coefficients[j] - proposed[j] > 0:
                        fraction = min(fraction, coefficients[j] / (coefficients[j] - proposed[j]))
                coefficients[:size] += fraction * (proposed - coefficients[:size])
                retained = 0
                for j in range(size):
                    if coefficients[j] > 1e-12:
                        ids[retained] = ids[j]
                        coefficients[retained] = coefficients[j]
                        retained += 1
                size = retained
                if size == 0:
                    break
            residual = target.copy()
            for j in range(size):
                residual -= coefficients[j] * rays[ids[j]]
        residuals[index] = -residual
        reduced = rays @ residual
        violation = max(0., np.max(reduced))
        for j in range(size):
            violation = max(violation, abs(reduced[ids[j]]))
        errors[index] = violation
        iterations[index] = iteration + 1
    return residuals, errors, iterations


def project_demands(rays, targets):
    rays = np.asarray(rays, float)
    targets = np.asarray(targets, float)
    if rays.ndim != 2 or targets.ndim != 2 or rays.shape[1] != targets.shape[1]:
        raise ValueError('Reaction and target dimensions must agree')
    if not np.isfinite(rays).all() or not np.isfinite(targets).all():
        raise ValueError('Finite reaction rays and targets required')
    lengths = np.linalg.norm(rays, axis=1)
    keep = lengths > 0
    normalized = np.unique(rays[keep] / lengths[keep, None], axis=0)
    normalization = max(1., float(np.linalg.norm(targets, axis=1).max(initial=0.)))
    if len(normalized):
        residuals, errors, iterations = _project(np.ascontiguousarray(normalized),
                                                np.ascontiguousarray(targets / normalization))
        fallback = np.flatnonzero(errors > 2e-8)
        for index in fallback:
            result = cone_projection(normalized, targets[index] / normalization)
            if result['kkt_max_violation'] > 2e-8:
                raise RuntimeError('Reaction-cone projection KKT unresolved')
            residuals[index] = result['residual']
            errors[index] = result['kkt_max_violation']
        residuals *= normalization
    else:
        residuals = -targets.copy()
        errors = np.zeros(len(targets))
        iterations = np.zeros(len(targets), int)
        fallback = []
    return dict(residuals=residuals, losses=.5*np.sum(residuals**2, axis=1),
                normalized_losses=.5*np.sum((residuals / normalization)**2, axis=1),
                maximum_kkt_violation=float(errors.max(initial=0.)),
                projection_iterations=int(iterations.sum()),
                projection_fallbacks=len(fallback), equilibrium_lps=0,
                method='nonnegative_active_set_projection_with_all_ray_KKT_check')


class DemandDistance:
    """Fixed original coupled demands; Gauss magnitude quadrature + boundaries."""
    def __init__(self, grid, magnitude_nodes=3):
        nodes, weights = np.polynomial.legendre.leggauss(magnitude_nodes)
        self.magnitudes = np.r_[0., (nodes+1)*grid.maximum/2, grid.maximum]
        self.targets = (grid.base + self.magnitudes[None, :, None] * grid.slopes[:, None, :]).reshape(-1, 7)
        self.weights = (grid.weights[:, None] * np.r_[0., weights/2, 0.]).ravel()
        self.metadata = dict(grid.metadata, magnitude_integrated_by_cone_segment_intersection=False,
                             magnitude_nodes=magnitude_nodes, magnitude_rule='Gauss-Legendre',
                             zero_and_maximum_boundaries_checked=True,
                             objective='integrated_squared_distance_to_original_reaction_cone',
                             coverage_is_quadrature_diagnostic=True)

    def evaluate(self, rays):
        result = project_demands(rays, self.targets)
        residual = np.max(np.abs(result['residuals']), axis=1)
        return dict(coverage=float(self.weights @ (residual <= 2e-8)),
                    residual_loss=float(self.weights @ result['normalized_losses']),
                    maximum_demand_residual=float(residual.max(initial=0.)),
                    maximum_kkt_violation=result['maximum_kkt_violation'],
                    projection_iterations=result['projection_iterations'],
                    projection_fallbacks=result['projection_fallbacks'],
                    lp_calls=0, maximum_interval_uncertainty=0.,
                    gravity_supported=bool(np.all(residual.reshape(-1, len(self.magnitudes))[:, 0] <= 2e-8)),
                    quadrature=self.metadata)
