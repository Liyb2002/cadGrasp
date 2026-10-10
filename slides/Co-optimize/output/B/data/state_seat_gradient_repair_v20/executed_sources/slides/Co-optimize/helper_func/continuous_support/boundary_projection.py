"""Positive Lobatto weights make the objective include checked force boundaries."""
import numpy as np
from .projection import DemandDistance,project_demands


class BoundaryDistance(DemandDistance):
    def __init__(self, grid):
        # Five-point Gauss-Lobatto integrates polynomials up to degree seven
        # in the ORIGINAL uniform magnitude measure, including both endpoints.
        nodes = np.array([-1., -np.sqrt(3/7), 0., np.sqrt(3/7), 1.])
        weights = np.array([1/20, 49/180, 16/45, 49/180, 1/20])
        self.magnitudes = (nodes+1)*grid.maximum/2
        self.targets = (grid.base+self.magnitudes[None,:,None]*grid.slopes[:,None,:]).reshape(-1,7)
        self.weights = (grid.weights[:,None]*weights).ravel()
        self.unique_targets,inverse=np.unique(self.targets,axis=0,return_inverse=True)
        self.unique_weights=np.bincount(inverse,weights=self.weights,minlength=len(self.unique_targets))
        self.gravity_indices=np.unique(inverse.reshape(-1,5)[:,0])
        self.metadata = dict(grid.metadata,magnitude_integrated_by_cone_segment_intersection=False,
            magnitude_nodes=5,magnitude_rule='Gauss-Lobatto',magnitude_polynomial_degree=7,
            zero_and_maximum_boundaries_checked=True,checked_boundaries_have_positive_objective_weight=True,
            objective='integrated_squared_distance_to_original_reaction_cone',coverage_is_quadrature_diagnostic=True)

    def evaluate(self,rays):
        result=project_demands(rays,self.unique_targets)
        residual=np.max(np.abs(result['residuals']),axis=1)
        return dict(coverage=float(self.unique_weights@(residual<=2e-8)),
            residual_loss=float(self.unique_weights@result['normalized_losses']),
            maximum_demand_residual=float(residual.max(initial=0.)),
            maximum_kkt_violation=result['maximum_kkt_violation'],
            projection_iterations=result['projection_iterations'],projection_fallbacks=result['projection_fallbacks'],
            lp_calls=0,maximum_interval_uncertainty=0.,gravity_supported=bool(np.all(residual[self.gravity_indices]<=2e-8)),
            quadrature=dict(self.metadata,unique_projected_demands=len(self.unique_targets),original_quadrature_demands=len(self.targets)))
