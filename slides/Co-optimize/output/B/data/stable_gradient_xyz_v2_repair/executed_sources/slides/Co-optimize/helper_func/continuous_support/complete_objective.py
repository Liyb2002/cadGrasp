"""Cached continuous guidance, with torque-extreme positions in fine checks."""
from .boundary_objective import ProductionObjective,TimedDistance
from .complete_grid import CompleteGrid


class CompleteObjective(ProductionObjective):
    def __init__(self,model,initial,quadrature_level=1,contact_depth=1):
        super().__init__(model,initial,quadrature_level,contact_depth)
        if quadrature_level>1:
            self.grids=[CompleteGrid.from_task(task,quadrature_level) for task in model.tasks]
            self.distances=[TimedDistance(grid) for grid in self.grids]
