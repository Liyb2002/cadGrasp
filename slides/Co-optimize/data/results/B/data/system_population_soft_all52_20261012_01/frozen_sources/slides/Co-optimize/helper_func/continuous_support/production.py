"""Cancellation-aware objective using cached exact two-dimensional shadows."""
import json
import os
import time
from .objective import CoverageObjective
from .projection import DemandDistance
from .incremental_surface import IncrementalSurfaceCuts
from run_all import output_root


class TimedDistance(DemandDistance):
    def __init__(self, grid):
        super().__init__(grid); self.seconds = 0.
    def evaluate(self, rays):
        started = time.monotonic()
        result = super().evaluate(rays)
        self.seconds += time.monotonic()-started
        return result


class ProductionObjective(CoverageObjective):
    def __init__(self, model, initial, quadrature_level=1, contact_depth=1):
        super().__init__(model, initial, quadrature_level, contact_depth)
        self.geometry.surface = IncrementalSurfaceCuts(model)
        self.distances = [TimedDistance(grid) for grid in self.grids]
        self.generation = output_root(model.name)/'data/whole_step4_active_run.json'
        self.run_token = os.environ.get('COOPT_WHOLE_RUN_TOKEN')
    def evaluate(self, layout):
        if self.run_token is not None and self.generation.exists():
            if json.loads(self.generation.read_text())['run_token'] != self.run_token:
                raise RuntimeError('Cancelled continuous optimization generation')
        return super().evaluate(layout)
    def timing(self):
        surface = self.geometry.surface
        return dict(surface_seconds=surface.seconds,
                    projection_seconds=sum(distance.seconds for distance in self.distances),
                    slice_cache_hits=surface.slice_hits, slice_cache_misses=surface.slice_misses)
