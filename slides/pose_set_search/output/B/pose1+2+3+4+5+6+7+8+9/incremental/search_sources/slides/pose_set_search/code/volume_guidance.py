"""Cheap nominal material occupancy for volume candidate screening only.

No Boolean reconstruction, pressure capacity, or final acceptance is performed
here. Finalists use the unchanged exact contact-core/clearance/load model.
"""
from collections import OrderedDict
from scipy.stats import qmc
from common import *


class VolumeGuidance:
    def __init__(self, model, layouts, power=15):
        self.model = model
        bounds = []
        for layout in layouts:
            for k in layout.active:
                points = C.transform_points(model.mesh.vertices, layout.placements[k])
                bounds.extend([points.min(0)-model.thickness, points.max(0)+model.thickness])
        bounds = np.asarray(bounds)
        self.low, self.high = bounds.min(0), bounds.max(0)
        unit = qmc.Sobol(3, scramble=True, seed=17).random_base2(power)
        self.points = self.low + unit*(self.high-self.low)
        self.box_cm3 = float(np.prod(self.high-self.low)*1e6)
        self.cache = OrderedDict()

    def contains(self, layout, k, kind):
        q = layout.placements[k]
        key = (kind, k if kind != 'body' else None, np.round(q, 11).tobytes())
        if key not in self.cache:
            local = C.transform_points(self.points, np.linalg.inv(q))
            ray = (self.model.ray if kind == 'body' else
                   self.model.work_rays[k] if kind == 'work' else self.model.wrap_rays[k])
            self.cache[key] = ray.contains_points(local)
            while len(self.cache) > 256:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def estimate(self, layout):
        alive = np.zeros(len(self.points), bool)
        for k in layout.active:
            alive |= self.contains(layout, k, 'wrap')
        for k in layout.active:
            alive &= ~self.contains(layout, k, 'body')
            alive &= ~self.contains(layout, k, 'work')
        seen = set()
        for k in layout.active:
            q = layout.placements[k]
            d = q[:3, :3].T @ layout.directions[k]
            key = (np.round(q, 11).tobytes(), np.round(d, 11).tobytes())
            if key in seen:
                continue
            seen.add(key)
            ids = np.flatnonzero(alive)
            if not len(ids):
                break
            points = C.transform_points(self.points[ids], np.linalg.inv(q))
            locations, rays, _ = self.model.ray.intersects_location(
                points, np.tile(-d, (len(points), 1)), multiple_hits=False)
            distance = (points[rays]-locations) @ d
            hit = rays[(distance >= 0) & (distance <= self.model.length)]
            alive[ids[hit]] = False
        return float(alive.mean()*self.box_cm3)
