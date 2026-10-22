"""Cheap contact/force guidance for continuous axial juxtaposition.

Centroids are search guidance only. Accepted fixtures use real contact polygons,
all 32,768 loads per task and the complete continuous outer-cover certificate.
"""
from collections import OrderedDict
import time
import numpy as np

from whole_task.step3.legacy import kernel as K
from whole_search.model import Model
from whole_search.delta_guidance import ContactDelta


class FamilyGuidance:
    locks = Model.locks
    material_at_points = Model.material_at_points
    contact_allowed = Model.contact_allowed

    def __init__(self, model, nodes=64):
        self.model = model
        self.points = model.mesh.triangles_center.copy()
        self.sources = np.arange(len(model.mesh.faces))
        self.point_normals = model.mesh.face_normals.copy()
        self.point_areas = model.mesh.area_faces.copy()
        self.lock_cache = OrderedDict(); self.fixed_lock_cache = OrderedDict(); self.seed_cache = OrderedDict()
        self.delta = ContactDelta(self)
        self.cache = OrderedDict()
        self.nodes = nodes
        self.extra_indices = {k: [] for k in range(len(model.poses))}
        self.extra_endpoint_indices = {k: [] for k in range(len(model.poses))}
        self.queries = 0; self.seconds = 0.

    def __getattr__(self, name):
        return getattr(self.model, name)

    def add_failures(self, result):
        for k, mask in result['masks'].items():
            failed = np.flatnonzero(~mask)
            if len(failed):
                indices = failed[np.minimum(len(failed)-1, ((np.arange(min(32, len(failed)))+.5)*len(failed)/min(32, len(failed))).astype(int))]
                self.extra_indices[k] = sorted(set(self.extra_indices[k]+indices.tolist()))[-128:]
        self.cache.clear()

    def add_endpoint_failures(self, failures):
        for k, ids in failures.items():
            merged = np.unique(np.r_[self.extra_endpoint_indices[k], ids]).astype(int)
            if len(merged) > 256:
                merged = merged[np.linspace(0, len(merged)-1, 256).astype(int)]
            self.extra_endpoint_indices[k] = merged.tolist()
        self.cache.clear()

    def score(self, layout):
        self.model.activate(layout)
        key = layout.key()
        if key in self.cache:
            return self.cache[key]
        began = time.monotonic(); self.queries += 1
        flags = self.delta.state(layout).available
        failures, areas = {}, {}
        for k in layout.active:
            task = self.model.tasks[k]
            ids = np.flatnonzero(flags[k])
            native = self.model.native[k]
            points = self.points[ids] @ native[:3, :3].T+native[:3, 3]
            normals = -self.point_normals[ids] @ native[:3, :3].T
            raw = np.c_[normals, np.cross(points-task.domain.com, normals)]
            supplies = K.I.merge_columns(self.model.floor_columns(layout, k), K.U.heads(raw, task.scale))
            samples = np.minimum(len(task.targets)-1, ((np.arange(self.nodes)+.5)*len(task.targets)/self.nodes).astype(int))
            samples = np.unique(np.r_[samples, self.extra_indices[k]]).astype(int)
            endpoint = self.model.base_oracle.loads[k]
            end_ids = np.minimum(len(endpoint)-1, ((np.arange(32)+.5)*len(endpoint)/32).astype(int))
            end_ids = np.unique(np.r_[end_ids, self.extra_endpoint_indices[k]]).astype(int)
            needs = np.vstack([task.targets[samples], endpoint[end_ids]*task.scale, np.r_[-task.domain.gravity, [0., 0., 0.]]*task.scale])
            passed, _ = K.J.classify(supplies, needs)
            failures[k] = int((~passed).sum())
            areas[k] = float(self.point_areas[ids].sum())
        cloud = np.vstack([self.points @ layout.placements[k, :3, :3].T+layout.placements[k, :3, 3]
                           for k in layout.active])
        extent = np.ptp(cloud, axis=0)
        rank = (sum(value > 0 for value in failures.values()), sum(failures.values()),
                round(float(np.prod(extent)), 12), -min(areas.values()))
        result = dict(rank=rank, failures=failures, areas=areas,
                      elapsed_s=time.monotonic()-began, guide_only=True)
        self.cache[key] = result
        while len(self.cache) > 256:
            self.cache.popitem(last=False)
        self.seconds += result['elapsed_s']
        return result

    def commit(self, layout):
        self.model.activate(layout)
        self.delta.commit(layout)
