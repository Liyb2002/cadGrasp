"""Step5.3: a deterministic connected enclosure, with work/body cuts only.

No exit sweep is constructed or subtracted. Floor legality belongs to
Step5.4. A later design-envelope request is an input shape, not a new
obstacle or a relaxation of work access. Units use the existing mm kernel.
"""
from itertools import product
from copy import copy

import numpy as np
from scipy.spatial import ConvexHull, QhullError

import stage5_common as C
import base_geometry as G
from work_access import WorkAccess
from policy import POLICY


def enclosure(mesh, data):
    """All demand hulls in the common frame, inside conv(actual material).

    This is an enclosure, not a claim that the material touches the floor.
    In a floor-legal seated state the same test reduces to ground coverage.
    """
    reports = []
    try:
        hull = ConvexHull(mesh.vertices)
    except (QhullError, ValueError):
        return dict(passed=False, per_pose=[], reason='no_volumetric_material_hull')
    for row in data['rows']:
        points = C.F.transform_points(row['hull'], np.linalg.inv(row['fixture_transform']))
        errors = points @ hull.equations[:, :3].T + hull.equations[:, 3]
        gap = max(0., float(errors.max()))
        reports.append(dict(pose=row['pose'], passed=gap <= 1e-9,
                            maximum_common_3d_hull_gap_m=gap))
    return dict(passed=all(r['passed'] for r in reports), per_pose=reports,
                kind='convex enclosure of actual connected material; not ground contact')


class WorkOnlyBase:
    def __init__(self, data, tasks, body, padding=.02, design_envelope=None):
        if not POLICY.forbid_entire_work_cones or POLICY.forbid_exit_paths:
            raise ValueError('The fixed Step5.3 work-only policy was changed')
        self.data, self.tasks, self.body = data, tasks, body
        self.placements = [np.linalg.inv(r['fixture_transform']) @ r['object_transform'] for r in data['rows']]
        points = [data['fixture'].vertices]
        for row, placement in zip(data['rows'], self.placements):
            points.append(C.F.transform_points(body.vertices, placement))
            points.append(C.F.transform_points(row['hull'], np.linalg.inv(row['fixture_transform'])))
        points = np.vstack(points)
        self.low, self.high = points.min(0)-padding, points.max(0)+padding
        self.window = np.array(list(product(*zip(self.low, self.high))))
        box = G.md.Manifold.cube(tuple((self.high-self.low)*1000)).translate(tuple(self.low*1000))
        self.stock = box if design_envelope is None else box ^ design_envelope
        self.old = G.solid(data['fixture'])
        self.work, self.guarded_work, bodies, self.allowed, self.exclusions = [], [], [], [], []
        self.work_guard_m = .0001
        local_body = G.solid(body)
        for task, row, placement in zip(tasks, data['rows'], self.placements):
            bodies.append(G.transform(local_body, placement))
            local_window = C.F.transform_points(self.window, np.linalg.inv(placement))
            access = WorkAccess(body, task.domain.work_ids, task.domain.data['load']['cone_half_deg'])
            length = access.required_length(local_window, minimum=.001)
            self.work.append(G.transform(G.from_legacy(access.solid(length)), placement))
            # Move each cone's triangular origin backward along its own
            # normal. Since the normal lies inside its cone, this enlarged
            # cone contains the original cone with positive side clearance.
            # Test collisions against the unchanged ORIGINAL cone below.
            guarded = WorkAccess(body, task.domain.work_ids, task.domain.data['load']['cone_half_deg'])
            guarded.triangles -= self.work_guard_m*guarded.normals[:, None, :]
            guarded_length = guarded.required_length(local_window, minimum=.001)
            self.guarded_work.append(G.transform(G.from_legacy(guarded.solid(guarded_length)), placement))
            # Allow every non-work face. Exit-facing contacts are permitted.
            self.allowed.append(np.setdiff1d(np.arange(len(body.faces)), task.domain.work_ids))
            self.exclusions.append(dict(pose=row['pose'], work_cap_m=length,
                cap_beyond_entire_design_window=True, definition=access.definition(),
                cone_origin_backward_guard_m=self.work_guard_m, exit_path_subtracted=False))
        self.work_cut, self.body_cut = G.combine(self.work), G.combine(bodies)
        self.guarded_work_cut = G.combine(self.guarded_work)
        self.free = self.stock - self.guarded_work_cut - self.body_cut

    def in_envelope(self, envelope):
        """A new Step5.3 design using Step5.4's returned envelope request."""
        result = copy(self)
        result.stock = self.stock ^ envelope
        result.free = result.stock - self.guarded_work_cut - self.body_cut
        return result

    def components(self):
        return sorted((p for p in self.free.decompose() if G.volume(p) > 1e-12),
                      key=lambda p: (-G.volume(p), tuple(G.unpack(p).bounds.ravel())))

    def checks(self, value):
        work = G.volume(value ^ self.work_cut)
        body = G.volume(value ^ self.body_cut)
        count = len(value.decompose())
        return dict(passed=work <= G.EPS_VOLUME and body <= G.EPS_VOLUME and count == 1,
            component_count=count, entire_work_cone_overlap_cm3=work*1e6,
            stationary_body_overlap_cm3=body*1e6, exit_paths_allowed=True,
            floor_checked_here=False, policy=POLICY.record(), exclusions=self.exclusions,
            material_volume_cm3=G.volume(value)*1e6,
            retained_old_material_cm3=G.volume(value ^ self.old)*1e6,
            added_material_cm3=G.volume(value-self.old)*1e6,
            omitted_old_material_cm3=G.volume(self.old-value)*1e6,
            stock_common_bounds_m=[self.low.tolist(), self.high.tolist()])
