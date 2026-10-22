"""Original Co-optimize freedoms with a vertically translating work window.

Per-task reference data stay immutable. Translating a task, its application points and COM
together leaves every paired COM wrench unchanged. Search uses grounded native
frames; positive seat heights automatically disable object-floor reactions.
"""
from copy import deepcopy
import json
import numpy as np

from .adapter import ComputedInputs, ComputedMeasure, ComputedOracle, TaskModel
from whole_task.step1.common import DEFAULT_OUTPUT
from whole_task.common.wrenches import WrenchDomain, endpoint_demands
from whole_task.step3.legacy import kernel as K
from whole_search.model import Model
from oracle_search import OracleModel
from continuous_support.stable_gradient import StableGradientModel
from fixture_state import installed_state


class MobileInputs(ComputedInputs):
    def __init__(self, *args, station=None, **kwargs):
        if station is None:
            from whole_task.step3.input_state import require_joint_initialization
            require_joint_initialization()
        super().__init__(*args, **kwargs)
        self.source_native = self.native.copy()
        self.source_heights = np.array([t.domain.mesh.bounds[0, 2] for t in self.tasks])
        self.native[:, 2, 3] -= self.source_heights
        self.floors = []
        self.floor_vertices = []
        for k, task in enumerate(self.tasks):
            delta = np.array([0., 0., -self.source_heights[k]])
            data = deepcopy(task.domain.data)
            data['geometry']['vertices_m'] = (task.domain.mesh.vertices + delta).tolist()
            data['frame']['moment_origin_m'] = (task.domain.com + delta).tolist()
            data['frame']['T_world_mesh'] = self.native[k].tolist()
            task.domain = WrenchDomain(data)
            # One ACTUAL lowest mesh vertex; never use the COM's projection.
            vertices = task.domain.mesh.vertices
            candidates = np.flatnonzero(vertices[:, 2] <= vertices[:, 2].min() + 1e-12)
            center = vertices[candidates, :2].mean(axis=0)
            index = candidates[np.argmin(np.linalg.norm(vertices[candidates, :2] - center, axis=1))]
            task.floor = vertices[index].copy()
            self.floor_vertices.append(int(index))
            self.floors.append(K.U.floor(K.FLOOR.columns(task.floor, task.domain.com), task.scale))
            # Check all saved samples, including provisional work faces.
            task.forward_nodes(np.arange(len(task.targets)))
        self.states = [(task, self.native[k], self.mesh) for k, task in enumerate(self.tasks)]
        self.station = station
        reach = self.station['workspace']['base_reach_estimate']
        self.reach_center = np.asarray(reach['center_world_m'])
        self.reach_radius = float(reach['usable_radius_m'])
        reference = np.asarray(self.station['reference_surface_point_world_m'])
        tcp = np.asarray(self.station['world_from_reference_tcp'])[:3, 3]
        flange = np.asarray(self.station['world_from_reference_flange'])[:3, 3]
        self.tool_length = float(np.linalg.norm(flange-tcp))
        self.flange_offset = flange-reference
        self.center_xy = reference[:2].copy()
        source_centers = np.einsum('kij,j->ki', self.source_native[:, :3, :3], self.mesh.center_mass)
        source_centers += self.source_native[:, :3, 3]
        self.center_shifts = np.zeros((len(self.tasks), 3))
        self.center_shifts[:, :2] = self.center_xy-source_centers[:, :2]
        self.flange_vertices = []
        for k, task in enumerate(self.tasks):
            points = task.domain.triangles.reshape(-1, 3)
            self.flange_vertices.append(points + self.flange_offset + self.center_shifts[k])
        # Bind new adapter sources to runtime cache identity as physical inputs.
        from pathlib import Path
        self.inputs += list(Path(__file__).parent.glob('*.py'))


class ContinuousFloorOracle(ComputedOracle):
    """Conservative continuous pressure envelope for the local force cones.

Every demand lies in the triangle-corner x outer-force-polytope hull.
Since Fz>0, its pressure center is the same convex combination with
weights renormalized by Fz. Endpoint pressure hulls therefore contain the
entire continuous task, including all saved samples, at ANY translated COM.
Outer targets may exceed the true cone slightly; containment is sufficient,
not an exact feasibility equivalence. Force classification keeps all samples.
"""
    def __init__(self, model):
        super().__init__(model)
        self.loads=[endpoint_demands(task.domain)['endpoint_need_wrench'] for task in model.tasks]
        if any(np.min(loads[:,2])<=0 for loads in self.loads):
            raise ValueError('Continuous pressure-hull reduction requires positive Fz')


class MobileTaskModel(OracleModel, ComputedMeasure, MobileInputs):
    def __init__(self, poses=None, name='B', output_root=DEFAULT_OUTPUT):
        self.support_floor_clearance_m = 0.
        StableGradientModel.__init__(self, poses, name=name, output_root=output_root)
        self.base_oracle = ContinuousFloorOracle(self)
        self.oracle_gate_active = False
        self.require_connected_core = True

    def validate_placement(self, layout, k):
        Model.validate_placement(self, layout, k)
        # Search XY is a gauge freedom. Only the centered execution is tested
        # against the unchanged base-reach estimate, including EVERY work vertex.
        height = self.workpiece_height(layout, k)
        flange = self.flange_vertices[k] + np.array([0., 0., height])
        if np.max(np.linalg.norm(flange-self.reach_center, axis=1)) > self.reach_radius + 1e-10:
            raise ValueError('Centered execution exceeds configured geometric flange reach')

    check_station = TaskModel.check_station
    work_length = TaskModel.work_length
    work_solid = TaskModel.work_solid
    verify_work = TaskModel.verify_work

    def execution(self, layout, k):
        """Apply ONE horizontal translation to object, support and work volume."""
        self.validate_placement(layout, k)
        T = installed_state(self.native, layout, layout.hosts[k])
        U = T @ layout.placements[k]
        D = np.eye(4)
        current_center = U[:3,:3] @ self.mesh.center_mass + U[:3,3]
        D[:2, 3] = self.center_xy-current_center[:2]
        support, body = D @ T, D @ U
        shift = body[:3, 3] - self.source_native[k, :3, 3]
        head = np.asarray(self.station['workspace']['spatial_cone']['apex_world_m']) + shift
        np.testing.assert_allclose(support @ layout.placements[k], body, atol=1e-12, rtol=0)
        np.testing.assert_allclose((body[:3,:3] @ self.mesh.center_mass + body[:3,3])[:2], self.center_xy, atol=1e-12, rtol=0)
        height = self.workpiece_height(layout, k)
        flange = self.flange_vertices[k] + np.array([0., 0., height])
        actual_floor = self.tasks[k].floor + body[:3, 3] - self.native[k, :3, 3]
        return dict(pose=self.poses[k], search_host=int(layout.hosts[k]),
            search_world_from_object=U.tolist(), search_world_from_support=T.tolist(),
            center_translation_m=D[:3, 3].tolist(), world_from_object=body.tolist(),
            object_COM_world_m=(body[:3,:3] @ self.mesh.center_mass + body[:3,3]).tolist(),
            world_from_support=support.tolist(), work_window_translation_m=shift.tolist(),
            work_window_center_m=(np.asarray(self.station['reference_surface_point_world_m'])+shift).tolist(),
            head_world_m=head.tolist(), needle_axis_world=self.tasks[k].domain.axis.tolist(),
            object_floor_gap_m=height, object_floor_reaction_enabled=self.floor_signature(layout, k),
            actual_lowest_vertex_index=self.floor_vertices[k], lowest_vertex_world_m=actual_floor.tolist(),
            maximum_flange_base_distance_m=float(np.max(np.linalg.norm(flange-self.reach_center, axis=1))),
            usable_base_reach_m=self.reach_radius, original_COM_wrenches_preserved=True)
