"""Mutable task orientation at juxtaposition, with explicit angle-aware states.

Object-space regions and finite head volumes are invariant. World geometry,
paired loads, floor reactions and robot reach are refreshed at every angle.
"""
from collections import OrderedDict
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .mobile import MobileTaskModel, ContinuousFloorOracle
from whole_task_2.common.poses import spin_pose
from whole_task_2.common.wrenches import WrenchDomain, endpoint_demands
from whole_task_2.step3.input_state import FAMILY_ROOT
from whole_task_2.step3.legacy import kernel as K
from whole_search.model import Layout
from whole_search.common import legal_direction
from whole_search.delta_guidance import ContactDelta
from fixture_state import installed_state
from juxtapose.reuse_search import JumpScreen


class AngularLayout(Layout):
    def __init__(self, placements, directions, hosts, active, state_offsets=None, angles_deg=None):
        super().__init__(placements, directions, hosts, active, state_offsets)
        self.angles_deg = (np.zeros(len(hosts)) if angles_deg is None
                           else np.asarray(angles_deg, dtype=float).copy())
        if self.angles_deg.shape != (len(hosts),) or not np.isfinite(self.angles_deg).all():
            raise ValueError('Expected one finite axial angle per work region')

    def copy(self):
        return type(self)(self.placements.copy(), self.directions.copy(), self.hosts.copy(),
                          self.active, self.state_offsets.copy(), self.angles_deg.copy())

    def key(self):
        return super().key()+(np.round(self.angles_deg % 360., 10).tobytes(),)

    def arrays(self, native=None):
        return dict(super().arrays(native), axial_angles_deg=self.angles_deg)

    @classmethod
    def from_arrays(cls, arrays):
        return cls(arrays['placements'], arrays['directions'], arrays['hosts'],
                   tuple(map(int, arrays['active'])), arrays['state_offsets'], arrays['axial_angles_deg'])


class AngularTaskModel(MobileTaskModel):
    def __init__(self, name='B', output_root=FAMILY_ROOT):
        from whole_task_2.step3.input_state import require_joint_initialization
        require_joint_initialization()

    def initial(self, active=None):
        q, metadata = super().initial(active)
        return AngularLayout(q.placements, q.directions, q.hosts, q.active, q.state_offsets), metadata

    def activate(self, layout):
        if not hasattr(self, 'current_angles'):
            return
        angles = np.asarray(getattr(layout, 'angles_deg', np.zeros(len(self.poses)))) % 360.
        changed = np.flatnonzero(np.abs(angles-self.current_angles) > 1e-10)
        if not len(changed):
            return
        for k in changed:
            source = spin_pose(self.reference_source_native[k], self.rotation_axis,
                               self.rotation_pivot, angles[k])
            vertices = self.mesh.vertices @ source[:3, :3].T+source[:3, 3]
            height = float(vertices[:, 2].min())
            native = source.copy(); native[2, 3] -= height
            vertices[:, 2] -= height
            data = deepcopy(self.reference_domain_data[k])
            data['geometry']['vertices_m'] = vertices.tolist()
            data['geometry']['inward_normals'] = (-self.mesh.face_normals[self.tasks[k].domain.work_ids] @ native[:3, :3].T).tolist()
            data['frame']['T_world_mesh'] = native.tolist()
            data['frame']['moment_origin_m'] = (native[:3, :3] @ self.mesh.center_mass+native[:3, 3]).tolist()
            domain = WrenchDomain(data)
            task = self.tasks[k]; task.domain = domain
            task.targets = domain.forward(task.work_face_index, task.parameters)*task.scale
            task.targets_7d = np.c_[task.targets, np.zeros(len(task.targets))]
            self.current_endpoint_targets[k] = endpoint_demands(domain)['endpoint_need_wrench']
            self.native[k] = native
            self.source_native[k] = source
            self.source_heights[k] = height
            candidates = np.flatnonzero(vertices[:, 2] <= vertices[:, 2].min()+1e-12)
            mean = vertices[candidates, :2].mean(axis=0)
            index = int(candidates[np.linalg.norm(vertices[candidates, :2]-mean, axis=1).argmin()])
            self.floor_vertices[k] = index
            task.floor = vertices[index].copy()
            self.floors[k] = K.U.floor(K.FLOOR.columns(task.floor, domain.com), task.scale)
            source_com = source[:3, :3] @ self.mesh.center_mass+source[:3, 3]
            self.center_shifts[k, :2] = self.center_xy-source_com[:2]
            self.flange_vertices[k] = domain.triangles.reshape(-1, 3)+self.flange_offset+self.center_shifts[k]
            points = self.points @ native[:3, :3].T+native[:3, 3]
            normals = -domain.mesh.face_normals[self.sources]
            self.point_rays[k] = K.U.heads(np.c_[normals, np.cross(points-domain.com, normals)], task.scale)
        self.states = [(task, self.native[k], self.mesh) for k, task in enumerate(self.tasks)]
        self.current_angles = angles.copy()
        self.angle_updates += 1
        # Legacy force keys assumed immutable task rotations. Invalidate these
        # explicitly; a contact-mask match must never reuse another angle's LP.
        for name, value in list(vars(self).items()):
            if 'cache' in name and isinstance(value, (dict, OrderedDict)):
                value.clear()
        self.basis_caches = {k: [] for k in range(len(self.poses))}
        self.contact_delta = ContactDelta(self)
        self.volume_delta = None
        self.adaptive_demands = {}
        self.quadrature_epoch += 1
        distances = [SimpleNamespace(unique_targets=t.targets_7d,
            unique_weights=np.full(len(t.targets), 1/len(t.targets)), gravity_indices=[]) for t in self.tasks]
        self.demand_screen = JumpScreen(SimpleNamespace(model=self, distances=distances), 32)
        self.original_scales = {k: max(1., float(np.linalg.norm(t.targets, axis=1).max()))
                                for k, t in enumerate(self.tasks)}
        from .family_oracle import FamilyFloorOracle
        self.base_oracle = FamilyFloorOracle(self)
        builder = getattr(self, 'connected_core_builder', None)
        if builder is not None:
            builder.parts.clear(); builder.shapes.clear()

    def validate_placement(self, layout, k):
        self.activate(layout)
        return super().validate_placement(layout, k)

    def evaluate(self, layout):
        self.activate(layout)
        return super().evaluate(layout)

    def proxy(self, layout, targets=()):
        self.activate(layout)
        return super().proxy(layout, targets)

    def floor_normal(self, layout, k):
        self.activate(layout)
        return super().floor_normal(layout, k)

    def workpiece_height(self, layout, k):
        self.activate(layout)
        return super().workpiece_height(layout, k)

    def execution(self, layout, k):
        self.activate(layout)
        row = super().execution(layout, k)
        row['axial_angle_deg'] = float(layout.angles_deg[k] % 360.)
        return row

    def spin_guest(self, layout, owner, angle_deg):
        """Change one task's continuous roll; preserve every other world task.

        The host installation may change too. Counter-transform all existing
        guests so changing a host angle cannot silently rotate their tasks.
        """
        self.activate(layout)
        previous_world = {k: installed_state(self.native, layout, int(layout.hosts[k])) @ layout.placements[k]
                          for k in layout.active}
        world_directions = {k: installed_state(self.native, layout, int(layout.hosts[k]))[:3, :3] @ layout.directions[k]
                            for k in layout.active}
        old_height = self.workpiece_height(layout, owner)
        target_xy = (previous_world[owner][:3, :3] @ self.mesh.center_mass+previous_world[owner][:3, 3])[:2]
        trial = layout.copy(); trial.angles_deg[owner] = angle_deg % 360.
        old_rotation = previous_world[owner][:3, :3].copy()
        self.activate(trial)
        rotation = self.native[owner, :3, :3]
        world = self.native[owner].copy()
        world[:2, 3] = target_xy-(rotation @ self.mesh.center_mass)[:2]
        world[2, 3] += old_height
        previous_world[owner] = world
        world_directions[owner] = rotation @ old_rotation.T @ world_directions[owner]
        for k in layout.active:
            installation = installed_state(self.native, trial, int(trial.hosts[k]))
            trial.placements[k] = np.linalg.inv(installation) @ previous_world[k]
            trial.directions[k] = legal_direction(installation[:3, :3].T @ world_directions[k], self.floor_normal(trial, k))
        self.check_station(trial)
        return trial


def roll_to_match(reference_rotation, target_rotation, axis):
    """Continuous best axial roll for a host-alignment proposal, in degrees."""
    axis = np.asarray(axis)
    cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    relative = reference_rotation @ target_rotation.T
    c = np.trace(relative)-axis @ relative @ axis
    s = np.trace(cross @ relative)
    return float(np.degrees(np.arctan2(s, c)))
