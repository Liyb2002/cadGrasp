"""Reuse Co-optimize search/geometry with current tasks and finite work access.

The two input mixins replace only the legacy task reader and angular demand
quadrature. The original search, contact and exit kernels remain imported.
"""
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import sys

from whole_task.step1.common import DEFAULT_OUTPUT, sha256
from whole_task.step3.initialize import register, supply
from whole_task.step3.legacy import kernel as K, COOPT
from whole_task.step3.work_access import RegisteredWorkVolumes, SurfaceSegments, read_source_cone
from whole_task.step3.validate import check_binding

for part in ('step4.2', 'step5.2', 'step5.3', 'step5', 'evaluation'):
    sys.path.insert(0, str(COOPT/part))
from whole_search.model import Model
from whole_search.fast_search import FastModel
from whole_search.common import ConservativeExitClearance, RayMeshIntersector, transform_mesh, result_mesh
from continuous_support.fast_gradient import GradientModel
from continuous_support.stable_gradient import StableGradientModel
from juxtapose.reuse_search import JumpScreen
from oracle_search import OracleModel
from base_oracle import BaseOracle
from fixed_work_planes import FixedWorkPlanes
from fixture_state import installed_state
import json


class FiniteAccess(SurfaceSegments):
    """Compatibility with cap-taking legacy APIs; finite ends never change."""
    half_angle_deg = 0.
    clearance = 0.

    def solid(self, length=None):
        return super().solid()

    def required_length(self, points, minimum=.5):
        return 0.


class ComputedInputs(Model):
    def __init__(self, poses=None, name='B', output_root=DEFAULT_OUTPUT, **kwargs):
        from whole_task.step3.input_state import require_joint_initialization
        require_joint_initialization()
        self.name = name
        self.output_root = Path(output_root).resolve()
        folder = self.output_root/name/'step3'
        upstream = json.loads((folder/'report.json').read_text())
        check_binding(folder, upstream)
        mesh, states, _, loads = register(self.output_root, name, poses or upstream['poses'])
        self.poses, self.states = list(states), list(states.values())
        self.tasks = [s[0] for s in self.states]
        self.native = np.array([s[1] for s in self.states])
        self.mesh, self.extent = mesh, float(mesh.extents.max())
        self.allow_z_translation = True  # Fixture translations are allowed; object U is locked below.
        self.length = max(.5, 4*self.extent)
        self.thickness = upstream['policy']['wrap_maximum_vertex_displacement_m']
        self.body = K.S.solid(mesh)
        self.clearance = ConservativeExitClearance(mesh)
        offsets = K.wrap_offsets(mesh, self.thickness)
        cells = [K.S.solid(K.G.hull_mesh(K.G.head_cell(mesh, tri, i, offsets)))
                 for i, tri in enumerate(mesh.triangles)]
        raw_wrap = K.union(cells)-self.body
        access = RegisteredWorkVolumes(mesh, states, read_source_cone(self.output_root, name))
        self.work_rays = [FiniteAccess(mesh, f.face_ids, f.head) for f in access.families]
        self.allowed = [np.setdiff1d(np.arange(len(mesh.faces)), t.domain.work_ids) for t in self.tasks]
        self.wraps = [raw_wrap-w.solid() for w in self.work_rays]
        # Each seat can regrow its own allowed wrap after a layout change.
        from whole_search.common import unpack_solid
        self.wrap_rays = [RayMeshIntersector(unpack_solid(w)) for w in self.wraps]
        self.ray = RayMeshIntersector(mesh)
        self.epsilon = self.extent*1e-7
        self.floors = []
        for k, task in enumerate(self.tasks):
            task.inputs = [Path(p) for p in loads['input_sha256']]
            # Compatibility-only point for legacy constructors; actual floor
            # columns below come exclusively from the computed world mesh.
            task.floor = np.r_[task.domain.com[:2], 0.]
            columns, _ = supply(task, self.native[k], np.empty((0,3,3)), np.array([],int), upstream['policy'])
            self.floors.append(columns)
        self.inputs = [Path(p) for p in upstream['input_sha256']]
        self.inputs += [folder/'report.json', folder/'step3.1/wrapped_support.obj']
        self.initialization_report = folder/'report.json'
        self.registered_seed = K.S.solid(K.trimesh.load(folder/'step3.1/wrapped_support.obj', force='mesh', process=False))
        self.exact_cache, self.lock_cache, self.seed_cache = OrderedDict(), OrderedDict(), OrderedDict()
        self.fixed_lock_cache = OrderedDict()
        self.exact_calls = self.proxy_calls = 0
        self.unresolved_layouts = {}
        self.startup_sources = {str(Path(__file__)): sha256(__file__)}
        self.upstream_report = upstream


class ComputedMeasure(GradientModel):
    def __init__(self, *args, demand_nodes=32, **kwargs):
        # Skip only GradientModel's old angular-domain reader.
        FastModel.__init__(self, *args, **kwargs)
        distances = []
        self._r6_forward_nodes = {}
        for k, task in enumerate(self.tasks):
            self._r6_forward_nodes[k] = SimpleNamespace(evaluate=task.forward_nodes)
            distances.append(SimpleNamespace(unique_targets=task.targets_7d,
                unique_weights=np.full(len(task.targets), 1/len(task.targets)), gravity_indices=[]))
        self.demand_screen = JumpScreen(SimpleNamespace(model=self, distances=distances), demand_nodes)
        self.gradient_seconds = 0.


class ComputedOracle(BaseOracle):
    def __init__(self, model):
        # The original constructor reads objects/B/poses and substitutes
        # unbounded normal prisms. Both inputs are replaced explicitly here.
        self.model, self.padding, self.foot_width = model, 2*model.extent, 0.
        self.work_mode = 'finite_surface_to_head_segments'
        self.loads = [t.targets/t.scale for t in model.tasks]
        for key in ('cache','hulls','family_cache','constraint_cache','slice_cache','floor_rejections','slice_planes'):
            setattr(self,key,OrderedDict())
        self.seconds=0.; self.queries=0; self.component_queries=0
        self.last_geometry=None; self.slice_cache_hits=0; self.slice_cache_misses=0
        self.equivalent_statistics=dict(floor_rejection_reuses=0,obstacle_plane_reuses=0,deferred_losing_oracles=0)
        self.work_access=model.work_rays
        self.fixed_work_planes=FixedWorkPlanes(self.work_access)

    def query(self, layout, *, connected=False):
        try:
            self.model.check_station(layout)
        except ValueError:
            return dict(passed=False,enclosure_passed=False,connected_passed=False,
                        status='robot_pose_changed',gap_m=1.)
        result = super().query(layout, connected=connected)
        result['work_access_definition']='finite union of selected triangles joined to the fixed head-side endpoint'
        return result


class TaskModel(OracleModel, ComputedMeasure, ComputedInputs):
    def __init__(self, poses=None, name='B', output_root=DEFAULT_OUTPUT):
        self.support_floor_clearance_m=0.
        StableGradientModel.__init__(self, poses, name=name, output_root=output_root)
        self.base_oracle=ComputedOracle(self)
        self.oracle_gate_active=False
        self.require_connected_core=True

    def check_station(self, layout):
        for k in layout.active:
            self.validate_placement(layout,k)

    def validate_placement(self, layout, k):
        U=installed_state(self.native,layout,layout.hosts[k]) @ layout.placements[k]
        if not np.allclose(U,self.native[k],atol=1e-10,rtol=0):
            raise ValueError('Candidate moves the reference robot work pose; couple fixture installation and seat instead')

    def floor_signature(self, layout, k):
        self.validate_placement(layout,k)
        return bool(abs(self.tasks[k].domain.mesh.bounds[0,2])<=1e-9)

    def floor_columns(self, layout, k):
        self.validate_placement(layout,k)
        return self.floors[k]

    def work_length(self, layout, k):
        return 0.

    def work_solid(self, layout, k):
        return self.work_rays[k].solid()

    def verify_work(self, result):
        layout=result['layout']; solid=K.S.solid(result_mesh(result)); rows=[]
        from whole_search.common import transform_solid
        self.check_station(layout)
        for k in layout.active:
            overlap=K.material_volume(solid ^ transform_solid(self.work_rays[k].solid(),layout.placements[k]))
            rows.append(dict(pose=self.poses[k],passed=overlap<=1e-10,finite_work_overlap_m3=overlap))
        return rows
