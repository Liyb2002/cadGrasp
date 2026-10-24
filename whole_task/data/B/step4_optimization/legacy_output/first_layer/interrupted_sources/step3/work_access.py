"""Finite access volume joining each current working triangle to the head side."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull

from .legacy import kernel as K, _contains


MODEL = "finite_segments_between_working_surface_and_head"
ENDPOINT_SOURCE = "step1.3.workspace.spatial_cone.apex_world_m"


def read_source_cone(output_root, object_name):
    """Reuse the existing head-side endpoint; do not extend or rebuild its cone."""
    root = Path(output_root)/object_name
    step1 = json.loads((root/"step1/step1.3/pose_coverage.json").read_text())
    step2 = json.loads((root/"step2/step2.1/object_poses.json").read_text())
    cone = step1["workspace"]["spatial_cone"]
    if cone != step2["fixed_station"]["workspace"]["spatial_cone"]:
        raise ValueError("Step2 station cone differs from current Step1.3; refresh Step2")
    return dict(cone)


class SurfaceSegments:
    """Union of conv{head, triangle}, with both ends closed and finite.

    Each triangle has its own four-vertex cell. Taking one hull of the entire
    region would fill unselected gaps and is deliberately not used here.
    Coordinates may be material or world coordinates, but must share a frame.
    """
    def __init__(self, mesh, face_ids, head):
        self.face_ids = np.asarray(face_ids, dtype=int)
        self.head = np.asarray(head, dtype=float)
        if self.head.shape != (3,) or not np.isfinite(self.head).all():
            raise ValueError("Expected a finite head-side endpoint in the mesh frame")
        self.triangles = np.asarray(mesh.triangles[self.face_ids]).copy()
        heads = np.broadcast_to(self.head, (len(self.face_ids), 1, 3))
        self.cells = np.concatenate((heads, self.triangles), axis=1)
        self.planes = np.empty((len(self.face_ids), 4, 4))
        for i, cell in enumerate(self.cells):
            distance = abs((self.head-cell[1]) @ mesh.face_normals[self.face_ids[i]])
            if distance <= 1e-12:
                raise ValueError(f"Head is coplanar with work face {self.face_ids[i]}; finite access volume unresolved")
            self.planes[i] = ConvexHull(cell).equations
        self.counts = np.full(len(self.face_ids), 4, dtype=np.int64)
        self._solid = None

    def contains_points(self, points):
        return _contains(np.ascontiguousarray(points, dtype=np.float64), self.planes, self.counts)

    def solid(self):
        if self._solid is None:
            self._solid = K.union([K.S.solid(K.G.hull_mesh(cell)) for cell in self.cells])
        return self._solid

    def definition(self):
        return dict(head_material_m=self.head.tolist(), working_face_ids=self.face_ids.tolist(),
            cell_count=len(self.cells), equation="union over p in working_area of {(1-t)*head + t*p : 0 <= t <= 1}",
            cell_geometry="one tetrahedron per selected triangle; union, not whole-region convex hull",
            extends_beyond_surface=False, extends_behind_head=False,
            includes_unselected_cone_space=False)


class RegisteredWorkVolumes:
    """Independent work regions pulled back to each installation of common S."""
    def __init__(self, mesh, states, source_cone):
        self.mesh, self.source_cone = mesh, dict(source_cone)
        self.head_world = np.asarray(source_cone["apex_world_m"], dtype=float)
        self.by_pose = {}
        for pose, (task, transform, _) in states.items():
            head_material = (self.head_world-transform[:3, 3]) @ transform[:3, :3]
            self.by_pose[pose] = SurfaceSegments(mesh, task.domain.work_ids, head_material)
        self.families = list(self.by_pose.values())
        self.ids = np.unique(np.concatenate([family.face_ids for family in self.families]))

    def for_pose(self, pose):
        return self.by_pose[pose]

    def solid_for_pose(self, pose):
        return self.for_pose(pose).solid()

    def combined_solid(self):
        return K.union([family.solid() for family in self.families])

    def definition(self):
        return dict(model=MODEL, endpoint_source=ENDPOINT_SOURCE, head_world_m=self.head_world.tolist(),
            endpoint_semantics="head-side gold-cone apex from Step1.3; access-envelope endpoint, not the translating contact TCP",
            source_cone=self.source_cone, segment_parameter_range=[0., 1.],
            poses={pose:family.definition() for pose, family in self.by_pose.items()},
            execution_constraint="for task i: installed_support(T_i S) intersect own_surface_to_head_volume(V_i_world) = empty",
            other_tasks_active_during_task_i=False,
            union_scope="only each pose's own pullback initializes common material S",
            rebuild_when_working_area_or_installations_change=True)
