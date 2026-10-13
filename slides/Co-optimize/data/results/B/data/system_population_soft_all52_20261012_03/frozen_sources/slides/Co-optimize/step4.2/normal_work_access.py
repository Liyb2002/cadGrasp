"""Oracle-only working space: each source triangle swept along its normal.

The access set is a union of unbounded outward triangular prisms. It has no
angular expansion, convex filling between faces, or geometric safety margin.
Force-demand angles and the original contact/initialization model are separate.
"""
from functools import lru_cache
import hashlib
import numpy as np
from co_common import S, G, union
from work_access import _contains, unpack_solid


class NormalWorkAccess:
    def __init__(self, mesh, face_ids):
        self.face_ids = np.asarray(face_ids, dtype=int)
        self.cell_face_ids = self.face_ids.copy()
        self.triangles = mesh.triangles[self.face_ids].copy()
        self.normals = mesh.face_normals[self.face_ids].copy()
        self.half_angle_deg = 0.
        self.clearance = 0.
        self.counts = np.full(len(self.face_ids), 4, dtype=np.int64)
        self.planes = np.zeros((len(self.face_ids), 4, 4))
        for k, (triangle, normal) in enumerate(zip(self.triangles, self.normals)):
            edges = np.roll(triangle, -1, axis=0)-triangle
            inward = np.cross(normal, edges)
            lengths = np.linalg.norm(inward, axis=1)
            if np.any(lengths == 0.) or np.linalg.norm(normal) == 0.:
                raise ValueError('Degenerate working triangle has no normal prism')
            inward /= lengths[:, None]
            # a*x+b<=0: three side planes, followed by the outward base.
            self.planes[k, :3, :3] = -inward
            self.planes[k, :3, 3] = np.einsum('ij,ij->i', inward, triangle)
            self.planes[k, 3, :3] = -normal
            self.planes[k, 3, 3] = normal@triangle[0]
        self._solids = {}

    def contains_points(self, points):
        return _contains(np.ascontiguousarray(points, dtype=np.float64), self.planes, self.counts)

    def required_length(self, points, minimum=.5):
        if not len(self.triangles):
            return float(minimum)
        height = np.asarray(points)[None]-self.triangles[:, 0, None]
        maximum = float(np.einsum('fpc,fc->fp', height, self.normals).max())
        # This cap only bounds a Boolean; all candidate points lie before it.
        return max(float(minimum), maximum+.001)

    def solid(self, length):
        length = float(length)
        if not np.isfinite(length) or length <= 0.:
            raise ValueError('Positive finite prism cap required')
        if length not in self._solids:
            parts = [S.solid(G.hull_mesh(np.vstack([triangle, triangle+length*normal])))
                     for triangle, normal in zip(self.triangles, self.normals)]
            self._solids[length] = union(parts)
            if len(self._solids) > 8:
                self._solids.pop(next(iter(self._solids)))
        return self._solids[length]

    def mesh(self, length):
        return unpack_solid(self.solid(length))

    def definition(self):
        return dict(mode='normal_ray_volume', source='original task working triangles and face normals',
            equation='union over ALL source points p of p + lambda*n(p), lambda>=0',
            half_angle_deg=0., total_opening_deg=0., work_face_count=len(self.face_ids),
            outward_half_rays=True, ray_parameter_min=0., bidirectional_lines=False,
            unbounded_during_search=True, whole_work_triangles=True,
            normals='constant original outward normal on each mesh triangle',
            angular_expansion=False, global_convex_hull=False, clearance_m=0.,
            plane_count_per_triangle=4, self_occluded_normal_rays_reserved=True,
            finite_mesh_cap_only_beyond_entire_candidate_seed=True,
            scope='base Oracle and saved base connection space only; force and contact models unchanged')


_MESHES = {}


@lru_cache(maxsize=128)
def _cached_access(mesh_key, face_ids):
    return NormalWorkAccess(_MESHES[mesh_key], face_ids)


def normal_work_access(mesh, original):
    digest = hashlib.sha256(np.asarray(mesh.vertices, dtype=np.float64).tobytes()+
                            np.asarray(mesh.faces, dtype=np.int64).tobytes()).hexdigest()
    _MESHES.setdefault(digest, mesh)
    return _cached_access(digest, tuple(map(int, original.face_ids)))
