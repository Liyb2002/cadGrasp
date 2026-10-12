"""Read-only adapters to immutable native tasks and existing geometry/physics."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = Path(os.environ.get('POSE_SET_REPO_ROOT',Path(__file__).resolve().parents[3]))
HERE = ROOT/'slides/pose_set_search'
CO = ROOT/'slides/Co-optimize'
sys.path.insert(0, str(CO/'helper_func'))
import co_common as C
from optimization.initial_directions import initialize_close_directions
from optimization.physics_guided_cone_iterative import cone_projection
from optimization.physics_guided_padded_sweep import ConservativeExitClearance
from exit_clearance import CONTACT_DEPTH_M, _PRISM_FACES
from trimesh.ray.ray_pyembree import RayMeshIntersector
from scipy.spatial import ConvexHull
import numpy as np

TOL = 1e-10


def transform_mesh(mesh, matrix):
    result = mesh.copy()
    result.apply_transform(matrix)
    return result


def unpack_solid(solid):
    """Own both arrays rather than retain a mesh adapter's borrowed buffers."""
    data = solid.to_mesh64()
    return C.trimesh.Trimesh(np.array(data.vert_properties[:, :3],copy=True)*C.S.SCALE,
                           np.array(data.tri_verts,copy=True),process=False)


def transform_solid(solid, matrix):
    matrix = np.asarray(matrix[:3]).copy()
    matrix[:, 3] /= C.S.SCALE
    return solid.transform(matrix)


def tangent_frame(normal):
    normal = np.asarray(normal)/np.linalg.norm(normal)
    axis = np.eye(3)[np.argmin(abs(normal))]
    u = np.cross(normal, axis)
    u /= np.linalg.norm(u)
    return np.column_stack([u, np.cross(normal, u)])


def legal_direction(direction, normal):
    direction = np.asarray(direction, float).copy()
    dot = direction @ normal
    if dot < 1e-6:
        direction += (1e-6-dot)*normal
    return direction/np.linalg.norm(direction)


def work_check(material, task):
    result = C.WORK.check(material, task)
    replay = 0;work_replay=0
    if not result['passed'] and result['forbidden_intersections'] and all(
            r['classification'] == 'solver_unresolved' for r in result['forbidden_intersections']):
        for cycle in [1, 2]:
            equivalent = material.copy()
            equivalent.faces = np.roll(material.faces, cycle, axis=1)
            result = C.WORK.check(equivalent, task)
            replay = cycle
            if result['passed']:
                break
        if not result['passed'] and not result['contained_work_face_ids'] and all(
                r['classification']=='solver_unresolved' for r in result['forbidden_intersections']):
            # Parameterize the SAME work triangles by another vertex, too.
            # Neither vertex coordinates, winding, working-face IDs, original
            # task data, nor the checker/tolerances are changed.
            for work_cycle in [1,2]:
                work_mesh=task.domain.mesh.copy()
                work_mesh.faces=np.roll(work_mesh.faces,work_cycle,axis=1)
                equivalent_task=SimpleNamespace(domain=SimpleNamespace(
                    mesh=work_mesh,work_ids=task.domain.work_ids))
                for cycle in [0,1,2]:
                    equivalent=material.copy()
                    equivalent.faces=np.roll(material.faces,cycle,axis=1)
                    attempt=C.WORK.check(equivalent,equivalent_task)
                    if attempt['passed']:
                        result=attempt;replay=cycle;work_replay=work_cycle
                        break
                    if any(r['classification']!='solver_unresolved' for r in attempt['forbidden_intersections']):
                        result=attempt
                        break
                if result['passed'] or any(r['classification']!='solver_unresolved'
                                           for r in result['forbidden_intersections']):
                    break
    return dict(passed=result['passed'], forbidden_intersections=result['forbidden_intersection_count'],
                contained_work_faces=result['contained_work_face_ids'], cyclic_vertex_replay=replay,
                cyclic_work_vertex_replay=work_replay,
                failure_classifications={kind:sum(r['classification']==kind for r in result['forbidden_intersections'])
                                         for kind in set(r['classification'] for r in result['forbidden_intersections'])})
