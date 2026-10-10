"""Working-face exclusion with the same surface meaning as Step3.

Adjacent non-working faces may share edges/vertices with a working triangle.
An intersection reaching the working triangle's interior is forbidden. Work
faces inside a closed support are also forbidden. No tool-ray volume is used.
"""
import numpy as np
from scipy.optimize import linprog
from trimesh.ray.ray_pyembree import RayMeshIntersector

from step4_connect_support.check_work_access import triangle_overlaps

TOLERANCE_M = 1e-9
LP_TOLERANCE = 1e-9


def intersection(work, support, scale):
    """Maximize intersection distance to all three work-triangle edges.

    Both points use triangle barycentric coordinates. A shared edge or vertex
    has optimum zero; an interior intersection has positive edge distance.
    Full triangles are checked, including small overlaps away from centroids.
    """
    a = np.asarray(work, float)
    b = np.asarray(support, float)
    e = (a[1:]-a[0])/scale
    f = (b[1:]-b[0])/scale
    rhs = (b[0]-a[0])/scale
    twice_area = np.linalg.norm(np.cross(e[0], e[1]))
    opposite = np.linalg.norm(np.roll(a, -1, axis=0)-np.roll(a, -2, axis=0), axis=1)/scale
    heights = twice_area/opposite
    equations = np.c_[e.T, -f.T, np.zeros(3)]
    constraints = np.array([
        [1,1,0,0,0], [0,0,1,1,0],
        [heights[0],heights[0],0,0,1],
        [-heights[1],0,0,0,1], [0,-heights[2],0,0,1]], float)
    result = linprog([0,0,0,0,-1], A_ub=constraints,
        b_ub=[1,1,heights[0],0,0], A_eq=equations, b_eq=rhs,
        bounds=[(0,1)]*4+[(0,None)], method='highs',
        options=dict(primal_feasibility_tolerance=LP_TOLERANCE,
                     dual_feasibility_tolerance=LP_TOLERANCE))
    if result.status == 2:
        return dict(classification='disjoint', passed=True)
    if not result.success:
        return dict(classification='solver_unresolved', passed=False,
            solver_status=int(result.status), solver_message=result.message)
    x = np.asarray(result.x, np.longdouble)
    wa = np.array([1-x[0]-x[1],x[0],x[1]], np.longdouble)
    sb = np.array([1-x[2]-x[3],x[2],x[3]], np.longdouble)
    point_a = wa@np.asarray(a, np.longdouble)
    point_b = sb@np.asarray(b, np.longdouble)
    residual = float(np.linalg.norm(point_a-point_b))
    edge_distance = float(np.min(wa*np.asarray(heights, np.longdouble))*scale)
    valid = residual <= TOLERANCE_M and min(wa.min(),sb.min()) >= -1e-8
    interior = valid and edge_distance > TOLERANCE_M
    return dict(classification='working_face_interior_contact' if interior else
        'shared_edge_or_vertex_only' if valid else 'solver_unresolved',
        passed=bool(valid and not interior), work_barycentric=[float(v) for v in wa],
        support_barycentric=[float(v) for v in sb],
        point_world_m=[float(v) for v in point_a], equality_residual_m=residual,
        minimum_work_edge_distance_m=edge_distance)


def check(mesh, task, *, closed_solid=True):
    tree = mesh.triangles_tree
    triangles = mesh.triangles
    scale = float(task.domain.mesh.extents.max())
    failures, boundary = [], []
    pairs = 0
    for face in task.domain.work_ids:
        tri = task.domain.mesh.triangles[face]
        ids = np.asarray(list(tree.intersection(np.r_[tri.min(0)-TOLERANCE_M,
            tri.max(0)+TOLERANCE_M])), int)
        pairs += len(ids)
        if not len(ids):
            continue
        candidates = ids[triangle_overlaps(tri,triangles[ids],TOLERANCE_M)]
        for index in candidates:
            detail = intersection(tri,triangles[index],scale)
            row = dict(work_face_id=int(face),support_triangle_id=int(index),**detail)
            if not detail['passed']:
                failures.append(row)
            elif detail['classification']=='shared_edge_or_vertex_only':
                boundary.append(row)
    contained = []
    if closed_solid:
        if not mesh.is_watertight or not mesh.is_winding_consistent:
            raise ValueError('Working-face solid check requires a closed consistently wound mesh')
        # Boundary crossings catch partial containment. A face wholly enclosed
        # without crossing the boundary is detected by an interior point.
        centers = task.domain.mesh.triangles_center[task.domain.work_ids]
        inside = RayMeshIntersector(mesh).contains_points(centers)
        contained = task.domain.work_ids[inside].tolist()
    return dict(passed=not failures and not contained,
        working_face_count=len(task.domain.work_ids),candidate_triangle_pairs=pairs,
        forbidden_intersection_count=len(failures),forbidden_intersections=failures,
        allowed_boundary_intersection_count=len(boundary),allowed_boundary_intersections=boundary,
        contained_work_face_ids=contained,tolerance_m=TOLERANCE_M,
        source_face_interiors_excluded=True,shared_edge_or_vertex_contact_allowed=True,
        processing_ray_volume_enforced=False,
        method='Full triangle intersection LP, long-double barycentric replay, and closed-solid containment')
