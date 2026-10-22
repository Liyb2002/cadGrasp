"""Complete near-enclosed surface gaps for region initialization.

The angular criterion is measured in the fixed needle's perpendicular plane.
Filled faces remain explicit heuristic additions, not certified tool targets.
"""
import numpy as np

from .step1_3 import _adjacency
from .work_disks import tangent_frame


def angular_surround_deg(point, segments):
    """Angular measure of directions hitting the projected boundary segments.

    Each segment subtends its shorter circular interval. Unioning intervals
    avoids double-counting overlapping boundaries and is independent of bins.
    """
    delta = np.asarray(segments)-np.asarray(point)
    angles = np.arctan2(delta[..., 1], delta[..., 0])
    difference = (angles[:, 1]-angles[:, 0]+np.pi) % (2*np.pi)-np.pi
    start = np.where(difference >= 0, angles[:, 0], angles[:, 1]) % (2*np.pi)
    finish = start+np.abs(difference)
    intervals = []
    for a, b in zip(start, finish):
        if b <= 2*np.pi:
            intervals.append((a, b))
        else:
            intervals.extend(((a, 2*np.pi), (0., b-2*np.pi)))
    covered, end = 0., 0.
    for a, b in sorted(intervals):
        if b > end:
            covered += b-max(a, end)
            end = b
    return min(360., float(np.degrees(covered)))


def _in_projected_triangles(point, triangles):
    # A different surface behind an already selected projected triangle is not
    # a visible gap. Only nondegenerate triangles define a filled projection.
    edges = np.roll(triangles, -1, axis=1)-triangles
    rel = point-triangles
    cross = edges[..., 0]*rel[..., 1]-edges[..., 1]*rel[..., 0]
    area = (triangles[:, 1, 0]-triangles[:, 0, 0])*(triangles[:, 2, 1]-triangles[:, 0, 1])-(triangles[:, 1, 1]-triangles[:, 0, 1])*(triangles[:, 2, 0]-triangles[:, 0, 0])
    return bool(np.any((np.abs(area) > 1e-16)
                       & (np.all(cross >= -1e-14, axis=1) | np.all(cross <= 1e-14, axis=1))))


def fill_surrounded_surface(mesh, selected_ids, outward_axis_object, eligible,
                            threshold_deg=270.):
    """One pass against the original region; keep only surface-connected fills.

    `eligible` enforces the caller's hard bounds. The first-layer workflow
    includes whole-face visibility, spatial and flange-position bounds here;
    enclosed but occluded faces can never be admitted by the angular rule.
    Centroid angular coverage is a region-initialization heuristic.
    """
    selected_ids = np.asarray(selected_ids, dtype=int)
    selected = np.zeros(len(mesh.faces), dtype=bool)
    selected[selected_ids] = True
    frame = tangent_frame(outward_axis_object)[:, :2]
    projected = mesh.vertices @ frame
    triangles = projected[mesh.faces[selected_ids]]
    edges = np.sort(mesh.faces[selected_ids][:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = projected[unique[counts == 1]]
    angles = np.zeros(len(mesh.faces))
    proposed = selected.copy()
    for face in np.flatnonzero(np.asarray(eligible, dtype=bool) & ~selected):
        point = projected[mesh.faces[face]].mean(axis=0)
        if _in_projected_triangles(point, triangles):
            continue
        angle = angular_surround_deg(point, boundary)
        if angle > threshold_deg+1e-8:
            proposed[face], angles[face] = True, angle
    # Do not color the other side of the object or a disconnected surface merely
    # because its projection lies in the same opening.
    neighbors = _adjacency(mesh)
    connected = np.zeros(len(mesh.faces), dtype=bool)
    stack = selected_ids.tolist()
    connected[selected_ids] = True
    while stack:
        current = stack.pop()
        for other in neighbors[current]:
            if proposed[other] and not connected[other]:
                connected[other] = True
                stack.append(other)
    additions = np.flatnonzero(connected & ~selected)
    return dict(face_ids=np.flatnonzero(connected), filled_face_ids=additions,
                filled_enclosure_angles_deg=angles[additions],
                threshold_deg=float(threshold_deg),
                method="one_pass_projected_boundary_angular_union_at_face_centroids",
                reference="plane_perpendicular_to_fixed_needle",
                requires_step2_verification=True)
