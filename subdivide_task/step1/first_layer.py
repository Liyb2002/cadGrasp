"""Whole-triangle first-hit visibility for a fixed, translating needle.

Orthographic triangle overlap plus linear depth comparison tests the entire
surface of each source face, including occluders outside the working window.
Partially occluded faces are conservatively excluded rather than relabelled.
"""
import numpy as np
import shapely
from shapely import STRtree

from .work_disks import tangent_frame


DEPTH_TOLERANCE_M = 1e-10
PROJECTED_AREA_TOLERANCE_M2 = 1e-16
FRONT_COSINE_EPSILON = 1e-10


def first_layer_faces(mesh, outward_axis, cache=None):
    axis = np.asarray(outward_axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    key = axis.tobytes()
    if cache is not None and key in cache:
        return cache[key].copy()
    frame = tangent_frame(axis)[:, :2]
    cosine = mesh.face_normals @ axis
    projected = mesh.triangles @ frame
    polygons = shapely.polygons(projected)
    valid = (np.abs(cosine) > FRONT_COSINE_EPSILON) & (shapely.area(polygons) > PROJECTED_AREA_TOLERANCE_M2)
    blockers = np.flatnonzero(valid)
    candidates = np.flatnonzero(valid & (cosine > FRONT_COSINE_EPSILON))
    visible = np.zeros(len(mesh.faces), dtype=bool)
    visible[candidates] = True
    if len(candidates):
        query, hit = STRtree(polygons[blockers]).query(polygons[candidates], predicate='intersects')
        own, other = candidates[query], blockers[hit]
        keep = own != other
        own, other = own[keep], other[keep]
        overlap = shapely.intersection(polygons[own], polygons[other])
        keep = shapely.area(overlap) > PROJECTED_AREA_TOLERANCE_M2
        own, other, overlap = own[keep], other[keep], overlap[keep]
        # z(x,y) is affine on each projected triangle. A positive depth
        # difference anywhere in the convex overlap occurs at a polygon vertex.
        coefficients = np.zeros((len(mesh.faces), 3))
        coefficients[valid, :2] = -(mesh.face_normals[valid] @ frame)/cosine[valid, None]
        coefficients[valid, 2] = np.einsum('ij,ij->i', mesh.face_normals[valid], mesh.triangles[valid, 0])/cosine[valid]
        xy, polygon_index = shapely.get_coordinates(overlap, return_index=True)
        delta = (coefficients[other]-coefficients[own])[polygon_index]
        ahead = np.einsum('ij,ij->i', xy, delta[:, :2])+delta[:, 2] > DEPTH_TOLERANCE_M
        visible[own[polygon_index[ahead]]] = False
    if cache is not None:
        cache[key] = visible.copy()
    return visible


def incidence_quality(mesh, face_ids, outward_axis):
    """Continuous area-weighted frontality score, never an angle cutoff."""
    ids = np.asarray(face_ids, dtype=int)
    cosine = np.clip(mesh.face_normals[ids] @ outward_axis, 0., 1.)
    area = mesh.area_faces[ids]
    return dict(area_weighted_incidence_cosine=float(np.average(cosine, weights=area)),
                effective_projected_area_m2=float(area @ cosine),
                maximum_incidence_angle_deg=float(np.degrees(np.arccos(cosine.min()))),
                used_as_hard_angle_limit=False)
