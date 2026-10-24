"""Compact circular envelopes clipped onto the original curved surface.

The envelope is a 256-sided polygon inscribed in a metric circle. Its
intersection with the legal surface may have small indentations. Every
result vertex is reconstructed from barycentric coordinates on its source
triangle; neither subdivision nor smoothing changes the input shape.
"""
import numpy as np
import shapely
from shapely.geometry import Point, Polygon
import trimesh


CIRCLE_SIDES = 256


def tangent_frame(normal):
    normal = np.asarray(normal, dtype=float)
    first = np.cross(normal, np.eye(3)[np.argmin(np.abs(normal))])
    first /= np.linalg.norm(first)
    return np.column_stack([first, np.cross(normal, first)])


def circle_polygon(radius):
    angles = np.arange(CIRCLE_SIDES)*2*np.pi/CIRCLE_SIDES
    return Polygon(radius*np.column_stack([np.cos(angles), np.sin(angles)]))


def project_surface(mesh, ids, center, normal, maximum_radius=None, minimum_fill=1):
    frame = tangent_frame(normal)
    projected = (mesh.triangles[ids]-center) @ frame
    footprint = shapely.union_all(shapely.polygons(projected))
    origin = Point(0, 0)
    if footprint.is_empty or not footprint.contains(origin):
        return frame, projected, 0.0
    inner = float(footprint.boundary.distance(origin))
    if maximum_radius is None or minimum_fill >= 1:
        return frame, projected, inner
    if not 0 < minimum_fill <= 1:
        raise ValueError("Circle fill must lie in (0, 1]")
    low, high = min(inner, maximum_radius), maximum_radius
    def fits(radius):
        envelope = circle_polygon(radius)
        return footprint.intersection(envelope).area >= minimum_fill*envelope.area
    if fits(high):
        return frame, projected, high
    # A bounded expansion heuristic: keep at least the configured fraction of
    # the envelope covered by the legal surface. Final geometry is rechecked.
    for _ in range(28):
        middle = (low+high)/2
        if fits(middle):
            low = middle
        else:
            high = middle
    return frame, projected, low


def _triangulate_on_source(geometry, projected_source, source_triangle):
    if geometry.is_empty:
        return [], []
    children = shapely.constrained_delaunay_triangles(geometry)
    basis = np.column_stack([projected_source[1]-projected_source[0],
                             projected_source[2]-projected_source[0]])
    triangles, barycentric = [], []
    for polygon in children.geoms:
        xy = np.asarray(polygon.exterior.coords)[:3]
        uv = np.linalg.solve(basis, (xy-projected_source[0]).T).T
        bary = np.column_stack([1-uv.sum(axis=1), uv])
        if bary.min() < -1e-8 or bary.max() > 1+1e-8:
            raise ValueError("Clipped vertex left its source triangle")
        triangle = bary @ source_triangle
        original_normal = np.cross(source_triangle[1]-source_triangle[0],
                                    source_triangle[2]-source_triangle[0])
        child_normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
        if np.linalg.norm(child_normal) <= 1e-24:
            continue
        if child_normal @ original_normal < 0:
            triangle = triangle[[0, 2, 1]]
            bary = bary[[0, 2, 1]]
        triangles.append(triangle)
        barycentric.append(bary)
    return triangles, barycentric


def disk_topology(triangles):
    surface = trimesh.Trimesh(vertices=triangles.reshape(-1, 3),
                              faces=np.arange(len(triangles)*3).reshape(-1, 3),
                              process=False)
    surface.merge_vertices(digits_vertex=12)
    edges, counts = np.unique(surface.edges_sorted, axis=0, return_counts=True)
    boundary = edges[counts == 1]
    if not len(boundary) or np.any(counts > 2):
        return None
    degree = np.bincount(boundary.ravel(), minlength=len(surface.vertices))
    if np.any(degree[degree > 0] != 2):
        return None
    loops = trimesh.graph.connected_components(boundary, nodes=np.unique(boundary))
    components = trimesh.graph.connected_components(surface.face_adjacency,
                                                     nodes=np.arange(len(surface.faces)))
    if len(loops) != 1 or len(components) != 1 or surface.euler_number != 1:
        return None
    return dict(boundary_segments=surface.vertices[boundary],
                connected_components=1, boundary_components=1, euler_number=1)


def clip_surface_disk(mesh, ids, center, normal, radius, minimum_fill=1):
    frame = tangent_frame(normal)
    projected = (mesh.triangles[ids]-center) @ frame
    circle = circle_polygon(radius)
    footprint = shapely.union_all(shapely.polygons(projected))
    expected_area = float(footprint.intersection(circle).area)
    if expected_area < minimum_fill*circle.area-1e-14:
        return None
    selected, selected_bary, source_ids, outside = [], [], [], []
    touched = []
    for source, xy in zip(ids, projected):
        polygon = Polygon(xy)
        inside = polygon.intersection(circle)
        if inside.is_empty or inside.area <= 1e-24:
            continue
        children, barycentric = _triangulate_on_source(inside, xy, mesh.triangles[source])
        if not children:
            continue
        selected.extend(children)
        selected_bary.extend(barycentric)
        source_ids.extend([int(source)]*len(children))
        complement, _ = _triangulate_on_source(polygon.difference(circle), xy,
                                               mesh.triangles[source])
        outside.extend(complement)
        touched.append(int(source))
    if not selected:
        return None
    triangles = np.asarray(selected)
    projected_children = (triangles-center) @ frame
    edges = projected_children[:, 1:]-projected_children[:, :1]
    projected_area = float(np.sum(np.abs(edges[:, 0, 0]*edges[:, 1, 1]
                                         - edges[:, 0, 1]*edges[:, 1, 0]))/2)
    # The clipped footprint must be covered exactly once by this surface sheet.
    if abs(projected_area-expected_area) > max(1e-14, circle.area*1e-8):
        return None
    topology = disk_topology(triangles)
    if topology is None:
        return None
    area = float(np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                         triangles[:, 2]-triangles[:, 0]), axis=1).sum()/2)
    return dict(triangles=triangles, barycentric=np.asarray(selected_bary),
                source_face_ids=np.array(source_ids, dtype=np.int64),
                touched_source_faces=np.array(touched, dtype=np.int64),
                outside_triangles=np.asarray(outside).reshape(-1, 3, 3),
                area_m2=area, projected_area_m2=projected_area,
                envelope_area_m2=float(circle.area),
                projected_circle_fill=projected_area/float(circle.area), **topology)
