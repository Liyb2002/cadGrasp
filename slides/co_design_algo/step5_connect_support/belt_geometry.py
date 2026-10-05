"""Shared solid checks and demand-derived open floor rings for the loose frame."""
from step1.needs import COORD
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from step2_local_support import geometry as H, insertion as D
from step5_connect_support import ground as G, ring as R, routing as T, solids as S
from step5_connect_support.surface_check import surface_distances


def union_parts(parts, scale):
    """Remove only zero-thickness Boolean duplicates in/on the main solid.

    Thin, detached REAL material is retained and fails connectivity. Source
    contact interfaces are additionally replayed by the assembly audit.
    """
    joined, record = S.union_parts(parts, scale)
    removed = []
    if record['component_count'] > 1:
        components = sorted(joined.split(only_watertight=False), key=lambda p: abs(p.volume), reverse=True)
        main = components[0]
        for part in components[1:]:
            singular = np.linalg.svd(part.vertices-part.vertices.mean(axis=0), compute_uv=False)
            if abs(part.volume) > scale**3*1e-18 or singular[-1] > scale*1e-11: break
            points = np.vstack([part.vertices, part.triangles.mean(axis=1),
                ((part.triangles+np.roll(part.triangles, 1, axis=1))/2).reshape(-1, 3)])
            distances = surface_distances(main, points)
            near = distances <= scale*1e-9
            internal = main.contains(points[~near]) if np.any(~near) else np.array([], bool)
            if not internal.all(): break
            distance = float(distances.max())
            removed.append(dict(face_count=len(part.faces), volume_m3=float(part.volume),
                maximum_distance_to_main_surface_m=distance,
                all_checked_points_on_or_inside_main=True, surface_tolerance_m=scale*1e-9))
        else:
            joined = main
            record.update(component_count=1, boundary_component_count=1,positive_boundary_shells=1,cavity_count=0,zero_boundary_shells=0,cavity_containment_verified=True,watertight=bool(main.is_watertight),
                consistently_wound=bool(main.is_winding_consistent), volume_m3=float(main.volume))
            record['one_solid'] = bool(main.is_watertight and main.is_winding_consistent and main.volume > 0)
        if record['component_count'] != 1: removed = []
    record['zero_thickness_surface_duplicates_removed'] = removed
    return joined, record

class Scene:
    def __init__(self, mesh):
        self.mesh = mesh
        self.scale = float(mesh.extents.max())
        self.origin = mesh.bounds.mean(axis=0)
        self.solid = G.solid64(mesh, self.origin, self.scale)
        self.cache = {}

    def volume(self, part):
        if np.any(part.bounds[1] < self.mesh.bounds[0]) or np.any(part.bounds[0] > self.mesh.bounds[1]):
            return 0.
        key = np.asarray(part.vertices, np.float64).tobytes()
        if key not in self.cache:
            try:
                self.cache[key] = abs(float((self.solid ^ G.solid64(part, self.origin, self.scale)).volume()))*self.scale**3
            except ValueError:
                # A malformed candidate is unverified, never collision-free.
                self.cache[key] = float('inf')
        return self.cache[key]

    def clear(self, part, ground=False):
        floor = -self.scale*1e-10 if ground else self.scale*1e-10
        return bool(part.vertices[:, 2].min() >= floor and self.volume(part) <= 1e-11*self.scale**3)

def open_ring(mesh, polygon, required, pivot, angle, expansion, cut_fraction, scene):
    """Design and thicken a demand-enclosing Step5 boundary, then remove a front cap.

    The hull of the REMAINING floor material must still contain the demand.
    Low object geometry can enlarge the ring for final-pose clearance.
    """
    scale = scene.scale; height = .04*scale; margin = .008*scale
    low_triangles = mesh.triangles[np.min(mesh.triangles[:, :, 2], axis=1) <= height+margin]
    clipped = [H.clip_plane(t, np.array([0., 0., 1., -height-margin])) for t in low_triangles]
    low = [COORD.floor(p) for p in clipped if len(p)]
    cloud = np.vstack([polygon]+low)
    hull = cloud[ConvexHull(cloud).vertices]
    padded = np.asarray(Polygon(hull).buffer(margin, join_style=2).exterior.coords)[:-1]
    padded = padded[ConvexHull(padded).vertices]
    ring = R.make(padded, expansion, scale, width_fraction=.05, height_fraction=.04)
    a = COORD.floor(G.frame(angle)[0])
    lo = float(np.max(np.asarray(required)@a))
    hi = float(np.max(ring['outer_xy_m']@a))
    cut = lo+cut_fraction*(hi-lo)
    polygons = [R.clip(p, a, cut) for p in ring['edge_strips_xy_m']]
    polygons = [p for p in polygons if len(p) >= 3 and Polygon(p).area > 1e-14*scale**2]
    if not polygons: return None
    points = np.vstack(polygons)
    covered, _ = G.hull_coverage(required, np.vstack([points, COORD.floor(pivot)]), scale*1e-9)
    if not covered.all(): return None
    parts = []
    for p in polygons:
        bottom = COORD.lift_floor(p)
        part = D.engine.hull_mesh(np.vstack([bottom, bottom+[0, 0, height]]))
        if not scene.clear(part, ground=True): return None
        parts.append(part)
    joined, solid = union_parts(parts, scale)
    if not solid['one_solid']: return None
    return parts, dict(bearing_deg=float(angle), expansion=float(expansion), cut_fraction=float(cut_fraction),
        cut_normal_xy=a.tolist(), cut_offset_m=cut, height_m=height,
        width_m=ring['width_m'], pads_xy_m=[p.tolist() for p in polygons],
        step5_seed_polygon_xy_m=np.asarray(polygon).tolist(),
        inner_xy_m=ring['inner_xy_m'].tolist(), outer_xy_m=ring['outer_xy_m'].tolist(),
        opening='front cap removed along +a', construction='Step5 demand-derived ring clipped to an open arc',
        actual_remaining_footprint_covers_demand=True, solid=solid)
