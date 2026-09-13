"""Surface-following open belts, clipped convex floor rings and short links.

The belt has a small stand-off: only the original Step3 interfaces contact the
workpiece. Graph paths avoid work faces. No machining-ray obstacle is added.
"""
from step1.needs import COORD
from heapq import heappop, heappush
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, LineString
from shapely.ops import nearest_points

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
        return bool(part.vertices[:, 1].min() >= floor and self.volume(part) <= 1e-11*self.scale**3)


def face_frame(mesh, face):
    triangle = mesh.triangles[face]
    x = triangle[1]-triangle[0]; x /= np.linalg.norm(x)
    return triangle[0], np.array([x, np.cross(mesh.face_normals[face], x)])


def safe_face_polygon(mesh, face, inner, outer, tolerance):
    weights = np.eye(3)
    for offsets in (inner, outer):
        heights = (mesh.vertices+offsets)[mesh.faces[face], 1]
        weights = H.clip_plane(weights, np.r_[-heights, tolerance])
        if len(weights) < 3: return np.empty((0, 3))
    return weights@mesh.triangles[face]


def ribbon_cell(mesh, face, first, second, width, inner, outer):
    """Clip each ribbon rectangle to its source triangle before offsetting."""
    origin, basis = face_frame(mesh, face)
    line = (np.array([first, second])-origin)@basis.T
    if np.linalg.norm(line[1]-line[0]) < width*1e-8:
        line[1, 0] += width*1e-7
    rectangle = LineString(line).buffer(width/2, cap_style=3, join_style=2)
    surface = safe_face_polygon(mesh, face, inner, outer, float(mesh.extents.max())*1e-9)
    if len(surface) < 3: return None
    polygon = rectangle.intersection(Polygon((surface-origin)@basis.T))
    if polygon.is_empty or polygon.area <= width**2*1e-12:
        return None
    points = np.asarray(polygon.exterior.coords)[:-1]@basis+origin
    a = H.head_cell(mesh, points, face, inner)[len(points):]
    b = H.head_cell(mesh, points, face, outer)[len(points):]
    return D.engine.hull_mesh(np.vstack([a, b]))


def belt(mesh, contacts, work_ids, depth, scene, blocked_faces=()):
    """Greedy shortest graph tree of ribbons; the belt need not form a loop."""
    blocked_faces = tuple(int(f) for f in blocked_faces)
    scale = scene.scale
    gap = min(.0008*scale, depth*.15)
    thickness = depth
    width = .025*scale
    inner, valid1 = H.vertex_offsets(mesh, gap)
    outer, valid2 = H.vertex_offsets(mesh, gap+thickness)
    allowed = np.all((valid1 & valid2)[mesh.faces], axis=1)
    allowed[np.asarray(work_ids, int)] = False
    allowed[np.asarray(blocked_faces, int)] = False
    # Clip near-floor faces instead of rejecting a whole face whose distant
    # vertex happens to touch the floor. Actual ribbons remain strictly above it.
    centers = mesh.triangles_center.copy()
    near_floor = np.min((mesh.vertices+outer)[mesh.faces, 1], axis=1) <= scale*1e-9
    for face in np.flatnonzero(allowed & near_floor):
        polygon = safe_face_polygon(mesh, face, inner, outer, scale*1e-9)
        if len(polygon) < 3: allowed[face] = False
        else: centers[face] = polygon.mean(axis=0)
    graph = [[] for _ in mesh.faces]; shared = {}
    for (a, b), edge in zip(mesh.face_adjacency, mesh.face_adjacency_edges):
        if allowed[a] and allowed[b]:
            low, high = 0., 1.
            for offsets in (inner, outer):
                z = (mesh.vertices+offsets)[edge, 1]-scale*1e-9
                if np.all(z <= 0): low, high = 1., 0.; break
                if z[0] <= 0: low = max(low, float(-z[0]/(z[1]-z[0])))
                if z[1] <= 0: high = min(high, float(-z[0]/(z[1]-z[0])))
            if high-low <= 1e-9: continue
            mid = mesh.vertices[edge[0]]+(low+high)/2*(mesh.vertices[edge[1]]-mesh.vertices[edge[0]])
            weight = float(np.linalg.norm(centers[a]-mid)+np.linalg.norm(centers[b]-mid))
            graph[a].append((int(b), weight)); graph[b].append((int(a), weight))
            shared[int(a), int(b)] = shared[int(b), int(a)] = mid
    analyzer = D.Analyzer(mesh, depth)
    groups = [analyzer.heads(c) for c in contacts]
    heads = [p for group in groups for p in group]
    report = dict(passed=False, head_ids=[c['candidate_id'] for c in contacts],
        gap_m=gap, thickness_m=thickness, width_m=width, paths=[],
        work_faces_excluded=True, closed_loop_required=False,
        method='greedy shortest paths on nonworking face adjacency; finite-width clipped offset ribbons',
        global_shortest_network_claimed=False)
    if not contacts or any(not scene.clear(p) for p in heads):
        report['status'] = 'head_geometry_not_clear'; return report, heads, [], []
    # A single Step3 contact can comprise disconnected source-face cells.
    # Join ALL of them using points strictly inside actual clipped triangles.
    anchors = []
    for c in contacts:
        for face in np.unique(c['source_faces']):
            ids = np.flatnonzero(c['source_faces'] == face)
            index = ids[np.argmax(c['triangle_areas_m2'][ids])]
            anchors.append((int(face), c['triangles_m'][index].mean(axis=0)))
    terminals = [face for face, point in anchors]
    report['rerouted_around_faces'] = list(blocked_faces)
    report['attached_head_cell_count'] = len(anchors)
    if not all(allowed[f] for f in terminals):
        report['status'] = 'head_center_outside_allowed_surface_graph'; return report, heads, [], []
    tree = {terminals[0]}; remaining = set(terminals[1:]); face_paths = []
    while remaining:
        queue = [(0., f) for f in sorted(tree)]; distance = {f: 0. for f in tree}; parent = {}
        target = None
        while queue:
            cost, face = heappop(queue)
            if cost != distance[face]: continue
            if face in remaining:
                target = face; break
            for other, weight in graph[face]:
                new = cost+weight
                if new < distance.get(other, np.inf):
                    distance[other] = new; parent[other] = face; heappush(queue, (new, other))
        if target is None:
            report['status'] = 'nonworking_surface_graph_disconnected'; return report, heads, [], []
        path = [target]
        while path[-1] not in tree: path.append(parent[path[-1]])
        path.reverse(); tree.update(path); remaining -= tree; face_paths.append(path)
        report['paths'].append(dict(source_faces=path, graph_length_m=distance[target]))
    segments = []
    for path in face_paths:
        for first, second in zip(path[:-1], path[1:]):
            mid = shared[first, second]
            segments.extend([(first, centers[first], mid), (second, mid, centers[second])])
    # Tie every original head to the graph at its center, even when multiple
    # terminals occupy one face. No source contact triangle is enlarged.
    for face, point in anchors:
        segments.append((face, point, centers[face]))
    cells = []; owners = []
    for face, first, second in segments:
        part = ribbon_cell(mesh, face, first, second, width, inner, outer)
        if part is None or not scene.clear(part):
            if face not in terminals and len(blocked_faces) < 16:
                return belt(mesh, contacts, work_ids, depth, scene, (*blocked_faces, int(face)))
            report.update(status='surface_ribbon_collision', failed_source_face=int(face))
            return report, heads, cells, owners
        cells.append(part); owners.append(int(face))
    # Adjacent offset cells meet on a side face. Add small positive-volume
    # joints there so floating-point face coincidences cannot split the belt.
    # The joint is centered halfway through the skin and independently cleared.
    joints = 0
    for path in face_paths:
        for first, second in zip(path[:-1], path[1:]):
            point = shared[first, second]
            mid_offsets = (inner+outer)/2
            center = H.head_cell(mesh, point[None], first, mid_offsets)[1]
            radius = min(gap*.2, width*.05, thickness*.1, center[1]*.25)
            if radius <= scale*1e-10: continue
            part = T.beam(center, center, radius, radius)
            if not scene.clear(part):
                report.update(status='surface_joint_collision', failed_source_face=int(first))
                return report, heads, cells, owners
            cells.append(part); owners.append(int(first)); joints += 1
    joined, solid = union_parts(heads+cells, scale)
    report.update(passed=solid['one_solid'], status='belt_connected' if solid['one_solid'] else 'belt_union_disconnected',
        solid=solid, ribbon_source_faces=owners, ribbon_cell_count=len(cells), seam_joint_count=joints)
    return report, heads, cells, owners


def open_ring(mesh, polygon, required, pivot, angle, expansion, cut_fraction, scene):
    """Design and thicken a demand-enclosing Step5 boundary, then remove a front cap.

    The hull of the REMAINING floor material must still contain the demand.
    Low object geometry can enlarge the ring for final-pose clearance.
    """
    scale = scene.scale; height = .04*scale; margin = .008*scale
    low_triangles = mesh.triangles[np.min(mesh.triangles[:, :, 1], axis=1) <= height+margin]
    clipped = [H.clip_plane(t, np.array([0., 1., 0., -height-margin])) for t in low_triangles]
    low = [COORD.floor(p) for p in clipped if len(p)]
    cloud = np.vstack([polygon]+low)
    hull = cloud[ConvexHull(cloud).vertices]
    padded = np.asarray(Polygon(hull).buffer(margin, join_style=2).exterior.coords)[:-1]
    padded = padded[ConvexHull(padded).vertices]
    ring = R.make(padded, expansion, scale, width_fraction=.05, height_fraction=.04)
    a = COORD.floor(G.frame(angle)[0])
    lo = float(np.max(np.asarray(required)@a))
    hi = float(np.max(ring['outer_xz_m']@a))
    cut = lo+cut_fraction*(hi-lo)
    polygons = [R.clip(p, a, cut) for p in ring['edge_strips_xz_m']]
    polygons = [p for p in polygons if len(p) >= 3 and Polygon(p).area > 1e-14*scale**2]
    if not polygons: return None
    points = np.vstack(polygons)
    covered, _ = G.hull_coverage(required, np.vstack([points, COORD.floor(pivot)]), scale*1e-9)
    if not covered.all(): return None
    parts = []
    for p in polygons:
        bottom = COORD.lift_floor(p)
        part = D.engine.hull_mesh(np.vstack([bottom, bottom+[0, height, 0]]))
        if not scene.clear(part, ground=True): return None
        parts.append(part)
    joined, solid = union_parts(parts, scale)
    if not solid['one_solid']: return None
    return parts, dict(bearing_deg=float(angle), expansion=float(expansion), cut_fraction=float(cut_fraction),
        cut_normal_xz=a.tolist(), cut_offset_m=cut, height_m=height,
        width_m=ring['width_m'], pads_xz_m=[p.tolist() for p in polygons],
        step5_seed_polygon_xz_m=np.asarray(polygon).tolist(),
        inner_xz_m=ring['inner_xz_m'].tolist(), outer_xz_m=ring['outer_xz_m'].tolist(),
        opening='front cap removed along +a', construction='Step5 demand-derived ring clipped to an open arc',
        actual_remaining_footprint_covers_demand=True, solid=solid)


def short_links(skin, base, scene, limit=8):
    """Broad shoulder into a 5%-D beam, with a 3.2%-D floor joint."""
    from shapely.geometry import Point
    radius = .025*scene.scale
    end_radius = min(radius, .4*base['height_m'])
    proposals = []
    polygons = [Polygon(p) for p in base['pads_xz_m']]
    for root_radius, index, start in T.roots(skin, count=24):
        closest, _, _ = scene.mesh.nearest.on_surface(start[None])
        normal = start-closest[0]
        if np.linalg.norm(normal) <= scene.scale*1e-10: continue
        normal /= np.linalg.norm(normal)
        # Use an area of the actual cell, not a tiny cubic point joint.
        root_points = start+.85*(skin[index].vertices-start)
        for distance in (.04, .07, .10):
            shoulder = start+normal*scene.scale*distance
            cap = D.engine.hull_mesh(np.vstack([root_points, shoulder+T.CORNERS*radius]))
            if not scene.clear(cap): continue
            for poly in polygons:
                inset = poly.buffer(-min(end_radius*.3, base['width_m']*.15))
                if inset.is_empty: inset=poly
                xy=np.asarray(nearest_points(Point(COORD.floor(shoulder)), inset)[1].coords[0])
                target=COORD.lift_floor(xy,base['height_m']*.5)
                bar=T.beam(shoulder,target,radius,end_radius)
                length=float(np.linalg.norm(shoulder-start)+np.linalg.norm(target-shoulder))
                proposals.append((length,index,cap,bar,np.array([start,shoulder,target])))
    proposals.sort(key=lambda p:p[0]); produced=0
    for length,index,cap,bar,path in proposals:
        if not scene.clear(bar): continue
        yield [cap,bar], dict(length_m=length,waypoints_m=path.tolist(),root_skin_cell=int(index),
            beam_width_m=2*radius, floor_joint_width_m=2*end_radius,
            shoulder_root='85 percent of actual source cell about its interior center',
            method='length-ordered broad shoulder and thick beam menu',global_shortest_claimed=False)
        produced+=1
        if produced>=limit:break
