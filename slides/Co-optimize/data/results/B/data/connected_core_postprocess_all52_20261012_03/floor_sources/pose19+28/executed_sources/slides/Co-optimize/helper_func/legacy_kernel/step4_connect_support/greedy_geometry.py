"""Nearest feasible head assignment, direct local lofts, and a fixed-graph MST.

The foot targets are an explicit heuristic initialization, not a theorem about
the minimum support footprint. Neither nearest assignment nor the MST solves
the globally minimum-volume, collision-constrained fixture problem.
"""
from dataclasses import dataclass
import numpy as np
import manifold3d as md
import trimesh
from scipy.spatial import ConvexHull, cKDTree
from shapely.geometry import MultiPoint

from step2_local_support import geometry as G
from step4_connect_support import geometry_kernel as K
from step4_connect_support import head_registration as H

VOLUME_TOL = 8e-14


def union(parts):
    return md.Manifold.batch_boolean(list(parts), md.OpType.Add)


def volume(value):
    return float(value.volume()) * K.SCALE**3


def contains(container, required):
    return abs(volume(required-container)) <= VOLUME_TOL


def component_containing(value, required):
    for component in sorted(value.decompose(), key=lambda q: -q.volume()):
        if all(contains(component, item) for item in required):
            return component
    return None


def kruskal(count, edges):
    """Return edge indices for a minimum additive-cost tree on a FIXED graph."""
    parent = list(range(count))
    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    result = []
    for index in sorted(range(len(edges)), key=lambda i: (edges[i][2], edges[i][0], edges[i][1])):
        a, b, _ = edges[index]
        a, b = root(a), root(b)
        if a != b:
            parent[a] = b
            result.append(index)
    if count and len(result) != count-1:
        raise RuntimeError('Legal bridge candidate graph is disconnected')
    return result


def nearest_order(head_vertices, point):
    distances = [float(np.linalg.norm(v-point, axis=1).min()) for v in head_vertices]
    return sorted(range(len(distances)), key=lambda i: (distances[i], i)), distances


def foot_targets(demand, outward=0., halfwidth=.004, family='hull'):
    """A square terminal centered at each vertex of a buffered demand hull.

    Buffering is an explicit candidate family. Exact original hull vertices
    remain covered by the *intended* footprints; the actual solid is rechecked.
    """
    polygon = MultiPoint(np.asarray(demand)).convex_hull
    if family == 'minimum_rectangle':
        polygon = polygon.minimum_rotated_rectangle
    elif family == 'axis_box':
        polygon = polygon.envelope
    elif family != 'hull':
        raise ValueError('Unknown enclosing target polygon family')
    if outward:
        polygon = polygon.buffer(outward, join_style=2)
    centers = np.asarray(polygon.exterior.coords)[:-1]
    square = halfwidth*np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    return [center+square for center in centers]


def actual_footprint(mesh, basis, offset):
    world = K.local_to_world(mesh.vertices, basis, offset)
    xy = np.unique(world[np.abs(world[:, 2]) <= 1e-9, :2], axis=0)
    if len(xy) < 3 or np.linalg.matrix_rank(xy-xy.mean(0), tol=1e-10) < 2:
        return np.empty((0, 2))
    return xy[ConvexHull(xy).vertices]


def footprint_coverage(footprint, pivot, demands):
    points = np.vstack([np.asarray(footprint).reshape(-1, 2), np.asarray(pivot)[:2]])
    if len(points) < 3 or np.linalg.matrix_rank(points-points.mean(0), tol=1e-10) < 2:
        return dict(passed=False, outside_sample_count=len(demands), max_outside_m=None)
    hull = ConvexHull(points)
    distances = np.max(np.asarray(demands)@hull.equations[:, :2].T+hull.equations[:, 2], axis=1)
    return dict(passed=bool(np.all(distances <= 1e-9)),
                outside_sample_count=int(np.count_nonzero(distances > 1e-9)),
                max_outside_m=float(max(0., distances.max())))


@dataclass
class Head:
    pose: int
    active_poses: tuple
    ident: str
    vertices: np.ndarray
    root: np.ndarray
    original: object


class Constructor:
    def __init__(self, groups, heads, directions, bases, offsets, forbidden,
                 root_depth=.008, terminal_depth=.003, bridge_radius=.004):
        self.bases, self.offsets = np.asarray(bases), np.asarray(offsets)
        self.forbidden = forbidden
        self.terminal_depth, self.bridge_radius = terminal_depth, bridge_radius
        self.heads, self.bodies = [], []
        registered, self.registration = H.register(groups, heads, bases, offsets)
        if not self.registration['all_heads_above_both_floors']:
            raise ValueError('An original head penetrates a task floor')
        for head in registered:
            k, local = head.pose, head.cells
            vertices = np.concatenate(local)
            vertices = vertices[ConvexHull(vertices).vertices]
            original = union(K.solid(G.hull_mesh(v)) for v in local)
            root = vertices+root_depth*np.asarray(directions[k])@bases[k]
            seed = self.carve(K.solid(G.hull_mesh(np.vstack([vertices, root]))))+original
            body = component_containing(seed, [original])
            if body is None:
                raise RuntimeError('A head seed is disconnected after clearance carving')
            self.heads.append(Head(k, head.active_poses, head.ident, vertices, root, original))
            self.bodies.append(body)
        self.assignments = []

    def carve(self, value):
        for basis, offset in zip(self.bases, self.offsets):
            value = value.trim_by_plane(basis[2].tolist(), float(basis[2]@offset/K.SCALE))
        return value-self.forbidden

    def assign(self, pose, polygon, index):
        basis, offset = self.bases[pose], self.offsets[pose]
        pad = np.c_[polygon, np.zeros(len(polygon))]@basis+offset
        terminal = np.vstack([pad, pad+self.terminal_depth*basis[2]])
        requested = K.solid(G.hull_mesh(terminal))
        pieces = sorted(self.carve(requested).decompose(), key=lambda c: -c.volume())
        required = pieces[0] if pieces else md.Manifold()
        actual = actual_footprint(K.unpack(required), basis, offset) if pieces else np.empty((0, 2))
        # Clipping may change a terminal. Require genuine coplanar area here,
        # and recheck enclosure from the actual final geometry, never requests.
        if len(actual) < 3 or MultiPoint(actual).convex_hull.area < 1e-6:
            raise RuntimeError(f'Foot target {pose}/{index} intersects forbidden space')
        order, distances = nearest_order([h.vertices for h in self.heads], pad.mean(0))
        rejected = []
        for i in order:
            head = self.heads[i]
            addition = self.carve(K.solid(G.hull_mesh(np.vstack([head.vertices, head.root, terminal]))))
            candidate = component_containing(self.bodies[i]+addition, [self.bodies[i], required])
            if candidate is None:
                rejected.append(i)
                continue
            added = volume(candidate)-volume(self.bodies[i])
            self.bodies[i] = candidate
            row = dict(floor_pose_index=pose, terminal=index, head_body=i,
                       head_pose_index=head.pose, candidate_id=head.ident,
                       active_pose_indices=list(head.active_poses),
                       polygon_xy_m=np.asarray(polygon).tolist(), distance_m=distances[i],
                       rejected_closer_heads=rejected, added_volume_cm3=added*1e6)
            self.assignments.append(row)
            return row
        raise RuntimeError(f'No head admits a direct connected loft to floor target {pose}/{index}')

    def connect(self):
        """Build the entire legal edge graph once, then run real Kruskal."""
        full = union(self.bodies)
        components = full.decompose()
        if len(components) == 1:
            return full, [], dict(component_count_before=1, candidate_edges=[], selected_edges=[])
        radius = self.bridge_radius
        bead = trimesh.creation.icosphere(subdivisions=1, radius=radius).vertices
        surfaces = []
        for comp in components:
            vertices = K.unpack(comp).vertices
            mask = np.ones(len(vertices), bool)
            for basis, offset in zip(self.bases, self.offsets):
                mask &= (vertices-offset)@basis[2] > radius*1.1
            surfaces.append(vertices[mask])
        up = self.bases[0][2]+self.bases[1][2]
        if np.linalg.norm(up) > 1e-8:
            up /= np.linalg.norm(up)
            detours = [np.zeros(3), .008*up, .016*up, .024*up]
        else:
            # Opposite parallel floors have no common "up". Try short lateral
            # doglegs in the slab, with both floor constraints still enforced.
            detours = [np.zeros(3)]+[s*.012*self.bases[0][axis] for axis in (0, 1) for s in (-1, 1)]
        records, solids, edges = [], [], []
        for i, a in enumerate(surfaces):
            for j in range(i):
                b = surfaces[j]
                if not len(a) or not len(b):
                    continue
                distances, near = cKDTree(b).query(a)
                pairs = []
                for q in np.argsort(distances):
                    x, y = a[q], b[near[q]]
                    if any(np.linalg.norm(x-u)+np.linalg.norm(y-v) < .008 for u, v in pairs):
                        continue
                    pairs.append((x, y))
                    if len(pairs) >= 10:
                        break
                candidates = []
                for x, y in pairs:
                    for detour in detours:
                        route = [x, y] if not np.any(detour) else [x, (x+y)/2+detour, y]
                        length = sum(float(np.linalg.norm(v-u)) for u, v in zip(route[:-1], route[1:]))
                        candidates.append((length, route))
                for length, route in sorted(candidates, key=lambda r: r[0]):
                    bridge = self.carve(union(K.solid(G.hull_mesh(np.vstack([x+bead, y+bead])))
                                             for x, y in zip(route[:-1], route[1:])))
                    joined = components[i]+components[j]+bridge
                    if component_containing(joined, [components[i], components[j]]) is None:
                        continue
                    if any(abs(volume(bridge^components[k])) < 1e-12 for k in (i, j)):
                        continue
                    records.append(dict(source=i, target=j, length_m=length,
                                        path_m=np.asarray(route).tolist(), radius_m=radius))
                    solids.append(bridge); edges.append((i, j, length))
                    break
        selected = kruskal(len(components), edges)
        bridges = [solids[i] for i in selected]
        result = union([full]+bridges)
        if len(result.decompose()) != 1:
            raise RuntimeError('Selected MST bridges did not produce one actual solid')
        return result, bridges, dict(component_count_before=len(components),
            candidate_edges=records, selected_edges=selected,
            total_selected_length_m=sum(edges[i][2] for i in selected),
            optimality_scope='minimum sum of edge lengths on the fixed tested candidate graph only')
