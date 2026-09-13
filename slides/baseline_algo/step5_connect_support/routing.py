"""Finite-width, bent head-to-ring connections with lazy continuous-sweep checks.

The centreline describes material, not the installation trajectory. Every bar is
checked at the installed pose against the work volume and over its full rigid
withdrawal against the object. A finite roadmap failure is never an impossibility
certificate. Sampling proposes shapes; acceptance checks the complete solids.
"""
from step1.needs import COORD
from heapq import heappop, heappush
from itertools import product
import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from step5_connect_support import ground as G, motion as M, ring as R
from step2_local_support import insertion as D

CORNERS = np.array(list(product([-1., 1.], repeat=3)))


def beam(first, second, first_radius, second_radius):
    """Convex tapered bar between two world-aligned cubic joints."""
    return D.engine.hull_mesh(np.vstack([first+CORNERS*first_radius,
                                        second+CORNERS*second_radius]))


def roots(heads, count=3):
    """Interior joints contained in actual head cells; no interface offsets."""
    choices = []
    for index, head in enumerate(heads):
        points = np.asarray(head.vertices)
        origin = points.mean(axis=0)
        scale = float(np.ptp(points, axis=0).max())
        planes = ConvexHull((points-origin)/scale).equations
        radius = float(np.min(-planes[:, 3]/np.abs(planes[:, :3]).sum(axis=1)))*scale*.75
        if radius > scale*1e-10:
            choices.append((radius, index, origin))
    choices.sort(key=lambda r: -r[0])
    return choices[:count]


def anchors(ring, contact, angle, count=6):
    """Independent ground attachment candidates, no mandatory withdrawal ray."""
    p = np.asarray(COORD.floor(contact['center_m']))
    choices = []
    hit = R.ray_exit(ring, p, -COORD.floor(G.frame(angle)[0]))
    if hit is not None:
        choices.append(hit[0])
    xy = ring['inner_xy_m']
    projected = []
    for first, second in zip(xy, np.roll(xy, -1, axis=0)):
        delta = second-first
        fraction = np.clip((p-first)@delta/(delta@delta), 0., 1.)
        projected.extend([first+fraction*delta, (first+second)/2])
    projected.sort(key=lambda point: float(np.linalg.norm(point-p)))
    # Retain distinct parts of the perimeter instead of many adjacent tiny edges.
    separation = max(ring['width_m']*1.5, np.ptp(xy, axis=0).max()*.08)
    for point in projected:
        if all(np.linalg.norm(point-other) > separation for other in choices):
            choices.append(point)
        if len(choices) >= count:
            break
    return choices


class Router:
    def __init__(self, mesh, work_volume, angle, width_fraction=.022, edge_budget=1600):
        self.mesh, self.work = mesh, work_volume
        self.scale = float(mesh.extents.max())
        self.direction = G.frame(angle)[0]
        self.basis = G.frame(angle)
        self.radius = width_fraction*self.scale/2
        self.edge_budget = edge_budget
        self.cache = {}
        self.checks = 0
        self.rejections = {'work_volume': 0, 'object_sweep_or_floor': 0}

    def clear(self, part):
        key = np.asarray(part.vertices, np.float64).tobytes()
        if key not in self.cache:
            self.checks += 1
            if part.vertices[:,2].min() <= self.scale*1e-10:
                self.rejections['object_sweep_or_floor'] += 1
                self.cache[key] = False
            elif self.work is not None and not self.work.check_parts([part])['passed']:
                self.rejections['work_volume'] += 1
                self.cache[key] = False
            elif not M.sweep_check(self.mesh, [part], self.direction)['passed']:
                self.rejections['object_sweep_or_floor'] += 1
                self.cache[key] = False
            else:
                self.cache[key] = True
        return self.cache[key]

    def edge(self, p, q, r0=None, r1=None):
        part = beam(np.asarray(p), np.asarray(q), self.radius if r0 is None else r0,
                    self.radius if r1 is None else r1)
        return part if self.clear(part) else None

    def polyline(self, points, root_radius):
        parts = []
        for index, (first, second) in enumerate(zip(points[:-1], points[1:])):
            part = self.edge(first, second, root_radius if index == 0 else self.radius)
            if part is None:
                return None
            parts.append(part)
        return parts

    def roadmap(self, start, target, root_radius, budget_start):
        """A* on a finite spatial lattice; lazy edges check finite-width solids."""
        step = .075*self.scale
        low = np.minimum(self.mesh.bounds[0], np.minimum(start, target))-.12*self.scale
        high = np.maximum(self.mesh.bounds[1], np.maximum(start, target))+.12*self.scale
        low[2] = self.radius+self.scale*1e-9
        high[2] = max(start[2], target[2])+.12*self.scale
        axes = [np.linspace(a, b, max(2, int(np.ceil((b-a)/step))+1)) for a, b in zip(low, high)]
        grid = np.array(np.meshgrid(*axes, indexing='ij')).reshape(3, -1).T
        # A short withdrawal waypoint gives the thin head joint room to widen.
        starters = [start-self.direction*d*self.scale for d in [.025, .05, .1]]
        landmarks = []
        for rear in starters:
            landmarks.extend([rear, np.array([rear[0], rear[1], target[2]]),
                              np.array([target[0], rear[1], rear[2]]),
                              np.array([rear[0], rear[1], target[2]])])
        sparse = np.vstack([start, target, landmarks])
        landmark_ids = list(range(2, len(sparse)))
        path = self.graph_path(sparse, root_radius, budget_start, landmark_ids)
        if path is not None:
            return path
        return self.graph_path(np.vstack([sparse, grid]), root_radius, budget_start, landmark_ids)

    def graph_path(self, nodes, root_radius, budget_start, landmark_ids):
        start, target = nodes[:2]
        tree = cKDTree(nodes)
        costs = {0: 0.}
        parents = {}
        queue = [(float(np.linalg.norm(start-target)), 0)]
        settled = set()
        while queue and self.checks-budget_start < self.edge_budget:
            _, i = heappop(queue)
            if i in settled:
                continue
            if i == 1:
                route = [1]
                while route[-1] != 0:
                    route.append(parents[route[-1]])
                points = nodes[route[::-1]]
                # Shortcut only when the actual widened shortcut is also clear.
                simplified = [points[0]]
                current = 0
                while current < len(points)-1:
                    for last in range(len(points)-1, current, -1):
                        if self.edge(points[current], points[last], root_radius if current == 0 else self.radius) is not None:
                            break
                    simplified.append(points[last]); current = last
                return np.asarray(simplified)
            settled.add(i)
            neighbors = list(np.atleast_1d(tree.query(nodes[i], k=min(22, len(nodes)))[1]))
            neighbors += [1]+landmark_ids
            for j in dict.fromkeys(neighbors):
                j = int(j)
                if j == i or j in settled:
                    continue
                cost = costs[i]+float(np.linalg.norm(nodes[j]-nodes[i]))
                if cost >= costs.get(j, np.inf):
                    continue
                if self.checks-budget_start >= self.edge_budget:
                    break
                if self.edge(nodes[i], nodes[j], root_radius if i == 0 else self.radius) is None:
                    continue
                costs[j] = cost; parents[j] = i
                heappush(queue, (cost+float(np.linalg.norm(nodes[j]-target)), j))
        return None

    def connect(self, heads, anchor, height, allow_roadmap=True):
        # Keep every routed bar strictly above the floor. Only the owned strips
        # supply the saved ground footprint; the terminal joint overlaps their top half.
        target = COORD.lift_floor(anchor, self.radius+height*.5)
        budget_start = self.checks
        if not self.clear(beam(target, target, self.radius, self.radius)):
            return None
        for root_radius, cell, start in roots(heads):
            start_radius = min(root_radius, self.radius)
            # Try compact shapes first, including early descent and lateral bends.
            paths = [[start, target]]
            for distance in [.03, .08, .16]:
                rear = start-distance*self.scale*self.direction
                paths.append([start, rear, target])
                low = rear.copy(); low[2] = target[2]
                paths.append([start, rear, low, target])
                for side in [-1., 1.]:
                    around = rear+side*.12*self.scale*self.basis[1]
                    down = around.copy(); down[2] = target[2]
                    paths.append([start, rear, around, down, target])
            for points in paths:
                parts = self.polyline(points, start_radius)
                if parts is not None:
                    return self.record(parts, points, cell, start_radius, 'direct_or_bent', budget_start)
        if allow_roadmap:
            for root_radius, cell, start in roots(heads, count=1):
                start_radius = min(root_radius, self.radius)
                points = self.roadmap(start, target, start_radius, budget_start)
                if points is not None:
                    parts = self.polyline(points, start_radius)
                    if parts is not None:
                        return self.record(parts, points, cell, start_radius, 'spatial_roadmap', budget_start)
        return None

    def record(self, parts, points, cell, root_radius, method, started):
        points = np.asarray(points)
        return dict(parts=parts, record=dict(method=method, root_head_cell=int(cell),
            root_joint_half_width_m=float(root_radius), member_width_m=2*self.radius,
            waypoints_m=points.tolist(), length_m=float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()),
            edge_checks=self.checks-started, finite_width_edges_verified=True,
            full_object_sweeps_checked=True, lattice_spacing_fraction=.075,
            edge_budget=self.edge_budget, global_path_existence_decided=False))
