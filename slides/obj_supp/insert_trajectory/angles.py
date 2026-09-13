"""Horizontal insertion angles: local inequalities and whole-head sweep checks.

All intervals use world +X=0 degrees, +Z=90 degrees. The support moves along
+a on insertion and along -a on withdrawal. Floor and other supports are absent.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import ConvexHull
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'baseline_algo'))
from step2_local_support import geometry as G
from step1.needs import COORD


def hull_mesh(points):
    """Preserve tiny clipped faces instead of merging vertices in metre units."""
    points=np.asarray(points,float)
    origin=points.mean(axis=0);scale=float(np.ptp(points,axis=0).max())
    local=(points-origin)/scale
    hull=ConvexHull(local)
    faces=hull.simplices.copy();tri=local[faces]
    wrong=(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])*hull.equations[:,:3]).sum(axis=1)<0
    faces[wrong]=faces[wrong][:,[0,2,1]]
    used,inverse=np.unique(faces,return_inverse=True)
    mesh=trimesh.Trimesh(points[used],inverse.reshape(-1,3),process=False)
    mesh.metadata['convex_volume_m3']=float(hull.volume*scale**3)
    return mesh


def direction(degrees):
    angle = np.deg2rad(degrees)
    return np.array([np.cos(angle), np.sin(angle), 0.])


def wrap_interval(start, end):
    if end-start >= 360-1e-10:
        return [[0., 360.]]
    start0 = start % 360
    end0 = start0+end-start
    return [[start0, end0]] if end0 <= 360 else [[start0, 360.], [0., end0-360.]]


def merge(intervals, tolerance=1e-9):
    result = []
    for low, high in sorted(intervals):
        if high <= low:
            continue
        if result and low <= result[-1][1]+tolerance:
            result[-1][1] = max(result[-1][1], float(high))
        else:
            result.append([float(low), float(high)])
    return result


def subtract(intervals, cuts):
    result = []
    for low, high in intervals:
        pieces = [[low, high]]
        for a, b in merge(cuts):
            next_pieces = []
            for x, y in pieces:
                if b <= x or a >= y:
                    next_pieces.append([x, y])
                else:
                    if a > x: next_pieces.append([x, a])
                    if b < y: next_pieces.append([b, y])
            pieces = next_pieces
        result.extend(pieces)
    return merge(result)


def intersection(first, second):
    return merge([[max(a, c), min(b, d)] for a, b in first for c, d in second
                  if min(b, d) > max(a, c)])


def local_angles(normals):
    """Analytic sign cells, including isolated tangent directions."""
    normals = np.asarray(normals)
    active = normals[np.linalg.norm(COORD.floor(normals), axis=1) > 1e-13]
    if not len(active):
        return dict(intervals_deg=[[0., 360.]], isolated_angles_deg=[])
    phase = np.rad2deg(np.arctan2(active[:, 2], active[:, 0]))
    events = np.unique(np.r_[0., 360., (phase-90)%360, (phase+90)%360])
    intervals = []
    for a, b in zip(events[:-1], events[1:]):
        if (active@direction((a+b)/2) <= 1e-12).all():
            intervals.append([a, b])
    intervals = merge(intervals)
    isolated = [float(a) for a in events[:-1]
                if (active@direction(a) <= 1e-12).all()
                and not any(x-1e-9 <= a <= y+1e-9 for x, y in intervals)
                and not (a == 0 and any(abs(y-360) < 1e-9 for _, y in intervals))]
    return dict(intervals_deg=intervals, isolated_angles_deg=isolated)


def obstacle_shadow(triangle, head, scale, guard_deg=1e-6):
    """An OPEN forbidden angular interval from one exact convex pair.

    q = triangle - head is the configuration obstacle for translations.
    Its z=0 section consists of horizontal shifts placing this object triangle
    in the head. A ray along -a through its interior is a collision somewhere
    along the complete straight path. Guarded endpoints remain unresolved.
    """
    points = (triangle[:, None, :]-head.vertices[None, :, :]).reshape(-1, 3)/scale
    eps = 1e-11
    if points[:, 2].min() >= -eps or points[:, 2].max() <= eps:
        return []
    hull = ConvexHull(points)
    edges = np.unique(np.sort(np.concatenate([hull.simplices[:, [0, 1]],
        hull.simplices[:, [1, 2]], hull.simplices[:, [2, 0]]]), axis=1), axis=0)
    cut = list(COORD.floor(points[np.abs(points[:, 2]) < eps]))
    for edge in edges:
        p, q = points[edge]
        if p[2]*q[2] < 0:
            cut.append(COORD.floor(p+(q-p)*(-p[2]/(q[2]-p[2]))))
    if len(cut) < 3:
        return []
    cut = np.unique(np.asarray(cut), axis=0)
    shape = ConvexHull(cut)
    if (shape.equations[:, 2] < -eps).all():
        raise ValueError('A head intersects the object at its final pose')
    vertices = cut[shape.vertices]
    vertices = vertices[np.linalg.norm(vertices, axis=1) > eps]
    angles = np.sort(np.rad2deg(np.arctan2(-vertices[:, 1], -vertices[:, 0])) % 360)
    gaps = np.diff(np.r_[angles, angles[0]+360])
    gap = int(np.argmax(gaps))
    start = float(angles[(gap+1)%len(angles)])
    width = 360-float(gaps[gap])
    if width > 180+1e-6:
        # Near-origin rounding or a degenerate section is not an all-angle proof.
        return []
    if width <= 2*guard_deg:
        return []
    return wrap_interval(start+guard_deg, start+width-guard_deg)


class AngleStudy:
    def __init__(self, mesh, heads):
        if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
            raise ValueError('Expected a closed, consistently oriented positive-volume object')
        if not heads or any(not h.is_watertight or h.metadata.get('convex_volume_m3',h.volume) <= 0 for h in heads):
            raise ValueError('Expected positive-volume convex head cells')
        self.mesh, self.heads = mesh, heads
        self.scale = float(mesh.extents.max())
        points = np.vstack([mesh.vertices, *[p.vertices for p in heads]])
        self.length = float(1.1*np.linalg.norm(np.ptp(points, axis=0)))
        self.tolerance = self.scale*1e-9
        self.clearance = G.Clearance(mesh, self.tolerance)
        self.queries = 0
        self.pair_cache = {}

    def envelope_hit(self, low, high):
        """The convex sector encloses EVERY sweep in this angle interval."""
        self.queries += 1
        middle = (low+high)/2
        half = np.deg2rad((high-low)/2)
        if high-low >= 180:
            raise ValueError('Split wide angular intervals before enclosing their sweep')
        shifts = np.array([np.zeros(3), -self.length*direction(low),
            -self.length*direction(high), -self.length/np.cos(half)*direction(middle)])
        for index, head in enumerate(self.heads):
            vertices = (head.vertices[None]+shifts[:, None]).reshape(-1, 3)
            face = self.clearance.obstruction(vertices)
            if face >= 0:
                return index, int(face)
        return None

    def test_angle(self, angle):
        pair = self.envelope_hit(angle, angle)
        return dict(angle_deg=float(angle%360), clear=pair is None,
                    obstruction=None if pair is None else dict(head_cell=pair[0], object_face=pair[1]))

    def classify(self, normals, minimum_width_deg=.01):
        local = local_angles(normals)
        pending = []
        for low, high in local['intervals_deg']:
            edges = np.linspace(low, high, max(1, int(np.ceil((high-low)/30)))+1)
            pending.extend(zip(edges[:-1], edges[1:]))
        clear, blocked, unresolved, witnesses = [], [], [], []
        while pending:
            low, high = map(float, pending.pop())
            remaining = subtract([[low, high]], blocked)
            if remaining != [[low, high]]:
                pending.extend(remaining)
                continue
            pair = self.envelope_hit(low, high)
            if pair is None:
                clear.append([low, high])
                continue
            if pair not in self.pair_cache:
                self.pair_cache[pair] = obstacle_shadow(self.mesh.triangles[pair[1]],
                                                      self.heads[pair[0]], self.scale)
            shadow = intersection([[low, high]], self.pair_cache[pair])
            if shadow:
                entire_shadow = intersection(local['intervals_deg'], self.pair_cache[pair])
                blocked = merge(blocked+entire_shadow)
                witnesses.append(dict(head_cell=pair[0], object_face=pair[1], blocked_intervals_deg=entire_shadow))
                pending.extend(subtract([[low, high]], shadow))
            elif high-low <= minimum_width_deg:
                unresolved.append([low, high])
            else:
                middle = (low+high)/2
                pending.extend([[low, middle], [middle, high]])
        isolated = [self.test_angle(a) for a in local['isolated_angles_deg']]
        if intersection(clear, blocked):
            raise ValueError('An angular interval has conflicting clear and blocked certificates')
        return dict(local=local, clear_intervals_deg=merge(clear),
            geometry_blocked_intervals_deg=merge(blocked), unresolved_intervals_deg=subtract(unresolved, blocked),
            isolated_direction_checks=isolated, obstruction_witnesses=witnesses,
            length_m=self.length, geometric_tolerance_m=self.tolerance,
            minimum_unresolved_cell_width_deg=minimum_width_deg,
            interval_envelope_queries=self.queries,
            whole_intervals_checked=True, floor_checked=False, rotation_allowed=False,
            boundary_convention='Closed clear intervals within geometry tolerance; blocked intervals have guarded ends. Unresolved bands are not classified as clear.',
            method='Adaptive full-sweep sector envelopes; forbidden angles from z=0 slices of triangle-minus-convex-head configuration obstacles')
