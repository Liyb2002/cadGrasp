"""The support's skin: its width along the object's surface, at least WIDTH everywhere.

A support is the outer layer of the object (Step3's fitted outer, 5 mm, minus the object) in each of its
seats, cut by every task's body, keep-out and exit sweep.  Over a seat its skin is a region of that object's
surface: the surface cells (the object subdivided to edges of at most CELL) whose mid-layer point lies in the
material.  Width is along the surface, not through the layer (that is the support's thickness).

The skin is WIDTH wide where a disc of radius r = WIDTH/2 inside it covers it (erosion, then dilation: the
opening).  The opening also rounds every convex corner, which is not a narrow part; so a piece the opening
loses is thin only if it is
  - a strip: it reaches further than a right-angled corner would, (sqrt 2 - 1) r, from the opened skin;
  - a neck: it meets the opened skin in two or more separate places;
  - or an island: it meets no opened skin at all.
Distances are straight lines between nearby cells that face the same way (dot of normals > 0), which along a
smooth surface is the distance in it; the rim of the skin is the midpoints of the edges between a covered and an
uncovered cell.

Seats overlap in space (a juxtaposed seat's object lies across the first one's), so material in one seat's
layer may be another seat's skin crossing it.  Each cell is judged by the seat whose object surface its
mid-layer point is nearest: a seat's thin cells are only its own; another seat's material over its surface
counts as covered (it is no rim), and that seat judges it.  A cell whose mid-layer point leaves the layer
(where the fitted outer folds over in a crevice) is judged by none.

pruned():  the support's largest piece without its skin narrower than WIDTH, cut once; nothing is added.  The
           Step4 oracle then judges what is left, and Step5 makes every skin narrower than its own least width
           (step5/peel.THICK) that thick.
Material is cut cell by cell: each cell's prism (the cell, its vertices moved along their normals from just
inside the object to past the layer) clipped to the seat's outer layer.
"""
from functools import lru_cache
import numpy as np
import trimesh
import manifold3d as md
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from subdivide_task.step3 import independent_geometry as G
from subdivide_task.step3.membranes import positive_components
from .merge import move

WIDTH = .020                    # Step4: skin narrower than this along the surface holds nothing (m)
CELL = .010/3                   # surface cells: the object subdivided to edges of at most this (m)
SEAM = 1e-9                     # edges shorter than this (m), left where many prisms meet, are collapsed
PROBE = 1e-5                    # the coverage cube must lie inside its cell's cutting prism
TETRA = np.array([[1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]])   # alternate corners of a cube
TETRA_FACES = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])               # outward


def transform(points, X):
    return np.asarray(points)@np.asarray(X)[:3, :3].T+np.asarray(X)[:3, 3]


class Skin:
    """The object's surface cells, their graph, mid-layer points and prisms (object frame)."""
    def __init__(self, source, outer, thickness):
        v, f = trimesh.remesh.subdivide_to_size(source.vertices, source.faces, CELL)
        m = trimesh.Trimesh(v, f, process=True)
        self.centers = m.triangles_center; self.normals = m.face_normals; self.area = m.area_faces
        self.tree = cKDTree(self.centers)
        # The layer's thickness at each cell: the distance to the fitted outer's surface (its points and triangle
        # centres, at most the thickness); the mid-layer point halfway along the cell's normal.
        depth = np.minimum(cKDTree(np.vstack([outer.vertices, outer.triangles_center])).query(self.centers)[0], thickness)
        self.mid = self.centers+.5*depth[:, None]*self.normals; self.tree_mid = cKDTree(self.mid)
        # Where the offsets fold over (a crevice) the mid-layer point leaves the layer: such a cell is judged by no seat.
        I = np.eye(4); self.reliable = self.covered(G.solid(outer), I) & ~self.covered(G.solid(source), I)
        T = m.vertices[m.faces]; N = m.vertex_normals[m.faces]; C = T.mean(1, keepdims=True)
        T = C+(T-C)*1.02                                                # a little wider: neighbours overlap, no slivers
        # At sharp bends the smoothed vertex normals can lean away from the
        # face-normal midpoint. Include the coverage probe so cutting a cell
        # always removes the material that made that cell count as covered.
        probe_corners = (np.array(list(np.ndindex(2, 2, 2)), float)-.5)*PROBE
        self.prisms = np.concatenate([T-.2*thickness*N, T+1.6*thickness*N,
                                      self.mid[:, None]+probe_corners[None]], axis=1)
        # Cells sharing an edge (the rim lies on these edges) and cells sharing a vertex (connectivity).
        self.pairs = m.face_adjacency; self.edge_mid = m.vertices[m.face_adjacency_edges].mean(1)
        k = np.repeat(np.arange(len(m.faces)), 3); incidence = coo_matrix((np.ones(len(k)), (k, m.faces.ravel()))).tocsr()
        self.graph = (incidence@incidence.T).tocsr(); self.graph.setdiag(0); self.graph.eliminate_zeros()
        self.spacing = float(np.median(np.linalg.norm(np.diff(self.centers[self.pairs], axis=1)[:, 0], axis=1)))

    def covered(self, support, X, probe=PROBE):
        """The cells of the seat X (support <- object) under material: a regular tetrahedron on alternate corners of
        a cube `probe` wide at each mid-layer point (inside the cell's cutting prism), intersected with the support
        (exact and repeatable, unlike a ray parity on these meshes; a third of a cube's triangles)."""
        value = support if isinstance(support, md.Manifold) else G.solid(support)
        P = transform(self.mid, X); n = len(P); corners = TETRA*probe/2; out = np.zeros(n, bool)
        box = np.asarray(value.bounding_box(), float).reshape(2, 3)*G.SCALE                  # only probes there can meet it
        P = P[((P >= box[0]-probe) & (P <= box[1]+probe)).all(1)]
        if not len(P): return out
        V = (P[:, None]+corners[None]).reshape(-1, 3); F = (TETRA_FACES[None]+4*np.arange(len(P))[:, None, None]).reshape(-1, 3)
        probes = md.Manifold(md.Mesh64(np.ascontiguousarray(V/G.SCALE), np.ascontiguousarray(F, dtype=np.uint64)))
        hit = (value ^ probes).to_mesh64().vert_properties[:, :3]*G.SCALE
        if len(hit): out[np.unique(self.tree_mid.query(transform(hit, np.linalg.inv(X)))[1])] = True
        return out

    def near(self, ids, points, normals, radius):
        """For every cell ids: is one of points (with normals) within radius, facing the same way?"""
        out = np.zeros(len(ids), bool)
        if not len(points) or not len(ids): return out
        tree = cKDTree(points); k = min(8, len(points))
        # The nearest few within radius first, all at once: one of them almost always faces the same way.  The ball
        # decides only where all of them are within radius and none does (the same answer, 2-3x faster).
        d, j = tree.query(self.centers[ids], k, distance_upper_bound=np.nextafter(radius, np.inf))
        d, j = d.reshape(len(ids), k), j.reshape(len(ids), k); found = np.isfinite(d)
        facing = np.zeros(d.shape, bool); facing[found] = np.einsum('ij,ij->i', normals[j[found]], np.repeat(self.normals[ids][:, None], k, 1)[found]) > 0
        out = facing.any(1); rest = np.flatnonzero(~out & found.all(1))
        for n, ball in zip(rest, tree.query_ball_point(self.centers[ids[rest]], radius)):
            out[n] = bool((normals[ball]@self.normals[ids[n]] > 0).any())
        return out

    def rim(self, R):
        """Distance from every cell of R to the rim of R (inf: R has no rim)."""
        a, b = self.pairs.T; edge = R[a] != R[b]; d = np.full(len(R), np.inf)
        if not edge.any(): return d
        points = self.edge_mid[edge]; normals = self.normals[a[edge]]+self.normals[b[edge]]
        ids = np.flatnonzero(R); k = min(16, len(points)); dist, j = cKDTree(points).query(self.centers[ids], k)
        dist, j = dist.reshape(len(ids), -1), j.reshape(len(ids), -1)
        facing = np.einsum('ikj,ij->ik', normals[j], self.normals[ids]) > 0
        d[ids] = np.where(facing.any(1), np.where(facing, dist, np.inf).min(1), dist[:, 0])
        return d

    def thin(self, R, radius=WIDTH/2, own=None):
        """The cells of R (of them, the seat's own) in a strip, neck or island narrower than 2 radius."""
        core = R & (self.rim(R) >= radius-.5*self.spacing); ids = np.flatnonzero(R)          # a cell is spacing wide
        opened = np.zeros(len(R), bool); opened[ids] = self.near(ids, self.centers[core], self.normals[core], radius)
        lost = R & ~opened; out = np.zeros(len(R), bool)
        if not lost.any(): return out
        corner = (np.sqrt(2)-1)*radius+.5*self.spacing
        n, label = connected_components(self.graph[lost][:, lost], directed=False); lost_ids = np.flatnonzero(lost)
        reach = np.full(len(R), np.inf)
        if opened.any():
            reach[lost_ids] = cKDTree(self.centers[opened]).query(self.centers[lost_ids])[0]
        for piece in range(n):
            cells = lost_ids[label == piece]
            touch = np.unique(self.graph[cells].indices); touch = touch[opened[touch]]
            if len(touch): places = connected_components(self.graph[touch][:, touch], directed=False)[0]
            else: places = 0
            if places != 1 or reach[cells].max() > corner: out[cells] = True
        return out if own is None else out & own

    def width(self, R, own=None):
        """The least width of R (judged on the seat's own cells) along the surface (inf: nothing narrower than
        4 WIDTH), and the area thinner than WIDTH."""
        if not R.any(): return np.inf, 0.
        lo, hi = 0., 2*WIDTH
        if not self.thin(R, hi, own).any(): return np.inf, 0.
        for _ in range(8):
            r = .5*(lo+hi)
            if self.thin(R, r, own).any(): hi = r
            else: lo = r
        return 2*lo, float(self.area[self.thin(R, own=own)].sum())

    def around(self, cells, radius):
        """The cells within radius of cells (bool), facing the same way."""
        found = np.unique(np.concatenate([np.zeros(0, int)]+[np.asarray(f, int) for f in self.tree.query_ball_point(self.centers[cells], radius)]))
        out = np.zeros(len(self.area), bool); out[found[self.near(found, self.centers[cells], self.normals[cells], radius)]] = True
        return out

    def cells(self, ids, X):
        """The prisms of cells ids at the seat X, joined."""
        return G.union([md.Manifold.hull_points(transform(self.prisms[k], X)/G.SCALE) for k in ids])


@lru_cache(maxsize=4)
def _skin(object_name):
    from .seats import context
    c = context(object_name)
    return Skin(c.base.source, c.base.outer, c.base.policy['mold_outward_displacement_m']), G.solid(c.base.outer)


def skin(c):
    return _skin(c.object_name)


def seats_of(slots):
    """[(X, its slots)] in order of first use."""
    out = []
    for k, s in enumerate(slots):
        X = np.asarray(s['X'], float)
        for Y, ks in out:
            if np.allclose(X, Y, atol=1e-9): ks.append(k); break
        else: out.append((X, [k]))
    return out


_nearer = {}


def nearer(c, A, B):
    """For the seat A's cells: is the mid-layer point no nearer to the seat B's object surface than to its own?
    Kept per pair of seats: a join adds one seat, so only its pairs are new."""
    key = (c.object_name, A.tobytes(), B.tobytes())
    if key not in _nearer:
        if len(_nearer) > 512: _nearer.pop(next(iter(_nearer)))
        S, _ = skin(c); half = np.linalg.norm(S.mid-S.centers, axis=1)       # the mid-layer point's distance to its own surface
        # Only whether the nearest cell is within half matters: the search stops beyond the largest half (inf: none).
        _nearer[key] = S.tree.query(transform(transform(S.mid, A), np.linalg.inv(B)), distance_upper_bound=half.max()*(1+1e-9))[0] >= half
    return _nearer[key]


def owners(c, slots):
    """Per seat, the cells it judges: its mid-layer point no nearer to another seat's object surface."""
    S, _ = skin(c); X = [x for x, _ in seats_of(slots)]; out = []
    for a, A in enumerate(X):
        own = S.reliable.copy()
        for b, B in enumerate(X):
            if b != a: own &= nearer(c, A, B)
        out.append(own)
    return out


def regions(c, slots, support):
    """Every seat's skin: [(X, covered cells, its own cells)]."""
    S, _ = skin(c); value = support if isinstance(support, md.Manifold) else G.solid(support)
    return [(X, S.covered(value, X), own) for (X, _), own in zip(seats_of(slots), owners(c, slots))]


def thin_cells(c, slots, support):
    """Per slot, its seat's own skin cells narrower than WIDTH, the ones pruned() cuts (bool over the cells): one
    coverage pass, nothing cut."""
    S, _ = skin(c); masks = [None]*len(slots)
    for (X, ks), (_, R, own) in zip(seats_of(slots), regions(c, slots, support)):
        T = S.thin(R, own=own)
        for k in ks: masks[k] = T
    return masks


def measure(c, slots, support):
    """The least skin width over every seat (inf: nothing narrower than 4 WIDTH) and the area (m^2) thinner than WIDTH."""
    S, _ = skin(c); out = [S.width(R, own) for _, R, own in regions(c, slots, support)]
    return min(w for w, _ in out), sum(a for _, a in out)


def facts(width):
    """measure() for a report: the least width (None: nothing narrower than 4 WIDTH) and the thin area."""
    return dict(least_width_m=float(width[0]) if np.isfinite(width[0]) else None, thin_m2=float(width[1]))


def legal(points, pad, material, forbidden):
    """Where material may go: the box of points widened by pad, above every installation's floor (material:
    each installation <- support), out of `forbidden`."""
    from subdivide_task.operations.floors import clip, planes
    return clip(G.box(points.min(0)-pad, points.max(0)+pad), planes(material))-forbidden


def one_piece(value, reason, crumbs=.005):
    """value's largest piece, its crumbs (each below `crumbs` of the material: prism seams) dropped;
    ValueError(reason) if another piece is more."""
    parts = sorted(positive_components(value), key=lambda p: -float(p.volume()))
    if not parts: raise ValueError(reason+': no material')
    total = sum(float(p.volume()) for p in parts)
    if any(float(p.volume()) >= crumbs*total for p in parts[1:]): raise ValueError(reason)
    return parts[0]


def piece(value, reason):
    """one_piece(value) as a mesh that reads back as one piece too (the export may part crumbs that touched it
    at a point: they are dropped again); ValueError(reason) if a real piece parts."""
    for _ in range(3):
        m = G.mesh(one_piece(value, reason).simplify(0.)); value = G.solid(m)
        if len(positive_components(value)) == 1: return m
    raise ValueError(reason+' (as a mesh)')


def cut(c, value, X, ids):
    """value minus the layer over cells ids at the seat X."""
    S, outer = skin(c)
    return value-(S.cells(ids, X) ^ move(X, outer)) if len(ids) else value


def cut_thin(c, slots, support, rounds=8):
    """support with every seat's thin skin cut away, again while a cut leaves new thin skin (a strip's stub):
    (solid, the area cut, m^2)."""
    S, _ = skin(c); value = G.solid(support); area = 0.
    for _ in range(rounds):
        thin = [(X, np.flatnonzero(S.thin(R, own=own))) for X, R, own in regions(c, slots, support)]
        if not any(len(ids) for _, ids in thin): break
        for X, ids in thin: value = cut(c, value, X, ids); area += float(S.area[ids].sum())
        support = piece(value, 'cutting the thin skin splits the support'); value = G.solid(support)
    return value, area


def largest_mesh(value, tries=3):
    """value's largest piece as a mesh that reads back as one piece: the export may part a crumb that touched it
    at a point, and it goes too (B+8pose S1 at 30 mm: 1e-4 cm^3, which the audit counts as a second body)."""
    for _ in range(tries):
        parts = positive_components(value)
        if not parts: raise ValueError('pruning leaves no support')
        support = G.mesh(max(parts, key=lambda p: float(p.volume())).simplify(0.)); value = G.solid(support)
        if len(positive_components(value)) == 1: break
    return support


def pruned(c, slots, support, forbidden=None):
    """support's largest piece without its skin narrower than WIDTH, cut once -- such material holds nothing: its
    contacts do not count -- and, given `forbidden`, the seams the prisms leave collapsed (SEAM) and `forbidden` cut
    again, so the support settles under further Booleans; then its largest piece: (support, record).  Nothing is
    added and nothing is cut again: the oracle judges what is left (narrow skin a cut leaves, Step5 makes thick)."""
    S, _ = skin(c); start = G.volume(G.solid(support)); support = largest_mesh(G.solid(support))
    value = G.solid(support); area = 0.
    for X, R, own in regions(c, slots, support):
        ids = np.flatnonzero(S.thin(R, own=own)); value = cut(c, value, X, ids); area += float(S.area[ids].sum())
    support = largest_mesh(value)
    if forbidden is not None: support = largest_mesh(G.solid(support).simplify(SEAM/G.SCALE)-forbidden)
    return support, dict(thin_cut_m2=area, removed_cm3=float(start-G.volume(G.solid(support)))*1e6)
