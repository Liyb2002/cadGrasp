"""Step5's peel, as Co-optimize's Step5.2 (prune_support.py): the support's material deleted a piece at a time,
each deletion kept when every task's paired forces and moments still hold; else that piece stays.

Deletions, in order: the outer material, by plane cuts on the support's oriented bbx toward the bunnies
(TRIM: fractions of the way); then boxes of a GRIDS subdivision of the support's bounds, the boxes holding the
most skin first.  At most BUDGET are tried.  A deletion is a real Boolean (the support minus the plane's far
side or the box); what it parts from the support falls off (the largest piece stays).  With it goes the skin
it leaves thinner than WIDTH (the cells' prisms, step4/skin.py's cut), again while that leaves more.  It is
kept when every task's paired forces and moments hold (below).  At the end the skin is probed in the
support itself again, and the thin skin left is cut (with the same checks); the thin skin whose cut they
refuse (it holds a load) is made THICK thick in the legal space instead: skin at least WIDTH wide or at least
THICK thick, where the legal space lets it be either (what it keeps both narrow and thin stays, recorded).
The skin is judged at its mid-layer, as Step4 judges it.  The feet grow after the peel.

It is not kept either when it takes the support off a table that feet alone cannot hold it on: the corners of
its ground faces in an installation where no ring of legal feet encloses the pressure hull (step5/combined.py).

What a deletion takes is known exactly from points, without another Boolean: the material at a point goes when
the point lies on the plane's far side, in the box, in a cut cell's prism and layer, or in a piece that falls
off.  So are tracked
  - the skin: step4/skin.py's mid-layer probe of every cell of every seat (its coverage, whose width counts);
  - every task's reactions: Step3's contact polygons of the support (actual, at the start) subdivided to
    cells, their corners each a frictionless normal reaction, a row of the paired-load LP (Step3's scale and
    floor row).  A corner that loses its material loses its row; the contact that stays holds every row left
    (and more), so a certificate on the rows left holds on the support.

The certificate of a task: for every continuous outer target (holding them certifies the whole continuous
domain), an LP basis of rows that holds it; as in Step3's classifier, a basis found for one target holds
every other target it reconstructs with nonnegative coefficients.  A target first tries the rows the
certificate already uses; else the cheapest basis over all rows left, a row costing exp(-d/WIDTH) at distance d
from the rim of its skin, so the bases lie deep in the skin.  A deletion that takes no row of a basis leaves
the certificate as it is; else the targets of the bases it takes are solved again on the rows left.  The
corners a kept certificate uses are probed in the support itself (Booleans can drop slivers the points do not
see): one without material there is lost and the certificate solved again.  The full audit at the end checks
the support itself.
"""
import time
import numpy as np
import trimesh
import manifold3d as md
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, identity
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree, ConvexHull

from whole_task_2.operations.geometry import moved_mesh, contact_surface
from whole_task_2.step3 import independent_geometry as G
from whole_task_2.step3.independent_physics import contacts
from whole_task_2.step3.membranes import positive_components
from whole_task_2.step3.legacy import kernel as K
from whole_task_2.step4.skin import WIDTH, CELL, PROBE, skin, seats_of, owners, transform, cut
from whole_task_2.step4.merge import move
from whole_task_2.step4.combine import largest

TRIM = (1., .5, .25, .125)          # plane cuts this fraction of the way from a box face to the bunnies
MID = (.5,)                         # the probe the peel tracks per cell: step4/skin.py's mid-layer coverage
THICK = WIDTH                       # skin too narrow where it cannot widen is made this thick instead
GRIDS = (3, 5)                      # box deletions: the support's bounds subdivided this many times per axis
BUDGET = 64                         # deletions tried at most
OPTIONS = dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9, presolve=False)

_discs, _prisms = {}, {}


def discs(c):
    """Cells x cells: within WIDTH/2 of each other (step4/skin.py's opening radius), facing the same way, each
    cell in its own disc."""
    if c.object_name not in _discs:
        S, _ = skin(c); n = len(S.area); radius = WIDTH/2
        D = S.tree.sparse_distance_matrix(S.tree, radius, output_type='coo_matrix')
        keep = np.einsum('ij,ij->i', S.normals[D.row], S.normals[D.col]) > 0
        _discs[c.object_name] = (coo_matrix((np.ones(keep.sum(), np.float32), (D.row[keep], D.col[keep])), shape=(n, n)).tocsr()
                                 + identity(n, np.float32, format='csr'))
    return _discs[c.object_name]


def prisms(c):
    """Every cell's cutting prism (step4/skin.py's, object frame) as its facets' equations (cells, facets, 4):
    a point p is in it where every [n, d].[p, 1] <= 0 (padding rows hold everywhere)."""
    if c.object_name not in _prisms:
        S, _ = skin(c); hulls = [ConvexHull(P).equations for P in S.prisms]; F = max(len(h) for h in hulls)
        E = np.zeros((len(hulls), F, 4)); E[:, :, 3] = -1.
        for k, h in enumerate(hulls): E[k, :len(h)] = h
        _prisms[c.object_name] = E
    return _prisms[c.object_name]


def probed(value, points, probe=PROBE):
    """Which of points (support frame) lie in value: a cube `probe` wide at each, intersected with it (as
    step4/skin.py's coverage)."""
    P = np.asarray(points, float); n = len(P); out = np.zeros(n, bool)
    if not n: return out
    corners = (np.array(list(np.ndindex(2, 2, 2)), float)-.5)*probe
    box = trimesh.creation.box((1., 1., 1.)); order = np.array([(v > 0).astype(int)@(4, 2, 1) for v in box.vertices])
    V = (P[:, None]+corners[None])[:, order].reshape(-1, 3); F = (box.faces[None]+8*np.arange(n)[:, None, None]).reshape(-1, 3)
    cubes = md.Manifold(md.Mesh64(np.ascontiguousarray(V/G.SCALE), np.ascontiguousarray(F, dtype=np.uint64)))
    hit = (value ^ cubes).to_mesh64().vert_properties[:, :3]*G.SCALE
    if len(hit): out[np.unique(cKDTree(P).query(hit)[1])] = True
    return out


def basis(A, target, cost):
    """Row ids of A whose nonnegative combination is target, the cheapest by cost (Step3's whitened LP); None
    if none is."""
    if not len(A): return None
    _, s, vt = np.linalg.svd(A, full_matrices=False)
    rank = int((s > s[0]*1e-10).sum()) if len(s) and s[0] > 0 else 0
    if not rank: return None
    B = vt[:rank].T
    if np.abs(target-target@B@B.T).max() > 2e-9: return None
    T = B/s[:rank]; b = target@T; scale = max(float(np.abs(b).max()), 1.)
    r = linprog(cost, A_eq=(A@T).T, b_eq=b/scale, bounds=(0, None), method='highs', options=OPTIONS)
    if r.status != 0: return None
    x = np.maximum(r.x*scale, 0.)
    if np.abs(A.T@x-target).max() > 2e-8: return None
    return np.flatnonzero(x > 1e-12*max(x.max(), 1e-300))


def held(B, targets, tolerance=K.J.C.FEASIBILITY_TOL):
    """Which targets the rows B hold with nonnegative coefficients (Step3's batch acceptance)."""
    coefficients = targets@np.linalg.pinv(B)
    return (coefficients >= 0).all(1) & (np.abs(coefficients@B-targets).max(1) <= tolerance)


class Hold:
    """A task's paired-load certificate on its contact corners: per outer target, the LP basis that holds it."""
    def __init__(self, t, support, depth):
        body = t['body_world']; domain = t['domain']; self.id = t['id']
        near = contact_surface(body, moved_mesh(support, np.linalg.inv(t['material'])))
        tri, src, _, scale = contacts(body, near, domain.work_ids, domain.com)
        v, f, index = trimesh.remesh.subdivide_to_size(tri.reshape(-1, 3), np.arange(3*len(tri)).reshape(-1, 3), CELL, return_index=True)
        q = v[f].reshape(-1, 3); n = np.repeat(-body.face_normals[src[index]], 3, axis=0)
        rows = K.U.heads(np.c_[n, np.cross(q-domain.com, n)], scale)
        rows, first = np.unique(np.round(rows, 13), axis=0, return_index=True)               # corners shared by triangles
        self.points = transform(q[first], t['material']); self.tree = cKDTree(self.points)  # support frame
        self.A = np.vstack([K.U.floor(np.empty((0, 6)), scale), rows]); self.alive = np.ones(len(self.A), bool)
        self.cost = np.r_[0., np.maximum(np.exp(-depth(self.points)/WIDTH), 1e-3)]
        self.targets = K.J.C.U.target(t['endpoints']['endpoint_need_wrench']*scale, self.A.shape[1])
        self.bases, self.of = [], np.full(len(self.targets), -1)
        found = self.solve(np.arange(len(self.targets)), self.alive, [])
        if found is None: raise ValueError(self.id+': its contact corners do not hold every outer target')
        self.bases, self.of = found

    def solve(self, which, alive, bases):
        """bases extended so that every target of which is held on the rows alive: (bases, of), or None."""
        bases = list(bases); of = self.of.copy(); pending = np.zeros(len(self.targets), bool); pending[which] = True
        while pending.any():
            j = np.flatnonzero(pending)[0]; used = np.zeros(len(self.A), bool)
            for b in bases:
                if b is not None: used[b] = True
            mine = np.flatnonzero(used & alive); ids = basis(self.A[mine], self.targets[j], np.zeros(len(mine)))
            if ids is not None: ids = mine[ids]
            else:
                rows = np.flatnonzero(alive); ids = basis(self.A[rows], self.targets[j], self.cost[rows])
                if ids is None: return None
                ids = rows[ids]
            bases.append(ids); p = np.flatnonzero(pending); p = p[held(self.A[ids], self.targets[p])]
            of[p] = len(bases)-1; pending[p] = False; pending[j] = False; of[j] = len(bases)-1
        return bases, of

    def after(self, alive):
        """The certificate on the rows alive, (bases, of), or None if a target is no longer held."""
        hit = [k for k, b in enumerate(self.bases) if b is not None and not alive[b].all()]
        if not hit: return self.bases, self.of
        return self.solve(np.flatnonzero(np.isin(self.of, hit)), alive, [None if k in hit else b for k, b in enumerate(self.bases)])


def layer_points(c, X, levels=MID):
    """Every cell's probes at the shares levels of its layer's depth, seat X (support frame), level by level."""
    S, _ = skin(c); half = np.linalg.norm(S.mid-S.centers, axis=1)[:, None]
    return transform(np.vstack([S.centers+2*f*half*S.normals for f in levels]), X)


def material(c, F):
    """The cells with material at one of their probes (F: per probe, as layer_points orders them)."""
    S, _ = skin(c); return F.reshape(-1, len(S.area)).any(0)


def fill(c, group, support, legal):
    """support with every hole of its skin filled, in the legal space: in every seat, the cells without material
    that the skin encloses (every piece of them on the object's surface but the largest).  (support, area)"""
    S, outer = skin(c); value = G.solid(support); area = 0.
    for X, _ in seats_of(group['slots']):
        free = np.flatnonzero(~material(c, probed(value, layer_points(c, X))))
        if not len(free): continue
        n, label = connected_components(S.graph[free][:, free], directed=False)
        if n < 2: continue
        ids = free[label != np.bincount(label, weights=S.area[free]).argmax()]
        value = value+(S.cells(ids, X) ^ move(X, outer) ^ legal); area += float(S.area[ids].sum())
    return (G.mesh(largest(value)[0]) if area else support), area


def thick(c, X, ids):
    """The cells ids of the seat X extruded from just inside the object to THICK out along their vertex
    normals (as step4/skin.py's cutting prisms, but THICK deep), joined; support frame."""
    S, _ = skin(c); t = c.base.policy['mold_outward_displacement_m']; P = S.prisms[ids]
    N = (P[:, 3:6]-P[:, :3])/(1.8*t); T = P[:, :3]+.2*t*N
    return G.union([md.Manifold.hull_points(transform(np.vstack([T[k]-.2*t*N[k], T[k]+1.05*THICK*N[k]]), X)/G.SCALE) for k in range(len(ids))])


def thicken(c, group, support, legal):
    """support with its thin skin (thin_cells) made THICK thick, in the legal space:
    (support, the area made thick)."""
    S, _ = skin(c); value = G.solid(support); area = 0.
    for X, ids, _ in thin_cells(c, group, support):
        if len(ids): value = value+(thick(c, X, ids) ^ legal); area += float(S.area[ids].sum())
    return (G.mesh(largest(value)[0]) if area else support), area


def too_thin(c, group, support):
    """The area (m^2) of thin skin (thin_cells) less than THICK thick (no material at
    0.95 THICK out along its cell's normal)."""
    S, _ = skin(c); value = G.solid(support); area = 0.
    for X, ids, _ in thin_cells(c, group, support):
        if len(ids): area += float(S.area[ids[~probed(value, transform(S.centers[ids]+.95*THICK*S.normals[ids], X))]].sum())
    return area


def thin_cells(c, group, support):
    """[(X, the cells of the seat X it owns that are thin, its cells with material)]: step4/skin.py's rule on the
    mid-layer skin, on support itself.  (Probes off the mid-layer read a neighbour's material at a cut's rim as
    skin: thickening those cells grew fins one cell wide.)"""
    S, _ = skin(c); value = G.solid(support); out = []
    for (X, _), own in zip(seats_of(group['slots']), owners(c, group['slots'])):
        M = material(c, probed(value, layer_points(c, X))); out.append((X, np.flatnonzero(S.thin(M, own=own)), M))
    return out


class Peel:
    """The support and what its points know: its skin (every seat's cells' mid-layer probes) and every task's
    contact corners."""
    def __init__(self, c, model, group, support, ground=np.zeros((0, 3))):
        S, _ = skin(c); self.c, self.S, self.group = c, S, group; self.seats = seats_of(group['slots'])
        self.own = owners(c, group['slots']); self.body = G.solid(support); self.disc = discs(c); self.E = prisms(c)
        self.half = np.linalg.norm(S.mid-S.centers, axis=1); self.ground = ground
        self.pts = [layer_points(c, X) for X, _ in self.seats]                              # support frame
        self.F = [probed(self.body, p) for p in self.pts]
        self.trees = [cKDTree(p) for p in self.pts]; rims = [S.rim(material(c, f)) for f in self.F]; mids = [cKDTree(transform(S.mid, X)) for X, _ in self.seats]

        def depth(points):
            """Distance of points (support frame) to the rim of the skin of the seat nearest them."""
            near = [t.query(points) for t in mids]; a = np.argmin([d for d, _ in near], 0)
            return np.array([rims[s][near[s][1][i]] for i, s in enumerate(a)])
        self.holds = [Hold(t, support, depth) for t in model.prepare(model.layout(support))]
        self.record = dict(tried=[], tasks=[dict(task=h.id, corners=len(h.A)-1, targets=len(h.targets),
                                                 bases=len(h.bases)) for h in self.holds])

    def thin(self, a, F):
        """Seat a's own cells with material (F: its probes) that step4/skin.py's opening loses (its core: at
        least WIDTH/2 less half a cell from the rim; opened: within WIDTH/2 of it), convex corners too: every cell
        step4/skin.py would call thin were all material its skin, and the corners it excuses."""
        M = material(self.c, F); core = M & (self.S.rim(M) >= WIDTH/2-.5*self.S.spacing)
        return np.flatnonzero(M & self.own[a] & ~(self.disc@core.astype(np.float32) > 0))

    def in_prisms(self, a, ids, points, tree):
        """Which of points (support frame; tree: theirs) lie in the prisms of seat a's cells ids, in its layer."""
        out = np.zeros(len(points), bool)
        if not len(ids) or not len(points): return out
        X = self.seats[a][0]; reach = 2*self.S.spacing+2*self.half[ids].max()
        near = np.unique(np.concatenate([np.asarray(f, int) for f in tree.query_ball_point(transform(self.S.centers[ids], X), reach)]))
        if not len(near): return out
        p = transform(points[near], np.linalg.inv(X)); _, k8 = cKDTree(self.S.centers[ids]).query(p, min(8, len(ids)))
        k8 = np.asarray(k8).reshape(len(p), -1); P1 = np.c_[p, np.ones(len(p))]; hit = np.zeros(len(p), bool)
        for j in range(k8.shape[1]):
            k = ids[k8[:, j]]; h = np.einsum('ij,ij->i', p-self.S.centers[k], self.S.normals[k])
            hit |= (np.einsum('ifj,ij->if', self.E[k], P1) <= 1e-12).all(1) & (h >= -1e-9) & (h <= 2*self.half[k])
        out[near[hit]] = True
        return out

    def attempt(self, value, gone, label):
        """Delete what makes self.body value (gone(points): the points whose material goes), with the thin skin
        that leaves and what falls off, if every certificate holds.  True if kept."""
        F = [f & ~gone(p) for f, p in zip(self.F, self.pts)]; alive = [h.alive & ~np.r_[False, gone(h.points)] for h in self.holds]
        row = dict(deletion=label); began = time.monotonic()
        for _ in range(4):                                       # the thin skin it leaves, again while that leaves more
            thin = [self.thin(a, f) for a, f in enumerate(F)]
            if not any(len(t) for t in thin): break
            for a, ids in enumerate(thin):
                if not len(ids): continue
                value = cut(self.c, value, self.seats[a][0], ids)
                F = [f & ~self.in_prisms(a, ids, p, t) for f, p, t in zip(F, self.pts, self.trees)]
                alive = [al & ~np.r_[False, self.in_prisms(a, ids, h.points, h.tree)] for al, h in zip(alive, self.holds)]
        clock = time.monotonic(); value = G.solid(G.mesh(value))        # as it is saved: a seam may part pieces
        parts = sorted(positive_components(value), key=lambda p: -float(p.volume()))
        if not parts: row['reason'] = 'nothing left'
        elif len(parts) > 1:                                     # what the deletion parts falls off
            value = parts[0]
            for part in parts[1:]:
                low, high = G.mesh(part).bounds; low, high = low-PROBE, high+PROBE
                for f, p in zip(F, self.pts):
                    k = np.flatnonzero(f & ((p >= low) & (p <= high)).all(1)); f[k[probed(part, p[k])]] = False
                for al, h in zip(alive, self.holds):
                    k = np.flatnonzero(al[1:] & ((h.points >= low) & (h.points <= high)).all(1)); al[1+k[probed(part, h.points[k])]] = False
        row['pieces_seconds'] = time.monotonic()-clock
        if 'reason' not in row and len(self.ground) and not probed(value, self.ground).all():
            row['reason'] = 'it takes the support off a table that feet alone cannot hold it on'
        if 'reason' not in row:
            found = []
            for h, al in zip(self.holds, alive):
                for _ in range(4):                               # the corners a certificate uses, probed in the support itself
                    f = h.after(al)
                    if f is None: break
                    used = np.unique(np.concatenate([b for b in f[0] if b is not None])); used = used[used > 0]
                    dead = used[~probed(value, h.points[used-1])]
                    if not len(dead): break
                    al[dead] = False; f = None
                if f is None: row['reason'] = h.id+': a paired force/moment target is no longer held'; break
                found.append(f)
        row.update(kept='reason' not in row, seconds=time.monotonic()-began); self.record['tried'].append(row)
        if not row['kept']: return False
        for h, al, (bases, of) in zip(self.holds, alive, found): h.alive, h.bases, h.of = al, bases, of
        self.body, self.F = value, F; return True

    def run(self, bunnies, legal):
        """The deletions in order (bunnies: points of every seat's object, support frame, for the plane cuts;
        legal: where material may be added): the support and the record."""
        began = time.monotonic(); start = G.volume(self.body)
        self.attempt(self.body, lambda p: np.zeros(len(p), bool), 'the thin skin the support starts with')
        T, _ = trimesh.bounds.oriented_bounds(G.mesh(self.body).vertices)
        target = transform(bunnies, T); target = np.array([target.min(0), target.max(0)])
        for fraction in TRIM:
            m = transform(G.mesh(self.body).vertices, T); b = np.array([m.min(0), m.max(0)])
            for axis, side in sorted(((ax, sd) for ax in range(3) for sd in range(2)),
                                     key=lambda r: -(b[r[1], r[0]]-target[r[1], r[0]])*(1 if r[1] else -1)):
                if len(self.record['tried']) >= BUDGET: break
                edge = transform(G.mesh(self.body).vertices, T)[:, axis]; edge = edge.max() if side else edge.min()
                if (edge-target[side, axis])*(1 if side else -1) <= 1e-8: continue
                limit = edge+fraction*(target[side, axis]-edge)
                normal, offset = (-T[axis, :3], T[axis, 3]-limit) if side else (T[axis, :3], limit-T[axis, 3])
                self.attempt(self.body.trim_by_plane(tuple(normal), float(offset/G.SCALE)),
                             lambda p, n=normal, o=offset: p@n < o, f'plane axis {axis} {"high" if side else "low"} {fraction}')
        for grid in GRIDS:
            low, high = G.mesh(self.body).bounds; size = (high-low)/grid; boxes = []
            skin_points = np.vstack([p[f] for p, f in zip(self.pts, self.F)])
            for index in np.ndindex(grid, grid, grid):
                a = low+np.array(index)*size; b = a+size; inside = int(((skin_points >= a) & (skin_points <= b)).all(1).sum())
                if inside: boxes.append((-inside, index, a, b))
            for _, index, a, b in sorted(boxes, key=lambda r: (r[0], r[1])):
                if len(self.record['tried']) >= BUDGET: break
                self.attempt(self.body-G.box(a, b), lambda p, a=a, b=b: ((p >= a) & (p <= b)).all(1), f'box {grid} {index}')
        for k in range(3):                                       # the skin probed in the support again
            self.F = [probed(self.body, p) for p in self.pts]
            if not any(len(self.thin(a, f)) for a, f in enumerate(self.F)): break
            self.attempt(self.body, lambda p: np.zeros(len(p), bool), f'the thin skin left, probed again ({k+1})')
        # The thin skin left (its cut refused: it holds a load) is made THICK thick instead, in the legal space
        # (adding never breaks a certificate).
        value, self.record['thickened_m2'] = thicken(self.c, self.group, G.mesh(self.body), legal)
        self.body = G.solid(value)
        self.record.update(removed_cm3=(start-G.volume(self.body))*1e6, seconds=time.monotonic()-began,
                           kept=sum(r['kept'] for r in self.record['tried']))
        return G.mesh(self.body), self.record
