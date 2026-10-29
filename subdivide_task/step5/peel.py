"""Step5's peel, as Co-optimize's Step5.2 (prune_support.py): the support's material deleted a piece at a time,
each deletion kept when every task's paired forces and moments still hold; else that piece stays.

Deletions, in order: the outer material, by plane cuts on the support's oriented bbx toward the bunnies
(TRIM: fractions of the way); then boxes of a GRIDS subdivision of the support's bounds, the boxes holding the
most skin first.  At most BUDGET are tried.  A deletion is a real Boolean (the support minus the plane's far
side or the box); what it parts from the support falls off (the largest piece stays).  No deletion takes the
skin narrower than THICK the support starts with (marked once at the start: its cells' prisms are cut out of
every deletion, so their material stays; step5/frame.py trims the narrow skin after); nor makes skin narrower
than WIDTH where it was not: where it would, the cells within WIDTH of that skin that had material keep it too
(their prisms cut out of the deletion as well), up to SPARE times, else the deletion is not made; nor one that
parts any of that skin from the support (it would fall off).  It is kept when every task's paired forces and
moments hold (below).  The skin is judged at its mid-layer, as Step4 judges it.  The feet grow after the frame.

It is not kept either when it takes the support off a table that feet alone cannot hold it on: the corners of
its ground faces in an installation where no ring of legal feet encloses the pressure hull (step5/combined.py).

What a deletion takes is known exactly from points, without another Boolean: the material at a point goes when
the point lies on the plane's far side, in the box, or in a piece that falls off.  So are tracked
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
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from subdivide_task.operations.geometry import moved_mesh, contact_surface
from subdivide_task.step3 import independent_geometry as G
from subdivide_task.step4.fast_contacts import contacts      # Step3's contact polygons, vectorized; the audit keeps the kernel's
from subdivide_task.step3.membranes import positive_components
from subdivide_task.step3.legacy import kernel as K
from subdivide_task.step4.skin import WIDTH, CELL, PROBE, SEAM, skin, seats_of, owners, transform

TRIM = (1., .5, .25, .125)          # plane cuts this fraction of the way from a box face to the bunnies
MID = (.5,)                         # the probe the peel tracks per cell: step4/skin.py's mid-layer coverage
THICK = .030                        # skin narrower than this the support starts with stays through the peel
GRIDS = (3, 5)                      # box deletions: the support's bounds subdivided this many times per axis
BUDGET = 64                         # deletions tried at most
SPARE = 3                           # times a deletion spares the material about the narrow skin it would leave
OPTIONS = dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9, presolve=False)

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


def within(value, points):
    """probed, the points beyond value's bounds left out (not in it)."""
    out = np.zeros(len(points), bool); low, high = G.mesh(value).bounds
    k = np.flatnonzero(((points >= low-PROBE) & (points <= high+PROBE)).all(1)); out[k] = probed(value, points[k])
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


def fresh(c, group, old, value, made=False, pieces=True):
    """The skin of value (a solid) narrower than WIDTH that the start (old: its thin_cells at WIDTH/2) has not -- a
    piece narrower than WIDTH holding no cell the start has narrower (not pieces: every cell narrower now that was
    not), and if made, a cell the start has wider now narrower with the cells within WIDTH/2 of it: [(X, its
    cells)] per seat, and their area."""
    S, _ = skin(c); out = []; area = 0.
    for (X, ids0, M0), (_, ids, _) in zip(old, thin_cells(c, group, G.mesh(value), WIDTH/2)):
        was = np.zeros(len(S.area), bool); was[ids0] = True
        n, label = connected_components(S.graph[ids][:, ids], directed=False)
        cells = ids[(np.bincount(label, weights=was[ids], minlength=n) == 0)[label]] if pieces else ids[~was[ids]]
        narrowed = ids[M0[ids] & ~was[ids]] if made else ids[:0]
        if len(narrowed): cells = np.union1d(cells, np.flatnonzero(S.around(narrowed, WIDTH/2)))
        out.append((X, cells)); area += float(S.area[cells].sum())
    return out, area


def prisms(c, found, grow=0.):
    """The prisms of the cells found ([(X, cells)]) joined, grown by `grow` every way (from a cube's corners)."""
    S, _ = skin(c); cube = grow*(np.array(list(np.ndindex(2, 2, 2)), float)*2-1)
    return G.union([md.Manifold.hull_points((transform(S.prisms[k], X)[:, None]+cube).reshape(-1, 3)/G.SCALE)
                    for X, cells in found for k in cells])


def too_thin(c, group, support):
    """The area (m^2) of skin narrower than THICK (thin_cells) less than THICK thick (no material at
    0.95 THICK out along its cell's normal): what the legal space kept narrow."""
    S, _ = skin(c); value = G.solid(support); area = 0.
    for X, ids, _ in thin_cells(c, group, support, THICK/2):
        if len(ids): area += float(S.area[ids[~probed(value, transform(S.centers[ids]+.95*THICK*S.normals[ids], X))]].sum())
    return area


def thin_cells(c, group, support, radius=WIDTH/2):
    """[(X, the cells of the seat X it owns narrower than 2 radius, its cells with material)]: step4/skin.py's rule
    on the mid-layer skin, on support itself.  (Probes off the mid-layer read a neighbour's material at a cut's rim as
    skin: thickening those cells grew fins one cell wide.)"""
    S, _ = skin(c); value = G.solid(support); out = []
    for (X, _), own in zip(seats_of(group['slots']), owners(c, group['slots'])):
        M = material(c, probed(value, layer_points(c, X))); out.append((X, np.flatnonzero(S.thin(M, radius, own)), M))
    return out


class Peel:
    """The support and what its points know: its skin (every seat's cells' mid-layer probes) and every task's
    contact corners."""
    def __init__(self, c, model, group, support, ground=np.zeros((0, 3))):
        S, _ = skin(c); self.c, self.S, self.group = c, S, group; self.seats = seats_of(group['slots'])
        self.own = owners(c, group['slots']); self.body = G.solid(support); self.ground = ground
        self.pts = [layer_points(c, X) for X, _ in self.seats]                              # support frame
        self.F = [probed(self.body, p) for p in self.pts]
        self.narrow = [np.zeros(len(S.area), bool) for _ in self.seats]                     # see run()
        rims = [S.rim(material(c, f)) for f in self.F]; mids = [cKDTree(transform(S.mid, X)) for X, _ in self.seats]

        def depth(points):
            """Distance of points (support frame) to the rim of the skin of the seat nearest them."""
            near = [t.query(points) for t in mids]; a = np.argmin([d for d, _ in near], 0)
            return np.array([rims[s][near[s][1][i]] for i, s in enumerate(a)])
        self.tasks = model.prepare(model.layout(support)); self.holds = [Hold(t, support, depth) for t in self.tasks]
        # The skin narrower than THICK it starts with, every seat's: its cells' prisms, cut out of every deletion (adding
        # its material back after one re-cuts it into the support, faces doubling a kept deletion), and the points in them.
        thin = [(X, np.flatnonzero(S.thin(material(c, f), THICK/2, o))) for (X, _), f, o in zip(self.seats, self.F, self.own)]
        cells = [S.cells(ids, X) for X, ids in thin if len(ids)]; self.kept = G.union(cells) if cells else md.Manifold()
        inside = (lambda p: probed(self.kept, p)) if cells else (lambda p: np.zeros(len(p), bool))
        self.kept_pts = [inside(p) for p in self.pts]; self.kept_corners = [inside(h.points) for h in self.holds]
        self.record = dict(tried=[], tasks=[dict(task=h.id, corners=len(h.A)-1, targets=len(h.targets),
                                                 bases=len(h.bases)) for h in self.holds])

    def remove(self, tool, gone, kept_pts, kept_corners, row):
        """The support less tool, with what falls off, and its skin's probes and the corners alive after (the points
        gone(points) or in kept_* stay): (support, F, alive); support None if nothing is left."""
        F = [f & ~(gone(p) & ~k) for f, p, k in zip(self.F, self.pts, kept_pts)]
        alive = [h.alive & ~np.r_[False, gone(h.points) & ~k] for h, k in zip(self.holds, kept_corners)]
        # As it is saved, its links thinner than SEAM collapsed (as settled() does): a piece hanging by one falls off.
        clock = time.monotonic(); value = G.solid(G.mesh(self.body-tool)).simplify(SEAM/G.SCALE)
        parts = sorted(positive_components(value), key=lambda p: -float(p.volume()))
        if not parts: row['reason'] = 'nothing left'; return None, F, alive
        value = parts[0]
        for part in parts[1:]:                                   # what the deletion parts falls off
            if any((within(part, p) & f & k).any() for f, p, k in zip(F, self.pts, self.kept_pts)):
                row['reason'] = 'it parts skin narrower than THICK'; return None, F, alive
            for f, p in zip(F, self.pts): f &= ~within(part, p)
            for al, h in zip(alive, self.holds): al[1:] &= ~within(part, h.points)
        row['pieces_seconds'] = row.get('pieces_seconds', 0.)+time.monotonic()-clock
        return value, F, alive

    def thin(self, a, F):
        """Seat a's own cells with material (F: its probes) in skin narrower than WIDTH (step4/skin.py's rule) -- but
        the narrow skin the support starts with (self.narrow: it stays)."""
        return np.flatnonzero(self.S.thin(material(self.c, F), WIDTH/2, self.own[a]) & ~self.narrow[a])

    def attempt(self, cut, gone, label):
        """Delete the support's material in cut but the kept prisms (gone(points): the points in cut), with what falls
        off, sparing the material about the skin narrower than WIDTH it would leave where it was not (up to SPARE
        times), if every certificate holds.  True if kept."""
        row = dict(deletion=label, spared_m2=0.); began = time.monotonic()
        keep, kept_pts, kept_corners = self.kept, self.kept_pts, self.kept_corners
        for spare in range(SPARE+1):
            value, F, alive = self.remove(cut-keep, gone, kept_pts, kept_corners, row)
            if value is None: break
            thin = [self.thin(a, f) for a, f in enumerate(F)]
            if not any(len(t) for t in thin): break
            # Not made there: the cells within WIDTH of that skin that had material keep it.
            ids = [np.flatnonzero(self.S.around(t, WIDTH) & material(self.c, f0) & ~material(self.c, f)) if len(t) else t
                   for t, f0, f in zip(thin, self.F, F)]
            if spare == SPARE or not any(len(i) for i in ids):
                row['reason'] = 'it leaves skin narrower than WIDTH'; break
            more = G.union([self.S.cells(i, X) for i, (X, _) in zip(ids, self.seats) if len(i)]); keep = keep+more
            kept_pts = [k | within(more, p) for k, p in zip(kept_pts, self.pts)]
            kept_corners = [k | within(more, h.points) for k, h in zip(kept_corners, self.holds)]
            row['spared_m2'] += float(sum(self.S.area[i].sum() for i in ids))
        if 'reason' not in row and len(self.ground) and not probed(value, self.ground).all():
            row['reason'] = 'it takes the support off a table that feet alone cannot hold it on'
        if 'reason' not in row:
            found = []
            for h, al in zip(self.holds, alive):
                while True:                                      # the corners a certificate uses, probed in the support itself,
                    f = h.after(al)                              # until it uses none that is gone (alive shrinks: it ends)
                    if f is None: break
                    used = np.unique(np.concatenate([b for b in f[0] if b is not None])); used = used[used > 0]
                    dead = used[~probed(value, h.points[used-1])]
                    if not len(dead): break
                    al[dead] = False
                if f is None: row['reason'] = h.id+': a paired force/moment target is no longer held'; break
                found.append(f)
        row.update(kept='reason' not in row, seconds=time.monotonic()-began); self.record['tried'].append(row)
        if not row['kept']: return False
        for h, al, (bases, of) in zip(self.holds, alive, found): h.alive, h.bases, h.of = al, bases, of
        self.body, self.F = value, F; return True

    def run(self, bunnies):
        """The deletions in order (bunnies: points of every seat's object, support frame, for the plane cuts): the
        support and the record."""
        began = time.monotonic(); start = G.volume(self.body)
        # Step4 cuts its thin skin once and lets what that leaves be: it stays.
        for a, f in enumerate(self.F): self.narrow[a][self.thin(a, f)] = True
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
                low, high = G.mesh(self.body).bounds                                    # the plane's far side:
                self.attempt(G.box(low-1, high+1).trim_by_plane(tuple(-normal), float(-offset/G.SCALE)),
                             lambda p, n=normal, o=offset: p@n < o, f'plane axis {axis} {"high" if side else "low"} {fraction}')
        for grid in GRIDS:
            low, high = G.mesh(self.body).bounds; size = (high-low)/grid; boxes = []
            skin_points = np.vstack([p[f] for p, f in zip(self.pts, self.F)])
            for index in np.ndindex(grid, grid, grid):
                a = low+np.array(index)*size; b = a+size; inside = int(((skin_points >= a) & (skin_points <= b)).all(1).sum())
                if inside: boxes.append((-inside, index, a, b))
            for _, index, a, b in sorted(boxes, key=lambda r: (r[0], r[1])):
                if len(self.record['tried']) >= BUDGET: break
                self.attempt(G.box(a, b), lambda p, a=a, b=b: ((p >= a) & (p <= b)).all(1), f'box {grid} {index}')
        self.record.update(removed_cm3=(start-G.volume(self.body))*1e6, seconds=time.monotonic()-began,
                           kept=sum(r['kept'] for r in self.record['tried']))
        return G.mesh(self.body), self.record
