"""Step5's frame: Step4's support kept as big contact faces, what is not needed deleted from the outside in, the
narrow skin trimmed, and the faces rebuilt clean and joined.

  - the peel (step5/peel.py): the material deleted from the outside in -- plane cuts toward the bunnies, then boxes
    -- each kept when every task's paired forces and moments still hold (and the ground) and it leaves no skin
    narrower than WIDTH where it was not (so the faces stay big);
  - the trim: every seat's skin opened by a disc WIDTH/2 in radius along the surface (its cells a disc fits in, and
    every cell within WIDTH/2 of them).  The pieces the opening loses (a strip, a neck, an island: the whiskers; a
    corner) go when every task still holds on the corners left -- a seat's pieces all at once, else each half in
    turn, down to one piece: what the loads need stays;
  - the faces, rebuilt: every seat's layer (Step3's fitted outer, 5 mm) where balls WIDTH/2 in radius over its
    opened skin (WIDTH/4 apart, a cell apart along its edge) reach, and the prisms of the pieces it needs, in the legal space.  Nothing of
    Step4's cut edges is left (no teeth, flaps or curtains): a face's edge is round, its faces the object's and the
    layer's.  A face has no holes: what it encloses of its layer is laid again (the deletions go from the outside in);
  - cones and rods: the pieces that are apart joined by straight rods FRAME['rod'] in radius between nodes,
    FRAME['node'] beyond the layer along a face's normal at its middle (clear of the layers, else of every object
    grown by NEAR), each node on its face by a cone (the hull of the face's cells within FRAME['stem'] of its
    middle, through the layer, and the node's ball), the shortest rods that join them all (Kruskal), each clear of
    every seat's layer and of the forbidden region (bodies, keep-outs, exit sweeps); where none is, one clear of
    every object grown by NEAR (it touches none); else through, cut.
At the end every certificate is solved again on its corners that lie on the frame's own contact polygons (a corner
in material may not: a rebuilt face's rim curves off the object), and judged on those polygons, as the audit judges; where the corners no longer hold, the corners the peel kept that the frame
lost and that a certificate on the peel's corners needs (the lost ones dear) get their seat's layer back about them
(where the corners hold and the polygons do not, the corners the certificate uses), in balls WIDTH/2 in radius over the
cells within GROW of them along the surface, farther each time one is lost again (a piece the join cannot reach is
dropped; grown along the surface it meets the seat's face), and the frame is joined again, at most FRAME['repairs']
times.  The frame is
one piece, its largest.
"""
import time
from functools import lru_cache
import numpy as np
import trimesh
import manifold3d as md
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from whole_task_2.operations.geometry import moved_mesh, contact_surface
from whole_task_2.step3 import independent_geometry as G
from whole_task_2.step3.independent_physics import certificate, contacts
from whole_task_2.step3.membranes import positive_components
from whole_task_2.step4.combine import largest
from whole_task_2.step4.merge import move
from whole_task_2.step4.seats import context
from whole_task_2.step4.skin import WIDTH, skin, legal, transform
from whole_task_2.step5 import peel as P

FRAME = dict(stem=.012,                 # a node's cone stands on its face's cells this near its middle
             node=.012,                 # a node this far beyond the layer
             rod=.004,                  # rods and the cones' tips this big in radius
             repairs=4)
GROW = (0., WIDTH/2, WIDTH, 2*WIDTH)    # a lost corner's layer given back this far about it along the surface, by repair
CLEAR = 1e-9                            # m^3: a rod meeting the layers or the forbidden region less than this is clear
NEAR = .55                              # material nearer another seat's object than this share of the layer is its skin


@lru_cache(maxsize=4)
def near(object_name):
    """The object grown by NEAR of the layer (Step3's fitted outer construction), object frame."""
    c = context(object_name); return G.fitted_outer(c.base.source, NEAR*c.base.policy['mold_outward_displacement_m'])


def ball(p, r, n=16):
    return md.Manifold.sphere(r/G.SCALE, n).translate(tuple(np.asarray(p, float)/G.SCALE))


def rod(a, b, r=FRAME['rod']):
    return md.Manifold.batch_hull([ball(a, r, 24), ball(b, r, 24)])


def spread(points, spacing):
    """Ids of points no two nearer than spacing, every point within spacing of one (greedy, in order)."""
    tree = cKDTree(points); taken = np.zeros(len(points), bool); covered = np.zeros(len(points), bool)
    for i in range(len(points)):
        if covered[i]: continue
        taken[i] = True; covered[tree.query_ball_point(points[i], spacing)] = True
    return np.flatnonzero(taken)


def trim(c, peel):
    """The trim (above): per seat (X, the centres of its balls, the cells it needs the opening loses), and the record.
    A corner of a seat's task is held where its cell is (it lies on that object)."""
    S, _ = skin(c); slots = {k: a for a, (_, ks) in enumerate(peel.seats) for k in ks}; out, record = [], []
    for a, ((X, ks), f) in enumerate(zip(peel.seats, peel.F)):
        R = P.material(c, f) & S.reliable
        core = R & (S.rim(R) >= WIDTH/2-.5*S.spacing)
        opened = S.around(np.flatnonzero(core), WIDTH/2) & R if core.any() else np.zeros(len(R), bool)
        lost = np.flatnonzero(R & ~opened); n, label = connected_components(S.graph[lost][:, lost], directed=False)
        mine = [k for k in range(len(peel.holds)) if slots[k] == a]
        at = [S.tree.query(transform(peel.holds[k].points, np.linalg.inv(X)))[1] for k in mine]
        gone = np.zeros(len(R), bool); stack = [[lost[label == j] for j in range(n)]] if n else []; needed = []
        while stack:
            group = stack.pop(); T = gone.copy(); T[np.concatenate(group)] = True
            als = [peel.holds[k].alive & ~np.r_[False, T[i]] for k, i in zip(mine, at)]; found = [peel.holds[k].after(al) for k, al in zip(mine, als)]
            if all(f is not None for f in found):
                for k, al, f in zip(mine, als, found): peel.holds[k].alive = al; peel.holds[k].bases, peel.holds[k].of = f
                gone = T
            elif len(group) > 1: stack += [group[len(group)//2:], group[:len(group)//2]]
            else: needed.append(group[0])
        # Balls over the opened skin: WIDTH/4 apart inside, a cell apart along the edge (they reach all of it).
        inner = np.flatnonzero(core); edge = np.flatnonzero(core & (S.rim(core) < 1.5*S.spacing))
        centres = np.vstack([S.mid[ids[spread(S.mid[ids], d)]] for ids, d in ((inner, WIDTH/4), (edge, S.spacing)) if len(ids)]+[np.zeros((0, 3))])
        out.append((X, transform(centres, X),
                    np.concatenate(needed) if needed else np.zeros(0, int)))
        record.append(dict(skin_m2=float(S.area[R].sum()), opened_m2=float(S.area[opened].sum()), pieces=n,
                           kept=len(needed), kept_m2=float(S.area[np.concatenate(needed)].sum()) if needed else 0.))
    return out, record


def join(c, value, rows, space, clear, touch, frame=FRAME):
    """value with its pieces joined by cones and rods (above): (the solid, record)."""
    S, _ = skin(c); thick = c.base.policy['mold_outward_displacement_m']
    pieces = sorted(positive_components(value), key=lambda p: -p.volume()); nodes = []
    if len(pieces) > 1:
        for X in rows:
            R = P.material(c, P.probed(value, P.layer_points(c, X))) & S.reliable; cells = np.flatnonzero(R)
            n_, lab = connected_components(S.graph[cells][:, cells], directed=False); rim = S.rim(R)
            for j in range(n_):
                part = cells[lab == j]; middle = part[np.argmax(rim[part])]; normal = X[:3, :3]@S.normals[middle]
                centre = transform(S.centers[middle][None], X)[0]
                node = next((centre+normal*(thick+d*frame['node']) for keep in (clear, touch) for d in (1., 1.5, .5, 2.)
                             if G.volume(ball(centre+normal*(thick+d*frame['node']), frame['rod']) ^ keep) < CLEAR), None)
                home = next((i for i, p in enumerate(pieces) if P.within(p, centre[None]).any()), None) if node is not None else None
                if home is not None: nodes.append((home, node, X, S.around(np.array([middle]), frame['stem']) & R))
    # Kruskal over the pieces: a rod between two nodes of pieces apart, the shortest first, in tiers.
    label = np.arange(len(pieces)); pairs = sorted((float(np.linalg.norm(a[1]-b[1])), i, j) for i, a in enumerate(nodes)
                                                   for j, b in enumerate(nodes[:i]) if a[0] != b[0])
    parts, used, edges = [value], set(), []
    for tier, keep in enumerate((clear, touch, None)):
        if len(np.unique(label)) == 1: break
        for d, i, j in pairs:
            if label[nodes[i][0]] == label[nodes[j][0]]: continue
            r = rod(nodes[i][1], nodes[j][1], frame['rod'])
            if keep is not None and G.volume(r ^ keep) >= CLEAR: continue
            if keep is None:
                r = r ^ space
                if len(positive_components(r)) != 1: continue
            edges.append((i, j, d, tier)); used |= {i, j}; parts.append(r)
            a, b = label[nodes[i][0]], label[nodes[j][0]]; label[label == a] = b
    for i in used:                                                          # the cones the rods stand on
        _, node, X, stem = nodes[i]
        parts.append(md.Manifold.batch_hull([md.Manifold.hull_points(transform(S.prisms[np.flatnonzero(stem)].reshape(-1, 3), X)/G.SCALE),
                                             ball(node, frame['rod'], 24)]) ^ space)
    out, share = largest((G.union(parts) ^ space).simplify(P.SEAM/G.SCALE))
    return out, dict(pieces=len(pieces), nodes=len(nodes), rods=len(edges), rods_by_tier=[sum(e[3] == t for e in edges) for t in range(3)],
                     rod_length_m=float(sum(e[2] for e in edges)), kept_share=float(share))


def certified(t, mesh):
    """Does mesh (support frame) hold task t on its own contact polygons, as the audit judges (more than
    step3/independent_physics.HELD of its paired samples)?"""
    world = moved_mesh(mesh, np.linalg.inv(t['material']))
    try: proof, _ = certificate(t['body_world'], contact_surface(t['body_world'], world), t['domain'], t['endpoints'], t['samples'])
    except RuntimeError: return False
    return bool(proof['passed'])


def on_contacts(t, mesh, hold, tolerance=1e-7):
    """Which of task t's corners (hold) lie on mesh's own contact polygons with it, as the audit finds them, on the
    corner's own object face?  (A corner in material need not: at the rim of a rebuilt face the material curves off the
    object, and a corner on an edge of the object is a corner of each face there, which a polygon may cover on one side
    only -- B+big_01 A1 held 98.5% where its corners in material held all.)"""
    world = moved_mesh(mesh, np.linalg.inv(t['material'])); out = np.zeros(len(hold.points), bool)
    try: tri, src, _, _ = contacts(t['body_world'], contact_surface(t['body_world'], world), t['domain'].work_ids, t['domain'].com)
    except RuntimeError: return out
    p = transform(hold.points, np.linalg.inv(t['material']))
    for f in np.intersect1d(src, hold.faces):
        T = tri[src == f]; k = np.flatnonzero(hold.faces == f); a = T[:, 0]
        e0, e1, d = T[:, 1]-a, T[:, 2]-a, p[k][:, None]-a[None]
        n = np.cross(e0, e1); area = np.linalg.norm(n, axis=1); good = area > 1e-20; n = n/np.where(good, area, 1.)[:, None]
        d00, d01, d11 = (e0*e0).sum(1), (e0*e1).sum(1), (e1*e1).sum(1); den = np.where(good, d00*d11-d01*d01, 1.)
        d20, d21 = (d*e0).sum(2), (d*e1).sum(2)
        v = (d11*d20-d01*d21)/den; w = (d00*d21-d01*d20)/den; slack = tolerance/np.sqrt(np.maximum(d00, d11))
        inside = (v >= -slack) & (w >= -slack) & (v+w <= 1+slack) & (np.abs((d*n[None]).sum(2)) <= tolerance) & good[None]
        out[k] = inside.any(1)
    return out


def frame(c, model, group, support, ground, F, frame=FRAME, stance=None):
    """The frame (above) from Step4's support (ground: the support's points the peel keeps on the tables; F: the
    forbidden region): (mesh, record)."""
    began = time.monotonic(); S, outer = skin(c); q = model.layout(support); thick = c.base.policy['mold_outward_displacement_m']
    reach = WIDTH+2*(thick+frame['node']+frame['rod'])
    space = legal(support.vertices, reach, q.material, F)
    bunnies = np.vstack([transform(model.source.vertices, X) for X in model.X])
    peel = P.Peel(c, model, group, support, ground, stance); _, peeled = peel.run(bunnies)
    found, trimmed = trim(c, peel)
    rows = [X for X, _ in peel.seats]; layers = [move(X, outer) for X in rows]
    faces = [G.union([ball(p, WIDTH/2) for p in centres]+([S.cells(cells, X)] if len(cells) else [])) ^ layer
             for (X, centres, cells), layer in zip(found, layers)]
    # A face has no holes (the deletions go from the outside in): every piece of the layer a seat's face encloses --
    # every piece of the cells without its material but the largest, reliable ones, and a cell about it -- is laid.
    filled = 0.
    for a, (X, layer) in enumerate(zip(rows, layers)):
        free = np.flatnonzero(~P.material(c, P.probed(faces[a], P.layer_points(c, X))))
        n, label = connected_components(S.graph[free][:, free], directed=False)
        if n < 2: continue
        hole = np.zeros(len(S.area), bool); hole[free[label != np.bincount(label, weights=S.area[free]).argmax()]] = True
        hole &= S.reliable; filled += float(S.area[hole].sum())
        if hole.any(): faces[a] = faces[a]+(S.cells(np.flatnonzero(hole | (S.graph@hole.astype(float) > 0)), X) ^ layer)
    # The points the stance stands on (ground: where the feet's legs start, step5/combined.grounds) stay: a ball WIDTH/2
    # of their seat's layer about each, as a repair gives a lost corner back.
    for a, layer in enumerate(layers):
        mine = ground[P.probed(layer, ground)] if len(ground) else ground
        if len(mine): faces[a] = faces[a]+(G.union([ball(p, WIDTH/2) for p in mine]) ^ layer)
    box = G.box(*(np.array(support.bounds)+[[-reach], [reach]]))
    clear = G.union([F ^ box]+layers)                                    # what a rod keeps clear of
    touch = G.union([F ^ box]+[move(X, near(c.object_name)) for X in rows])   # else at least of this: it touches no object
    slots = {k: a for a, (_, ks) in enumerate(peel.seats) for k in ks}; repairs, tries = [], {}
    for repair in range(frame['repairs']+1):
        out, joined = join(c, G.union(faces) ^ space, rows, space, clear, touch, frame)
        # Every certificate on its corners on the frame's own contact polygons (the bookkeeping's alive ones that are
        # there); where that fails, on the polygons themselves (the audit's: their cut edges hold reactions too).
        dead, failed = [], []; mesh = G.mesh(out)
        for k, (h, t) in enumerate(zip(peel.holds, peel.tasks)):
            alive = h.alive & np.r_[True, on_contacts(t, mesh, h)]; f = h.after(alive)
            if f is not None:
                h.alive, (h.bases, h.of) = alive, f
                if certified(t, mesh): continue
                # The corners hold it, the frame's own contact polygons do not (their rebuilt edges moved off corners
                # that are still in material: B+big_01 A1 turned into B1's seat, 98.5%): the corners its certificate
                # uses get their seat's layer back about them.
                used = np.unique(np.concatenate([b for b in h.bases if b is not None]))
                failed.append(h.id); dead.append((k, used[used > 0]-1)); continue
            if certified(t, mesh): continue
            # The corners to give back: the lost ones a certificate on the corners the peel kept uses, the lost ones
            # costing a thousand times more (so it uses as few as it can).
            cost = h.cost; lost = h.alive & ~alive; h.cost = cost*np.where(lost, 1e3, 1.)
            need = h.solve(np.arange(len(h.targets)), h.alive, []); h.cost = cost
            used = np.unique(np.concatenate([b for b in need[0] if b is not None])) if need is not None else np.flatnonzero(lost)
            failed.append(h.id); dead.append((k, used[lost[used]]-1))
        if not failed: break
        if repair == frame['repairs']: raise ValueError(', '.join(failed)+': the rebuilt frame no longer holds it')
        # A corner the peel kept that the frame lost gets its seat's layer back about it: balls WIDTH/2 in radius over
        # its seat's cells within GROW of it, farther each time it is lost again (a piece the join cannot reach is
        # dropped: grown along the surface, it meets the seat's face).
        for k, ids in dead:
            a = slots[k]; X = rows[a]; cells = []
            for i in ids:
                n = tries.get((k, i), 0); tries[(k, i)] = n+1
                cell = np.atleast_1d(S.tree.query(transform(peel.holds[k].points[i][None], np.linalg.inv(X)))[1])
                r = GROW[min(n, len(GROW)-1)]; cells.append(np.flatnonzero(S.around(cell, r)) if r else cell)
            if cells:
                p = transform(S.mid[np.unique(np.concatenate(cells))], X)
                faces[a] = faces[a]+(G.union([ball(q, WIDTH/2) for q in p[spread(p, WIDTH/4)]]) ^ layers[a])
        repairs.append(dict(failed=failed, corners=int(sum(len(i) for _, i in dead))))
    record = dict(peel=dict(kept=peeled['kept'], tried=len(peeled['tried']), removed_cm3=peeled['removed_cm3'], seconds=peeled['seconds']),
                  trim=trimmed, holes_filled_m2=filled, join=joined, repairs=repairs, volume_cm3=G.volume(out)*1e6, seconds=time.monotonic()-began)
    return G.mesh(out), record
