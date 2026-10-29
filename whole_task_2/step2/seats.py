"""Step2: seats -- the fewest supports, each with one exit, the robot left to Step6.

A seat is one support the object sits in, opened along an exit e (object frame).  Every task working on it shares
its support and its exit, so how many installations the work arm needs is not the seat's concern: turning the
support on the table or moving it costs no material, only another installation (Step6 picks them for the robot, on
the finished support).  What a seat needs is a base -- the face it rests on, the object's up u (object frame) -- for
every face it serves: on a base the station's needle, turned about the vertical, sweeps a cone about its fixed fall
and reaches the faces whose screened needle axes lie within the tilt of that cone.  More bases turn the support
about a non-vertical axis (combine's Rotational reuse); they are added only for faces the bases taken do not reach.
The loads are judged without the robot (MODEL): the push within the tilt cone about each face's own needle axis, the
weight along -u of the face's base.

  1. Seats (the big cut): candidate exits e (Fibonacci); e takes the faces in front of it (n.e > 0) that some base on
     which e rises (combine's exit_rise_deg) reaches.  A greedy set cover picks the fewest exits whose fronts,
     overlapping, cover the area -- an exit need not take its whole front -- swapped one at a time for others that keep
     the area covered and share it more evenly.  The faces then go to them in about equal, connected blocks (user,
     2026-10-28): each seat grows along the surface from the face of its front that faces its
     exit most squarely, a face to the seat of least distance less the seat's weight, the weights balanced; a piece cut
     off from its seat, or enclosed by another, goes to the neighbour it shares the longest border with.
  2. Bases of a seat (the small cut): greedily the base reaching most of the seat's faces not yet served, judged
     exactly on the seat's widest support (the 5 mm outer layer less the object and the exit sweep along e; every
     support of the seat lies in it, so loads that fail there fail on all): its faces' loads hold (more than 99% of the
     paired samples, step3/independent_physics.HELD) and it keeps combine's floor rule with every base taken (no
     required pressure point below another installation's resting plane).  The best candidates are judged together;
     one that fails is not tried again for the seat; a base after the first must take more than policy['assumed'] of
     the target area (fewer faces are assumed reachable, not worth turning the support over).  The faces then go to the
     bases in connected areas (each to the
     first base that reaches it, a cut-off piece to the neighbouring base that reaches all of it) and Step1's exact
     needle and head test, its own base's turns first (the object alone; a face that fails tries the seat's other
     needles, then the other seats', the squarest first; a face no needle reaches is seated anew, once).  A seat's
     faces on one base are one area (Step3's task).
  3. The seat's check: Step3's construction on ONE support for all its areas (the 5 mm outer layer less the object,
     every face's needle and head and the exit sweep along e -- every area's exit is e in the plan, Step3 builds its
     membrane along it and combine's Rotational reuse leaves through it): one closed piece, the exit clear, no body or
     keep-out overlap; its skin narrower than step4/skin.WIDTH cut, as combine cuts it.  Then every area: its own
     support passes Step3's audit as Step3 builds it, and narrowed as combine starts it, it holds and stands; its loads
     hold on the seat's support; every two areas keep the floor rule.  A failing area is dropped, and bases that break
     the floor rule with a larger base; the seat is built again from what is left.  Dropped faces are seated anew; a
     seat left with nothing keeps its block with the next exit that covers it (the nearest first, at most
     policy['exit_retries'] exits), then its block is cut again without the exits it tried.  Hard-to-reach faces are
     assumed reachable and join the nearest area, uncertified: a face no needle axis reaches, a single face that fails,
     and the faces of seats too small to open, the smallest first within policy['assumed'] (0.5%) of the target area.

The plan (Step1's format, step2/split.py) marks every area robot-free: Step3, Step4 and Step5 skip the KUKA and judge
the loads by MODEL; Step6 splits every area into the work arm's installations on the finished support.  Area ids are
the seat's letter (A: the largest seat) and the base's rank in it.

  python -m whole_task_2.step2.seats B+whole_shape_bases --task objects/B/big_tasks/whole_shape.json --config robots_config/random/07_kuka.json
"""
import argparse
import string
import time
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from whole_task_2.step1.common import DEFAULT_OUTPUT
from whole_task_2.step1 import plan_reach as P1
from whole_task_2.operations.geometry import moved_mesh, contact_surface
from whole_task_2.operations.state import Layout
from whole_task_2.step3 import independent_geometry as G
from whole_task_2.step3.independent_physics import certificate
from whole_task_2.step4.skin import Skin, largest_mesh
from . import split as S

FOLDER = 'step2_seats'
MODEL = 'local_axis_cone'           # robot-free loads (common/wrenches.py): the push within the tilt cone about each
                                    #     face's own needle axis (the needle that reaches it on its base)
POLICY = dict(exits=240,            # candidate seat exits e (Fibonacci)
              ups=240,              # candidate bases u (Fibonacci)
              spins=24,             # the needle's turns about the vertical on a base (what a base reaches)
              exit_rise_deg=10.,    # step4/combine.py: an exit rises at least this much in every installation
              candidates=8,         # bases judged together in a greedy step
              exit_candidates=48,   # exits a seat's may be swapped for, to share the area evenly (the largest fronts)
              swaps=4,              # sweeps of swaps at most
              exit_retries=3,       # exits a seat that fails whole tries for its block before the block is cut again
              assumed=.005,         # hard-to-reach faces up to this share of the target are taken as reachable and join
                                    #     the nearest area, uncertified, instead of opening seats (user, 2026-10-28)
              rounds=400)           # auction rounds


def fibonacci(n):
    k = np.arange(n)+.5; z = 1-2*k/n; r = np.sqrt(1-z*z); phi = k*np.pi*(3-5**.5)
    return np.c_[r*np.cos(phi), r*np.sin(phi), z]


def options(x, policy):
    """The installations a seat may use, once per run: bases u (Fibonacci, object frame) at the needle's turns about
    the vertical.  Arrays: each one's base (index into ups), its turn about the needle for split.placement (deg), its
    needle (object frame) and the faces it reaches (the screen's axes within the tilt cone of the needle); by_up:
    what each base reaches; zero: each base's installation at turn 0 (its pose in the plan)."""
    if 'options' in x: return x['options']
    a = S.unit(x['s']['working_direction_world']); ups = fibonacci(policy['ups']); cosb = np.cos(np.radians(x['s']['max_tilt_deg']))
    base, angle, needle = [], [], []
    for j, u in enumerate(ups):
        T = S.toward(u, np.asarray([0., 0., 1.]))
        for phi in 2*np.pi*np.arange(policy['spins'])/policy['spins']:
            R = Rotation.from_rotvec([0., 0., phi]).as_matrix()@T; v = R.T@a
            psi = Rotation.from_matrix(R@S.toward(v, a).T).as_rotvec()@a       # R = turn(a, psi) toward(v, a)
            base.append(j); angle.append(float(np.degrees(psi))); needle.append(v)
    needle = np.asarray(needle); sets = np.asarray([x['reach'][x['axes']@v >= cosb-1e-12].any(0) for v in needle])
    base = np.asarray(base)
    x['options'] = dict(ups=ups, base=base, angle=np.asarray(angle), needle=needle, sets=sets,
                        by_up=np.asarray([sets[base == j].any(0) for j in range(len(ups))]),
                        zero=np.asarray([np.flatnonzero(base == j)[0] for j in range(len(ups))]))
    return x['options']


def cover(sets, faces, area):
    """Greedy set cover of faces by the boolean rows of sets, area-weighted: the chosen rows in order."""
    left = np.zeros(sets.shape[1], bool); left[faces] = True; chosen = []
    while True:
        gain = (sets & left)@area; k = int(np.argmax(gain))
        if gain[k] <= 0: return chosen
        chosen.append(k); left &= ~sets[k]


def auction(cost, area, rounds):
    """Each face to the option of least cost plus price.  Prices rise for options above the mean area and fall below it
    (by the excess as a share of the mean, the step shrinking), so the options come out about the same size where the
    faces allow; the most even round (the smallest largest option) is kept.  cost: (faces, options), inf where an
    option cannot take a face."""
    k = cost.shape[1]; price = np.zeros(k); mean = area.sum()/k; best = None
    for r in range(rounds):
        label = np.argmin(cost+price, 1); load = np.bincount(label, area, k)
        if best is None or load.max() < best[0]: best = (load.max(), label)
        price += .5*(load-mean)/mean/np.sqrt(r+1)
    return best[1]


def squareness(mesh, faces, directions):
    """(faces, directions): the angle between each face's inward normal and each direction."""
    return np.arccos(np.clip(-mesh.face_normals[faces]@np.asarray(directions).T, -1, 1))


def rising(o, e, policy):
    """The bases on which exit e rises at least exit_rise_deg."""
    return np.flatnonzero(o['ups']@e >= np.sin(np.radians(policy['exit_rise_deg'])))


def fronts(x, policy):
    """(candidate exits E, what each takes: the faces in front of it that a base on which it rises reaches), once per run."""
    if 'fronts' not in x:
        mesh = x['mesh']; o = options(x, policy); E = fibonacci(policy['exits'])
        x['fronts'] = E, np.asarray([o['by_up'][rising(o, e, policy)].any(0) & (mesh.face_normals@e > 0) for e in E])
    return x['fronts']


def seat_plan(x, faces, policy, banned=()):
    """[(exit index, the seat's faces)]: as many exits (not banned) as a greedy set cover needs for their fronts,
    overlapping, to cover faces, chosen to share them evenly: one exit at a time is swapped for a candidate (the
    policy['exit_candidates'] with the largest fronts on faces) while every face stays covered and the largest block of
    the even, connected partition (blocks) shrinks."""
    mesh = x['mesh']; E, sets = fronts(x, policy); area = mesh.area_faces[faces]
    sets = sets.copy(); sets[list(banned)] = False
    chosen = cover(sets, faces, mesh.area_faces)
    if not chosen: return []

    def worst(c):
        if not sets[c][:, faces].any(0).all(): return np.inf, None
        label = blocks(x, faces, sets[c], E[c], policy); ok = label >= 0
        return float(np.bincount(label[ok], area[ok], len(c)).max()), label
    best = worst(chosen); candidates = list(np.argsort(-(sets[:, faces]@area))[:policy['exit_candidates']])
    for _ in range(policy['swaps']):
        before = best[0]
        for j in range(len(chosen)):
            for c in candidates:
                if c in chosen: continue
                trial = chosen[:j]+[int(c)]+chosen[j+1:]; got = worst(trial)
                if got[0] < best[0]-1e-12: chosen, best = trial, got
        if best[0] >= before-1e-12: break
    label = best[1]
    return [(int(c), faces[label == j]) for j, c in enumerate(chosen) if (label == j).any()]


def blocks(x, faces, sets, E, policy):
    """Each face's seat (index into E): about equal, connected blocks.  A seat grows from its seed -- the face of its
    front that faces its exit most squarely -- over the faces it may take (sets), along the surface (face adjacency,
    the distance between face centres); a face goes to the seat of least distance less the seat's weight (an
    additively weighted geodesic Voronoi partition, whose cells are connected), the weights raised for seats below the
    mean area and lowered above it, like the auction's prices; the most even round is kept.  Then every seat keeps its
    largest connected piece: its other pieces, and faces no seat reaches, go to the neighbouring seat they share the
    longest border with (a piece a seat encloses is that seat's)."""
    mesh = x['mesh']; n, k = len(faces), len(E); area = mesh.area_faces[faces]
    index = np.full(len(mesh.faces), -1); index[faces] = np.arange(n)
    pairs = index[mesh.face_adjacency]; pairs = pairs[(pairs >= 0).all(1)]
    length = np.linalg.norm(mesh.triangles_center[faces[pairs[:, 0]]]-mesh.triangles_center[faces[pairs[:, 1]]], axis=1)
    D = np.full((k, n), np.inf)
    for s_ in range(k):
        mine = sets[s_][faces]
        if not mine.any(): continue
        ok = mine[pairs[:, 0]] & mine[pairs[:, 1]]
        seed = int(np.flatnonzero(mine)[np.argmax(mesh.face_normals[faces[mine]]@E[s_])])
        D[s_] = dijkstra(coo_matrix((length[ok], (pairs[ok, 0], pairs[ok, 1])), shape=(n, n)).tocsr(), directed=False, indices=seed)
    reach = np.isfinite(D).any(0); scale = float(np.median(D[np.isfinite(D)])); w = np.zeros(k); mean = area.sum()/k; best = None
    for r in range(policy['rounds']):
        label = np.where(reach, np.argmin(D-w[:, None], 0), -1); load = np.bincount(label[reach], area[reach], k)
        if best is None or load.max() < best[0]: best = (load.max(), label)
        w += .5*scale*(mean-load)/mean/np.sqrt(r+1)
    return connected(best[1].copy(), pairs, area, k)


def connected(label, pairs, area, k, allowed=None):
    """label made connected (blocks): faces no seat reaches (-1) and every piece of a seat but its largest go to the
    neighbouring seat they share the longest border with -- with allowed (k, n), only to one allowed every face of the
    piece; a piece with no such neighbour keeps its seat (or, reached by none, goes to the first)."""
    n = len(label)
    for _ in range(100):
        changed = False
        for s_ in [-1]+list(range(k)):
            mask = label == s_
            if not mask.any(): continue
            inner = pairs[mask[pairs[:, 0]] & mask[pairs[:, 1]]]
            _, comp = connected_components(coo_matrix((np.ones(len(inner)), (inner[:, 0], inner[:, 1])), shape=(n, n)), directed=False)
            pieces = np.unique(comp[mask]); size = {c: float(area[mask & (comp == c)].sum()) for c in pieces}
            keep = None if s_ < 0 else max(size, key=size.get)
            for c in pieces:
                if c == keep: continue
                piece = mask & (comp == c); border = pairs[piece[pairs[:, 0]] ^ piece[pairs[:, 1]]]
                other = label[np.where(piece[border[:, 0]], border[:, 1], border[:, 0])]; other = other[other >= 0]
                if allowed is not None: other = other[allowed[other][:, piece].all(1)]
                if len(other): label[piece] = np.bincount(other, minlength=k).argmax(); changed = True
                elif s_ < 0: label[piece] = 0; changed = True
        if not changed: break
    return label


# ---------------------------------------------------------------------------------------------
# The workers' side: supports and their loads, without the robot.

def loads_policy():
    """Step3's policy with the robot-free loads."""
    return dict(S._W['policy'], force_direction_model=MODEL)


def record(name, faces, axes, P, angle, keepout=True):
    """A robot-free area (step3/tool_membranes reads robot=False: no KUKA), axes its faces' needle axes (object frame,
    the loads' too: MODEL); without keepout its faces are loaded but cut no needle or head."""
    faces = [int(f) for f in faces]; mine = faces if keepout else []; axes = np.asarray(axes, float).reshape(-1, 3)
    return dict(id=name, direction_index=0, working_face_ids=faces, certified_face_ids=mine, pose=P, axes=axes[:len(mine)],
                force_axes=axes, joints=np.zeros((len(mine), 4, 7)), angle_deg=angle, robot=False)


def widest(e):
    """A seat's widest support, object frame: the 5 mm outer layer less the object and the exit sweep along e
    (step4/combine.exact_sweep, combine's construction length); every support of the seat lies in it."""
    from whole_task_2.step4.combine import exact_sweep
    W = S._W; length = float(np.ceil(4*np.linalg.norm(W['source'].extents)/.1)*.1)
    m = largest_mesh(G.solid(W['outer'])-G.solid(W['source'])-exact_sweep(W['source'], S.unit(e), length))
    return m.vertices, m.faces


def shape_checks(model, candidate):
    """The audit's checks before the loads (step3/membranes.MembraneModel.audit): one closed connected piece, every
    exit, no body or keep-out overlap.  All hold in the support's frame, whatever the installation.  The complaint, or None."""
    support = candidate['support']; solid = G.solid(support)
    if (not support.is_watertight or not support.is_winding_consistent or support.volume <= 0 or
            model.material_component_count(support, solid) != 1): return 'not one closed connected piece'
    if not all(r['passed'] for r in model.path_checks(solid, model.exit_paths(candidate))[0]): return 'an exit is blocked'
    for t in candidate['tasks']:
        if max(G.volume(solid ^ G.solid(t['body'])), G.volume(solid ^ t['work'])) > model.tolerance: return t['id']+': body or keep-out overlap'
    return None


def built(members, e, audit=False):
    """Step3's construction, object frame, of one support for members [(id, faces, axes, v, turn, pose)], all leaving
    along e (object frame): (its mesh or None, the complaint).  Judged by shape_checks, or with audit by Step3's own
    (step3/tool_membranes.initialize_one: the full audit), so that Step3 passes the area at its pose."""
    W = S._W; recs = [record(i, f, ax, P, a) for i, f, ax, v, a, P in members]
    model = (S.TM.ToolMembraneModel if audit else S.Free)(W['source'], W['outer'], recs, W['setup'], W['physical'], loads_policy(), W['run'])
    P = np.asarray([r['pose'] for r in recs]); d = [r['pose'][:3, :3]@S.unit(e) for r in recs]
    # An area audited alone is built exactly as Step3 builds it (tool_membranes.initialize_one: world frame, the exit
    # as the plan writes it) -- Booleans near a tolerance depend on the frame -- and brought to the object frame as
    # combine reads it (combine.membrane); a seat is built in the object frame, the support's.
    q = Layout(P, np.tile(np.eye(4), (len(P), 1, 1)) if audit else np.linalg.inv(P), np.asarray(d))
    try:
        candidate = model.build(q, facewise=True)
        if audit: model.audit(candidate); why = None
        else: why = shape_checks(model, candidate)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as ex: return None, str(ex)[:200]
    if why: return None, why
    return (moved_mesh(candidate['support'], np.linalg.inv(P[0])) if audit else candidate['support']), None


def pruned(support):
    """step4/skin.pruned for one seat in the object frame: the largest piece without its skin narrower than skin.WIDTH,
    cut once -- such skin holds nothing, and combine cuts it before it audits or stands a support."""
    W = S._W
    if 'skin' not in W: W['skin'] = Skin(W['source'], W['outer'], W['policy']['mold_outward_displacement_m'])
    K = W['skin']; value = G.solid(largest_mesh(G.solid(support)))
    ids = np.flatnonzero(K.thin(K.covered(value, np.eye(4)), own=K.reliable))
    if len(ids): value = value-(K.cells(ids, np.eye(4)) ^ G.solid(W['outer']))
    return largest_mesh(value)


def shell(members, e, audit=False):
    """One support for members (a seat's, or an area's own when alone), object frame, without its narrow skin as
    combine prunes it: {'mesh': (vertices, faces) or None, 'reason', 'volume_cm3', 'seconds'}."""
    began = time.monotonic(); mesh, why = built(members, e, audit)
    if mesh is not None:
        try: mesh = pruned(mesh)
        except ValueError as ex: mesh, why = None, 'pruned: '+str(ex)[:120]
    return dict(mesh=None if mesh is None else (mesh.vertices, mesh.faces), reason=why,
                volume_cm3=None if mesh is None else float(mesh.volume*1e6), seconds=time.monotonic()-began)


def holds(model, P, support):
    """The model's one area's paired forces and moments on support (object frame), the object at P: its world data, or None."""
    data = model.world_data(model.records[0], P); world = moved_mesh(support, P)
    try:
        proof, _ = certificate(data['body_world'], contact_surface(data['body_world'], world), data['domain'], data['endpoints'],
                               data['samples'], search=True)
    except RuntimeError: return None
    return data if proof['passed'] else None


def row(data, P, support):
    """An installation's row for the floor rule (step4/combine.floor_gap): its required pressure points on its table,
    support (object frame) resting on its lowest point.  None if they cannot be had."""
    R = P[:3, :3]; low = float((support.vertices@R.T)[:, 2].min()); world = R@S._W['source'].center_mass; world[2] -= low
    wrench = data['endpoints']['endpoint_need_wrench']
    try: ground = np.c_[G.pressure_points(wrench, world), np.full(len(wrench), low)]@R
    except ValueError: return None
    return dict(passed=True, Q=R, low=low, ground=ground)


def screen(job):
    """A candidate base on a seat's widest support (step 2 of the rule): (its row, None) or (None, the reason)."""
    j, v, angle, faces, axes, support = job; W = S._W; support = trimesh.Trimesh(*support, process=False)
    P = S.placement(W['source'], v, S.nominal(W['source'], np.asarray(faces))[1], angle, W['setup'])
    model = S.Free(W['source'], W['outer'], [record(f'u{j}', faces, axes, P, angle, keepout=False)], W['setup'], W['physical'],
                   loads_policy(), W['run'])
    data = holds(model, P, support)
    if data is None: return None, 'its loads fail on the widest support'
    r = row(data, P, support)
    return (r, None) if r is not None else (None, 'no pressure points')


def hold(job):
    """One area of a seat at its pose: it stands alone (step4/combine.stand: its own support reaches below the object by
    more than 1.1 needle radii), holds alone and holds on the seat's support; then its row for the floor rule."""
    member, seat, own = job; W = S._W; i, faces, axes, v, angle, P = member
    seat, own = [trimesh.Trimesh(*m, process=False) for m in (seat, own)]; R = P[:3, :3]
    below = float((W['source'].vertices@R.T)[:, 2].min()-(own.vertices@R.T)[:, 2].min())
    if below <= 1.1*W['setup']['needle_radius_m']: return dict(reason=f'alone it reaches {1e3*below:.1f} mm below the object')
    model = S.Free(W['source'], W['outer'], [record(i, faces, axes, P, angle)], W['setup'], W['physical'], loads_policy(), W['run'])
    if holds(model, P, own) is None: return dict(reason='its loads fail alone')
    data = holds(model, P, seat)
    if data is None: return dict(reason="its loads fail on the seat's support")
    return row(data, P, seat) or dict(reason='no pressure points')


def same_base(a, b):
    """Do two installations rest the support on one face (the same up in the support's frame)?"""
    return bool(a['Q'][2]@b['Q'][2] >= 1-1e-9)


def gap(a, b):
    """step4/combine.floor_gap for two installations: the lowest height of either's required pressure points above the
    other's resting plane (0 when both rest on one face)."""
    if same_base(a, b): return 0.
    return min(float((a['ground']@b['Q'].T)[:, 2].min()-b['low']), float((b['ground']@a['Q'].T)[:, 2].min()-a['low']))


def floor(rows, size):
    """The areas the floor rule keeps: bases the largest first, each kept when it keeps the rule with every one kept."""
    kept = []
    for k in sorted([k for k, r in enumerate(rows) if r.get('passed')], key=lambda k: -size[k]):
        if all(gap(rows[k], rows[j]) >= -1e-8 for j in kept): kept.append(k)
    return kept


# ---------------------------------------------------------------------------------------------
# The cut.

def bases(pool, x, seat, policy):
    """The seat's bases (step 2 of the rule), greedily: [base index]."""
    mesh = x['mesh']; o = options(x, policy); ups = rising(o, seat['e'], policy); area = mesh.area_faces
    left = np.zeros(len(mesh.faces), bool); left[seat['faces']] = True; chosen, rows, out = [], [], set()
    small = policy['assumed']*float(area[x['target']].sum())    # a further base must take more: less is assumed reachable
    while True:
        gain = (o['by_up'][ups] & left)@area
        order = [int(ups[i]) for i in np.argsort(-gain) if gain[i] > (small if chosen else 0) and int(ups[i]) not in out][:policy['candidates']]
        if not order: return chosen
        jobs = []
        for j in order:
            faces = np.flatnonzero(o['by_up'][j] & left); turns = np.flatnonzero(o['base'] == j)
            # Each face loaded about the needle of the base's turn that reaches it most squarely.
            fit = np.where(o['sets'][turns][:, faces].T, -mesh.face_normals[faces]@o['needle'][turns].T, -np.inf)
            jobs.append(pool.submit(screen, (j, o['needle'][o['zero'][j]], o['angle'][o['zero'][j]], faces,
                                             o['needle'][turns[np.argmax(fit, 1)]], seat['widest'])))
        took = None
        for j, f in zip(order, jobs):                    # the best gain that holds and keeps the floor rule
            r, why = f.result()
            if r is not None and all(gap(r, q) >= -1e-8 for q in rows): took = (j, r); break
            out.add(j)
        for f in jobs: f.cancel()
        if took is None: continue
        share = float(area[o['by_up'][took[0]] & left].sum()/area[seat['faces']].sum())
        chosen.append(took[0]); rows.append(took[1]); left &= ~o['by_up'][took[0]]
        print('STEP2 SEAT', seat['name'], 'base', len(chosen), 'takes', round(100*share, 1), '% of its faces', flush=True)


def needles(x, seat, policy):
    """The seat's needles: every turn of its bases.  Its faces go to its bases in connected areas -- each to the first
    base (in the order they were taken) that reaches it, then made connected (connected: a piece goes to the
    neighbouring base that reaches all of it) -- and every face tries its own base's turns first, the squarest first,
    then the other bases' (exact's inputs)."""
    mesh = x['mesh']; o = options(x, policy); faces = np.asarray(seat['faces']); n = len(faces); taken = list(seat['bases'])
    reach = o['by_up'][taken][:, faces]; label = np.where(reach.any(0), np.argmax(reach, 0), -1)
    index = np.full(len(mesh.faces), -1); index[faces] = np.arange(n)
    pairs = index[mesh.face_adjacency]; pairs = pairs[(pairs >= 0).all(1)]
    label = connected(label, pairs, mesh.area_faces[faces], len(taken), reach)
    chosen = np.flatnonzero(np.isin(o['base'], taken)); own = np.asarray([taken.index(int(j)) for j in o['base'][chosen]])
    cost = np.where(o['sets'][chosen][:, faces].T, squareness(mesh, faces, o['needle'][chosen]), np.inf)
    order = np.argsort(cost+4*np.pi*(own[None, :] != label[:, None]), 1)              # its own base's turns first
    seat.update(option=chosen, needles=list(o['needle'][chosen]), order=order, reachable=np.isfinite(cost))


def exact(pool, x, seats):
    """Step1's exact needle and head test for every face at its needle (the least tilt about the needle that clears; the
    object alone).  A face that fails tries its seat's next needle that may reach it, then the other seats' (the
    squarest first), so a face its own seat cannot reach joins another seat instead of opening a new one.  Each seat
    gets its reached faces [(option, faces, axes)] and its faces no needle reaches (missed)."""
    s = x['s']; mesh, axes, reach = x['mesh'], x['axes'], x['reach']; cosb = np.cos(np.radians(s['max_tilt_deg']))
    flat = [(k, j) for k, seat in enumerate(seats) for j in range(len(seat['needles']))]; index = {kj: n for n, kj in enumerate(flat)}
    V = np.asarray([seats[k]['needles'][j] for k, j in flat]).reshape(-1, 3)
    reaches = np.asarray([reach[axes@v >= cosb-1e-12].any(0) for v in V]).reshape(len(V), -1)
    choices = {}
    for k, seat in enumerate(seats):
        for i, f in enumerate(seat['faces']):
            f = int(f); own = [index[(k, int(j))] for j in seat['order'][i, :int(seat['reachable'][i].sum())]]
            others = [int(n) for n in np.argsort(squareness(mesh, [f], V)[0]) if flat[n][0] != k and reaches[n, f]] if len(V) else []
            choices[f] = own+others
    got = {}; tried = dict.fromkeys(choices, 0)
    while True:
        jobs = []
        for f, opts in choices.items():
            if f in got or tried[f] >= len(opts): continue
            n = opts[tried[f]]; C = axes@V[n]; a = np.flatnonzero((C >= cosb-1e-12) & reach[:, f]); a = a[np.argsort(-C[a])]
            jobs.append((f, [(n, int(b), axes[b]) for b in a], s['needle_length_m'], s['head_radius_m'], s['needle_radius_m'],
                         x['eps'], s['max_incidence_deg']))
        if not jobs: break
        for face, n, b, *_ in pool.map(P1._certify, jobs, chunksize=16):
            if n is None: tried[face] += 1
            else: got[face] = (n, axes[b])
    for k, seat in enumerate(seats):
        seat['reached'] = []
        for j in range(len(seat['needles'])):
            mine = sorted(f for f, (n, _) in got.items() if n == index[(k, j)])
            if mine: seat['reached'].append((int(seat['option'][j]), np.asarray(mine), np.asarray([got[f][1] for f in mine])))
        seat['missed'] = np.asarray(sorted(f for f in map(int, seat['faces']) if f not in got), int)


def areas(x, seat, policy):
    """A seat's areas: its reached faces on one base, each at the base's installation at turn 0, its pose as
    split.placement puts it (its faces' centre over the work point, the object on the table): (id, faces, axes, v, turn, pose)."""
    o = options(x, policy); by = {}
    for k, faces, axes in seat['reached']:
        j = int(o['base'][k]); f, a = by.get(j, ([], [])); by[j] = (f+list(faces), a+list(axes))
    out = []
    for n, (j, (faces, axes)) in enumerate(sorted(by.items(), key=lambda b: -x['mesh'].area_faces[b[1][0]].sum())):
        faces = np.asarray(faces); z = o['zero'][j]; v, a = o['needle'][z], float(o['angle'][z])
        out.append((f"{seat['name']}{n+1}", faces, np.asarray(axes), v, a, S.placement(x['mesh'], v, S.nominal(x['mesh'], faces)[1], a, x['s'])))
    return out


def check(pool, x, pending):
    """Every pending seat's check (step 3 of the module's rule), the seats in parallel: its support and its areas' own,
    then every area at its pose; failing areas and bases dropped and the seat built again until it holds.  Leaves each
    seat its members that hold ([] if none), its 'result' (the support) and its 'drops' (area, faces, reason)."""
    area = x['mesh'].area_faces
    for seat in pending: seat['drops'] = []; seat.setdefault('own', {})
    active = [seat for seat in pending if seat['members']]
    while active:
        made = {seat['name']: pool.submit(shell, seat['members'], seat['e']) for seat in active}
        alone = {(seat['name'], m[0]): pool.submit(shell, [m], seat['e'], True) for seat in active for m in seat['members'] if m[0] not in seat['own']}
        for (name, task), f in alone.items(): next(s_ for s_ in active if s_['name'] == name)['own'][task] = f.result()
        futures = {}
        for seat in active:
            seat['result'] = r = made[seat['name']].result()
            if r['mesh'] is None or any(seat['own'][m[0]]['mesh'] is None for m in seat['members']): continue
            for k, m in enumerate(seat['members']): futures[(seat['name'], k)] = pool.submit(hold, (m, r['mesh'], seat['own'][m[0]]['mesh']))
        following = []
        for seat in active:
            r, members = seat['result'], seat['members']; size = [float(area[m[1]].sum()) for m in members]
            bad = {k: 'alone: '+seat['own'][m[0]]['reason'] for k, m in enumerate(members) if seat['own'][m[0]]['mesh'] is None}
            if bad: drop = bad
            elif r['mesh'] is None: drop = {int(np.argmin(size)): 'the seat with it: '+(r['reason'] or '')}
            else:
                rows = [futures[(seat['name'], k)].result() for k in range(len(members))]
                drop = {k: row_['reason'] for k, row_ in enumerate(rows) if not row_.get('passed')}
                kept = floor(rows, size)
                drop.update({k: 'floor rule against a larger base' for k in range(len(members)) if k not in kept and k not in drop})
                if not drop: continue
            seat['drops'] += [dict(task=members[k][0], faces=members[k][1], reason=why) for k, why in drop.items()]
            seat['members'] = [m for k, m in enumerate(members) if k not in drop]
            print('STEP2 SEAT', seat['name'], 'drops', [(members[k][0], why[:60]) for k, why in drop.items()], flush=True)
            if seat['members']: following.append(seat)
        active = following


def plan(pool, x, policy):
    """The seats: (passed seats, every attempt's row, faces that fail alone, faces assumed reachable).  Hard-to-reach
    faces -- no needle axis reaches them, or a seat would hold only them -- are assumed reachable, the smallest seats
    first while they stay within policy['assumed'] of the target area, and so are the faces that fail alone whatever
    their area; run() joins them to the nearest area."""
    mesh = x['mesh']; total = float(mesh.area_faces[x['target']].sum()); problems = [x['root']]; seats, rows, alone = [], [], []
    missed_once = set(); made = 0; options(x, policy); E, front = fronts(x, policy)
    merged = [int(f) for f in x['unreachable']]; budget = policy['assumed']*total-float(mesh.area_faces[merged].sum())

    def again(seat):
        """A seat that failed whole keeps its block with the next exit whose front covers it (the nearest to its own
        first), at most policy['exit_retries'] exits; then its block is cut again without the exits it tried."""
        tried = seat['tried']; whole = front[:, seat['faces']].all(1); whole[list(tried)] = False
        if whole.any() and len(tried) < policy['exit_retries']:
            c = int(np.flatnonzero(whole)[np.argmax(E[whole]@E[seat['c']])]); return dict(faces=seat['faces'], c=c, tried=tried | {c})
        return dict(faces=seat['faces'], banned=tried)
    while problems:
        offers = []
        for p_ in problems:
            if isinstance(p_, dict) and 'c' in p_: offers.append((p_['c'], p_['faces'], p_['tried']))
            else:
                faces, banned = (p_['faces'], p_['banned']) if isinstance(p_, dict) else (p_, set())
                offers += [(c, f, set(banned) | {c}) for c, f in seat_plan(x, faces, policy, banned)]
        pending = []
        for c, f, tried in sorted(offers, key=lambda o_: mesh.area_faces[o_[1]].sum()):
            if mesh.area_faces[f].sum() <= budget: merged += [int(g) for g in f]; budget -= float(mesh.area_faces[f].sum()); continue
            pending.append(dict(name=f'S{made}', c=c, e=E[c], faces=f, tried=tried)); made += 1
        if not pending: break
        widths = {seat['name']: pool.submit(widest, seat['e']) for seat in pending}; problems = []
        for seat in pending: seat['widest'] = widths[seat['name']].result(); seat['bases'] = bases(pool, x, seat, policy)
        failed = [seat for seat in pending if not seat['bases']]; pending = [seat for seat in pending if seat['bases']]
        for seat in failed:
            row_ = dict(seat=seat['name'], exit=seat['e'].tolist(), faces=len(seat['faces']), reason='no base holds on its widest support',
                        area_fraction_of_target=float(mesh.area_faces[seat['faces']].sum()/total))
            if len(seat['faces']) == 1: alone.append(dict(face=int(seat['faces'][0]), reason=row_['reason'])); row_['verdict'] = 'fails alone'
            else: problems.append(again(seat)); row_['verdict'] = 'again'
            rows.append(row_); print('STEP2 SEAT', seat['name'], round(100*row_['area_fraction_of_target'], 1), '%:', row_['verdict'], row_['reason'], flush=True)
        for seat in pending: needles(x, seat, policy)
        if pending: exact(pool, x, pending)
        for seat in pending:
            back = [int(f) for f in seat['missed'] if int(f) not in missed_once]; missed_once |= set(back)
            if back: problems.append(np.asarray(back))
            alone += [dict(face=int(f), reason='no needle of its seat reaches it') for f in seat['missed'] if int(f) not in back]
            seat['members'] = areas(x, seat, policy)
            seat['faces'] = np.concatenate([m[1] for m in seat['members']]) if seat['members'] else np.zeros(0, int)
        check(pool, x, pending)
        for seat in pending:
            if not len(seat['faces']): continue
            row_ = dict(seat=seat['name'], exit=seat['e'].tolist(), bases=len(seat['bases']), faces=len(seat['faces']),
                        area_fraction_of_target=float(mesh.area_faces[seat['faces']].sum()/total),
                        dropped=[dict(task=d['task'], area_fraction_of_target=float(mesh.area_faces[d['faces']].sum()/total),
                                      reason=d['reason']) for d in seat['drops']])
            if seat['members']:
                seat['faces'] = np.concatenate([m[1] for m in seat['members']]); seats.append(seat)
                if seat['drops']: problems.append(np.concatenate([d['faces'] for d in seat['drops']]))
                row_.update(verdict='seat', kept=len(seat['members']), kept_fraction_of_target=float(mesh.area_faces[seat['faces']].sum()/total),
                            support_cm3=seat['result']['volume_cm3'])
            elif len(seat['faces']) == 1:
                alone.append(dict(face=int(seat['faces'][0]), reason=seat['drops'][-1]['reason'])); row_.update(verdict='fails alone')
            else:
                problems.append(again(seat)); row_.update(verdict='again')
            rows.append(row_)
            print('STEP2 SEAT', row_['seat'], row_['bases'], 'bases', round(100*row_['area_fraction_of_target'], 1), '%:', row_['verdict'],
                  f"{row_['kept']} kept, {round(100*row_['kept_fraction_of_target'], 1)} %" if 'kept' in row_ else '',
                  f"({len(row_['dropped'])} dropped)" if row_['dropped'] else '', flush=True)
    return seats, rows, alone, merged+[d['face'] for d in alone]


def ups_of(tasks):
    """[(up, [area ids])]: the faces the support rests on, the up (object frame) at each area's pose."""
    out = []
    for t in tasks:
        up = t['pose'][2, :3]
        for b in out:
            if up@b[0] >= 1-1e-9: b[1].append(t['id']); break
        else: out.append((up, [t['id']]))
    return out


def run(run_name, task, config, workers=22, output_root=DEFAULT_OUTPUT, policy=POLICY):
    began = time.monotonic(); x = S.prepare(run_name, task, config); mesh = x['mesh']
    pool = S.workers_pool(run_name, x['shape'], x['s'], x['config'], workers)
    try:
        seats, rows, alone, merged = plan(pool, x, policy)
        seats.sort(key=lambda seat: -mesh.area_faces[seat['faces']].sum()); tasks = []
        for k, seat in enumerate(seats):
            seat['name'] = string.ascii_uppercase[k] if k < 26 else f'Z{k}'; r = seat['result']; V, F = r['mesh']
            for n, (_, faces, axes, v, a, P) in enumerate(sorted(seat['members'], key=lambda m: -mesh.area_faces[m[1]].sum())):
                tasks.append(dict(id=f"{seat['name']}{n+1}", seat=seat['name'], exit=seat['e'], faces=faces, d=v, centre=S.nominal(mesh, faces)[1],
                                  axes=axes, angle=a, pose=P, robot=False, volume_cm3=r['volume_cm3'], vertices=V@P[:3, :3].T+P[:3, 3],
                                  triangles=F, merged=[]))
        # Faces assumed reachable join the area with the nearest face (uncertified: Step3 counts their loads, no keep-out).
        owner = np.concatenate([np.full(len(t['faces']), j) for j, t in enumerate(tasks)])
        near = cKDTree(np.concatenate([mesh.triangles_center[t['faces']] for t in tasks]))
        for f in merged: tasks[owner[near.query(mesh.triangles_center[f])[1]]]['merged'].append(int(f))
        plan_, validation = S.finish(x, pool, tasks, FOLDER, __file__, robot=False)
    finally:
        pool.shutdown(cancel_futures=True)
    total = float(mesh.area_faces[x['target']].sum()); summary = []
    for seat in seats:
        mine = [t for t in tasks if t['seat'] == seat['name']]; ups = ups_of(mine)
        summary.append(dict(seat=seat['name'], exit=seat['e'].tolist(), support_cm3=seat['result']['volume_cm3'],
                            area_fraction_of_target=float(mesh.area_faces[seat['faces']].sum()/total), areas=[t['id'] for t in mine],
                            bases=[dict(up=u.tolist(), areas=ids) for u, ids in ups],
                            base_angles_deg=[float(np.degrees(np.arccos(np.clip(a@b, -1, 1)))) for i, (a, _) in enumerate(ups) for b, _ in ups[i+1:]]))
    assumed = float(mesh.area_faces[merged].sum()/total) if merged else 0.
    print('STEP2 SEATS', len(seats), 'seats,', len(tasks), 'areas; seat shares', [round(100*s_['area_fraction_of_target'], 1) for s_ in summary],
          '; bases per seat', [len(s_['bases']) for s_ in summary], '; assumed reachable', round(100*assumed, 2), '%', flush=True)
    return S.record(x, tasks, plan_, validation, FOLDER, __doc__.split('\n\n')[1:3], began, output_root,
                    policy=policy, load_model=MODEL, seats=summary, attempts=rows, failing_alone=alone,
                    assumed_reachable=dict(face_count=len(merged), fraction_of_target=assumed,
                                           areas={t['id']: len(t['merged']) for t in tasks if t['merged']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run'); p.add_argument('--task', required=True); p.add_argument('--config', required=True)
    p.add_argument('--workers', type=int, default=22); p.add_argument('--render-only', action='store_true')
    a = p.parse_args()
    S.render(a.run, folder=FOLDER) if a.render_only else run(a.run, a.task, a.config, a.workers)
