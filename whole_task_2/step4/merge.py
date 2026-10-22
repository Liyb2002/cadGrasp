"""Step4: merge the chosen seats into one support with Step4's two operations.

A group is one support with slots.  A slot holds the object at X (support <- object) for its
members: the tasks of one seat of step4/seats.py, turned on the table between them.  Groups A and B
merge by a rigid motion C of the whole of B:

  Juxtaposition     C only translates B;
  Rotational reuse  C turns B about any axis (SO(3)) and translates it.

The merged support is the union of the two parent supports minus every task's body, keep-out
(needle, head and KUKA, step4/keepout.py) and exit sweep at its slot, largest piece kept.  So one
slot's bunny cuts the other's material where they cross, and the cut faces are contact for it.  A
merge is accepted when it saves material, V(AB) <= rho (V(A)+V(B)), every task passes the full audit
on it, and it stands on the table in every pose: with one pose, the standing construction of
seats.grounded_seat (post and plate) for all slots, audited again in full.

A pose is a direction of the support resting down; a rotation that turns B's down onto A's down
(about any axis) adds no pose.  Candidates are built exactly (Booleans only) and screened by the gate,
by connection and by each slot's contact skin: the parent's material within a fifth of the membrane
thickness of that slot's object.  The parents' loads are proven on contacts in that skin; when no
skin loses material the proofs carry over (only material is added there), and a seat proven on
lean contacts fails if as little as 0.5% of its volume is cut where it touches.  So placements that
keep every skin are audited first, the least material first, then the others by the least skin lost.
Among the accepted: fewest poses, then smallest footprint on the table, then least material.
"""
import argparse
import hashlib
import itertools
import json
import time
from functools import lru_cache
from pathlib import Path
import numpy as np
import trimesh
import manifold3d as md
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from whole_task_2.step1.common import ROOT, sha256, write_json
from whole_task_2.operations.geometry import moved_mesh
from whole_task_2.operations.state import Layout
from whole_task_2.step3 import independent_geometry as G
from whole_task_2.step3.membranes import MembraneModel, positive_components
from .seats import SeatModel, context, read_mesh, settled, pool, SAME_DOWN, FOLDER as SEATS
from .join import JoinedModel, rank

FOLDER = 'step4_merge'
POLICY = dict(rho=.6,                   # merge gate V(AB) <= rho (V(A)+V(B)), as for seat sharing
              turn_step_deg=15.,        # turns about a shared down direction
              rotations=48,             # uniform SO(3) samples (fixed seed)
              directions=64,            # offset directions (Fibonacci sphere)
              radii=(.1, .2, .3, .4, .5, .6),   # offsets, fractions of the object diagonal
              connected=.9,             # the kept piece holds at least this share of the merged material
              contact_skin=.2,          # contact skin depth, share of the membrane thickness
              audits=96,                # candidates audited per merge
              keep=2,                   # accepted first merges carried to the next merge
              covers=1,                 # covers tried, in join.rank order
              seed=7, chunk=64)


def move(C, solid):
    return solid.transform(np.c_[C[:3, :3], C[:3, 3]/G.SCALE])


def flat(group):
    """Members, turns and placements, one row per task."""
    rows = [(i, a, np.asarray(s['X'], float)) for s in group['slots'] for i, a in zip(s['members'], s['angles'])]
    return [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]


def downs(c, group):
    """The group's distinct down directions in its support frame (its poses)."""
    out = []
    for i, a, X in zip(*flat(group)):
        g = X[:3, :3]@c.down(i, a)
        if not any(1-g@h < SAME_DOWN for h in out): out.append(g)
    return out


def fewest(D):
    out = []
    for g in D:
        if not any(1-g@h < SAME_DOWN for h in out): out.append(g)
    return out


class PlacedModel(SeatModel):
    """Every task on its slot X_k (support <- object), the support installed in any orientation,
    world_from_support = P_k X_k^-1, lowered so that its lowest point rests on the table (z = 0)."""
    def __init__(self, c, members, angles, X):
        rows = [c.record(i, a) for i, a in zip(members, angles)]
        super().__init__(c.base, [r for r, _ in rows], [k for _, k in rows], [c.sweeps[i] for i in members])
        self.X = [np.asarray(x, float) for x in X]; self.placed = {}

    def solids(self, k):
        if k not in self.placed:
            X = self.X[k]
            self.placed[k] = (moved_mesh(self.source, X), G.solid(moved_mesh(self.keepouts[k], X)), move(X, self.sweeps[k]))
        return self.placed[k]

    def layout(self, support):
        poses, material = [], []
        for k, r in enumerate(self.records):
            P = np.asarray(r['pose']).copy(); W = P@np.linalg.inv(self.X[k])
            P[2, 3] -= float((support.vertices@W[:3, :3].T+W[:3, 3])[:, 2].min())
            poses.append(P); material.append(self.X[k]@np.linalg.inv(P))
        return Layout(np.asarray(poses), np.asarray(material), np.tile(self.exit, (len(poses), 1)))

    def prepare(self, layout):
        rows = []
        for k, r in enumerate(self.records):
            P, M, d = layout.poses[k], layout.material[k], layout.exits[k]
            lift = P@np.linalg.inv(r['pose'])
            if not (np.allclose(lift[:3, :3], np.eye(3), atol=1e-9) and np.allclose(lift[:2, 3], 0, atol=1e-9)):
                raise ValueError(r['id']+': pose is not the certified one moved vertically')
            if not np.allclose(M@P, self.X[k], atol=1e-9): raise ValueError(r['id']+': not on its slot')
            body, work, sweep = self.solids(k)
            rows.append(dict(self.world_data(r, P), body=body, work=work, sweep=sweep, direction=M[:3, :3]@d,
                             length=self.length, material=M, pose=P, index=k, id=r['id'],
                             working_face_ids=np.asarray(r['working_face_ids'])))
        return rows

    def audit(self, candidate, *, arm=True, **options):
        result = MembraneModel.audit(self, candidate, **options); began = time.monotonic()
        for t, row in zip(candidate['tasks'], result['tasks']):
            r = self.records[t['index']]
            hits = self.arm_hits(moved_mesh(candidate['support'], np.asarray(r['pose'])@np.linalg.inv(self.X[t['index']])), r)
            row['arm'] = dict(configurations=4*len(r['certified_face_ids']), faces_hit=hits)
            if hits and arm: raise ValueError(t['id']+': KUKA links meet the support at faces '+str(hits[:10]))
        result['arm_seconds'] = time.monotonic()-began
        return result


# ---------------------------------------------------------------------------------------------
# Per process: each group's support, forbidden region, slot layers and contact skins, in its own frame.

_groups = {}


@lru_cache(maxsize=None)
def dilated(object_name):
    """The object grown by the contact skin depth (Step3's fitted outer skin construction)."""
    c = context(object_name)
    return G.fitted_outer(c.base.source, POLICY['contact_skin']*c.base.policy['mold_outward_displacement_m'])


def solids(object_name, group):
    key = group['file']
    if key not in _groups:
        c = context(object_name); s = G.solid(read_mesh(group['file'])); stock = G.solid(c.base.outer); F = []
        for i, a, X in zip(*flat(group)):
            F += [G.solid(moved_mesh(c.base.source, X)), G.solid(moved_mesh(c.record(i, a)[1], X)), move(X, c.sweeps[i])]
        X = [np.asarray(sl['X'], float) for sl in group['slots']]
        _groups[key] = dict(s=s, F=G.union(F), layers=[(L, G.volume(s ^ L)) for L in (move(x, stock) for x in X)],
                            skins=[s ^ move(x, dilated(object_name)) for x in X])
    return _groups[key]


def merged(object_name, A, B, C):
    """(A ∪ C B) minus both groups' forbidden regions (each parent already excludes its own), the
    largest piece, and the whole before the piece was chosen."""
    a, b = solids(object_name, A), solids(object_name, B)
    value = (a['s']-move(C, b['F']))+(move(C, b['s'])-a['F'])
    parts = positive_components(value)
    if not parts: return None, value
    keep = max(parts, key=lambda p: float(p.volume()))
    return keep, value


def screen(args):
    """Exact construction of each placement: kept volume, whole volume, each slot's surviving layer share
    and the volume of each slot's contact skin it loses (m^3)."""
    object_name, A, B, Cs = args; a, b = solids(object_name, A), solids(object_name, B); rows = []
    for C in Cs:
        C = np.asarray(C); keep, whole = merged(object_name, A, B, C)
        if keep is None: continue
        layers = [G.volume(keep ^ L)/v for L, v in a['layers']]+[G.volume(keep ^ move(C, L))/v for L, v in b['layers']]
        lost = [G.volume(S-keep) for S in a['skins']]+[G.volume(move(C, S)-keep) for S in b['skins']]
        rows.append(dict(C=C.tolist(), volume_cm3=float(keep.volume())*G.SCALE**3*1e6,
                         whole_cm3=float(whole.volume())*G.SCALE**3*1e6, layers=layers, skin_lost_m3=lost))
    return rows


def combined(A, B, C):
    """The merged group's slots: A's, then B's moved by C."""
    return [dict(s) for s in A['slots']]+[dict(s, X=(C@np.asarray(s['X'], float)).tolist()) for s in B['slots']]


def stand(c, slots, membrane):
    """seats.grounded_seat for several slots with one down direction: the membrane resting at plate
    height in the table frame T, an upright post under its lowest part up to a needle radius below the
    lowest object, and a plate under the post and every task's pressure hull, widened by the edge
    margin.  Settled.  Returns (standing support, slots in the table frame, the model, its layout)."""
    from .ground_oracle import pressure_rows
    from whole_task_2.step3.tool_keepout import disc
    p = c.base.policy; h = p['plate_thickness_m']; t = p['mold_outward_displacement_m']; clear = 1.1*c.base.setup['needle_radius_m']
    s0 = slots[0]; X0 = np.asarray(s0['X'], float)
    R = np.asarray(c.record(s0['members'][0], s0['angles'][0])[0]['pose'])[:3, :3]@X0[:3, :3].T
    T = np.eye(4); T[:3, :3] = R; T[2, 3] = h-float((membrane.vertices@R.T)[:, 2].min())
    table = [dict(s, X=(T@np.asarray(s['X'], float)).tolist()) for s in slots]
    placed = moved_mesh(membrane, T)
    low = min(float(moved_mesh(c.base.source, np.asarray(s['X'])).vertices[:, 2].min()) for s in table)
    top = min(h+t, low-clear); V = placed.vertices[placed.vertices[:, 2] <= top]
    if top <= h or len(V) < 3: raise ValueError('no membrane below the objects to stand on')
    model = JoinedModel(c, [s['members'] for s in table], [s['angles'] for s in table], [s['X'] for s in table]); q = model.layout()
    rows = pressure_rows(model, q, model.prepare(q))
    ring = disc(np.array([0., 0., 1.]), p['plate_edge_margin_m'], 16)[:, :2]
    F = (np.vstack([V[:, :2], *[r['ground_material'][:, :2] for r in rows]])[:, None]+ring[None]).reshape(-1, 2)
    post = md.Manifold.hull_points(np.vstack([np.c_[V[:, :2], np.zeros(len(V))], np.c_[V[:, :2], np.full(len(V), top)]])/G.SCALE)
    plate = md.Manifold.hull_points(np.vstack([np.c_[F, np.zeros(len(F))], np.c_[F, np.full(len(F), h)]])/G.SCALE)
    support = settled(G.mesh(G.union([G.solid(placed), post, plate])))
    if support is None: raise ValueError('standing support does not settle under manifold')
    return support, table, model, q


def footprint(support):
    V = support.vertices[np.abs(support.vertices[:, 2]) <= 1e-9][:, :2]
    return float(ConvexHull(V).volume)


def accept(args):
    """One placement in full: build, settle, refine exits, audit every task; with one pose, the standing
    support built and audited in full with its ground faces enclosing every pressure hull."""
    from .ground_oracle import pressure_rows, ground_enclosure, foot_regions
    object_name, A, B, C, file, limit = args; began = time.monotonic(); c = context(object_name); C = np.asarray(C)
    out = dict(C=C.tolist(), file=None)
    try:
        keep, _ = merged(object_name, A, B, C)
        membrane = settled(G.mesh(keep.simplify(0.)))
        if membrane is None: raise ValueError('merged membrane does not settle under manifold')
        slots = combined(A, B, C); model = PlacedModel(c, *flat(dict(slots=slots)))
        candidate = dict(layout=model.layout(membrane), support=membrane, construction={})
        candidate['tasks'] = model.prepare(candidate['layout'])
        model.refine_exits(candidate)
        membrane = settled(candidate['support'])
        if membrane is None: raise ValueError('refined membrane does not settle under manifold')
        candidate.update(support=membrane, layout=model.layout(membrane)); candidate['tasks'] = model.prepare(candidate['layout'])
        if membrane.volume*1e6 > limit: raise ValueError('the merge does not save enough material')
        model.audit(candidate)
        out.update(membrane_cm3=float(membrane.volume*1e6), poses=len(downs(c, dict(slots=slots))))
        if out['poses'] == 1:
            support, table, joined, q = stand(c, slots, membrane)
            standing = dict(layout=q, support=support, tasks=joined.prepare(q), construction={})
            joined.audit(standing)
            rows = ground_enclosure(support, pressure_rows(joined, q, standing['tasks']))
            if not all(r['passed'] for r in rows): raise ValueError('a pressure hull leaves the plate')
            out.update(standing_cm3=float(support.volume*1e6), footprint_cm2=1e4*footprint(support))
            np.savez_compressed(file, vertices=membrane.vertices, faces=membrane.faces,
                                standing_vertices=support.vertices, standing_faces=support.faces, table_from_support=np.asarray(
                                    np.asarray(table[0]['X'])@np.linalg.inv(np.asarray(slots[0]['X']))))
        else:
            found = next(foot_regions(model, candidate), None)
            if found is None: raise ValueError('no foot region')
            prints = []
            for g in downs(c, dict(slots=slots)):
                from whole_task_2.common.force_cone import tangent_frame
                u, v = tangent_frame(g); x = np.vstack([membrane.vertices, *[r['ground_material'] for r in found[2]]])
                prints.append(1e4*float(ConvexHull(np.c_[x@u, x@v]).volume))
            out.update(footprint_cm2=max(prints), foot_padding_m=found[0])
            np.savez_compressed(file, vertices=membrane.vertices, faces=membrane.faces)
        out.update(passed=True, file=str(file), slots=slots)
    except (ValueError, RuntimeError, AssertionError, np.linalg.LinAlgError) as ex:
        out.update(passed=False, reason=str(ex)[:300])
    out['seconds'] = time.monotonic()-began
    return out


# ---------------------------------------------------------------------------------------------
# Placements.

def sphere(n):
    i = np.arange(n)+.5; phi = np.arccos(1-2*i/n); th = np.pi*(1+5**.5)*i
    return np.c_[np.cos(th)*np.sin(phi), np.sin(th)*np.sin(phi), np.cos(phi)]


def turn(axis, deg):
    return Rotation.from_rotvec(axis/np.linalg.norm(axis)*np.radians(deg)).as_matrix()


def rotations(c, A, B, policy):
    """(operation, R): the identity (Juxtaposition); every rotation turning one of B's downs onto one of
    A's, then about it in steps (Rotational reuse, no new pose); uniform SO(3) samples (Rotational reuse)."""
    out = [('juxtaposition', np.eye(3))]
    for d, e in itertools.product(downs(c, B), downs(c, A)):
        R0 = Rotation.align_vectors([e], [d])[0].as_matrix()
        for deg in np.arange(0., 360., policy['turn_step_deg']): out.append(('rotational_reuse', turn(e, deg)@R0))
    for R in Rotation.random(policy['rotations'], random_state=policy['seed']).as_matrix(): out.append(('rotational_reuse', R))
    return out


def placements(c, A, B, policy):
    """Every rotation with B's slot centres put at A's slot centres plus offsets by object size."""
    D = float(np.linalg.norm(c.base.source.extents)); com = c.base.source.center_mass
    offsets = [np.zeros(3)]+[r*D*u for r in policy['radii'] for u in sphere(policy['directions'])]
    pa = [np.asarray(s['X'])[:3, :3]@com+np.asarray(s['X'])[:3, 3] for s in A['slots']]
    pb = [np.asarray(s['X'])[:3, :3]@com+np.asarray(s['X'])[:3, 3] for s in B['slots']]
    gA = downs(c, A); gB = downs(c, B); out = []
    for operation, R in rotations(c, A, B, policy):
        poses = len(fewest(gA+[R@g for g in gB]))
        for x, y, o in itertools.product(pa, pb, offsets):
            C = np.eye(4); C[:3, :3] = R; C[:3, 3] = x+o-R@y; out.append((operation, poses, C))
    return out


def merge(object_name, A, B, folder, workers, policy=POLICY, executor=None):
    """Merge B into A: screen every placement, audit the best at the fewest poses reachable, return the
    accepted ones best first (fewest poses, smallest footprint, least material) and the statistics."""
    began = time.monotonic(); c = context(object_name)
    limit = policy['rho']*(A['volume_cm3']+B['volume_cm3']); rows = placements(c, A, B, policy)
    chunks = [rows[k:k+policy['chunk']] for k in range(0, len(rows), policy['chunk'])]
    screened = executor.map(screen, [(object_name, A, B, [C for *_, C in chunk]) for chunk in chunks])
    found = []
    for chunk, result in zip(chunks, screened):
        lookup = {np.asarray(C).tobytes(): (op, n) for op, n, C in chunk}
        for r in result:
            op, n = lookup[np.asarray(r['C']).tobytes()]; found.append(dict(r, operation=op, poses=n))
    within = [r for r in found if r['volume_cm3'] <= limit and r['volume_cm3'] >= policy['connected']*r['whole_cm3']]
    intact = [r for r in within if max(r['skin_lost_m3']) <= c.base.tolerance]
    stats = dict(placements=len(rows), built=len(found), within_gate_and_connected=len(within), limit_cm3=limit,
                 skins_intact={int(n): sum(r['poses'] == n for r in intact) for n in sorted({r['poses'] for r in intact})},
                 by_poses={int(n): sum(r['poses'] == n for r in within) for n in sorted({r['poses'] for r in within})},
                 by_operation={op: sum(r['operation'] == op for r in within) for op in ('juxtaposition', 'rotational_reuse')})
    accepted = []
    for n in sorted({r['poses'] for r in within}):
        # Skins intact first, the least material first; then the least skin lost.
        level = sorted([r for r in within if r['poses'] == n], key=lambda r: (
            max(r['skin_lost_m3']) > c.base.tolerance, max(r['skin_lost_m3']) if max(r['skin_lost_m3']) > c.base.tolerance
            else r['volume_cm3']))[:policy['audits']]
        # Groups of one beam may share a name; the parents' files tell them apart.
        tag = hashlib.sha1((A['file']+B['file']).encode()).hexdigest()[:8]; (folder/'candidates').mkdir(parents=True, exist_ok=True)
        jobs = [(object_name, A, B, r['C'], folder/'candidates'/f"{A['name']}+{B['name']}_{tag}_{n}_{k:03d}.npz", limit)
                for k, r in enumerate(level)]
        results = list(executor.map(accept, jobs))
        for r, s in zip(results, level): r.update(operation=s['operation'], screened_cm3=s['volume_cm3'], layers=s['layers'],
                                                  skin_lost_m3=s['skin_lost_m3'])
        stats[f'audited_{n}_poses'] = len(results); stats[f'accepted_{n}_poses'] = sum(r['passed'] for r in results)
        stats.setdefault('failures', {})[n] = sorted({r['reason'].split(' {')[0] for r in results if not r['passed']})
        accepted = sorted([r for r in results if r['passed']], key=lambda r: (r['poses'], r['footprint_cm2'], r['membrane_cm3']))
        if accepted: break
    stats['seconds'] = time.monotonic()-began
    print('MERGE', A['name'], '+', B['name'], 'placements', len(rows), 'within gate', len(within), 'accepted', len(accepted),
          *(['best', accepted[0]['operation'], accepted[0]['poses'], 'poses', round(accepted[0]['footprint_cm2']), 'cm2',
             round(accepted[0]['membrane_cm3'], 1), 'cm3'] if accepted else []), round(stats['seconds']), 's', flush=True)
    return accepted, stats


# ---------------------------------------------------------------------------------------------

def seat_group(c, members, angles):
    """A seat of step4/seats.py as a group: one slot at the support origin, its certified membrane."""
    file = c.folder/'membranes'/(c.name(members)+'.npz')
    return dict(name=c.name(members), slots=[dict(members=list(members), angles=list(angles), X=np.eye(4).tolist())],
                file=str(file), volume_cm3=float(read_mesh(file).volume*1e6))


def as_group(result, name, parents):
    """An accepted merge as a group; `chain` lists every merge that made it, gate ratio included."""
    ratio = result['membrane_cm3']/sum(p['volume_cm3'] for p in parents)
    step = dict(name=name, parents=[p['name'] for p in parents], parent_cm3=[p['volume_cm3'] for p in parents],
                operation=result['operation'], volume_cm3=result['membrane_cm3'], ratio=ratio, poses=result['poses'],
                footprint_cm2=result['footprint_cm2'], rotation_deg=float(np.degrees(Rotation.from_matrix(
                    np.asarray(result['C'])[:3, :3]).magnitude())), translation_m=np.asarray(result['C'])[:3, 3].tolist(),
                file=result['file'], slots=result['slots'], parent_files=[p['file'] for p in parents],
                parent_slots=[p['slots'] for p in parents])
    return dict(name=name, slots=result['slots'], file=result['file'], volume_cm3=result['membrane_cm3'],
                footprint_cm2=result['footprint_cm2'], standing_cm3=result.get('standing_cm3'),
                chain=[x for q in parents for x in q.get('chain', [])]+[step])


def best_first(c, groups):
    return sorted(groups, key=lambda g: (len(downs(c, g)), g['footprint_cm2'], g['volume_cm3']))


def run(object_name='B', workers=20, policy=POLICY):
    """For each cover (join.rank order, fewest seats): a beam search over merge orders.  The beam starts
    with the best accepted merges of every pair of seats; each step merges one more remaining seat into
    every group of the beam and keeps the best `keep` groups.  The best one-piece result over all covers."""
    began = time.monotonic(); c = context(object_name); folder = c.folder.parent/FOLDER; folder.mkdir(parents=True, exist_ok=True)
    covers, _ = rank(c.folder/'sets.json', len(c.base.ids), rho=policy['rho'])
    fewest_seats = min(len(cv['seats']) for cv in covers)
    covers = [cv for cv in covers if len(cv['seats']) == fewest_seats][:policy['covers']]
    finals, log = [], []
    with pool(workers) as executor:
        def step(n, A, B):
            accepted, stats = merge(object_name, A, B, folder/f'cover{n}', workers, policy, executor)
            log.append(dict(cover=n, merge=[A['name'], B['name']], **stats))
            return [dict(as_group(r, A['name']+'+'+B['name'], [A, B]), seats=A.get('seats', [A['name']])+[B['name']]) for r in accepted]
        for n, cover in enumerate(covers):
            seats = {g['name']: g for g in (seat_group(c, T, a) for T, a in zip(cover['seats'], cover['angles']))}
            if len(seats) == 1: finals += [dict(g, cover=n, footprint_cm2=float('inf')) for g in seats.values()]; continue
            beam = best_first(c, [g for A, B in itertools.combinations(seats.values(), 2) for g in step(n, A, B)])[:policy['keep']]
            while beam and len(beam[0]['seats']) < len(seats):
                beam = best_first(c, [g for A in beam for name in seats if name not in A['seats'] for g in step(n, A, seats[name])])
                unique = {}
                for g in beam: unique.setdefault(tuple(sorted(g['seats'])+[round(g['footprint_cm2'], 3)]), g)
                beam = list(unique.values())[:policy['keep']]
            finals += [dict(g, cover=n) for g in beam]
            write_json(folder/'search.json', dict(log=log, finals=finals, seconds=time.monotonic()-began))
    if not finals: raise ValueError('No cover merges into one support')
    best = best_first(c, finals)[0]
    report = dict(status='merged', object=object_name, policy=policy, cover=covers[best['cover']], group=best,
                  poses=len(downs(c, best)), finals=finals, log=log, seconds=time.monotonic()-began)
    write_json(folder/'report.json', report)
    print('STEP4 MERGED', best['name'], 'poses', report['poses'], 'footprint', round(best['footprint_cm2']), 'cm2, membrane',
          round(best['volume_cm3'], 1), 'cm3', flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--object', default='B'); p.add_argument('--workers', type=int, default=20)
    p.add_argument('--covers', type=int, default=POLICY['covers'])
    a = p.parse_args(); run(a.object, a.workers, dict(POLICY, covers=a.covers))
