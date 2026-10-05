"""Two operations: nearest physical-head union, exit-driven CSG removal.

Independent DSL initialization is immutable; no head-center or direction-first
head search. Force generators are rebuilt after real surface cuts. Numerical
gradients use actual continuous-sweep collision volumes, not area rewards.
All writes are under THIS directory; DSL_algo is a read-only dependency.
"""
import argparse
import contextlib
from dataclasses import dataclass, replace
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import manifold3d as md
import numpy as np
import trimesh
from scipy.spatial import cKDTree, ConvexHull

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'DSL_algo'
sys.path.append(str(OLD))
from step3_scheculer import contacts as I, operation_dsl as F
from step3_scheculer import shared_direction_paths as P, additive_gpu_v24 as GPU
from step3_scheculer.initialize_current_v16 import current_task
from step3_scheculer.feasible_seating import restore
from step3_scheculer.shared_direction_search import bounded_lps
from step3_scheculer.operation_growth_recovery import RecoveryGrow
from step3_scheculer import build_shared_fixture_v35 as BUILD
from step3_scheculer.compare_shared_volume_v33 import GROUPS
from step4_connect_support import working_surface as WS
from step2_local_support import geometry as G, surface as SURF
from step5_current.evaluate import measure

TOL = 8e-14


def volume(s):
    return abs(float(s.volume()))*F.S.SCALE**3


def nonempty_parts(s):
    # Negative oriented shells enclose voids, not separate pieces of material.
    return [p for p in s.decompose() if float(p.volume())*F.S.SCALE**3 > 1e-16]


def cube(center, radius):
    return md.Manifold.cube([2*radius/F.S.SCALE]*3).translate(
        (np.asarray(center)-radius)/F.S.SCALE)


def planes(center, radius):
    return [(sign*np.eye(3)[i], sign*center[i]+radius)
            for i in range(3) for sign in (-1., 1.)]


def outside_convex(poly, constraints):
    """Partition a polygon minus a convex cutter, retaining holes/disjoint parts."""
    remaining = np.asarray(poly); result = []
    for normal, bound in constraints:
        if len(remaining) < 3:
            break
        values = remaining@normal-bound
        if values.max() <= 1e-12:
            continue
        if values.min() >= -1e-12:
            result.append(remaining); remaining = np.empty((0, 3)); break
        outside = G.clip_plane(remaining, np.r_[-normal, bound])
        if len(outside) >= 3:
            result.append(outside)
        remaining = G.clip_plane(remaining, np.r_[normal, -bound])
    return result


def cut_contact(contact, basis, offset, cutters):
    triangles = []; faces = []
    for tri, face in zip(contact['triangles_m'], contact['source_faces']):
        pieces = [tri@basis+offset]
        for center, radius in cutters:
            if radius > 1e-10:
                pieces = [q for p in pieces for q in outside_convex(p, planes(center, radius))]
        for poly in pieces:
            native = (poly-offset)@basis.T
            # Vertex fan: unchanged triangles must remain ONE triangle.
            # SURF.fan inserts a centroid and triples them on every edit.
            fan = np.asarray([[native[0], native[j], native[j+1]]
                              for j in range(1, len(native)-1)])
            good = SURF.areas(fan) > 1e-18
            triangles.extend(fan[good]); faces.extend([int(face)]*int(good.sum()))
    if not triangles:
        return None
    c = {k: v for k, v in contact.items() if k not in ('wrench_generators', 'wrench_com_m')}
    c.update(triangles_m=np.asarray(triangles), source_faces=np.asarray(faces, int))
    c['triangle_areas_m2'] = SURF.areas(c['triangles_m'])
    c['center_is_historical_reference'] = True
    return c


def name(task, contact):
    return task.pose+':'+contact['candidate_id']


def contact_cells(task, c, unit_offsets):
    """Convex face polygons when possible, triangles when a face has holes.

    Taking the convex hull of ALL surviving vertices would refill deleted holes.
    Area comparison prevents that, while avoiding unnecessary initial fan cells.
    """
    cells = []
    for face in np.unique(c['source_faces']):
        tris = c['triangles_m'][c['source_faces'] == face]
        points = np.unique(tris.reshape(-1, 3), axis=0)
        source = task.domain.mesh.triangles[face]
        x = source[1]-source[0]; x /= np.linalg.norm(x)
        y = np.cross(task.domain.mesh.face_normals[face], x)
        uv = np.c_[points@x, points@y]
        hull = ConvexHull(uv); polygon = points[hull.vertices]
        target_area = float(SURF.areas(tris).sum())
        hull_area = float(SURF.areas(SURF.fan(polygon)).sum())
        polygons = [polygon] if abs(hull_area-target_area) <= max(1e-16, 1e-9*target_area) else tris
        cells.extend(G.head_cell(task.domain.mesh, p, int(face),
                                c['head_depth_m']*unit_offsets) for p in polygons)
    return cells


def subtract_cell_box(cell, center, radius):
    cell = np.asarray(cell)
    # A disjoint box must not partition this cell into unrelated pieces.
    if np.any(cell.max(0) <= np.asarray(center)-radius+1e-12) or np.any(cell.min(0) >= np.asarray(center)+radius-1e-12):
        return [cell]
    remaining = F.S.solid(G.hull_mesh(cell)); result = []
    for normal, bound in planes(center, radius):
        if volume(remaining) <= 1e-16:
            break
        vertices = F.S.unpack(remaining).vertices; values = vertices@normal-bound
        if values.max() <= 1e-12:
            continue
        if values.min() >= -1e-12:
            result.append(vertices); remaining = md.Manifold(); break
        outside = remaining.trim_by_plane(normal.tolist(), bound/F.S.SCALE)
        if volume(outside) > 1e-16:
            result.append(F.S.unpack(outside).vertices)
        remaining = remaining.trim_by_plane((-normal).tolist(), -bound/F.S.SCALE)
    return result


@dataclass(frozen=True)
class Head:
    ident: str
    members: tuple
    solid: object
    raw: object
    cutters: tuple = ()
    parents: tuple = ()


def cut_head(head, cutters):
    solid = head.solid
    for center, radius in cutters:
        if radius > 1e-10:
            solid = solid-cube(center, radius)
    return replace(head, solid=solid, cutters=head.cutters+tuple(cutters))


def merge_head(a, b):
    cloud = np.vstack([F.S.unpack(h.solid).vertices for h in (a, b)])
    connector = F.S.solid(G.hull_mesh(cloud))
    raw = a.solid+b.solid+connector
    return Head(a.ident+'_'+b.ident, a.members+b.members, raw, raw, (), (a.ident, b.ident))


def compact_cost(heads, scale, weight):
    spans = [float(np.linalg.norm(F.S.unpack(h.solid).extents)) for h in heads]
    return len(heads)+weight*sum((d/scale)**2 for d in spans), spans


def direction_value(directions, mask):
    mask = np.asarray(mask, bool); directions = np.asarray(directions)
    if not mask.any(1).all():
        return dict(valid=False, common_count=int(mask.all(0).sum()))
    cosines = directions@directions.T
    angles = np.array([np.arccos(np.clip(cosines[:, row].max(1), -1, 1)) for row in mask])
    means = angles.mean(0); maxima = angles.max(0)
    reference = min(range(len(directions)), key=lambda j: (round(float(means[j]), 12), float(maxima[j]), j))
    selected = [int(np.flatnonzero(row)[np.argmax(cosines[reference, row])]) for row in mask]
    return dict(valid=True, common_count=int(mask.all(0).sum()), reference_index=reference,
                reference_world_xyz=directions[reference].tolist(), selected_indices=selected,
                mean_target_angle_deg=float(np.rad2deg(means[reference])),
                max_target_angle_deg=float(np.rad2deg(maxima[reference])))


class PhysicalGrow(RecoveryGrow):
    """Grow support ports from retained head surfaces, preserving every CSG cut."""
    def __init__(self, *args, physical_heads=(), port_witnesses=None, **kwargs):
        self.port_witnesses = port_witnesses or {}
        self.physical_heads = physical_heads
        super().__init__(*args, **kwargs)
        self.removed = F.F.union([h.raw for h in physical_heads])-F.F.union([h.solid for h in physical_heads])
        actual = F.F.union([h.solid for h in physical_heads])
        # Padded sweeps guide NEW branches. The edited heads themselves are
        # contact exceptions only after exact raw sweep and floor certificates.
        vertices = F.S.unpack(actual).vertices
        for b, o in zip(self.bases, self.offsets):
            if ((vertices-o)@b.T)[:, 2].min() < -1e-9:
                raise RuntimeError('Edited head penetrates an installed floor')
        for mesh, b, o in zip(self.sweep_meshes, self.bases, self.offsets):
            if volume(actual^F.F.transform(F.S.solid(mesh), b, o)) > TOL:
                raise RuntimeError('Edited whole head blocks selected full exit')
        self.mask = (self.mask-self.removed)+actual

    def legal(self, solid):
        return super().legal(solid) and volume(solid^self.removed) <= TOL

    def roots(self):
        began = time.monotonic()
        self.original_roots = [h.solid for h in self.physical_heads]
        for h in self.physical_heads:
            mesh = F.S.unpack(h.solid)
            faces = np.argsort(-mesh.area_faces)[:128]
            attached = False
            witness = self.port_witnesses.get(h.ident)
            if witness is not None:
                ball, extra, start = witness['ball'], witness['extra'], witness['start']
                seed = witness['seed'] - self.removed
                if self.legal(ball) and volume(extra-self.mask) <= TOL and volume(seed ^ self.removed) <= TOL and len(nonempty_parts(seed)) == 1:
                    index = len(self.seed_positions); self.seed_positions.append(start)
                    self.terminals.append(dict(name=h.ident, node=index, solid=seed, kind='physical_head'))
                    self.core_solids.extend([(h.ident+':retained_head', h.solid), (h.ident+':start_ball', ball)])
                    self.thickness.append(dict(name=h.ident, start_m=start.tolist(), guaranteed_core_diameter_mm=self.guaranteed_radius*2000, transition='retained initial support attachment'))
                    continue
            for face in faces:
                tri = mesh.triangles[face]; normal = mesh.face_normals[face]
                for depth in (.0036, .0061, .0101, .0151):
                    start = tri.mean(0)+depth*normal; ball = self.sphere(start)
                    if not self.legal(ball):
                        continue
                    transition = F.S.solid(G.hull_mesh(np.vstack([tri, start+self.bead])))^self.mask
                    seed = h.solid+transition+ball
                    if len(nonempty_parts(seed)) != 1 or volume(seed^self.removed) > TOL:
                        continue
                    index = len(self.seed_positions); self.seed_positions.append(start)
                    self.terminals.append(dict(name=h.ident, node=index, solid=seed, kind='physical_head'))
                    self.core_solids.extend([(h.ident+':retained_head', h.solid), (h.ident+':start_ball', ball)])
                    self.thickness.append(dict(name=h.ident, start_m=start.tolist(),
                        guaranteed_core_diameter_mm=self.guaranteed_radius*2000,
                        original_contact_transition_length_mm=depth*1000,
                        transition_normal='actual retained physical-head face', transition_may_taper=True))
                    attached = True; break
                if attached:
                    break
            if not attached:
                raise RuntimeError('No cut-preserving full-core support port for '+h.ident)
        self.timings['contact_starts'] = time.monotonic()-began
        self.save_stage('roots', F.F.union(self.original_roots))


class Solver:
    def __init__(self, group, config):
        self.group = group; self.config = config; self.started = time.monotonic()
        self.base = HERE/'output/B'/group
        self.out = self.base/'step3_scheculer'/config['stage']; self.out.mkdir(parents=True, exist_ok=True)
        self.tasks = [current_task('B', 'pose_'+p) for p in group.removeprefix('pose').split('+')]
        self.inputs = []; rows = []; paths = []
        for task in self.tasks:
            source = OLD/'output/B/independent_poses_cached_v17'/task.pose/'step3_scheculer/schedule.json'
            r = I.check_report(source)
            if not r['passed']:
                raise RuntimeError('Independent initialization failed: '+task.pose)
            cp = source.parent/f'final_contacts_{task.pose}.npz'
            rows.append(tuple(I.read_contacts(cp))); paths.append(F.ray(-np.asarray(r['result']['object_exit_world'])))
            self.inputs += [source, cp]+task.inputs
        self.seed = F.State(tuple(rows), np.repeat(np.eye(3)[None], len(rows), axis=0),
                            np.zeros((len(rows), 3)), tuple(paths))
        self.backends = [GPU.GPUClassifier(t.targets, config['device']) for t in self.tasks]
        self.force_cache = {}; self.sweeps = {}; self.events = []
        self.unit_offsets = [G.vertex_offsets(t.domain.mesh, 1.)[0] for t in self.tasks]
        n = config['directions']; k = np.arange(n); z = (k+.5)/n; az = k*np.pi*(3-np.sqrt(5))
        sphere = np.c_[np.sqrt(1-z*z)*np.cos(az), np.sqrt(1-z*z)*np.sin(az), z]
        az = np.arange(12)*np.pi/6
        self.directions = np.unique(np.round(np.vstack([sphere, np.c_[np.cos(az), np.sin(az), np.zeros(12)],
            [0, 0, 1], [p['initial_object_exit_world'] for p in paths]]), 12), axis=0)

    def loads(self, state, cpu=False):
        rows = []
        for i, (task, contacts) in enumerate(zip(self.tasks, state.groups)):
            digest = hashlib.sha256(b''.join(c['triangles_m'].tobytes()+c['source_faces'].tobytes() for c in contacts)).hexdigest()
            cache = i, digest, cpu
            if cache not in self.force_cache:
                full = task.supply([{k: v for k, v in c.items() if k not in ('wrench_generators', 'wrench_com_m')} for c in contacts])
                mask, info = F.J.classify(full, task.targets) if cpu else self.backends[i].classify(full)
                self.force_cache[cache] = dict(pose=task.pose, passed=bool(mask.all()),
                                               covered=int(mask.sum()), total=len(mask), info=info)
            rows.append(self.force_cache[cache])
        return all(r['passed'] for r in rows), rows

    def make_heads(self, state):
        rows = []; heads = []
        for task, contacts, b, o, unit in zip(self.tasks, state.groups, state.bases, state.offsets, self.unit_offsets):
            updated = []
            for original in contacts:
                c = dict(original)
                points = c['triangles_m'].reshape(-1, 3)@b+o
                height = min(float(((points-oj)@bj.T)[:, 2].min()) for bj, oj in zip(state.bases, state.offsets))
                if height <= 1e-10:
                    raise RuntimeError('Native contact has no all-floor clearance')
                ids = np.unique(task.domain.mesh.faces[np.unique(c['source_faces'])])
                ratio = float(np.linalg.norm(unit[ids], axis=1).max())
                c['head_depth_m'] = min(F.ROOT_DEPTH, .5*height/ratio)
                cells = contact_cells(task, c, unit)
                solid = F.F.union([F.S.solid(G.hull_mesh(v@b+o)) for v in cells])
                if len(nonempty_parts(solid)) != 1:
                    raise RuntimeError('Independent head is disconnected')
                heads.append(Head('H'+str(len(heads)).zfill(3), (name(task, c),), solid, solid))
                updated.append(c)
            rows.append(tuple(updated))
        merged = []
        for head in heads:
            changed = True
            while changed:
                changed = False
                for j, other in enumerate(merged):
                    if len(nonempty_parts(head.solid+other.solid)) == 1:
                        joined = head.solid+other.solid
                        head = Head(other.ident+'_'+head.ident, other.members+head.members, joined, joined)
                        merged.pop(j); changed = True; break
            merged.append(head)
        return replace(state, groups=tuple(rows)), merged

    def sweep(self, i, j):
        if (i, j) not in self.sweeps:
            task = self.tasks[i]; b = self.state.bases[i]; o = self.state.offsets[i]
            self.sweeps[i, j] = F.F.transform(F.S.solid(P.sweep_mesh(task.domain.mesh,
                F.ray(-self.directions[j], j))), b, o)
        return self.sweeps[i, j]

    def readout(self, heads, solid=None):
        solid = F.F.union([h.solid for h in heads]) if solid is None else solid
        mesh = F.S.unpack(solid); overlaps = np.zeros((len(self.tasks), len(self.directions)))
        mask = np.zeros_like(overlaps, bool)
        for i, (task, b, o) in enumerate(zip(self.tasks, self.state.bases, self.state.offsets)):
            native = (mesh.vertices-o)@b.T
            for j, d in enumerate(self.directions):
                overlap = volume(solid^self.sweep(i, j)); overlaps[i, j] = overlap
                end = task.domain.mesh.bounds+.5*d
                separated = bool(np.any(end[0] > native.max(0)+1e-10) or np.any(end[1] < native.min(0)-1e-10))
                mask[i, j] = overlap <= TOL and separated and d[2] >= -1e-10
        value = direction_value(self.directions, mask)
        return dict(directions_world_xyz=self.directions.tolist(), per_pose_mask=mask.tolist(),
            collision_volume_m3=overlaps.tolist(), per_pose_count=mask.sum(1).tolist(),
            common_mask=mask.all(0).tolist(), **value, continuous_sweep_length_m=.5,
            finite_readout_is_not_all_angles=True)

    def geometry(self, heads, state):
        mesh = F.S.unpack(F.F.union([h.solid for h in heads])); rows = []
        for task, b, o in zip(self.tasks, state.bases, state.offsets):
            native = trimesh.Trimesh((mesh.vertices-o)@b.T, mesh.faces, process=False)
            minimum = float(native.vertices[:, 2].min())
            work = WS.check(native, task)
            rows.append(dict(pose=task.pose, minimum_floor_height_m=minimum,
                             work_surface=work, passed=minimum >= -1e-9 and work['passed']))
        return all(r['passed'] for r in rows), rows

    def initialize(self):
        candidates = []
        for stage in ('shared_fixture_seating_v35', 'shared_fixture_seating_v34', ''):
            path = OLD/'output/B'/self.group/'step4'/stage/'report.json'
            if path.exists():
                record = json.loads(path.read_text()); placement = record.get('placement', {})
                if len(placement.get('bases', [])) == len(self.tasks):
                    self.inputs.append(path)
                    historical = replace(self.seed, bases=np.array(placement['bases']), offsets=np.array(placement['offsets']))
                    candidates.append(('historical_placement_only', historical)); break
        candidates.append(('identity_placement', self.seed))
        attempts = []
        for label, state in candidates:
            for repair in (-1, 0, 1):
                proposed = state
                try:
                    if repair >= 0:
                        result = restore(self.tasks, state, F.roots, attempt=repair, iterations=32)
                        if result is None:
                            attempts.append(dict(label=label, repair=repair, passed=False, error='Bounded seating LP repair failed')); continue
                        proposed = result[0]
                    proposed, heads = self.make_heads(proposed)
                    self.state = proposed; self.sweeps.clear()
                    ok, checks = self.loads(proposed)
                    sets = self.readout(heads)
                    legal, geometry = self.geometry(heads, proposed) if sets['valid'] else (False, [])
                    passed = ok and legal and sets['valid']
                    attempts.append(dict(label=label, repair=repair, passed=passed,
                                         sets=sets, loads=checks, geometry=geometry))
                    if passed:
                        self.heads = heads; self.sets = sets
                        self.initial_heads = list(heads); self.initial_state = proposed; self.initial_sets = sets
                        self.typical_span = float(np.median([np.linalg.norm(F.S.unpack(h.solid).extents) for h in heads]))
                        self.source = self.out/'initialization.json'
                        I.save(self.source, dict(complete=True, passed=True, attempts=attempts,
                            independent_contacts_unchanged=True, placement_frozen_after_initialization=True,
                            provenance=dict(inputs=I.hashes(self.inputs), code=I.hashes([Path(__file__)]))))
                        self.save_head_state(self.out/'initial', heads, proposed, sets, checks)
                        return
                except (RuntimeError, ValueError) as error:
                    attempts.append(dict(label=label, repair=repair, passed=False, error=str(error)))
        I.save(self.out/'initialization_failure.json', dict(complete=True, passed=False, attempts=attempts))
        raise RuntimeError('Bounded installation initialization failed; no heads were re-searched')

    def nearest(self):
        result = []
        for i, a in enumerate(self.heads):
            vertices = F.S.unpack(a.solid).vertices
            for j in range(i+1, len(self.heads)):
                distance = float(cKDTree(vertices).query(F.S.unpack(self.heads[j].solid).vertices)[0].min())
                result.append((distance, i, j))
        return sorted(result, key=lambda p: (p[0], self.heads[p[1]].ident, self.heads[p[2]].ident))

    def cut_state(self, state, head, cutters):
        rows = []
        for task, contacts, b, o in zip(self.tasks, state.groups, state.bases, state.offsets):
            row = []
            for c in contacts:
                q = cut_contact(c, b, o, cutters) if name(task, c) in head.members else c
                if q is not None:
                    row.append(q)
            rows.append(tuple(row))
        return replace(state, groups=tuple(rows))

    def refine_directions(self):
        """Adapt the checked world directions between current feasible witnesses."""
        selected = self.directions[self.sets['selected_indices']]
        reference = self.directions[self.sets['reference_index']]
        pairs = [(reference, d) for d in selected]
        pairs += [(a,b) for i,a in enumerate(selected) for b in selected[i+1:]]
        extra = []
        for a,b in pairs:
            for t in (.25,.5,.75):
                d = (1-t)*a+t*b; norm = np.linalg.norm(d)
                if norm < 1e-10:
                    continue
                d = d/norm
                existing = np.vstack([self.directions]+([np.asarray(extra)] if extra else []))
                if d[2] >= -1e-10 and np.max(existing@d) < np.cos(np.deg2rad(.1)):
                    extra.append(d)
                if len(extra) >= 18:
                    break
            if len(extra) >= 18:
                break
        if extra:
            self.directions = np.vstack([self.directions, extra])
            self.sets = self.readout(self.heads)
            print('REFINE', self.group, 'checked_directions',len(self.directions),
                  'common',self.sets['common_count'], flush=True)

    def release_all(self):
        """Cut physical material once and update every affected pose contact."""
        denominator = np.array([abs(t.domain.mesh.volume) for t in self.tasks])
        for step in range(self.config['gradient_steps']):
            if step in (0,8,16):
                self.refine_directions()
            if self.sets['common_count']:
                break
            overlap = np.asarray(self.sets['collision_volume_m3'])
            scores = np.max(overlap/denominator[:, None], axis=0)
            eligible = np.flatnonzero(self.directions[:, 2] >= -1e-10)
            target = int(eligible[np.argmin(scores[eligible])])
            physical = F.F.union([h.solid for h in self.heads])
            obstruction = F.F.union([physical ^ self.sweep(i,target) for i in range(len(self.tasks))])
            if volume(obstruction) <= TOL:
                self.events.append(dict(operation='global_exit_release',index=step,accepted=False,
                    reason='No volumetric blocker: terminal separation or numerical tolerance prevents this checked ray'))
                break
            mesh = F.S.unpack(obstruction)
            centers = [mesh.triangles_center[f] for f in np.argsort(-mesh.area_faces)[:self.config['cutters']]]
            accepted = False; trials = []
            for center in centers:
                def loss(radius):
                    solid = physical if radius <= 1e-10 else physical-cube(center,radius)
                    values = np.array([volume(solid ^ self.sweep(i,target))/denominator[i]
                                       for i in range(len(self.tasks))])
                    return float(values.max()+.1*values.mean())
                current = loss(0.); eps = self.config['difference_step']
                gradient = (loss(eps)-current)/eps
                if gradient >= -1e-12:
                    trials.append(dict(gradient=gradient,reason='no_descent'));continue
                for bt in range(self.config['backtracks']):
                    radius = self.config['cut_step']*.5**bt; value = loss(radius)
                    if value >= current-1e-12:
                        continue
                    cuts = ((center,radius),)
                    all_members = tuple(m for h in self.heads for m in h.members)
                    state = self.cut_state(self.state,Head('physical_union',all_members,physical,physical),cuts)
                    members = {name(t,c) for t,row in zip(self.tasks,state.groups) for c in row}
                    proposed = []
                    for h in self.heads:
                        edited = cut_head(h,cuts)
                        edited = replace(edited,members=tuple(m for m in h.members if m in members))
                        if edited.members:
                            proposed.append(edited)
                    connected = bool(proposed) and all(len(nonempty_parts(h.solid))==1 for h in proposed)
                    proposed_union = F.F.union([h.solid for h in proposed])
                    removed = F.F.union([h.raw for h in self.initial_heads])-proposed_union
                    port_preserved = True
                    for h in proposed:
                        w = self.port_witnesses.get(h.ident)
                        if w is not None:
                            port_preserved &= volume(w['ball'] ^ removed) <= TOL and len(nonempty_parts(w['seed']-removed))==1
                    loads_ok, checks = self.loads(state)
                    sets = self.readout(proposed) if connected and loads_ok and port_preserved else None
                    legal, geometry = self.geometry(proposed,state) if sets and sets['valid'] else (False,[])
                    entry = dict(center_fixture_m=center.tolist(),radius_m=radius,gradient=gradient,
                        difference_step_m=eps,backtrack=bt,loss_before=current,loss_after=value,
                        connected=connected,support_attachment_preserved=bool(port_preserved),
                        all_loads_passed=loads_ok,load_checks=checks,
                        each_pose_has_exit=bool(sets and sets['valid']),geometry_passed=legal)
                    trials.append(entry)
                    if not legal:
                        continue
                    before = self.sets; count = len(self.heads)
                    self.heads,self.state,self.sets = proposed,state,sets
                    event = dict(operation='global_exit_release',index=step,accepted=True,
                        reference_world_xyz=self.directions[target].tolist(),exits_before=before,
                        exits_after=sets,trials=trials,heads_before=count,heads_after=len(proposed),
                        same_physical_cut_applied_to_all_contacts=True)
                    self.events.append(event)
                    self.save_head_state(self.out/f'release_{step:03d}',proposed,state,sets,checks)
                    I.save(self.out/f'release_{step:03d}'/'event.json',event)
                    print('RELEASE',self.group,step,'physical_cut','loss',current,value,
                          'common',sets['common_count'],flush=True)
                    accepted=True;break
                if accepted:
                    break
            if not accepted:
                event=dict(operation='global_exit_release',index=step,accepted=False,
                    reference_world_xyz=self.directions[target].tolist(),trials=trials,
                    reason='No legal physical-union descent in this finite parameter budget')
                self.events.append(event);I.save(self.out/f'release_rejected_{step:03d}.json',event);break

    def descent(self, raw, others, index):
        target = self.sets['reference_index']
        target_ids = [target]+[j for j in np.argsort(-(self.directions@self.directions[target]))
                               if j != target][:1]
        fixed = F.F.union([h.solid for h in others])
        denominator = np.array([abs(t.domain.mesh.volume) for t in self.tasks])
        fixed_overlap = np.array([[volume(fixed^self.sweep(i, j)) for j in target_ids]
                                  for i in range(len(self.tasks))])
        collision = F.F.union([raw.solid^self.sweep(i, j) for i in range(len(self.tasks)) for j in target_ids])
        if volume(collision) <= TOL:
            return raw, self.state, []
        mesh = F.S.unpack(collision); order = np.argsort(-mesh.area_faces)
        centers = [mesh.center_mass]; gap = max(.002, self.typical_span*.15)
        for face in order:
            point = mesh.triangles_center[face]-.0001*mesh.face_normals[face]
            if all(np.linalg.norm(point-q) >= gap for q in centers):
                centers.append(point)
            if len(centers) >= self.config['cutters']:
                break
        centers = np.asarray(centers); radii = np.zeros(len(centers)); history = []
        def build(x):
            return cut_head(raw, tuple((c, float(r)) for c, r in zip(centers, x)))
        def loss(x):
            head = build(x); exclusive = head.solid-fixed
            overlap = fixed_overlap+np.array([[volume(exclusive^self.sweep(i, j)) for j in target_ids]
                                             for i in range(len(self.tasks))])
            worst = np.max(overlap/denominator[:, None], axis=0)
            temperature = .01
            minimum = float(worst.min())
            return minimum-temperature*np.log(np.exp(-(worst-minimum)/temperature).sum())
        current = loss(radii); best_head = raw; best_state = self.state
        for step in range(self.config['gradient_steps']):
            eps = self.config['difference_step']; gradient = np.array([
                (loss(radii+eps*np.eye(len(radii))[j])-current)/eps for j in range(len(radii))])
            direction = np.maximum(-gradient, 0.)
            if direction.max() <= 1e-12:
                break
            direction /= direction.max(); accepted = False
            for backtrack in range(self.config['backtracks']):
                rate = self.config['cut_step']*.5**backtrack
                trial = radii+rate*direction; nextloss = loss(trial)
                if nextloss >= current-1e-10:
                    continue
                candidate = build(trial); state = self.cut_state(self.state, raw,
                    tuple((c, float(r)) for c, r in zip(centers, trial)))
                ok, checks = self.loads(state); connected = len(nonempty_parts(candidate.solid)) == 1
                entry = dict(step=step, backtrack=backtrack, gradient=gradient.tolist(),
                    finite_difference_step_m=eps, rate_m=rate, centers_fixture_m=centers.tolist(),
                    cut_halfwidths_m=trial.tolist(), collision_loss_before=current, collision_loss_after=nextloss,
                    all_loads_passed=ok, load_checks=checks, connected=connected, accepted=bool(ok and connected),
                    actual_head_volume_m3=volume(candidate.solid), contact_area_m2=I.area([c for row in state.groups for c in row]))
                history.append(entry)
                if entry['accepted']:
                    best_head = candidate; best_state = state; radii = trial; current = nextloss; accepted = True
                    folder = self.out/'proposals'/f'{index:03d}'; folder.mkdir(parents=True, exist_ok=True)
                    F.SPACE.export_exact_obj(F.S.unpack(candidate.solid), folder/f'cut_{step:02d}.obj')
                    break
            if not accepted:
                break
        return best_head, best_state, history

    def candidate(self, pair, index):
        distance, ai, bi = pair; a, b = self.heads[ai], self.heads[bi]
        raw = merge_head(a, b); others = [h for j, h in enumerate(self.heads) if j not in (ai, bi)]
        folder = self.out/'proposals'/f'{index:03d}'; folder.mkdir(parents=True, exist_ok=True)
        F.SPACE.export_exact_obj(F.S.unpack(raw.solid), folder/'raw_merge.obj')
        # Original contact generators are UNCHANGED by additive union. Only
        # post-cut geometry gets new generators and original-load acceptance.
        reduced, state, history = self.descent(raw, others, index)
        remaining = {name(t, c) for t, row in zip(self.tasks, state.groups) for c in row}
        reduced = replace(reduced, members=tuple(m for m in reduced.members if m in remaining))
        proposed = others+([reduced] if reduced.members else [])
        sets = self.readout(proposed); ok, checks = self.loads(state)
        legal, geometry = self.geometry(proposed, state) if sets['valid'] else (False, [])
        before_cost, before_spans = compact_cost(self.heads, self.typical_span, self.config['span_weight'])
        after_cost, after_spans = compact_cost(proposed, self.typical_span, self.config['span_weight'])
        direction_ok = sets['valid'] and sets['mean_target_angle_deg'] <= self.sets['mean_target_angle_deg']+1e-8 \
            and sets['max_target_angle_deg'] <= self.sets['max_target_angle_deg']+1e-8 \
            and sets['common_count'] >= self.sets['common_count']
        improves = direction_ok and (sets['mean_target_angle_deg'] < self.sets['mean_target_angle_deg']-1e-8
            or sets['common_count'] > self.sets['common_count'] or after_cost < before_cost-1e-10)
        accepted = bool(ok and legal and direction_ok and improves and after_cost <= before_cost+1e-10)
        event = dict(index=index, pair=[a.ident, b.ident], distance_m=distance, accepted=accepted,
            force_preserved_by_raw_union=True, raw_exit_improvement_required=False,
            raw_head_volume_m3=volume(raw.solid), trimmed_head_volume_m3=volume(reduced.solid),
            heads_before=len(self.heads), heads_after=len(proposed), compact_cost_before=before_cost,
            compact_cost_after=after_cost, spans_before_m=before_spans, spans_after_m=after_spans,
            exits_before=self.sets, exits_after=sets, gradient_history=history,
            all_loads_passed=ok, load_checks=checks, geometry_passed=legal, geometry_checks=geometry,
            direction_nonregression=bool(direction_ok), compactness_nonregression=after_cost <= before_cost+1e-10)
        self.events.append(event); I.save(folder/'event.json', event)
        if accepted:
            self.heads = proposed; self.state = state; self.sets = sets
            self.save_head_state(self.out/f'accepted_{index:03d}', proposed, state, sets, checks)
        print('MERGE', self.group, index, event['pair'], 'accepted', accepted,
              'heads', len(self.heads), 'angle', sets.get('mean_target_angle_deg'), flush=True)
        return accepted

    def save_head_state(self, folder, heads, state, sets, checks):
        folder.mkdir(parents=True, exist_ok=True)
        for task, row in zip(self.tasks, state.groups):
            I.save_contacts(folder/f'contacts_{task.pose}.npz', row)
        physical = []
        arrays = {}
        for i, h in enumerate(heads):
            m = F.S.unpack(h.solid); raw = F.S.unpack(h.raw)
            arrays.update({f'h{i}_vertices':m.vertices, f'h{i}_faces':m.faces,
                           f'h{i}_raw_vertices':raw.vertices, f'h{i}_raw_faces':raw.faces})
            owners = sorted({member.split(':')[0] for member in h.members})
            physical.append(dict(id=h.ident, members=list(h.members), owner_poses=owners,
                parents=list(h.parents), volume_m3=volume(h.solid), span_m=float(np.linalg.norm(m.extents)),
                cutters=[dict(center_fixture_m=c.tolist(), halfwidth_m=r) for c, r in h.cutters]))
        np.savez_compressed(folder/'head_geometry.npz', **arrays)
        F.SPACE.export_exact_obj(F.S.unpack(F.F.union([h.solid for h in heads])), folder/'heads.obj')
        I.save(folder/'state.json', dict(complete=True, physical_heads=physical, physical_head_count=len(heads),
            contact_region_count=sum(map(len, state.groups)), poses=[t.pose for t in self.tasks],
            placement=dict(bases=state.bases.tolist(), offsets=state.offsets.tolist()), load_checks=checks,
            source_centers_are_references_only_after_cut=True, exits=sets))
        I.save(folder/'exit_sets.json', sets)
        self.picture(folder/'overview.png', heads, state, sets)

    def picture(self, path, heads, state, sets):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        fig = plt.figure(figsize=(13, 6)); cmap = plt.get_cmap('tab20')
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        for i, h in enumerate(heads):
            mesh = F.S.unpack(h.solid)
            ax.add_collection3d(Poly3DCollection(mesh.triangles, facecolor=cmap(i%20), alpha=.95, edgecolor='none'))
        task = self.tasks[0]; b, o = state.bases[0], state.offsets[0]
        triangles = task.domain.mesh.triangles@b+o
        ax.add_collection3d(Poly3DCollection(triangles[::3], facecolor='#999999', alpha=.10, edgecolor='none'))
        cloud = np.vstack([triangles.reshape(-1, 3)]+[F.S.unpack(h.solid).vertices for h in heads])
        lo, hi = cloud.min(0), cloud.max(0); center = (lo+hi)/2; extent = max(hi-lo)*.55
        ax.set_xlim(center[0]-extent, center[0]+extent); ax.set_ylim(center[1]-extent, center[1]+extent)
        ax.set_zlim(center[2]-extent, center[2]+extent); ax.set_box_aspect((1,1,1)); ax.set_axis_off()
        ax.view_init(elev=30, azim=30); ax.set_title(f'{len(heads)} real physical heads; reference object: {task.pose}')
        sphere = fig.add_subplot(1, 2, 2, projection='3d'); mask = np.asarray(sets['per_pose_mask'])
        for i, row in enumerate(mask):
            d = self.directions[row]
            sphere.scatter(d[:,0], d[:,1], d[:,2], color=cmap(i%20), s=25, label=f'{self.tasks[i].pose}: {int(row.sum())}')
        common = self.directions[mask.all(0)]
        if len(common):
            sphere.scatter(common[:,0], common[:,1], common[:,2], color='black', marker='*', s=95, label='Common rays')
        sphere.set_xlim(-1.1,1.1); sphere.set_ylim(-1.1,1.1); sphere.set_zlim(-.1,1.1)
        sphere.set_box_aspect((1,1,1)); sphere.set_title('World XYZ: continuous-sweep checked rays')
        sphere.legend(fontsize=8); sphere.set_xlabel('X'); sphere.set_ylabel('Y'); sphere.set_zlabel('Z')
        fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

    def root_builder(self, tasks, state):
        seeds = []; owned = {}
        for task, row, b, o, unit in zip(tasks, state.groups, state.bases, state.offsets, self.unit_offsets):
            group = []
            for c in row:
                h = next(h for h in self.heads if name(task, c) in h.members)
                pieces = [v@b+o for v in contact_cells(task, c, unit)]
                for center, radius in h.cutters:
                    if radius > 1e-10:
                        pieces = [q for v in pieces for q in subtract_cell_box(v, center, radius)]
                if not pieces:
                    raise RuntimeError('Contact has no positive-volume supporting geometry')
                for v in pieces:
                    if volume(F.S.solid(G.hull_mesh(v))-h.solid) > TOL:
                        raise RuntimeError('Derived contact cell would restore cut material')
                group.append([(v-o)@b.T for v in pieces])
                owned[c['candidate_id']] = (0, pieces, c['triangles_m']@b+o)
            seeds.append(group)
        return seeds, owned, 0.

    def construct(self, heads, state, sets, label):
        folder = self.base/'step4'/self.config['stage']/label; folder.mkdir(parents=True, exist_ok=True)
        mask = np.asarray(sets['per_pose_mask']); common = np.flatnonzero(mask.all(0))
        selections = [[int(j)]*len(self.tasks) for j in common[:2]]
        selections.append(sets['selected_indices'])
        if label != 'initial' and all(mask[i,j] for i,j in enumerate(self.initial_build_indices)):
            selections.append(self.initial_build_indices)
        seen = set(); errors = []
        old_roots, old_analyzer = F.roots, F.E.PathAnalyzer
        old_legacy = F.S.W.Analyzer
        class MetadataAnalyzer(old_legacy):
            def test(self, heads, direction):
                cells = [SimpleNamespace(vertices=h.vertices, metadata=dict(getattr(h,'metadata',{}))) for h in heads]
                return super().test(cells, direction)
        try:
            F.S.W.Analyzer = MetadataAnalyzer
            F.roots = self.root_builder; F.E.PathAnalyzer = P.PathAnalyzer
            for indices in selections:
                if tuple(indices) in seen:
                    continue
                seen.add(tuple(indices)); trial = folder/f'attempt_{len(seen)-1}'; trial.mkdir(exist_ok=True)
                proposed = replace(state, paths=tuple(F.ray(-self.directions[j], j) for j in indices))
                try:
                    F.save_state(trial, self.tasks, proposed, dict(passed=True, scope='fresh original-load proof'))
                    cloud = np.vstack([t.domain.mesh.vertices@b+o for t,b,o in zip(self.tasks,state.bases,state.offsets)])
                    lo, hi = cloud.min(0)-.05, cloud.max(0)+.05
                    guide = trimesh.creation.box(extents=hi-lo, transform=trimesh.transformations.translation_matrix((lo+hi)/2))
                    F.SPACE.export_exact_obj(guide, trial/'navigation_guide.obj')
                    grow = PhysicalGrow(SimpleNamespace(name=self.group), self.tasks, proposed, trial,
                        trial/'navigation_guide.obj', dict(complete=True, passed=False, construction={},
                        provenance=dict(inputs={},code={})), physical_heads=heads, port_witnesses=getattr(self, 'port_witnesses', {}))
                    grow.case.paths.append(self.source); grow.recheck_export = False
                    body = grow.run(.008)
                    ok, checks = self.loads(proposed, cpu=True)
                    if not ok:
                        raise RuntimeError('Full original CPU load check failed')
                    fullsets = self.readout(heads, solid=grow.final_solid)
                    body.update(original_loads_passed=True, load_checks=checks, physical_head_count=len(heads),
                        selected_exit_directions_world_xyz=[self.directions[j].tolist() for j in indices],
                        full_fixture_exit_sets=fullsets, cut_material_restored_m3=volume(grow.final_solid^grow.removed))
                    if body['cut_material_restored_m3'] > TOL:
                        raise RuntimeError('Full constructor restored removed head material')
                    I.save(trial/'report.json', body)
                    return (body, F.S.unpack(grow.final_solid), proposed, grow, trial), errors
                except Exception as error:
                    record = dict(error=str(error), traceback=traceback.format_exc(), indices=indices)
                    errors.append(record); I.save(trial/'failure.json', record)
        finally:
            F.roots = old_roots; F.E.PathAnalyzer = old_analyzer; F.S.W.Analyzer = old_legacy
        return None, errors

    def run(self):
        with bounded_lps():
            self.initialize()
            self.heads = list(self.initial_heads)
            self.initial_build_indices = list(self.initial_sets['selected_indices'])
            initial_body, initial_errors = self.construct(self.heads, self.state, self.sets, 'initial')
            self.port_anchors = {}; self.port_witnesses = {}
            if initial_body is not None:
                grow = initial_body[3]
                for terminal, head in zip(grow.terminals[:len(self.heads)], self.heads):
                    self.port_anchors[head.ident] = terminal['solid'] - head.solid
                    ball = next(core for key, core in grow.core_solids if key == head.ident+':start_ball')
                    start = np.asarray(next(r['start_m'] for r in grow.thickness if r['name'] == head.ident))
                    self.port_witnesses[head.ident] = dict(extra=self.port_anchors[head.ident], ball=ball, start=start, seed=terminal['solid'])
            if self.config['global_release']:
                self.release_all()
            attempted = set()
            for index in range(self.config['merge_trials']):
                choices = [p for p in self.nearest() if (self.heads[p[1]].ident,self.heads[p[2]].ident) not in attempted]
                if not choices:
                    break
                pair = choices[0]; attempted.add((self.heads[pair[1]].ident,self.heads[pair[2]].ident))
                if self.candidate(pair, index):
                    attempted.clear()
            # Compare initialization with the SAME final adaptive direction menu.
            self.initial_sets = self.readout(self.initial_heads)
            ok, checks = self.loads(self.state, cpu=True)
            if not ok:
                raise RuntimeError('Final head program failed full original CPU loads')
            self.save_head_state(self.out/'final', self.heads, self.state, self.sets, checks)
            final_body, errors = self.construct(self.heads, self.state, self.sets, 'final') if self.events and any(e['accepted'] for e in self.events) else (initial_body, initial_errors)
        initial_cost, initial_spans = compact_cost(self.initial_heads, self.typical_span, self.config['span_weight'])
        final_cost, final_spans = compact_cost(self.heads, self.typical_span, self.config['span_weight'])
        owners = [{m.split(':')[0] for m in h.members} for h in self.heads]
        report = dict(complete=True, passed=final_body is not None, head_program_passed=True,
            group=self.group, config=self.config, initial_physical_heads=len(self.initial_heads),
            final_physical_heads=len(self.heads), initial_contact_regions=sum(map(len,self.initial_state.groups)),
            final_contact_regions=sum(map(len,self.state.groups)), multi_pose_modules=sum(len(p)>1 for p in owners),
            initial_contact_area_m2=I.area([c for row in self.initial_state.groups for c in row]),
            final_contact_area_m2=I.area([c for row in self.state.groups for c in row]),
            initial_exits=self.initial_sets, final_head_exits=self.sets,
            initial_compact_cost=initial_cost, final_compact_cost=final_cost,
            initial_spans_m=initial_spans, final_spans_m=final_spans, typical_head_span_m=self.typical_span,
            original_cpu_load_checks=checks, events=self.events, initial_full_fixture_passed=initial_body is not None,
            initial_construction_errors=initial_errors, final_construction_errors=errors,
            contact_centers_moved=False, installation_frozen_during_operations=True,
            gradient='forward numerical derivative of real continuous-sweep CSG collision volumes',
            contact_area_is_objective=False, global_optimum_claim=False, seconds=time.monotonic()-self.started,
            provenance=dict(inputs=I.hashes(self.inputs+[self.source]), code=I.hashes(
                [Path(__file__),Path(F.__file__),Path(P.__file__),Path(GPU.__file__),Path(WS.__file__)])))
        step5 = self.base/'step5_evaluate'/self.config['stage']; step5.mkdir(parents=True,exist_ok=True)
        if final_body is not None:
            body, mesh, state, grow, trial = final_body
            metric, perpose, _ = measure([t.domain.mesh.vertices for t in self.tasks],mesh.vertices,state.bases,state.offsets)
            public = self.base/'step4'/self.config['stage']
            F.SPACE.export_exact_obj(mesh, public/'shape.obj')
            BUILD.pictures(public,step5,self.tasks,state,mesh,grow,metric,self.group)
            report.update(final_full_fixture_exit_sets=body['full_fixture_exit_sets'],
                selected_exit_directions_world_xyz=body['selected_exit_directions_world_xyz'],
                final_material_volume_cm3=body['volume_cm3'],aggregate=metric,per_pose=perpose,
                full_fixture_witness=str((trial/'report.json').relative_to(I.ROOT)))
            if initial_body is not None:
                _, initialmesh, initialstate, _, _ = initial_body
                initialmetric, _, _ = measure([t.domain.mesh.vertices for t in self.tasks],initialmesh.vertices,
                                             initialstate.bases,initialstate.offsets)
                old = initialmetric['object_and_support_poses']['box_volume_cm3']
                new = metric['object_and_support_poses']['box_volume_cm3']
                report.update(initial_aggregate=initialmetric, box_reduction_percent=100*(1-new/old))
        report['seconds'] = time.monotonic()-self.started
        I.save(self.out/'report.json',report); I.save(step5/'report.json',report)
        I.save(self.base/'step4'/self.config['stage']/'report.json',report)
        return {k:report[k] for k in ('group','passed','head_program_passed','initial_physical_heads',
            'final_physical_heads','multi_pose_modules','seconds')} | dict(
            initial_angle_deg=self.initial_sets['mean_target_angle_deg'],final_angle_deg=self.sets['mean_target_angle_deg'],
            initial_common=self.initial_sets['common_count'],final_common=self.sets['common_count'])


def solve(group, config):
    folder = HERE/'output/B'/group/'step3_scheculer'/config['stage']; folder.mkdir(parents=True,exist_ok=True)
    began = time.monotonic()
    with (folder/'pipeline.log').open('w',buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            return Solver(group,config).run()
        except Exception as error:
            result = dict(complete=True,passed=False,head_program_passed=False,group=group,error=str(error),
                          traceback=traceback.format_exc(),seconds=time.monotonic()-began)
            I.save(folder/'failure.json',result); return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--global-release', action='store_true')
    p.add_argument('--sets',nargs='+',default=['pose3+6','pose2+12+15','pose5+7'])
    p.add_argument('--all-sets',action='store_true'); p.add_argument('--jobs',type=int,default=1)
    p.add_argument('--stage',default='merge_release_v1'); p.add_argument('--merge-trials',type=int,default=6)
    p.add_argument('--gradient-steps',type=int,default=6); p.add_argument('--cutters',type=int,default=4)
    p.add_argument('--backtracks',type=int,default=5); p.add_argument('--directions',type=int,default=16)
    p.add_argument('--difference-step',type=float,default=.001); p.add_argument('--cut-step',type=float,default=.006)
    p.add_argument('--span-weight',type=float,default=.08); p.add_argument('--device',default='cuda')
    a = p.parse_args(); config = vars(a).copy()
    if a.all_sets:
        saved=json.loads((I.ROOT/'objects/B/pose_sets.json').read_text())['sets']
        groups=list(dict.fromkeys(GROUPS+[s['id'] for s in saved]))
    else:
        groups=a.sets
    began=time.monotonic(); results=[]
    if a.jobs == 1:
        for group in groups:
            result=solve(group,config);results.append(result);print(json.dumps(result),flush=True)
    else:
        with ProcessPoolExecutor(a.jobs) as pool:
            for future in as_completed([pool.submit(solve,g,config) for g in groups]):
                result=future.result();results.append(result);print(json.dumps(result),flush=True)
    out=HERE/'output/B'/groups[-1]/'step5_evaluate'/a.stage;out.mkdir(parents=True,exist_ok=True)
    I.save(out/'batch.json',dict(complete=True,groups=results,config=config,wall_seconds=time.monotonic()-began))


if __name__ == '__main__':
    main()
