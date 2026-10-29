"""Step3: one independent 5 mm membrane per Step1 task, cut by the tool Step1 certified.

Tasks come from the validated Step1 plan (data/<object>/step1_reach): each input
pose uses its exclusive face_task region, matching the pink faces in Step1.3.
Input poses with no assigned faces need no membrane. A membrane is the
5 mm outer layer of the object minus the object, the task's tool keep-out (needle
and head along every certified face's own axis, tool_keepout.py) and the continuous
exit of the whole nonconvex object; the largest piece is kept.  It must
  - hold the object under gravity and every paired processing load (forces in the
    max-tilt cone about the nominal needle direction, the cone Step1 tilts in);
  - let the object leave along its exit without touching it;
  - keep clear of the KUKA: every link hull at the joints Step1 stored for every
    certified face (step1.arm's exact test).
The pose stays Step1's.  Only if no membrane passes there is the object turned about
the nominal direction through the work point, and the arm is certified again there.
"""
import argparse
import hashlib
import json
import shutil
import time
from copy import deepcopy
from pathlib import Path
import numpy as np
import trimesh
import manifold3d as md

from robots_config.constraints import load_config
from subdivide_task.step1.common import ROOT, DEFAULT_OUTPUT, load_shape, object_of, sha256, write_json
from subdivide_task.step1.arm import Check
from subdivide_task.common.poses import spin_pose
from subdivide_task.common.wrenches import resolve_policy
from subdivide_task.operations.geometry import moved_mesh
from subdivide_task.operations.state import Layout
from subdivide_task.operations.sweep_cache import sweep_mesh, implementation_key
from . import independent_geometry as G
from .exit_geometry import whole_object_sweep
from .membranes import MembraneModel
from .run_independent import PHYSICS_FILES
from .tool_keepout import disc, keepout

HERE = Path(__file__).parent
POLICIES = [HERE/'independent_policy.json', HERE/'membrane_policy.json', HERE/'tool_policy.json']
SOURCES = [Path(__file__), HERE/'tool_keepout.py', HERE/'membranes.py',
           HERE/'render_tool_membranes.py', ROOT/'subdivide_task/step1/arm.py', *POLICIES]


def stem(task):
    """File-safe name of a task: +X -> PX, -X -> NX."""
    return task.replace('+', 'P').replace('-', 'N')


def step1_tasks(object_name):
    """Nonempty exclusive task regions and each pose's own reach witnesses."""
    work = ROOT/'subdivide_task/data'/object_name/'step1_reach'
    plan = json.loads((work/'plan.json').read_text()); check = json.loads((work/'validation.json').read_text())
    if check['status'] != 'passed' or check['plan_sha256'] != sha256(work/'plan.json') or plan['arrays_sha256'] != sha256(work/'plan.npz'):
        raise ValueError('Step1 plan is not the validated one; rerun subdivide_task.step1.plan_reach')
    with np.load(work/'plan.npz') as z:
        a = {k: z[k] for k in ('face_task', 'reach', 'reach_axis', 'joints')}
    records = []
    for j, area in enumerate(plan['areas']):
        assigned = a['face_task'] == j
        faces = np.flatnonzero(assigned)
        if faces.tolist() != area['face_ids']:
            raise ValueError(area['id']+': task assignment does not match Step1 face_ids')
        if not len(faces):continue
        W = np.eye(4); W[:3, :3] = area['placement']['rotation_world_object']; W[:3, 3] = area['placement']['translation_world_m']
        certified = np.flatnonzero(assigned & a['reach'][:,j])
        records.append(dict(id=area['id'], direction_index=j, working_face_ids=faces.tolist(), certified_face_ids=certified.tolist(),
                            pose=W, axes=a['reach_axis'][certified,j], joints=a['joints'][certified,j], angle_deg=0))
    if not records:raise ValueError('Step1 has no assigned tasks to initialize')
    key = next(k for k in plan['input_sha256'] if k.startswith('robots_config/'))
    if sha256(ROOT/key) != plan['input_sha256'][key]: raise ValueError('Robot config changed since Step1')
    return plan, records, [work/'plan.json', work/'plan.npz', work/'validation.json', ROOT/key]


def public(record):
    """A record without its arrays, for the report."""
    return dict(id=record['id'], working_face_ids=record['working_face_ids'], certified_face_ids=record['certified_face_ids'],
                direction_index=record['direction_index'],
                world_from_object=np.asarray(record['pose']).tolist(), angle_deg=record['angle_deg'])


def task_partition(records, source):
    """Audit actual downstream face lists, independently of the Step1 labels."""
    counts = np.zeros(len(source.faces), dtype=int)
    for record in records:
        np.add.at(counts, np.asarray(record['working_face_ids'], int), 1)
    if np.any(counts > 1):raise ValueError('Downstream tasks repeat source faces')
    return dict(source='Step1 exclusive face_task partition', duplicate_face_count=int((counts > 1).sum()),
                duplicate_area_m2=float(source.area_faces[counts > 1].sum()),
                assigned_face_count=int((counts > 0).sum()),
                assigned_area_fraction=float(source.area_faces[counts > 0].sum()/source.area),
                task_area_fractions=[float(source.area_faces[r['working_face_ids']].sum()/source.area) for r in records])


def rigid(matrix):
    return (np.isfinite(matrix).all() and np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-12, rtol=0) and
            np.allclose(matrix[:3, :3].T@matrix[:3, :3], np.eye(3), atol=1e-12, rtol=0) and abs(np.linalg.det(matrix[:3, :3])-1) <= 1e-12)


class ToolMembraneModel(MembraneModel):
    """The membrane model with each task's own certified tool as its keep-out, and the KUKA checked."""
    def __init__(self, source, outer, records, setup, physical, policy, name='B'):
        self.source = source; self.outer = outer; self.records = deepcopy(records)
        self.ids = [r['id'] for r in records]; self.setup = deepcopy(setup)
        self.axis = np.asarray(setup['working_direction_world'], float)
        # build_domain reads the load cone's axis from here.
        self.workspace = dict(nominal_axis_world=self.axis.tolist(), work_point_m=list(setup['work_point_m']))
        self.physical = physical; self.policy = deepcopy(policy); self.name = name
        self.tolerance = policy['boolean_volume_tolerance_m3']; self.use_floor_constraints = False
        self.pose_cache = {}; self.geometry_cache = {}; self.path_cache = {}
        self.world_sweep_cache = {}; self.evaluations = 0; self.inputs = []
        self.sweep_cache_implementation = implementation_key()
        self.reach = float(np.linalg.norm(source.extents))+2*policy['mold_outward_displacement_m']
        self.tools = {}; self.arm = None

    def tool(self, i):
        """Task i's tool keep-out, object frame."""
        r = self.records[i]
        if r['id'] not in self.tools:
            self.tools[r['id']] = G.mesh(keepout(self.source, r['certified_face_ids'], r['axes'], self.setup,
                                                 self.reach, self.policy['tool_polygon_sides']))
        return self.tools[r['id']]

    def prepare(self, layout):
        rows = []
        for i, record in enumerate(self.records):
            P, M, d = layout.poses[i], layout.material[i], layout.exits[i]
            if not (rigid(P) and rigid(M)): raise ValueError('Expected a proper rigid pose/installation')
            if not np.isfinite(d).all() or abs(np.linalg.norm(d)-1) > 1e-12 or d[2] < .05:
                raise ValueError('Invalid upper-hemisphere exit')
            # The needle, head and arm are certified at this pose only.
            if not np.allclose(P, record['pose'], atol=1e-12, rtol=0):
                raise ValueError(record['id']+': pose without a certified tool and arm')
            data = self.world_data(record, P)
            key = (record['id'], P.tobytes(), M.tobytes(), d.tobytes())
            if key not in self.geometry_cache:
                # The construction travel is deliberately longer than any
                # local membrane; the final distance is derived from S.
                length = float(np.ceil(4*np.linalg.norm(self.source.extents)/.1)*.1)
                sweep = sweep_mesh(ROOT/'subdivide_task/data'/self.name/'step3_membrane_sweeps', data['body_world'], d, length,
                                   lambda: G.mesh(whole_object_sweep(data['body_world'], d, length)[0]),
                                   self.sweep_cache_implementation)
                self.geometry_cache[key] = dict(body=moved_mesh(data['body_world'], M),
                    stock=G.solid(moved_mesh(data['stock_world'], M)), work=G.solid(moved_mesh(self.tool(i), M@P)),
                    sweep=G.solid(moved_mesh(sweep, M)), direction=M[:3, :3]@d, length=length, material=M, pose=P,
                    index=i, id=record['id'], working_face_ids=np.asarray(record['working_face_ids']))
                if len(self.geometry_cache) > 128: self.geometry_cache.pop(next(iter(self.geometry_cache)))
            rows.append(dict(data, **self.geometry_cache[key]))
        return rows

    def arm_hits(self, world, record):
        """Certified faces at whose stored joints a KUKA link hull meets the support (world frame).
        Long triangles (a plate, a rail) are split first: the same surface, but Check's neighbour
        search radius grows with the longest triangle and would otherwise test every triangle."""
        edge = self.policy['arm_check_edge_fraction']*float(np.linalg.norm(self.source.extents))
        world = trimesh.Trimesh(*trimesh.remesh.subdivide_to_size(world.vertices, world.faces, edge, max_iter=20), process=False)
        if self.arm is None: self.arm = Check(world, self.setup)
        self.arm.mesh = world; self.arm.placed = {}; I = np.eye(4)
        return [int(f) for f, Q in zip(record['certified_face_ids'], record['joints']) if any(self.arm.hits(q, I) for q in Q)]

    def audit(self, candidate, *, arm=True, **options):
        """MembraneModel's audit (body, tool keep-out, exit, loads) and the KUKA against the support."""
        result = super().audit(candidate, **options); began = time.monotonic()
        for t, row in zip(candidate['tasks'], result['tasks']):
            record = self.records[t['index']]
            hits = self.arm_hits(moved_mesh(candidate['support'], np.linalg.inv(t['material'])), record)
            row['arm'] = dict(configurations=4*len(record['certified_face_ids']), faces_hit=hits)
            if hits and arm: raise ValueError(t['id']+': KUKA links meet the support at faces '+str(hits[:10]))
        result['arm_seconds'] = time.monotonic()-began
        return result

    def subset(self, indices):
        model = ToolMembraneModel(self.source, self.outer, [self.records[i] for i in indices], self.setup,
                                  self.physical, self.policy, self.name)
        model.world_sweep_cache = self.world_sweep_cache; model.tools = self.tools
        return model


def turned(record, source, setup, angle):
    """record turned by angle about the nominal direction through the work point, its KUKA certified
    again for every certified face (None: some face has no clear arm there).  The tilt to the nominal
    direction, and so the needle and head certificate, do not change under this turn."""
    P = spin_pose(record['pose'], setup['working_direction_world'], setup['work_point_m'], angle)
    arm = Check(source, setup); joints = []; previous = None
    for f, axis in zip(record['certified_face_ids'], record['axes']):
        code, Q = arm.pair(f, axis, P, previous)
        if code: return None
        joints.append(Q); previous = Q[-1]
    return dict(record, pose=P, joints=np.asarray(joints).reshape(len(joints), 4, -1), angle_deg=angle)


def row_axis(setup):
    """The row's direction on the table: across the arm's reach (horizontal, perpendicular to base -> work
    point), so the row of fixtures and its plate pass beside the work window, never into the base."""
    u = np.subtract(setup['work_point_m'], setup.get('base_position_world_m', (0., 0., 0.)))[:2]
    u = u/np.linalg.norm(u); return np.array([-u[1], u[0], 0.])


def row(model, q, shells, gaps):
    """Every fixture's place in the row: (placed membrane, its world -> plate translation, foot top, foot
    footprint).  gaps[i]: room added between fixtures i and i+1 beyond row_gap_m."""
    p = model.policy; h = p['plate_thickness_m']; t = p['mold_outward_displacement_m']
    clear = 1.1*model.setup['needle_radius_m']; rows = []; left = 0.
    e = row_axis(model.setup); across = np.array([-e[1], e[0], 0.])
    for i, shell in enumerate(shells):
        # The row keeps every object's whole exit (it rises from the object) clear of its neighbours:
        # the range along the row of the membrane, the object and the object at the end of a full exit.
        body = moved_mesh(model.source, q.poses[i]).vertices; reach = 4*float(np.linalg.norm(model.source.extents))
        X = np.r_[shell.vertices@e, body@e, (body+reach*q.exits[i])@e]
        shift = (left-X.min())*e+[0., 0., h-shell.bounds[0, 2]]
        placed = shell.copy(); placed.apply_translation(shift); left = X.max()+shift@e+p['row_gap_m']+gaps[i]
        top = min(h+t, body[:, 2].min()+shift[2]-clear)
        V = placed.vertices[placed.vertices[:, 2] <= top]
        if top <= h or len(V) < 3: raise ValueError(model.ids[i]+': no membrane below the object to stand on')
        # The support slides freely on the table, so every foot is centred on one line along the row.
        dy = -((V@across).min()+(V@across).max())/2; shift += dy*across; placed.apply_translation(dy*across); V = V+dy*across
        rows.append((placed, shift, top, V))
    return rows


def arm_meets_neighbour(model, rows):
    """The first (task, fixture) whose work arm, at any of the task's certified joints, meets that other
    fixture's membrane in the row (None: every arm is clear of its neighbours)."""
    check = None
    for i, record in enumerate(model.records):
        for j, (placed, _, _, _) in enumerate(rows):
            if j == i: continue
            world = placed.copy(); world.apply_translation(-rows[i][1])
            if check is None: check = Check(world, model.setup)
            check.mesh = world; check.placed = {}
            if any(check.hits(Q, np.eye(4)) for Q in np.asarray(record['joints']).reshape(-1, check.arm.dof)): return i, j
    return None


def connected(model, q, shells):
    """The separate membranes, each unchanged, on one base plate on the ground, joined.

    The work arm rises and falls, so a fixture simply drops until its membrane's lowest point
    rests on the plate; the whole support slides on the table to bring a seat to the station, so
    the fixtures stand in a row along row_axis (across the arm's reach), row_gap_m apart, their feet
    on one line.  Where a task's work arm still meets a neighbouring membrane, the gap on that side
    widens by row_gap_m until every arm is clear (arm_meets_neighbour).
    Under each membrane a foot -- an upright post over the footprint of the membrane's lowest part, from the
    floor up into the membrane -- joins it to the plate.  The foot ends a needle radius below the object, and the
    tool keep-out and the exit both rise from the object, so it meets none of them.  The plate
    is the convex hull of the feet's footprints widened by plate_edge_margin_m, plate_thickness_m
    thick.  Returns the joined mesh and every installation's translation (its world -> plate frame).
    """
    p = model.policy; h = p['plate_thickness_m']
    gaps = np.zeros(len(shells))
    for _ in range(64):
        rows = row(model, q, shells, gaps); hit = arm_meets_neighbour(model, rows)
        if hit is None: break
        i, j = hit; gaps[i if j > i else i-1] += p['row_gap_m']
    else: raise ValueError('The work arms still meet neighbouring fixtures in the row')
    model.connected_gaps = gaps
    pieces = []; prints = []; shifts = []; tops = []
    for placed, shift, top, V in rows:
        # The foot: an upright post over the footprint of that part, from the floor to `top`.
        pieces += [G.solid(placed), md.Manifold.hull_points(np.vstack([np.c_[V[:, :2], np.zeros(len(V))], np.c_[V[:, :2], np.full(len(V), top)]])/G.SCALE)]
        prints.append(V[:, :2]); tops.append(top); shifts.append(shift)
    ring = disc(np.array([0., 0., 1.]), p['plate_edge_margin_m'], 16)[:, :2]
    F = (np.vstack(prints)[:, None]+ring[None]).reshape(-1, 2)
    pieces.append(md.Manifold.hull_points(np.vstack([np.c_[F, np.zeros(len(F))], np.c_[F, np.full(len(F), h)]])/G.SCALE))
    value = G.union(pieces); forbidden = []
    for i, shift in enumerate(shifts):
        M = np.eye(4); M[:3,3] = shift; P = M@q.poses[i]
        body = moved_mesh(model.world_data(model.records[i], q.poses[i])['body_world'], M)
        forbidden.extend([G.solid(body), G.solid(moved_mesh(model.tool(i),P))])
    # Joining and serializing can leave tiny tool intersections. Recut the same
    # exclusions, keeping a subset of the joined solid and the same tolerance.
    # The body is placed exactly as prepare() places it for the audit: the membrane touches it, so a
    # differently rounded copy leaves contact slivers above the tolerance.
    # The resulting whole base still undergoes all load, arm and exit audits.
    joined, conditioning = G.condition(value,forbidden,dict(p,maximum_conditioning_rounds=3),subset=value)
    model.connected_conditioning = conditioning
    return joined, np.asarray(shifts), np.asarray(tops)


def joined_audit(model, q, joined, shells, shifts, tops, separate):
    """The joined support for every task: it stands on the ground; object and tool keep-out overlap, loads
    and the KUKA checked on the whole support; exits by joined_exits."""
    if abs(joined.bounds[0, 2]) > 1e-9: raise ValueError('The joined support does not stand on the ground plane')
    cq = q.copy(); cq.material[:, :3, 3] = shifts
    proof = model.audit(dict(layout=cq, support=joined, tasks=model.prepare(cq)), continuous=False)
    proof['exits'] = joined_exits(model, q, shells, shifts, tops, separate, [r['exit_length_m'] for r in proof['tasks']])
    return proof


def joined_exits(model, q, shells, shifts, tops, separate, joined):
    """Every object's continuous exit in the joined support, by exact bounds instead of Booleans.

    The support is its own membrane, the other membranes, the feet and the plate.  Its own membrane
    is the one already certified on its own, moved rigidly; past that certificate's travel the object
    is wholly above it.  Every other membrane, and its foot inside its footprint, lies outside the range
    along the row (row_axis) the object sweeps.  The plate and its own foot end below the object, and the exit rises.
    separate, joined: exit lengths of each task alone and in the joined support."""
    rows = []; h = model.policy['plate_thickness_m']; e = row_axis(model.setup)
    for i, (shell, shift, d) in enumerate(zip(shells, shifts, q.exits)):
        body = moved_mesh(model.source, q.poses[i]).vertices+shift; low = body[:, 2].min()
        x = np.r_[body@e, (body+joined[i]*d)@e]
        gaps = [max(x.min()-((s.vertices@e).max()+t@e), ((s.vertices@e).min()+t@e)-x.max())
                for j, (s, t) in enumerate(zip(shells, shifts)) if j != i]
        own_top = shell.vertices[:, 2].max()+shift[2]
        row = dict(id=model.ids[i], exit_rises=bool(d[2] > 0), x_gap_to_other_membranes_m=float(min(gaps)),
                   plate_and_own_foot_below_object_m=float(low-max(h, tops[i])),
                   above_own_membrane_after_its_certified_travel_m=float(low+separate[i]*d[2]-own_top),
                   separate_length_m=float(separate[i]), joined_length_m=float(joined[i]))
        row['passed'] = bool(row['exit_rises'] and min(gaps) > 0 and row['plate_and_own_foot_below_object_m'] > 0 and
                             row['above_own_membrane_after_its_certified_travel_m'] > 0 and joined[i] >= separate[i])
        if not row['passed']: raise ValueError(model.ids[i]+': joined exit not certified '+str(row))
        rows.append(row)
    return rows


def initialize_one(args):
    vertices, faces, ov, of, record, setup, physical, policy, name, folder = args
    source = trimesh.Trimesh(vertices, faces, process=False); outer = trimesh.Trimesh(ov, of, process=False)
    path = Path(folder)/(stem(record['id'])+'.npz'); meta = path.with_suffix('.json')
    if path.exists() and meta.exists():
        saved = json.loads(meta.read_text())
        if saved.get('signature') == policy['initialization_signature'] and saved.get('sha256') == sha256(path):
            return saved
    trace = []; direction = -np.asarray(setup['working_direction_world'], float)
    for angle in policy['initial_axial_angles_deg']:
        r = record if angle == 0 else turned(record, source, setup, angle)
        if r is None:
            trace.append(dict(angle_deg=angle, status='rejected', reason='KUKA cannot hold every certified face'))
            print(record['id'], 'REJECT', angle, 'arm', flush=True); continue
        model = ToolMembraneModel(source, outer, [r], setup, physical, policy, name)
        q = Layout(r['pose'][None], np.eye(4)[None], direction[None])
        try:
            candidate = model.build(q, facewise=True); model.audit(candidate)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as ex:
            trace.append(dict(angle_deg=angle, status='rejected', reason=str(ex)))
            print(record['id'], 'REJECT', angle, str(ex)[:130], flush=True); continue
        # How much of the membrane the tool removes: the stock minus everything but the tool.
        task = candidate['tasks'][0]; stock = task['stock']-G.solid(task['body'])
        cut = G.volume(stock^task['work'])*1e6
        np.savez_compressed(path, **q.arrays(), vertices=candidate['support'].vertices, faces=candidate['support'].faces,
                            joints=r['joints'])
        result = dict(id=record['id'], angle_deg=angle, volume_cm3=candidate['volume_cm3'], tool_cut_from_stock_cm3=cut,
                      audit=candidate['audit'], sha256=sha256(path), signature=policy['initialization_signature'], trace=trace)
        write_json(meta, result)
        print(record['id'], 'MEMBRANE PASS', round(candidate['volume_cm3'], 3), flush=True)
        return result
    write_json(meta, dict(status='failed', trace=trace))
    raise RuntimeError(record['id']+': no legal thin-membrane initialization')


def inputs_and_policy(object_name):
    plan, records, files = step1_tasks(object_name); setup = plan['setup']
    policy = {}
    for p in POLICIES: policy.update(json.loads(p.read_text()))
    policy = resolve_policy(policy, load_config(files[-1])); policy['task_count'] = len(records)
    if abs(policy['direction_half_angle_deg']-setup['max_tilt_deg']) > 1e-12:
        raise ValueError("The load cone must be Step1's tilt cone")
    # The fitted outer shell and the object belong to the object, whatever station a run (B+<config>) uses.
    outer_path = ROOT/'subdivide_task/data'/object_of(object_name)/'step3_mold_plate/fitted_outer.npz'
    objects = [ROOT/'objects'/object_of(object_name)/'mesh.stl', ROOT/'objects'/object_of(object_name)/'meta.json']
    inputs = {str(p): sha256(p) for p in [*files, *objects, *SOURCES, *(ROOT/p for p in PHYSICS_FILES)]}
    return plan, records, setup, policy, outer_path, inputs


def load_model(folder):
    """The model and layout from disk: tasks and tools rebuilt from Step1, poses and turned joints from Step3."""
    folder = Path(folder); report = json.loads((folder/'report.json').read_text())
    plan, records, setup, policy, outer_path, inputs = inputs_and_policy(report['object'])
    source = load_shape(ROOT/'objects'/object_of(report['object'])/'mesh.stl')
    if len(records) != len(report['tasks']):raise ValueError('Step3 task count differs from Step1')
    task_partition(report['tasks'], source)
    with np.load(folder/'initialization.npz') as z:
        outer = trimesh.Trimesh(z['outer_vertices'], z['outer_faces'], process=False); q = Layout.read(z)
        for i, (r, saved) in enumerate(zip(records, report['tasks'])):
            if r['id'] != saved['id'] or r['working_face_ids'] != saved['working_face_ids']:
                raise ValueError('Step3 tasks differ from Step1')
            r['pose'] = q.poses[i]; r['joints'] = z[stem(r['id'])+'_joints']; r['angle_deg'] = saved['angle_deg']
    return ToolMembraneModel(source, outer, records, setup, report['physical_properties'], report['policy'], report['object']), q, report


def validate(folder):
    """Disk replay: every input unchanged, each saved membrane audited again with tools rebuilt from Step1,
    a turned pose's arm replayed by forward kinematics; the joined support rebuilt from the saved membranes,
    equal to the saved one, and audited again for every task."""
    folder = Path(folder); model, q, report = load_model(folder); proofs = []
    for path, digest in report['input_sha256'].items():
        if sha256(path) != digest: raise ValueError('Changed Step3 input: '+path)
    arm = Check(model.source, model.setup)
    with np.load(folder/'initialization.npz') as z:
        for i, name in enumerate(model.ids):
            r = model.records[i]
            if r['angle_deg'] != 0 and not all(arm.replay(f, a, r['pose'], Q) for f, a, Q in zip(r['certified_face_ids'], r['axes'], r['joints'])):
                raise ValueError(name+': turned pose arm replay failed')
            part = model.subset([i]); qi = Layout(q.poses[i:i+1], q.material[i:i+1], q.exits[i:i+1])
            support = trimesh.Trimesh(z[stem(name)+'_vertices'], z[stem(name)+'_faces'], process=False)
            proofs.append(part.audit(dict(layout=qi, support=support, tasks=part.prepare(qi))))
            print(name, 'DISK REPLAY PASS', flush=True)
        shells = [trimesh.Trimesh(z[stem(name)+'_vertices'], z[stem(name)+'_faces'], process=False) for name in model.ids]
    # The joined support is rebuilt from the saved membranes and must be the saved one.
    joined, shifts, tops = connected(model, q, shells)
    with np.load(folder/'connected.npz') as c:
        if not (np.array_equal(joined.vertices, c['vertices']) and np.array_equal(joined.faces, c['faces'])):
            raise ValueError('The saved joined support is not the one built from the saved membranes')
    joint = joined_audit(model, q, joined, shells, shifts, tops, [p['tasks'][0]['exit_length_m'] for p in proofs])
    print('JOINED DISK REPLAY PASS', flush=True)
    result = dict(status='passed', independent_fixture_count=len(proofs), tasks=proofs, connected=joint,
                  task_partition=task_partition(report['tasks'], model.source),
                  initialization_sha256=sha256(folder/'initialization.npz'), connected_sha256=sha256(folder/'connected.npz'),
                  report_sha256=sha256(folder/'report.json'))
    write_json(folder/'validation.json', result); return result


def run(output_root=DEFAULT_OUTPUT, object_name='B', workers=6):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    began = time.monotonic(); output_root = Path(output_root).resolve()
    plan, records, setup, policy, outer_path, inputs = inputs_and_policy(object_name)
    source = load_shape(ROOT/'objects'/object_of(object_name)/'mesh.stl')
    physical = json.loads((ROOT/'objects'/object_of(object_name)/'meta.json').read_text())
    if not outer_path.exists():
        outer = G.mesh(G.fitted_outer(source, policy['mold_outward_displacement_m']))
        outer_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(outer_path, vertices=outer.vertices, faces=outer.faces)
    with np.load(outer_path) as z: outer = trimesh.Trimesh(z['vertices'], z['faces'], process=False)
    inputs[str(outer_path)] = sha256(outer_path)
    policy['initialization_signature'] = hashlib.sha256(json.dumps(dict(inputs=inputs, policy=policy), sort_keys=True).encode()).hexdigest()
    folder = ROOT/'subdivide_task/data'/object_name/'step3_tool_membranes'; folder.mkdir(parents=True, exist_ok=True)
    args = [(source.vertices, source.faces, outer.vertices, outer.faces, r, setup, physical, policy, object_name, folder) for r in records]
    with ProcessPoolExecutor(max_workers=min(workers, len(records)), mp_context=multiprocessing.get_context('spawn')) as pool:
        results = list(pool.map(initialize_one, args))
    poses = []; exits = []; shells = []; saved = {}
    for r, result in zip(records, results):
        with np.load(folder/(stem(r['id'])+'.npz')) as z:
            poses.append(z['poses'][0]); exits.append(z['exits'][0])
            r['pose'] = z['poses'][0]; r['joints'] = z['joints']; r['angle_deg'] = result['angle_deg']
            shells.append(trimesh.Trimesh(z['vertices'], z['faces'], process=False))
            saved.update({stem(r['id'])+'_vertices': z['vertices'], stem(r['id'])+'_faces': z['faces'], stem(r['id'])+'_joints': z['joints']})
    n = len(records); q = Layout(np.asarray(poses), np.tile(np.eye(4), (n, 1, 1)), np.asarray(exits))
    np.savez_compressed(folder/'initialization.npz', **q.arrays(), **saved, source_vertices=source.vertices,
                        source_faces=source.faces, outer_vertices=outer.vertices, outer_faces=outer.faces)
    model = ToolMembraneModel(source, outer, records, setup, physical, policy, object_name)
    joined, shifts, tops = connected(model, q, shells)
    np.savez_compressed(folder/'connected.npz', vertices=joined.vertices, faces=joined.faces, shifts=shifts, tops=tops)
    joint = joined_audit(model, q, joined, shells, shifts, tops, [r['audit']['tasks'][0]['exit_length_m'] for r in results])
    uncertified = sum(len(r['working_face_ids'])-len(r['certified_face_ids']) for r in records)
    report = dict(status='passed', object=object_name, method='pose_contact_membranes_on_one_common_base',
        task_count=n, independent_fixture_count=n, tasks=[public(r) for r in records], task_results=results, setup=setup,
        input_task_count=len(plan['areas']),
        skipped_empty_task_ids=[area['id'] for area in plan['areas'] if not area['face_ids']],
        task_definition='nonempty exclusive Step1 face_task regions, matching Step1.3; no repeated source faces',
        policy=policy, physical_properties=physical, input_sha256=inputs,
        task_partition=task_partition(records, source),
        material_volume_cm3=sum(r['volume_cm3'] for r in results),
        tool_keepout=dict(model=policy['tool_keepout'], head_reach_m=model.reach, uncertified_faces_without_keepout=uncertified),
        connected_initialization_required=True, common_floor_or_head_height_required=False,
        original_sample_count=sum(r['audit']['tasks'][0]['force']['original_sample_count'] for r in results),
        continuous_outer_target_count=sum(r['audit']['tasks'][0]['force']['continuous_outer_target_count'] for r in results),
        connected=dict(volume_cm3=float(joined.volume*1e6), installation_shifts_m=shifts.tolist(),
            conditioning=model.connected_conditioning,
            plate_and_feet_cm3=float(joined.volume*1e6)-sum(r['volume_cm3'] for r in results),
            plate_thickness_m=policy['plate_thickness_m'], row_gap_m=policy['row_gap_m'], rule=policy['connected'],
            row_axis=row_axis(setup).tolist(), gaps_widened_for_arms_m=model.connected_gaps.tolist(),
            arm_seconds=joint['arm_seconds']),
        seconds=time.monotonic()-began)
    write_json(folder/'report.json', report)
    # Independent reload: tools from Step1, everything else from the saved arrays.
    validate(folder)
    from .render_tool_membranes import render_connected
    render_connected(model, q, joined, shifts, folder)
    joined.export(folder/'base.obj')
    target = output_root/object_name/'step3'; target.mkdir(parents=True, exist_ok=True)
    # Publish one primary image of the common base, with and without all poses.
    for old in ('baseline_result.png','initialization_connected.png','initialization_independent.png'):
        (target/old).unlink(missing_ok=True); (folder/old).unlink(missing_ok=True)
    images = ('initialization.png',)
    for name in images: shutil.copy2(folder/name, target/name)
    shutil.copy2(folder/'base.obj',target/'base.obj')
    write_json(folder/'images.json', {name: sha256(folder/name) for name in images})
    print('STEP3 TOOL MEMBRANES PASS', report['material_volume_cm3'], 'joined', report['connected']['volume_cm3'], flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--workers', type=int, default=6)
    p.add_argument('--validate-only', action='store_true'); a = p.parse_args()
    if a.validate_only: validate(ROOT/'subdivide_task/data/B/step3_tool_membranes')
    else: run(workers=a.workers)
