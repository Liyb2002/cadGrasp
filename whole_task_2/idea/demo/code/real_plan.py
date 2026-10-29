"""Read existing designs and plan fixed-table SO101 contact stops, without changing Step1--Step5.

  PYTHONPATH=. .venv/bin/python -B whole_task_2/idea/demo/code/real_plan.py
  PYTHONPATH=. .venv/bin/python -B whole_task_2/idea/demo/code/real_plan.py --fixture-run B+9pose

No renderer, carrier arm, or robot connection. The execution agent builds and checks continuous trajectories.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from whole_task_2.step1.common import ROOT, load_shape, sha256, write_json
from whole_task_2.step1 import reach as R
from whole_task_2.step1.arm import order
if __package__:
    from .real_arm import RealArm, SceneCheck, tcp_matrix
else:
    from real_arm import RealArm, SceneCheck, tcp_matrix

HERE = Path(__file__).resolve().parent
STATION = HERE.parent / 'real_station_so101.json'
PRINT = 1.01                     # The existing Step5 package's print scale, without importing its rendering stack.
_CONTEXT = None
_REPLAY_CHECKS = {}


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    return value


def core_hashes():
    return {str(p.relative_to(ROOT)): sha256(p) for i in range(1, 6)
            for p in sorted((ROOT / f'whole_task_2/step{i}').rglob('*.py'))}


def load_station(path=STATION):
    s = json.loads(Path(path).read_text())
    tcp_matrix(s['feasibility']['tcp_transform'])
    bounds = np.asarray(s['placement_bounds_xy_m'], float)
    if (bounds.shape != (2, 2) or not np.isfinite(bounds).all() or not (bounds[0] < bounds[1]).all() or
            not np.isfinite(s['table_top_z_m']) or not 0 < s['maximum_joint_speed_deg_s'] <= 60):
        raise ValueError('Invalid measured station bounds, tabletop, or joint speed')
    return s


def placed_on_table(source, rotation, xy, station, support=None, seat=None):
    """Pure translations preserve fixture/object gravity and contact orientation."""
    P = np.eye(4); P[:3, :3] = rotation
    v = source.vertices @ rotation.T
    P[:2, 3] = np.asarray(xy) - (v.min(0) + v.max(0))[:2] / 2
    W = None if support is None else P @ np.linalg.inv(seat)
    low = v[:, 2].min() if W is None else (support.vertices @ W[:3, :3].T + W[:3, 3])[:, 2].min()
    P[2, 3] += station['table_top_z_m'] - low
    if W is not None: W = P @ np.linalg.inv(seat)
    return P, W


def within_table(source, P, station, support=None, W=None):
    vertices = [trimesh.transform_points(source.vertices, P)]
    if support is not None: vertices.append(trimesh.transform_points(support.vertices, W))
    v = np.vstack(vertices); lo, hi = np.asarray(station['placement_bounds_xy_m'])
    return bool((v[:, :2] >= lo - 1e-9).all() and (v[:, :2] <= hi + 1e-9).all() and
                v[:, 2].min() >= station['table_top_z_m'] - 1e-9)


def cone_axes(station):
    d = np.asarray(station['working_direction_world']); u, v = R.frame(d)
    axes = [d]
    for t in np.radians([station['cone_half_angle_deg']/2, station['cone_half_angle_deg']]):
        for a in np.arange(8) * np.pi / 4:
            axes.append(np.cos(t)*d + np.sin(t)*(np.cos(a)*u + np.sin(a)*v))
    return np.asarray(axes)


def screen_cache(source, station):
    shape = ROOT/'objects/B/mesh.stl'
    for path in (ROOT/'whole_task_2/data/B/step1_reach/screen.npz',
                 ROOT/'whole_task_2/data/B+9pose/step1_reach/screen.npz'):
        if path.exists():
            with np.load(path) as z:
                key = json.loads(str(z['key']))
                if (key['shape'] == sha256(shape) and key['reach'] == sha256(ROOT/'whole_task_2/step1/reach.py') and
                        all(np.isclose(key[k], station[n]) for k, n in
                            [('length', 'needle_length_m'), ('radius', 'head_radius_m'), ('needle_radius', 'needle_radius_m')])):
                    return z['axes'].copy(), z['reach'].copy()
    return None


def translations(station):
    lo, hi = np.asarray(station['placement_bounds_xy_m'])
    for x in np.arange(lo[0] + .07, hi[0] - .06, .02):
        for y in np.arange(lo[1] + .07, hi[1] - .06, .04):
            yield np.array([x, y])


def rank(source, arm, station, rotation, ids, axes_world, masks, support=None, seat=None, cube=True):
    """Rank translations with the random FK cloud; only exact checks can accept them."""
    test = R.test_points(source)[ids] @ rotation.T; centres = test[:, 3]
    rows = []; edge = station['work_cube_edge_m']
    for xy in translations(station):
        P, W = placed_on_table(source, rotation, xy, station, support, seat)
        if not within_table(source, P, station, support, W): continue
        points = centres + P[:3, 3]
        distances = np.full(masks.shape, np.inf)
        for j, axis in enumerate(axes_world):
            m = masks[j]
            if m.any(): distances[j, m] = arm.nearest(points[m], axis)[0]
        cost = distances.min(0); potential = cost < .018
        if not potential.any(): continue
        windows = [None]
        if cube:
            weights = source.area_faces[ids] * potential
            mean = np.average(points, weights=weights, axis=0)
            anchors = points[np.flatnonzero(potential)[::max(1, int(potential.sum())//12)]]
            windows = [mean, *anchors]
        for centre in windows:
            keep = potential.copy(); bounds = None
            if centre is not None:
                bounds = np.array([centre-edge/2, centre+edge/2])
                keep &= ((test + P[:3, 3] >= bounds[0]-1e-9) &
                         (test + P[:3, 3] <= bounds[1]+1e-9)).all(axis=(1, 2))
            score = float(source.area_faces[ids[keep]].sum())
            if score:
                rows.append(dict(score=score, pose=P, support_pose=W, keep=keep, bounds=bounds,
                                 axis_order=np.argsort(distances, axis=0), distances=distances))
    return sorted(rows, key=lambda r: -r['score'])


def _init(context):
    global _CONTEXT
    _CONTEXT = context


def _face(job):
    face, axes = job
    source, arm, station, P, support, W = _CONTEXT
    check = SceneCheck(source, arm, station, support, W)
    for axis_world in axes:
        axis = P[:3, :3].T @ axis_world
        if not R.certify(source, face, axis, station['needle_length_m'], station['head_radius_m'],
                         station['needle_radius_m'], .0005, None)[0]: continue
        Q = check.face(face, axis, P)
        if Q is not None: return face, axis_world, Q
    return face, None, None


def certify(source, arm, station, candidate, ids, axes_world, masks, workers):
    jobs = []
    for k in np.flatnonzero(candidate['keep']):
        options = [axes_world[j] for j in candidate['axis_order'][:, k]
                   if masks[j, k] and candidate['distances'][j, k] < .025]
        jobs.append((int(ids[k]), options))
    context = (source, arm, station, candidate['pose'], candidate['support'], candidate['support_pose'])
    accepted = []
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(context,)) as pool:
        for n, row in enumerate(pool.map(_face, jobs, chunksize=4)):
            if row[1] is not None: accepted.append(row)
            if (n+1) % 100 == 0: print('CONTACT CHECK', n+1, '/', len(jobs), 'passed', len(accepted), flush=True)
    if not accepted: return None
    faces = np.array([r[0] for r in accepted]); walk = order(trimesh.transform_points(source.triangles_center[faces], candidate['pose']))
    faces = faces[walk]; axes = np.array([r[1] for r in accepted])[walk]; Q = np.array([r[2] for r in accepted])[walk]
    return dict(candidate, faces=faces, axes=axes, joints=Q, attempted=len(jobs), rejected=len(jobs)-len(faces))


def single(source, arm, station, workers, candidates):
    stable, prob = source.compute_stable_poses(n_samples=1); best = int(np.argmax(prob)); base = stable[best, :3, :3]
    axes = cone_axes(station); cache = screen_cache(source, station); ranking = []; ids = np.arange(len(source.faces))
    screening = None
    if cache is None:
        pixel = .004 * np.linalg.norm(source.extents)
        screening = R.Screen(source, station['needle_length_m'], station['head_radius_m'],
                             station['needle_radius_m'], pixel, int(6*source.area/pixel**2))
    for yaw in range(0, 360, 45):
        rotation = Rotation.from_euler('z', yaw, degrees=True).as_matrix() @ base
        local = axes @ rotation
        if cache is not None:
            nearest = (local @ cache[0].T).argmax(1); masks = cache[1][nearest].copy()
        else: masks = np.array([screening(a) for a in local])
        rows = rank(source, arm, station, rotation, ids, axes, masks)
        if rows:
            ranking.append(dict(rows[0], masks=masks, support=None, yaw_deg=yaw))
    for proposal in sorted(ranking, key=lambda r: -r['score'])[:candidates]:
        print('SINGLE CANDIDATE', proposal['yaw_deg'], 'estimated cm2', proposal['score']*1e4, flush=True)
        result = certify(source, arm, station, proposal, ids, axes, proposal['masks'], workers)
        if result is not None:
            result.update(id='S1', original_faces=ids, stable_pose_index=best, stable_pose_probability=float(prob[best]),
                          stable_yaw_deg=proposal['yaw_deg'], seat=None, exit=-np.asarray(station['working_direction_world']), exit_length=0.)
            return [result], None, dict(mode='single_stable_pose_without_support', stable_probability=float(prob[best]))
    raise ValueError('No certified contact in the searched stable-pose placements')


def fixture_inputs(run, result, source):
    data = ROOT/'whole_task_2/data'/run
    report_path = data/'step5_combined/report.json'; report = json.loads(report_path.read_text())
    row = next(r for r in report['results'] if r['name'] == result and r['status'] == 'passed')
    path = data/f'step5_combined/{result}.npz'
    if sha256(path) != row['support_sha256']: raise ValueError('Changed Step5 support')
    with np.load(path) as z: support = trimesh.Trimesh(z['vertices'], z['faces'], process=False)
    O = support.bounds.mean(0); printed = support.copy(); printed.vertices = O + PRINT*(support.vertices-O)
    combined_path = data/'step4_combine/report.json'; combined = json.loads(combined_path.read_text())
    group = next(g for g in combined['finals'] if g['file'] == row['step4_file'])
    init_path = data/'step3_tool_membranes/report.json'; init = json.loads(init_path.read_text())
    plan_path = data/'step1_reach/plan.npz'; tasks = []
    with np.load(plan_path) as z:
        for s in group['slots']:
            for i, angle in zip(s['members'], s['angles']):
                if angle != 0: raise ValueError('This adapter requires original zero-turn records')
                record = init['tasks'][i]; ids = np.asarray(record['certified_face_ids'], int)
                X = np.asarray(s['X'], float); Xprint = X.copy()
                Xprint[:3, 3] += (PRINT-1)*(trimesh.transform_points(source.center_mass[None], X)[0]-O)
                P = np.asarray(record['world_from_object'], float); j = record['direction_index']
                axes = z['reach_axis'][ids, j] @ P[:3, :3].T
                proof = next(t for t in row['tasks'] if t['id'] == record['id'])
                tasks.append(dict(id=record['id'], ids=ids, axes=axes, rotation=P[:3, :3], seat=Xprint,
                                  original_seat=X, exit=P[:3, :3] @ s['exit'], exit_length=proof['exit_length_m']))
    return printed, tasks, dict(mode='existing_fixture_repositioned_without_redesign', run=run, result=result,
        print_scale=PRINT, source_volume_cm3=float(support.volume)*1e6,
        source_files={str(p.relative_to(ROOT)): sha256(p) for p in [path, report_path, combined_path, init_path, plan_path]})


def fixture(source, arm, station, run, result, workers, candidates):
    support, records, origin = fixture_inputs(run, result, source); results = []
    for record in records:
        ids = record['ids']; axes = record['axes']
        # Each face keeps its original process direction; one sparse mask per distinct axis.
        unique, inverse = np.unique(np.round(axes, 12), axis=0, return_inverse=True)
        unique /= np.linalg.norm(unique, axis=1)[:, None]
        masks = np.arange(len(unique))[:, None] == inverse
        rows = rank(source, arm, station, record['rotation'], ids, unique, masks, support, record['seat'], cube=False)
        found = None
        for proposal in rows[:candidates]:
            proposal['support'] = support
            print('FIXTURE CANDIDATE', record['id'], 'estimated faces', int(proposal['keep'].sum()), flush=True)
            found = certify(source, arm, station, proposal, ids, unique, masks, workers)
            if found is not None: break
        if found is None:
            results.append(dict(id=record['id'], status='unreachable_in_search', original_faces=ids,
                                seat=record['seat'], exit=record['exit'], exit_length=record['exit_length']))
            print('FIXTURE TASK', record['id'], 'no accepted contacts', flush=True); continue
        found.update(id=record['id'], original_faces=ids, seat=record['seat'], exit=record['exit'], exit_length=record['exit_length'])
        results.append(found)
        print('FIXTURE TASK', record['id'], len(found['faces']), '/', len(ids), 'certified', flush=True)
    return results, support, origin


def mm(T):
    if T is None: return None
    T = np.array(T, float); T[:3, 3] *= 1e3; return T


def export(source, tasks, support, origin, station, out, station_path, before):
    out.mkdir(parents=True, exist_ok=True); source_mm = source.copy(); source_mm.vertices *= 1e3; source_mm.export(out/'object.stl')
    files = [out/'object.stl']; arrays = {}; rows = []; seats = []
    if support is not None:
        printed = support.copy(); printed.vertices *= 1e3; printed.export(out/'support.stl'); files.append(out/'support.stl')
    for task in tasks:
        name = task['id']; original = np.asarray(task['original_faces'], int)
        if 'faces' not in task:
            rows.append(dict(id=name, status=task['status'], certified_faces=0, original_faces=len(original),
                             unprocessed_face_ids=original.tolist(), exit_direction_world=task['exit']))
            continue
        P, W, faces, axes, joints = (task[k] for k in ['pose', 'support_pose', 'faces', 'axes', 'joints'])
        points = trimesh.transform_points(R.test_points(source)[faces].reshape(-1, 3), P).reshape(-1, 4, 3)
        key = name.replace('+', 'P').replace('-', 'N'); joint_deg = np.degrees(joints)
        arrays.update({f'{key}_certified_faces': faces, f'{key}_needle_direction_world': axes,
                       f'{key}_arm_joints_deg': joint_deg, f'{key}_world_from_object_mm': mm(P),
                       f'{key}_test_points_world_mm': points*1e3, f'{key}_stop_tip_world_mm': points[:, 3]*1e3,
                       f'{key}_stop_needle_direction_world': axes, f'{key}_stop_arm_joints_deg': joint_deg[:, 3],
                       f'{key}_world_from_support_mm': np.empty((0, 4, 4)) if W is None else mm(W)})
        seat_index = None
        if task['seat'] is not None:
            seat_index = next((i for i, s in enumerate(seats) if np.allclose(s['support_from_object'], mm(task['seat']))), len(seats))
            if seat_index == len(seats): seats.append(dict(seat=seat_index, support_from_object=mm(task['seat']), tasks=[]))
            seats[seat_index]['tasks'].append(name)
        k = int(np.argmin(np.linalg.norm(source.triangles_center[faces]-source.triangles_center[faces].mean(0), axis=1)))
        rows.append(dict(id=name, status='contacts_certified', seat=seat_index, turn_deg=0.,
            world_from_object=mm(P), world_from_support=mm(W), support_from_object=mm(task['seat']),
            support_lowest_z_mm=None if W is None else float(trimesh.transform_points(support.vertices, W)[:, 2].min())*1e3,
            object_lowest_z_mm=float(trimesh.transform_points(source.vertices, P)[:, 2].min())*1e3,
            object_lift_mm=0., arm_base_offset_mm=0., exit_direction_world=task['exit'], exit_length_mm=task['exit_length']*1e3,
            original_faces=len(original), certified_faces=len(faces), working_faces=len(faces),
            working_area_cm2=float(source.area_faces[faces].sum())*1e4,
            retained_fraction_of_original_area=float(source.area_faces[faces].sum()/source.area_faces[original].sum()),
            unprocessed_face_ids=np.setdiff1d(original, faces),
            stable_pose_probability=task.get('stable_pose_probability'), stable_yaw_deg=task.get('stable_yaw_deg'),
            work_cube_bounds_world_mm=None if task['bounds'] is None else task['bounds']*1e3,
            central_face=dict(face=int(faces[k]), test_points_world_mm=points[k]*1e3, needle_direction_world=axes[k], arm_joints_deg=joint_deg[k]),
            checks=dict(contacts_only=True, exact_tool_object=True, exact_arm_object_and_support=True,
                        fixed_table=True, continuous_motion_verified=False, load_model_recomputed=False)))
    np.savez_compressed(out/'tasks.npz', **arrays); files.append(out/'tasks.npz')
    spec = json.loads((ROOT/station['source_station']).read_text())['robot_specification']; arm = RealArm(station)
    support_info = None if support is None else dict(print_scale=origin['print_scale'],
        print_volume_cm3=float(support.volume)*1e6, print_dimensions_mm=support.extents*1e3,
        print_bounds_mm=support.bounds*1e3, volume_cm3=origin['source_volume_cm3'], geometry_redesigned=False)
    info = dict(schema='real_so101_contacts_v1', object='B', result='single' if support is None else origin['result'],
        units='lengths mm (4x4 matrices: translation mm), angles deg', origin=origin,
        frames=dict(world='SO101 URDF base_link; fixed base; table z = -2.4 mm', object='object.stl; original mesh triangle order'),
        support=support_info, seats=seats, tasks=rows,
        station=dict(id=station['id'], config=str(Path(station_path).resolve().relative_to(ROOT)),
            robot=dict(model=spec['model'], dof=5, joint_order=spec['joint_order'], joint_limits_deg=np.degrees(np.c_[arm.lower, arm.upper]),
                       maximum_joint_speed_deg_s=[station['maximum_joint_speed_deg_s']]*5),
            arm_base_world_mm=np.asarray(station['base_position_world_m'])*1e3, arm_base_rotation_world=station['base_rotation_world'],
            arm_base_offsets_mm=[0.]*len(rows), table_top_z_mm=station['table_top_z_m']*1e3,
            placement_bounds_xy_mm=np.asarray(station['placement_bounds_xy_m'])*1e3,
            manual_repositioning=True, carrier_robot=None, working_direction_world=station['working_direction_world'],
            tilt_cone_half_angle_deg=station['cone_half_angle_deg'], camera=station['camera'],
            tool=dict(needle_length_mm=station['needle_length_m']*1e3, needle_radius_mm=station['needle_radius_m']*1e3,
                head_radius_mm=station['head_radius_m']*1e3, config_tcp_transform=station['feasibility']['tcp_transform'],
                tcp_transform_unit='m', tcp_frame='gripper_frame_from_process_tcp; +z into surface')),
        execution=dict(trajectory_generated=False, joint_offsets_applied=False, planner='offline contact and placement planner',
            next='Execution agent re-solves calibrated IK and checks every interpolated frame before program.json execution.',
            transition_rules=dict(straight_back_m=[.012, .006, 0.], retreat_tilt_deg=[45, 60, 75], retreat_search_m=.08,
                maximum_joint_speed_deg_s=station['maximum_joint_speed_deg_s'], base_vertical_motion_allowed=False)),
        input_sha256={str(ROOT/'objects/B/mesh.stl'): sha256(ROOT/'objects/B/mesh.stl'), str(Path(station_path).resolve()): sha256(station_path)},
        core_source_sha256=before, files={p.name: sha256(p) for p in files})
    write_json(out/'info.json', plain(info))
    return info


def _replay_init(context):
    global _CONTEXT, _REPLAY_CHECKS
    _CONTEXT = context; _REPLAY_CHECKS = {}


def _replay(job):
    name, face, axis, qs, ps, P, W = job
    source, arm, station, support = _CONTEXT
    if name not in _REPLAY_CHECKS: _REPLAY_CHECKS[name] = SceneCheck(source, arm, station, support, W)
    check = _REPLAY_CHECKS[name]; errors = []; position = []; angle = []; local = P[:3, :3].T @ axis
    if not R.certify(source, int(face), local, station['needle_length_m'], station['head_radius_m'], station['needle_radius_m'], .0005)[0]:
        errors.append(f'{name}:{face}: tool/object')
    if axis @ np.asarray(station['working_direction_world']) < np.cos(np.radians(station['cone_half_angle_deg']))-1e-9:
        errors.append(f'{name}:{face}: outside process cone')
    if not check.tool_above_table(int(face), local, P): errors.append(f'{name}:{face}: tool/table')
    allowance = float(R.tip_contact(station['needle_radius_m'], -source.face_normals[face] @ local))
    for q, p in zip(qs, ps):
        tcp, d, _, _ = arm.forward(q); position.append(float(np.linalg.norm(tcp-p))); angle.append(float(np.linalg.norm(d-axis)))
        if np.any(q < arm.lower-1e-9) or np.any(q > arm.upper+1e-9) or not check.clear(q, P, allowance):
            errors.append(f'{name}:{face}: arm/floor/scene')
    return errors, max(position), max(angle), len(qs)


def validate(out, source, support, station, workers=8):
    info = json.loads((out/'info.json').read_text()); arm = RealArm(station); errors = []; count = 0; positional = []; angular = []
    jobs = []
    for name, digest in info['files'].items():
        if sha256(out/name) != digest: raise ValueError('Changed exported '+name)
    mesh_errors = {}
    for name, expected in [('object.stl', source), ('support.stl', support)]:
        if expected is None: continue
        saved = trimesh.load(out/name, force='mesh', process=False)
        if saved.triangles.shape != expected.triangles.shape:
            errors.append(name+': exported triangle count differs'); continue
        deviation = float(np.abs(saved.triangles-expected.triangles*1e3).max())
        mesh_errors[name] = deviation
        if deviation > 5e-5: errors.append(name+': exported face order, geometry, or mm units differ')
    with np.load(out/'tasks.npz') as z:
        for task in info['tasks']:
            if task['status'] != 'contacts_certified': continue
            key = task['id'].replace('+', 'P').replace('-', 'N'); faces = z[f'{key}_certified_faces']; axes = z[f'{key}_needle_direction_world']
            Q = np.radians(z[f'{key}_arm_joints_deg']); P = np.array(task['world_from_object']); P[:3, 3] /= 1e3
            W = task['world_from_support']
            if W is not None: W = np.array(W); W[:3, 3] /= 1e3
            if not within_table(source, P, station, support, W): errors.append(task['id']+': outside measured table region')
            actual_low = min(trimesh.transform_points(source.vertices if support is None else support.vertices, P if support is None else W)[:, 2])
            if not np.isclose(actual_low, station['table_top_z_m'], atol=1e-9): errors.append(task['id']+': does not rest on measured table')
            points = R.test_points(source)[faces] @ P[:3, :3].T + P[:3, 3]
            if not np.allclose(z[f'{key}_test_points_world_mm'], points*1e3): errors.append(task['id']+': serialized contact points differ')
            bounds = task['work_cube_bounds_world_mm']
            if bounds is not None and not ((points*1e3 >= np.asarray(bounds)[0]-1e-6) &
                                           (points*1e3 <= np.asarray(bounds)[1]+1e-6)).all():
                errors.append(task['id']+': contact outside its fixed work cube')
            for face, axis, qs, ps in zip(faces, axes, Q, points):
                jobs.append((key, face, axis, qs, ps, P, W))
    with ProcessPoolExecutor(workers, initializer=_replay_init, initargs=((source, arm, station, support),)) as pool:
        for n, (bad, position, angle, checked) in enumerate(pool.map(_replay, jobs, chunksize=4)):
            errors.extend(bad); positional.append(position); angular.append(angle); count += checked
            if (n+1) % 100 == 0: print('DISK REPLAY', n+1, '/', len(jobs), flush=True)
    if not count: errors.append('No certified contacts')
    if positional and max(positional) > 1e-4: errors.append('FK position mismatch')
    if angular and max(angular) > 1e-3: errors.append('FK axis mismatch')
    if info['core_source_sha256'] != core_hashes(): errors.append('Step1--Step5 source changed during planning')
    report = dict(status='passed' if not errors else 'failed', scope='static contacts; execution trajectories not certified',
        contact_configurations=count, max_fk_position_error_mm=max(positional, default=0.)*1e3,
        max_fk_axis_error=max(angular, default=0.), fixed_table_top_z_mm=station['table_top_z_m']*1e3,
        exported_mesh_max_deviation_mm=mesh_errors,
        arm_base_offsets_mm=info['station']['arm_base_offsets_mm'], core_sources_unchanged=info['core_source_sha256']==core_hashes(),
        errors=errors[:30], info_sha256=sha256(out/'info.json'))
    write_json(out/'validation.json', report)
    if errors: raise ValueError(str(errors[:5]))
    return report


def run(args):
    began = time.monotonic(); station = load_station(args.station); source = load_shape(ROOT/'objects/B/mesh.stl'); before = core_hashes()
    name = f'{args.fixture_run}_{args.result}' if args.fixture_run else 'single'
    out = Path(args.output) if args.output else HERE.parent/'real_so101'/name
    if args.validate_only:
        support = fixture_inputs(args.fixture_run, args.result, source)[0] if args.fixture_run else None
        return validate(out, source, support, station, args.workers)
    arm = RealArm(station); arm.sample(); print('REAL STATION fixed table', station['table_top_z_m'], 'base offsets 0', flush=True)
    if args.fixture_run:
        tasks, support, origin = fixture(source, arm, station, args.fixture_run, args.result, args.workers, args.candidates)
        name = f'{args.fixture_run}_{args.result}'
    else:
        tasks, support, origin = single(source, arm, station, args.workers, args.candidates); name = 'single'
    info = export(source, tasks, support, origin, station, out, args.station, before)
    result = validate(out, source, support, station, args.workers)
    print('REAL PACKAGE', out, 'tasks', [(r['id'], r['certified_faces']) for r in info['tasks']],
          result['status'], round(time.monotonic()-began), 's', flush=True)
    return info


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--station', type=Path, default=STATION)
    p.add_argument('--fixture-run'); p.add_argument('--result', default='r00'); p.add_argument('--output', type=Path)
    p.add_argument('--validate-only', action='store_true', help='Replay the saved package with FK and collision checks, without searching or solving IK')
    p.add_argument('--workers', type=int, default=8); p.add_argument('--candidates', type=int, default=3)
    run(p.parse_args())
