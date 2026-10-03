"""Run nearest-feasible assignment / local loft / MST on saved pose pairs.

Inputs are the unchanged successful Step3 triples and original Step4 demands.
Five physical heads are registered once; one complete contact patch is shared.
No head selection, load augmentation, or contact-area optimization occurs here.
"""
import argparse
from contextlib import redirect_stdout
import itertools
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import trimesh
import manifold3d as md

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.needs import ROOT, OUTPUTS
from step2_local_support import geometry as G, circles as P, withdrawal as W
from step3_scheculer import contacts as I
from step3_scheculer.pair_tasks import read_task, pair_folder
from step0_pose_selection.floor_points import pressure_centers
from step4_connect_support.baseline_current import geometry_kernel as K
from step4_connect_support.baseline_current import greedy_geometry as GG
from step4_connect_support.baseline_current.fixture_view import cells_for, pack, export_viewer
from step4_connect_support.baseline_current import head_registration as H
from step4_connect_support.baseline_current.refresh_shared_geometry_view import write_viewer

HERE = Path(__file__).resolve().parent
SCHEMA = 'registered_five_heads_local_loft_mst_v2'


def log(*args):
    print(*args, flush=True)


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k:plain(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def read_case(source):
    schedule = json.loads((source/'schedule.json').read_text())
    if not schedule['passed']:
        raise ValueError('Only successful Step3 particles are accepted')
    pair = source.parents[4]
    summary = json.loads((source.parent/'schedule.json').read_text())
    poses, name = summary['poses'], summary['object']
    tasks = [read_task(name, pose, poses) for pose in poses]
    groups = [I.read_contacts(source/f'contacts_{p}.npz') for p in poses]
    identifiers = set(c['candidate_id'] for group in groups for c in group)
    shared_ids = set(c['candidate_id'] for c in groups[0]) & set(c['candidate_id'] for c in groups[1])
    if identifiers != set(schedule['selected_ids']) or shared_ids != {schedule['shared_head']['selected_id']}:
        raise ValueError('Saved Step3 head IDs disagree with the active contact triples')
    paths = [source/'schedule.json', source.parent/'schedule.json']+[source/f'contacts_{p}.npz' for p in poses]
    catalogues = []
    for pose in poses:
        path = pair/'step2_local_support/sequential_3plus2'/source.parent.parent.name/'terminal_expansion'/f'candidates_{pose}.json'
        catalogues.append(np.asarray(json.loads(path.read_text())['direction_catalogue']['vectors']))
        paths.append(path)
    frames = [np.asarray(p.domain.data['frame']['T_world_mesh']) for p in tasks]
    owned, heads = {}, []
    for k, (task, group) in enumerate(zip(tasks, groups)):
        depth = G.vertex_offsets(task.domain.mesh, P.DEPTH_FRACTION*task.domain.mesh.extents.max())[0]
        row = []
        for contact in group:
            ident = contact['candidate_id']
            if ident not in owned:
                owned[ident] = k, cells_for(contact, task.domain, depth)
            owner, pieces = owned[ident]
            transform = frames[k]@np.linalg.inv(frames[owner])
            row.append([v@transform[:3, :3].T+transform[:3, 3] for v in pieces])
        heads.append(row)
    demands = []
    for task in tasks:
        floor_path = pair/'step0_pose_selection/sequential_3plus2'/source.parent.parent.name/'terminal_expansion'/f'floor_contact_{task.pose}.npz'
        data = np.load(floor_path)
        xy = data['floor_demands_xy_m']
        np.testing.assert_allclose(data['load_wrenches'], task.targets/task.scale, atol=1e-13, rtol=0)
        np.testing.assert_allclose(xy, pressure_centers(task.targets/task.scale, task.domain.com)[0], atol=1e-13, rtol=0)
        if len(xy) != 32768:
            raise ValueError('Expected the fixed original 32768 floor demands')
        demands.append(xy); paths.append(floor_path)
        paths.extend(task.inputs)
    return SimpleNamespace(name=name, poses=poses, pair=pair, tasks=tasks, groups=groups,
        heads=heads, catalogues=catalogues, source=source, schedule=schedule,
        paths=paths, demands=demands)


def placements(case, maximum=8):
    """Only withdrawal directions vary; shared-patch registration fixes placement."""
    bases, offsets = H.fixed_placements(case.tasks)
    registered, registration = H.register(case.groups, case.heads, bases, offsets)
    if not registration['all_heads_above_both_floors']:
        return
    menus = [r['common_direction_ids'] for r in case.schedule['geometry']['per_pose']]
    # Check every unique original head, including those inactive for this task.
    legal = []
    for k, task in enumerate(case.tasks):
        analyzer = W.Analyzer(task.domain.mesh, P.DEPTH_FRACTION*task.domain.mesh.extents.max(),
                              dict(vectors=case.catalogues[k].tolist()))
        cells = [SimpleNamespace(vertices=K.local_to_world(v, bases[k], offsets[k]))
                 for h in registered for v in h.cells]
        valid = []
        for ident in menus[k]:
            direction = case.catalogues[k][ident]
            if analyzer.test(cells, direction)['clear']:
                valid.append(ident)
        legal.append(valid)
    for count, ids in enumerate(itertools.product(*legal)):
        if count >= maximum:
            break
        yield dict(kind='exact_shared_patch_registration', bases=bases, offsets=offsets,
            directions=np.asarray([c[i] for c, i in zip(case.catalogues, ids)]),
            direction_ids=list(ids), registration=registration)


def forbidden_space(case, placement, work):
    work.mkdir(parents=True, exist_ok=True)
    sweeps = []
    for k, task in enumerate(case.tasks):
        basis, offset, direction = placement['bases'][k], placement['offsets'][k], placement['directions'][k]
        local = trimesh.Trimesh(task.domain.mesh.vertices@basis+offset, task.domain.mesh.faces, process=False)
        sweep = K.swept_solid(local, -K.SWEEP_LENGTH*direction@basis)
        sweeps.append(sweep)
    box = md.Manifold.cube([2*K.RELIEF/K.SCALE]*3).translate([-K.RELIEF/K.SCALE]*3)
    forbidden = GG.union(K.solid(sweep).minkowski_sum(box) for sweep in sweeps)
    return sweeps, forbidden


def export_result(case, placement, constructor, full, bridges, mst, checks, certificate, attempts, out):
    mesh = K.unpack(full)
    solid_check = dict(component_count=len(full.decompose()), watertight=bool(mesh.is_watertight),
        consistently_wound=bool(mesh.is_winding_consistent), volume_m3=float(mesh.volume))
    solid_check['one_solid'] = bool(solid_check['component_count'] == 1 and solid_check['watertight']
                                  and solid_check['consistently_wound'] and mesh.volume > 0)
    if not solid_check['one_solid']:
        raise RuntimeError('Output must be one closed, oriented, positive-volume solid')
    mesh.export(out/'fixture.obj', file_type='obj', digits=17, include_normals=False)
    mm = mesh.copy(); mm.apply_scale(1000)
    (out/'fixture_mm.stl').write_text(trimesh.exchange.stl.export_stl_ascii(mm))
    restored = trimesh.load(out/'fixture_mm.stl', force='mesh')
    if not restored.is_watertight or not restored.is_winding_consistent:
        raise RuntimeError('STL reload failed')
    np.testing.assert_allclose(restored.extents, mm.extents, atol=1e-10, rtol=0)
    np.savez_compressed(out/'geometry.npz', vertices_m=mesh.vertices, faces=mesh.faces,
                        rotations=placement['bases'], local_offsets_m=placement['offsets'])
    np.savez_compressed(out/'equilibrium.npz', **certificate)
    records = []
    for i, (head, body) in enumerate(zip(constructor.heads, constructor.bodies)):
        name = f'body{i}.obj'
        K.unpack(body).export(out/name, file_type='obj', digits=17, include_normals=False)
        records.append(dict(index=i, head_pose=case.poses[head.pose], active_poses=[case.poses[k] for k in head.active_poses], candidate_id=head.ident, file=name,
                            volume_cm3=GG.volume(body)*1e6))
    if bridges:
        K.unpack(GG.union(bridges)).export(out/'bridges.obj', file_type='obj', digits=17, include_normals=False)
    design = dict(method=SCHEMA, placement={k:v for k,v in placement.items() if k != 'head_checks'},
        automatic_assignment=True, fixed_head_count_per_floor=False, floor_perimeter_constructed=False,
        forced_opposite_floor_projection=False, local_bodies=records, assignments=constructor.assignments, mst=mst)
    design = plain(design)
    I.save(out/'design.json', design)
    artifacts = ['fixture.obj', 'fixture_mm.stl', 'geometry.npz', 'equilibrium.npz', 'design.json']
    artifacts += [r['file'] for r in records]+(['bridges.obj'] if bridges else [])
    report = dict(schema=SCHEMA, object=case.name, poses=case.poses, particle=case.schedule['particle'],
        complete=True, passed=True, status='geometry_and_fixed_sample_equilibrium_passed',
        successful_step3_particles_tried=len({a['particle'] for a in attempts}),
        body_construction_attempted=True, full_equilibrium_rerun=True,
        same_rigid_solid_in_both_poses=True, contact_groups=len(case.schedule['selected_ids']), contact_patch_count=len(records),
        shared_branch_definition='five unique physical heads; exactly one physical contact patch shared by both poses',
        physical_head_definition='five_unique_heads_one_shared_patch',
        registration=constructor.registration,
        original_task_poses_changed=False, original_active_contacts_preserved=True, contact_area_optimization_performed=False,
        support_mass_ignored=True, extra_loads_added=False, structural_strength_verified=False,
        dimensions_mm=(mesh.extents*1000).tolist(), volume_cm3=float(mesh.volume*1e6), solid=solid_check, checks=checks,
        source_schedule=str((case.source/'schedule.json').relative_to(ROOT)),
        task_fixture_transforms=[dict(pose=p, rotation=b.tolist(), translation_m=(-b@o).tolist())
            for p,b,o in zip(case.poses, placement['bases'], placement['offsets'])],
        body_design=design, attempts=attempts,
        presentation_description='最近可行头分配 → 局部 loft → 最小生成树连接。彩色是原接触头附近，灰白色是新增身体。五个实体头；橙色是同一块共享接触面。',
        exports=dict(obj_units='m', stl_units='mm', stl_encoding='ASCII', reloaded_stl_watertight=True),
        artifacts={f:I.sha256(out/f) for f in artifacts},
        provenance=dict(inputs=I.hashes(case.paths), code=I.hashes([Path(__file__), Path(GG.__file__), Path(K.__file__), Path(H.__file__),
            HERE/'fixture_view.py', HERE/'translation_sweep.py', HERE/'surface_check.py',
            HERE/'refresh_shared_geometry_view.py', HERE/'shared_geometry_viewer.html',
            Path(K.Q.__file__), Path(G.__file__), Path(W.__file__), Path(K.FLOOR.__file__),
            Path(K.bearing_rays.__code__.co_filename)])))
    I.save(out/'report.json', report)
    colors = {case.schedule['shared_head']['selected_id']:'#dc9d47'}
    colors.update(zip([i for i in case.schedule['selected_ids'] if i not in colors],
                      ['#ac7098', '#7196c0', '#50a59b', '#77a76a']))
    visual = [(head.ident, G.hull_mesh(head.vertices)) for head in constructor.heads]
    export_viewer(out, mesh, visual, report, case.tasks, placement['bases'], placement['offsets'], colors)
    html = (out/'index.html').read_text()
    data, _ = json.JSONDecoder().raw_decode(html.split('const DATA=', 1)[1])
    data['local_bodies'] = [dict(label=f'{r["candidate_id"]} · {r["head_pose"]}', id=r['candidate_id'],
                               head_pose=r['head_pose'], **pack(K.unpack(body)))
                            for r,body in zip(records, constructor.bodies)]
    write_viewer(out, data)
    return report


def run_pair(pair, maximum_placements=5, only_particle=None):
    began = time.monotonic()
    out = pair/'step4'; out.mkdir(parents=True, exist_ok=True)
    roots = sorted((pair/'step3_scheculer/sequential_3plus2').glob('from_*/terminal_expansion'))
    sources = [p.parent for root in roots for p in sorted(root.glob('particle_*/schedule.json'))
               if json.loads(p.read_text())['passed']]
    if only_particle is not None:
        sources = [s for s in sources if int(s.name.split('_')[1]) == only_particle]
    sources.sort(key=lambda s: (not (pair.name == 'pose1+3' and s.name == 'particle_009'), s.name))
    cases = [read_case(source) for source in sources]
    clear_previous_outputs(out)
    I.save(out/'report.json', dict(schema=SCHEMA, complete=False, passed=False, status='running'))
    attempts, passed = [], None
    preview = None
    registrations = []
    for case in cases:
        log(pair.name, case.source.name, 'registering five physical heads')
        try:
            bases, offsets = H.fixed_placements(case.tasks)
            registered, registration = H.register(case.groups, case.heads, bases, offsets)
            floor_check = H.floor_compatibility(case.tasks, case.demands, bases, offsets)
        except ValueError as error:
            attempts.append(dict(particle=case.schedule['particle'], stage='head_registration',
                                 passed=False, reason=str(error)))
            continue
        registrations.append(dict(particle=case.schedule['particle'],
            source_schedule=str((case.source/'schedule.json').relative_to(ROOT)),
            registration=registration, floor_compatibility=floor_check))
        if preview is None:
            preview = case, bases, offsets, registered, registration, floor_check
        if not floor_check['passed'] or not registration['all_heads_above_both_floors']:
            attempts.append(dict(particle=case.schedule['particle'], stage='floor_compatibility', passed=False,
                reason='Original fixed contact registration conflicts with a task floor',
                registration=registration, floor_compatibility=floor_check))
            log(pair.name, 'REJECT:', [r['violating_sample_count'] for r in floor_check['per_pose']],
                'original floor demands outside the legal halfplanes')
            continue
        direction_count = 0
        for placement_id, placement in enumerate(placements(case, maximum_placements)):
            direction_count += 1
            log(pair.name, source.name, 'placement', placement_id, placement['kind'])
            try:
                sweeps, forbidden = forbidden_space(case, placement, out)
            except Exception as error:
                attempts.append(dict(particle=case.schedule['particle'], placement=placement_id,
                    stage='sweeps', reason=str(error)))
                I.save(out/'attempts.json', attempts); continue
            foot_trials = [(family, outward) for outward in (0., .01, .025, .04, .06)
                           for family in ('minimum_rectangle', 'axis_box', 'hull')]
            for family, outward in foot_trials:
                row = dict(particle=case.schedule['particle'], placement=placement_id,
                           placement_kind=placement['kind'], target_polygon_family=family,
                           outward_m=outward, passed=False)
                attempts.append(row)
                try:
                    constructor = GG.Constructor(case.groups, case.heads, placement['directions'],
                        placement['bases'], placement['offsets'], forbidden)
                    targets = [GG.foot_targets(xy, outward, family=family) for xy in case.demands]
                    # Target geometry order is deterministic and does not force
                    # a fixed number of heads, active or inactive, per pose.
                    for k, polygons in enumerate(targets):
                        for index, polygon in enumerate(polygons):
                            constructor.assign(k, polygon, index)
                    full, bridges, mst = constructor.connect()
                    mesh = K.unpack(full)
                    coverage = [GG.footprint_coverage(GG.actual_footprint(mesh, b, o), p.floor, xy)
                        for p,b,o,xy in zip(case.tasks, placement['bases'], placement['offsets'], case.demands)]
                    row['floor_coverage'] = coverage
                    if not all(r['passed'] for r in coverage):
                        raise RuntimeError('Actual final footprint does not enclose original demands')
                    checks, certificate = K.verify(case.tasks, case.groups, case.heads, placement['directions'],
                        placement['bases'], placement['offsets'], mesh, sweeps)
                    for check, cover in zip(checks, coverage):
                        check['floor_coverage'] = cover
                    row['passed'] = True
                    passed = export_result(case, placement, constructor, full, bridges, mst,
                        checks, certificate, attempts, out)
                except (RuntimeError, ValueError) as error:
                    row['reason'] = str(error)
                    if hasattr(error, 'checks'):
                        row['checks'] = error.checks
                    log(pair.name, 'candidate failed:', str(error))
                I.save(out/'attempts.json', attempts)
                if passed:
                    break
            if passed:
                break
        if passed:
            break
        if not direction_count:
            attempts.append(dict(particle=case.schedule['particle'], stage='head_withdrawal', passed=False,
                reason='No original withdrawal direction clears all five registered heads'))
    I.save(out/'attempts.json', attempts)
    I.save(out/'registration_checks.json', registrations)
    elapsed = time.monotonic()-began
    if passed:
        log('RESULT', pair.name, 'PASS', passed['volume_cm3'], 'cm3', round(elapsed, 1), 's')
        return dict(pair=pair.name, passed=True, particle=passed['particle'], volume_cm3=passed['volume_cm3'],
                    seconds=elapsed, attempt_count=len(attempts))
    floor_conflict = bool(attempts) and all(a.get('stage') == 'floor_compatibility' for a in attempts)
    report = dict(schema=SCHEMA, object=pair.parent.name, poses=[f'pose_{p}' for p in pair.name[4:].split('+')],
        complete=True, passed=False, status='fixed_contact_floor_conflict' if floor_conflict else 'registered_construction_failed',
        complete_fixture_verified=False, search_exhaustion_is_infeasibility_proof=False,
        infeasible_under_fixed_contact_correspondence=floor_conflict,
        physical_head_definition='five_unique_heads_one_shared_patch',
        contact_groups=5, contact_patch_count=5, geometry_candidate_exported=False,
        body_construction_attempted=any('target_polygon_family' in a for a in attempts),
        full_equilibrium_rerun=any('checks' in a for a in attempts),
        original_sample_count_per_pose=32768, extra_loads_added=False, attempts=attempts,
        seconds=elapsed, successful_step3_particles_tried=len(sources),
        provenance=dict(inputs=I.hashes(list(dict.fromkeys(p for c in cases for p in c.paths))),
            code=I.hashes([Path(__file__), Path(H.__file__), Path(GG.__file__), Path(K.__file__)])),
        artifacts={name:I.sha256(out/name) for name in ('attempts.json', 'registration_checks.json')})
    I.save(out/'report.json', report)
    if preview is not None:
        from step4_connect_support.baseline_current.registration_view import export_failure
        export_failure(out, report, *preview)
    log('RESULT', pair.name, 'FAIL', len(attempts), 'attempts', round(elapsed, 1), 's')
    return dict(pair=pair.name, passed=False, seconds=elapsed, attempt_count=len(attempts))


def clear_previous_outputs(out):
    """A failed rerun must never leave a previous fixture or PASS image active."""
    audit = out/'shared_head_audit.json'
    if audit.exists():
        if (out/'historical_shared_head_audit.json').exists():
            audit.unlink()
        else:
            audit.replace(out/'historical_shared_head_audit.json')
    names = ['fixture.obj', 'fixture_mm.stl', 'geometry.npz', 'equilibrium.npz', 'design.json',
        'bridges.obj', 'index.html', 'overview.png', 'fixture.png', 'local_body.png', 'separate.png',
        'independent_audit.json', 'audit.log', 'candidate_replay.log', 'viewer_check.json',
        'case_summary.json', 'batch.html', 'batch.png', 'batch_viewer_check.json', 'batch_publish.log',
        'head_layout.obj', 'head_layout.npz', 'floor_conflict.png', 'head_layout.png',
        'registration_replay.json']
    names += [p.name for p in out.glob('body[0-9]*.obj')]
    for name in names:
        (out/name).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', default='B')
    parser.add_argument('--pair', nargs='*', help='For example pose1+3; default all saved pairs')
    parser.add_argument('--placements', type=int, default=5, help='Maximum withdrawal-direction pairs; registered placement is fixed')
    parser.add_argument('--particle', type=int)
    args = parser.parse_args()
    pairs = [OUTPUTS/args.object/p for p in args.pair] if args.pair else sorted((OUTPUTS/args.object).glob('pose*+*'))
    if args.placements < 1:
        parser.error('--placements must be positive')
    results = []
    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, value):
            for stream in self.streams:
                stream.write(value)
            return len(value)
        def flush(self):
            for stream in self.streams:
                stream.flush()
    for pair in pairs:
        out = pair/'step4'; out.mkdir(parents=True, exist_ok=True)
        with (out/'run.log').open('w') as stream, redirect_stdout(Tee(sys.stdout, stream)):
            results.append(run_pair(pair, args.placements, args.particle))
    log(json.dumps(dict(results=results, passed=sum(r['passed'] for r in results), total=len(results)), indent=2))
    return 0 if all(r['passed'] for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
