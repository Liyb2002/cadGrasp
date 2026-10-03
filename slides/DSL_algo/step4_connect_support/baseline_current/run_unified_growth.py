"""Joint material growth for all heads and both floors; original loads only."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np
import trimesh
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.needs import ROOT, OUTPUTS
from step3_scheculer import contacts as I
from step4_connect_support.baseline_current import material_graph as M, material_network as N
from step4_connect_support.baseline_current import build_coupled_saddle as S
from step4_connect_support.baseline_current.run_greedy import read_case, plain, clear_previous_outputs
from step4_connect_support.baseline_current.run_local_bodies import Tee
from step4_connect_support.baseline_current.fixture_view import export_viewer

HERE = Path(__file__).resolve().parent
SCHEMA = 'unified_material_group_growth_v1'


def reaction_groups(case, graph, checks, certificate):
    """Separate a failed ORIGINAL load from the current reaction cone.

    If A.T@y >= 0 but b@y < 0, a feasible extension needs a new floor ray
    with negative dot product. Candidate material cells providing such rays
    become one more group in the same connected-cover problem.
    """
    groups = []; records = []
    rays = np.array([[64., 0, 1], [-64, 0, 1], [0, 64, 1], [0, -64, 1]])
    for k, check in enumerate(checks):
        if check['coupled_equilibrium_passed']: continue
        task = case.tasks[k]; name = task.pose
        matrix = certificate[f'{name}_matrix']
        assignment = certificate[f'{name}_assignment']
        index = int(np.flatnonzero(assignment < 0)[0])
        target = np.r_[task.targets[index], np.zeros(6)]
        result = linprog(target, A_ub=-matrix.T, b_ub=np.zeros(matrix.shape[1]),
                         bounds=[(-1., 1.)]*12, method='highs-ds', options=S.Q.OPTIONS)
        if not result.success or result.fun >= -1e-8:
            raise RuntimeError('Original-load numerical failure has no reliable separating reaction witness')
        dual = result.x; min_dot = float((matrix.T@dual).min())
        if min_dot < -2e-8:
            raise RuntimeError('Reaction separator violates the current cone')
        members = []
        for i, floors in enumerate(graph['floors']):
            if not len(floors[k]): continue
            p = np.c_[floors[k], np.zeros(len(floors[k]))]
            force = np.broadcast_to(rays, (len(p), 4, 3))
            moment = np.cross(p[:, None]-task.domain.com, force)
            wrench = np.concatenate([force, moment], axis=2)*task.scale
            if (wrench@dual[6:]).min() < -1e-8:
                members.append(i)
        if not members:
            raise RuntimeError('No candidate floor material resolves a separating original-load constraint')
        groups.append(N.Group(np.asarray(members), 'mechanics', k, dual.tolist(), -1e-8))
        records.append(dict(pose=name, original_sample_index=index, dual=dual.tolist(),
            target_dot=float(result.fun), minimum_current_ray_dot=min_dot,
            candidate_nodes=len(members), new_physical_load=False))
    return groups, records


def benchmark(graph, groups, solutions, seconds):
    chosen = solutions[0][1]
    # Bounded comparison on the union of greedy solutions and a one-hop
    # neighborhood. The reported lower bound applies only to this subgraph.
    keep = np.zeros(len(chosen), bool)
    for solution in solutions: keep |= solution[1]
    near = np.any(keep[graph['edges']], axis=1)
    keep[np.unique(graph['edges'][near])] = True
    ids = np.flatnonzero(keep); mapping = np.full(len(keep), -1); mapping[ids] = np.arange(len(ids))
    edge = graph['edges'][np.all(keep[graph['edges']], axis=1)]
    subgroups = [N.Group(mapping[g.members[keep[g.members]]], g.kind, g.pose, g.direction, g.threshold) for g in groups]
    candidate, record = N.flow_milp(graph['cost'][ids], mapping[edge], subgroups,
        mapping[graph['heads']].tolist(), int(mapping[solutions[0][2]]), chosen[ids], seconds)
    record.update(scope='union_of_greedy_solutions_plus_one_hop', full_graph_nodes=len(keep),
                  greedy_volume_cm3=float(graph['cost'][chosen].sum()))
    if candidate is None: return None, record
    selected = np.zeros(len(keep), bool); selected[ids] = candidate
    return selected, record


def inspect(case, context, graph, selected):
    full = M.material_solid(graph, selected)
    full, regularization = M.regularize_export(full, context['heads'])
    mesh = S.unpack(full)
    solid = dict(component_count=len(full.decompose()), watertight=bool(mesh.is_watertight),
                 consistently_wound=bool(mesh.is_winding_consistent), volume_m3=float(mesh.volume))
    solid['one_solid'] = bool(solid['component_count'] == 1 and solid['watertight'] and solid['consistently_wound'] and mesh.volume > 0)
    if not solid['one_solid']:
        raise RuntimeError(f'Actual material union is not one closed solid: {solid}')
    floor_checks = []
    for k in (0, 1):
        world = (mesh.vertices-context['offsets'][k])@context['bases'][k].T
        ground = world[np.abs(world[:, 2]) < 1e-9, :2]
        floor_checks.append(N.containment(ground, case.demands[k]))
    if not all(c['passed'] for c in floor_checks):
        raise RuntimeError('Actual exported body lost the predicted ground hull containment')
    passed = True
    try:
        checks, certificate = S.verify(case.tasks, case.groups, case.heads, context['directions'],
            context['bases'], context['offsets'], mesh, context['sweeps'])
    except RuntimeError as error:
        if not hasattr(error, 'certificate'): raise
        checks, certificate = error.checks, error.certificate; passed = False
    return dict(full=full, mesh=mesh, solid=solid, checks=checks, certificate=certificate,
                floor_checks=floor_checks, passed=passed, selected=selected, regularization=regularization)


def preserve_reference(out):
    if (out/'reference_report.json').exists(): return
    report = json.loads((out/'report.json').read_text())
    if report.get('schema') == SCHEMA: return
    # Flat named comparison files, not an archive/stage directory migration.
    for name in ('report.json', 'overview.png', 'fixture_mm.stl', 'geometry.npz', 'independent_audit.json'):
        source = out/name
        if source.exists(): shutil.copy2(source, out/f'reference_{name}')


def publish(out, work, case, context, placement, graph, result, design, reference,
            *, schema=SCHEMA, extra_code=(), presentation_description=None):
    mesh = result['mesh']; stage = work/'publish'; stage.mkdir(exist_ok=True)
    mesh.export(stage/'fixture.obj', file_type='obj', digits=17, include_normals=False)
    mm = mesh.copy(); mm.apply_scale(1000)
    (stage/'fixture_mm.stl').write_text(trimesh.exchange.stl.export_stl_ascii(mm))
    loaded = trimesh.load(stage/'fixture_mm.stl', force='mesh')
    welding = 'trimesh_default'
    if not loaded.is_watertight or not loaded.is_winding_consistent:
        # STL repeats coordinates per triangle. Weld exactly equal coordinates,
        # avoiding proximity merging of different sub-nanometre mesh vertices.
        raw = trimesh.load(stage/'fixture_mm.stl', force='mesh', process=False)
        vertices, inverse = np.unique(raw.vertices, axis=0, return_inverse=True)
        loaded = trimesh.Trimesh(vertices, inverse[raw.faces], process=False)
        welding = 'exact_coordinate_weld'
    if not loaded.is_watertight or not loaded.is_winding_consistent or len(loaded.split()) != 1:
        raise RuntimeError('STL round trip changed closed connected topology')
    np.testing.assert_allclose(loaded.extents, mm.extents, atol=1e-9, rtol=0)
    np.savez_compressed(stage/'geometry.npz', vertices_m=mesh.vertices, faces=mesh.faces,
                        rotations=context['bases'], local_offsets_m=context['offsets'])
    certname = 'equilibrium.npz' if result['passed'] else 'equilibrium_diagnostic.npz'
    np.savez_compressed(stage/certname, **result['certificate'])
    records = []
    for head, row in zip(context['heads'], context['records']):
        record = dict(row, file=f"body{row['index']}.obj")
        hm = S.unpack(head); hm.export(stage/record['file'], file_type='obj', digits=17, include_normals=False)
        record['volume_cm3'] = float(hm.volume*1e6); records.append(record)
    selected = result['selected']; centers = np.array([np.mean(v, axis=0) for v in graph['vertices']])
    np.savez_compressed(stage/'material_graph.npz', edges=graph['edges'], cost_cm3=graph['cost'],
        centers_m=centers, selected=selected, head_nodes=graph['heads'])
    design.update(local_bodies=records, material_ownership_per_head=False,
                  total_selected_material_nodes=int(selected.sum()), graph=graph['stats'],
                  selected_cell_cost_cm3=float(graph['cost'][selected].sum()),
                  fixed_head_volume_cm3=float(M.union(context['heads']).volume()*S.SCALE**3*1e6),
                  assembly_coordinate_snap_m=1e-12, export_regularization=result['regularization'])
    I.save(stage/'design.json', plain(design))
    artifacts = ['fixture.obj', 'fixture_mm.stl', 'geometry.npz', certname, 'material_graph.npz', 'design.json']+[r['file'] for r in records]
    code = [Path(__file__), Path(M.__file__), Path(N.__file__), Path(S.__file__),
        Path(S.swept_solid.__code__.co_filename), Path(S.surface_distances.__code__.co_filename),
        Path(S.bearing_rays.__code__.co_filename), Path(S.Q.__file__), Path(S.W.__file__), Path(S.G.__file__),
        Path(S.FLOOR.__file__), HERE/'run_greedy.py', HERE/'fixture_view.py', HERE/'refresh_shared_geometry_view.py',
        HERE/'shared_geometry_viewer.html', HERE/'export_shared_geometry.cjs']+list(extra_code)
    report = dict(schema=schema, object=case.name, poses=case.poses, particle=case.schedule['particle'],
        complete=True, passed=result['passed'], status='full_acceptance_passed' if result['passed'] else 'connected_candidate_failed_original_loads',
        same_rigid_solid_in_both_poses=True, physical_head_definition='six_retained_contact_patches',
        contact_groups=5, contact_patch_count=6, original_active_contacts_preserved=True,
        original_task_poses_changed=False, support_mass_ignored=True, structural_strength_verified=False,
        extra_loads_added=False, source_schedule=str((case.source/'schedule.json').relative_to(ROOT)),
        placement=plain(placement), task_fixture_transforms=[dict(pose=p, rotation=b.tolist(), translation_m=(-b@o).tolist())
            for p, b, o in zip(case.poses, context['bases'], context['offsets'])],
        dimensions_mm=(mesh.extents*1000).tolist(), volume_cm3=float(mesh.volume*1e6), solid=result['solid'],
        checks=result['checks'], actual_ground_containment=result['floor_checks'], body_design=plain(design),
        comparison=dict(reference_method=reference.get('schema'), reference_volume_cm3=reference['volume_cm3'],
            reference_passed=reference['passed'], placement_unchanged=True,
            volume_change_percent=100*(mesh.volume*1e6/reference['volume_cm3']-1), strength_equivalence_claimed=False),
        presentation_description=(presentation_description or '共享材料图统一生长：头、两种地面覆盖和连接在同一个循环选择。彩色为原头附近，灰白为共同材料；保留六块接触面、五个标识，橙色有两面。')+('' if result['passed'] else '当前是未通过原始载荷验收的候选。'),
        exports=dict(obj_units='m', stl_units='mm', stl_encoding='ASCII', stl_welding=welding, reloaded_stl_watertight=True),
        provenance=dict(inputs=I.hashes(case.paths), code=I.hashes(code)),
        artifacts={f:I.sha256(stage/f) for f in artifacts})
    I.save(stage/'report.json', plain(report))
    colors = {case.schedule['shared_head']['selected_id']:'#dc9d47'}
    colors.update(zip([i for i in case.schedule['selected_ids'] if i not in colors], ['#ac7098','#7196c0','#50a59b','#77a76a']))
    export_viewer(stage, mesh, [(r['candidate_id'], S.unpack(h)) for r, h in zip(records, context['heads'])],
                  plain(report), case.tasks, context['bases'], context['offsets'], colors)
    preserve_reference(out); clear_previous_outputs(out)
    (out/'equilibrium_diagnostic.npz').unlink(missing_ok=True)
    for name in artifacts+['report.json', 'index.html']: shutil.copy2(stage/name, out/name)
    return report


def run_pair(pair, step=.008, padding=.03, milp_seconds=20., max_feedback=64):
    out = pair/'step4'; work = Path(tempfile.gettempdir())/'cadgrasp_unified_growth'/pair.name
    work.mkdir(parents=True, exist_ok=True)
    source = out/'reference_report.json' if (out/'reference_report.json').exists() else out/'report.json'
    reference = json.loads(source.read_text()); case = read_case((ROOT/reference['source_schedule']).parent)
    placement = reference['placement']; context = M.prepare(case, placement, work)
    graph = M.build_graph(case, context, work, step, padding)
    groups = None; history = []; best = None; benchmark_record = None; seed = None
    began = time.monotonic()
    for iteration in range(max_feedback):
        solutions, groups, separation = N.solve_cover(graph['cost'], graph['edges'], graph['floors'],
            case.demands, graph['heads'], groups=groups, seed=seed)
        chosen = solutions[0][1]
        print('UNIFIED', pair.name, 'round', iteration, 'groups', len(groups), 'added cm3', solutions[0][0], flush=True)
        result = inspect(case, context, graph, chosen); best = result
        history.append(dict(iteration=iteration, volume_cm3=result['mesh'].volume*1e6,
            passed=result['passed'], root=solutions[0][2], growth_steps=solutions[0][3], geometric_separation=separation))
        if result['passed']:
            if milp_seconds > 0:
                candidate, benchmark_record = benchmark(graph, groups, solutions, milp_seconds)
                print('MILP', benchmark_record, flush=True)
                if candidate is not None and graph['cost'][candidate].sum() < graph['cost'][chosen].sum()-1e-7:
                    coverage = [N.containment(N.floor_points(candidate, graph['floors'], k), case.demands[k]) for k in (0, 1)]
                    if all(c['passed'] for c in coverage):
                        try:
                            other = inspect(case, context, graph, candidate)
                            benchmark_record['full_actual_acceptance_passed'] = other['passed']
                            if other['passed']: best = other
                        except RuntimeError as error:
                            benchmark_record['full_actual_acceptance_passed'] = False
                            benchmark_record['rejection'] = str(error)
                    else: benchmark_record['actual_ground_containment_passed'] = False
            break
        new, records = reaction_groups(case, graph, result['checks'], result['certificate'])
        history[-1]['original_load_separation'] = records
        changed = False
        for group in new:
            if not any(g.kind == group.kind and g.pose == group.pose and np.array_equal(g.members, group.members) for g in groups):
                groups.append(group); changed = True
        print('Added original-load groups:', records, flush=True)
        if not changed: raise RuntimeError('Repeated original-load failure without a new material group')
        seed = chosen
    design = dict(method=SCHEMA, forced_opposite_floor_projection=False, preassigned_head_floor_roles=False,
        separate_final_connection_phase=False, finite_material_graph=True, optimality_claim=False,
        monotone_original_load_feedback=True, maximum_feedback_iterations=max_feedback,
        groups=[dict(kind=g.kind, pose=g.pose, members=g.members.tolist(), direction=g.direction, threshold=g.threshold) for g in groups],
        iterations=history, milp_comparison=benchmark_record, solve_seconds=time.monotonic()-began)
    report = publish(out, work, case, context, placement, graph, best, design, reference)
    print('RESULT', pair.name, report['passed'], report['volume_cm3'], report['comparison'], flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', nargs='+', default=['pose1+3','pose1+6','pose2+8'])
    parser.add_argument('--step-mm', type=float, default=8.)
    parser.add_argument('--padding-mm', type=float, default=30.)
    parser.add_argument('--milp-seconds', type=float, default=20.)
    parser.add_argument('--max-feedback', type=int, default=64)
    args = parser.parse_args(); status = []
    for name in args.pair:
        pair = OUTPUTS/'B'/name
        with (pair/'step4/unified_growth.log').open('w') as f, redirect_stdout(Tee(sys.stdout, f)):
            report = run_pair(pair, args.step_mm/1000, args.padding_mm/1000, args.milp_seconds, args.max_feedback)
            status.append(dict(pair=name, passed=report['passed'], volume_cm3=report['volume_cm3']))
    print(json.dumps(status, indent=2))
    return 0 if all(r['passed'] for r in status) else 1


if __name__ == '__main__': raise SystemExit(main())
