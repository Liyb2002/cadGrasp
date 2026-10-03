"""Connect independent pose heads using separate fixture-relative placements."""
import argparse
import contextlib
import itertools
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step3_scheculer.run_independent import root, SCHEMA
from step3_scheculer.floor_margin import SCHEMA as FLOOR_MARGIN_SCHEMA
from step3_scheculer.pair_tasks import read_task
from step4_connect_support.baseline_current import head_registration as H, build_local_bodies as L
from step4_connect_support.baseline_current import build_coupled_saddle as S
from step4_connect_support.baseline_current.fixture_view import cells_for, frame, local_to_world
from step4_connect_support.baseline_current.run_sequential_k import foot_menu, plain
from step0_pose_selection.floor_points import pressure_centers
from step2_local_support import geometry as G, withdrawal as W


def read_case(name, poses, output, *, independent_root=None):
    source = output/'data/independent_input'
    source.mkdir(parents=True, exist_ok=True)
    tasks, groups, heads, catalogues, menus, paths, reports = [], [], [], [], [], [], []
    source_root = Path(independent_root) if independent_root is not None else root(name)
    for pose in poses:
        folder = source_root/pose
        report_path = folder/'step3_scheculer/schedule.json'
        report = I.check_report(report_path)
        if report['schema'] not in (SCHEMA,FLOOR_MARGIN_SCHEMA) or report['poses'] != [pose] or report['shared_heads']:
            raise ValueError('Expected one independent head group per pose')
        task = read_task(name, pose, folder=folder/'step_1_needs')
        contact_path = report_path.parent/f'final_contacts_{pose}.npz'
        contacts = I.read_contacts(contact_path)
        if not contacts:
            raise ValueError(f'{pose} has no selected heads')
        metadata = {h['id']:h for h in report['heads']}
        solids = [cells_for(c,task.domain,G.vertex_offsets(task.domain.mesh,
                  metadata[c['candidate_id']]['normal_depth_m'])[0]) for c in contacts]
        description = folder/'step2_local_support'/f'candidates_{pose}.json'
        catalogue = json.loads(description.read_text())['direction_catalogue']
        menu = report['result']['geometry']['per_pose'][0]['common_direction_ids']
        if not menu:
            raise ValueError(f'{pose} has no certified active-head withdrawal')
        shutil.copyfile(contact_path, source/f'contacts_{pose}.npz')
        tasks.append(task); groups.append(contacts); heads.append(solids)
        catalogues.append(np.asarray(catalogue['vectors'])); menus.append(menu)
        paths.extend([report_path,contact_path,description]+list(task.inputs)); reports.append(report)
    identifiers = [c['candidate_id'] for group in groups for c in group]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError('Independent groups must not share a head ID')
    schedule = dict(complete=True, passed=all(r['result']['passed'] for r in reports),
        particle=0, selected_ids=identifiers, poses=poses,
        active_ids_by_pose=[[c['candidate_id'] for c in g] for g in groups],
        covered_counts=[r['result']['covered_counts'][0] for r in reports],
        source_reports=[str((source_root/p/'step3_scheculer/schedule.json').relative_to(I.ROOT)) for p in poses],
        head_model=reports[0].get('head_model', 'finite_thickness_contact_head'),
        model='Independent groups; placements are Step4 variables; no shared heads',
        inherited_cross_pose_directions=False)
    I.save(source/'schedule.json', schedule)
    paths += [source/'schedule.json']+list(source.glob('contacts_*.npz'))
    return SimpleNamespace(name=name, poses=poses, pair=output.parent, source=source,
        tasks=tasks, groups=groups, heads=heads, schedule=schedule, paths=paths,
        catalogues=catalogues, menus=menus, source_reports=reports,
        demands=[pressure_centers(t.targets/t.scale,t.domain.com)[0] for t in tasks])


def placement_for(case, direction_ids, gap=.04):
    """Parallel exit lanes; each pose retains its original world orientation.

    Row vectors: fixture = task_world @ basis + offset. A separate yaw/XY
    placement per task aligns its certified head exit to fixture +X. The object
    exits toward fixture -X. Separate Y lanes leave inactive heads elsewhere.
    This is a simple placement construction, not a compactness optimizer.
    """
    vectors = [cat[i] for cat,i in zip(case.catalogues,direction_ids)]
    bases = np.asarray([frame(d) for d in vectors])
    offsets, cursor, lane_bounds = [], 0., []
    for task, groups, demand, basis in zip(case.tasks,case.heads,case.demands,bases):
        points = np.vstack([task.domain.mesh.vertices]+[v for hs in groups for v in hs])@basis
        # The largest foot menu extends by 110 mm; reserve its lateral span.
        feet = np.c_[demand,np.zeros(len(demand))]@basis
        low = min(points[:,1].min(), feet[:,1].min()-.114)
        high = max(points[:,1].max(), feet[:,1].max()+.114)
        offset = np.array([-points[:,0].mean(), cursor-low, 0.])
        offsets.append(offset); lane_bounds.append([float(cursor),float(cursor+high-low)])
        cursor += high-low+gap
    offsets = np.asarray(offsets)
    offsets[:,1] -= cursor*.5
    return dict(bases=bases, offsets=offsets, directions=np.asarray(vectors),
        direction_ids=list(direction_ids), layout='independent_parallel_exit_lanes',
        lane_gap_m=gap, uncentered_lane_bounds_m=lane_bounds)


def verify_placed_heads(case, placement):
    bases, offsets = placement['bases'],placement['offsets']
    registered, registration = H.register(case.groups,case.heads,bases,offsets,general_layout=True)
    checks = []
    for k, task in enumerate(case.tasks):
        cells = [SimpleNamespace(vertices=local_to_world(v,bases[k],offsets[k]))
                 for h in registered for v in h.cells]
        analyzer = W.Analyzer(task.domain.mesh,.01*task.domain.mesh.extents.max(),
                              dict(vectors=[placement['directions'][k].tolist()]))
        check = analyzer.test(cells,placement['directions'][k])
        checks.append(dict(pose=task.pose,**check))
    return dict(passed=registration['passed'] and registration['all_heads_above_all_floors']
                and all(c['clear'] for c in checks), registration=registration, per_pose=checks)


def placement_menu(case):
    # Preserve independent Step3 menus. Only the actual Step4 placement decides
    # whether idle material blocks an exit; never map it via object pose frames.
    ranks = [[ids[len(ids)//2]]+ids for ids in case.menus]
    ranks = [list(dict.fromkeys(ids)) for ids in ranks]
    choices = [tuple(ids[0] for ids in ranks)]
    # A small, deterministic menu changes every lane's own choice together.
    choices += [tuple(ids[min(j,len(ids)-1)] for ids in ranks) for j in range(1,4)]
    for ids in dict.fromkeys(choices):
        yield placement_for(case,ids)


def run(name, poses, output=None):
    label = 'pose'+'+'.join(p.split('_')[1] for p in poses)
    output = Path(output) if output else I.OUTPUTS/name/label/'step4'
    output.mkdir(parents=True,exist_ok=True)
    data = output/'data'; data.mkdir(exist_ok=True)
    case = read_case(name,poses,output)
    report = dict(schema='independent_pose_connected_shape_v1', complete=False,
        object=name,poses=poses,constructed=False,passed=False,
        step3_passed=case.schedule['passed'],covered_counts=case.schedule['covered_counts'],
        shared_head_count=0, physical_head_count=len(case.schedule['selected_ids']),
        source_schedule=str((case.source/'schedule.json').relative_to(I.ROOT)), attempts=[])
    I.save(data/'independent_report.json',report)
    success = False
    for placement_index,placement in enumerate(placement_menu(case)):
        head_check = verify_placed_heads(case,placement)
        if not head_check['passed']:
            report['attempts'].append(dict(placement=plain(placement),head_check=head_check))
            continue
        for feet_index,feet in enumerate(foot_menu(case,placement['bases'],placement['offsets'])):
            attempt = dict(placement_index=placement_index,feet_index=feet_index,
                placement=plain(placement),head_check=head_check)
            report['attempts'].append(attempt)
            cache = data/'independent_cache'/f'placement_{placement_index}'
            with tempfile.TemporaryDirectory(prefix='cadgrasp-independent-body-') as tmp:
                work = Path(tmp)
                try:
                    L.build(work,feet,case=case,placement=placement,general_layout=True,
                        skip_unreachable=True,floor_policy='nearest',verify=True,allow_failed=True,
                        cache_dir=cache,export_stl=False)
                except (RuntimeError,ValueError) as error:
                    attempt.update(constructed=False,error=str(error))
                    I.save(data/'independent_report.json',report)
                    print('ATTEMPT FAILED',placement_index,feet_index,str(error),flush=True)
                    continue
                built = json.loads((work/'report.json').read_text())
                attempt.update(constructed=True,passed=built['passed'],volume_cm3=built['volume_cm3'])
                # Always save an actual constructed shape, including when the
                # fixed input heads cannot certify every load. Never call a
                # failed equilibrium a successful final fixture.
                if not success or built['passed']:
                    destination=data/'independent_body'
                    if destination.exists():shutil.rmtree(destination)
                    shutil.copytree(work,destination)
                    shutil.copyfile(work/'fixture.obj',output/'shape.obj')
                    shutil.copyfile(work/'index.html',output/'shape.html')
                    report.update(constructed=True,passed=bool(built['passed']),
                        complete_fixture_verified=bool(built['passed']),
                        status='verified_connected_shape' if built['passed'] else 'connected_shape_failed_acceptance',
                        placement=plain(placement),head_check=head_check,construction=built,
                        volume_cm3=built['volume_cm3'],solid=built['solid'])
                    success=True
                I.save(data/'independent_report.json',report)
                if built['passed'] or not case.schedule['passed']:
                    break
        if report.get('passed') or (success and not case.schedule['passed']):break
    report.update(complete=True,verification=dict(performed=success,mode='full' if success else 'construction_failed'),
        provenance=dict(inputs=I.hashes(case.paths),code=I.hashes([Path(__file__),Path(H.__file__),Path(L.__file__),Path(S.__file__)])))
    if not success:report['status']='body_construction_failed'
    if success:
        report['artifacts']={f'../{n}':I.sha256(output/n) for n in ('shape.obj','shape.html')}
    I.save(data/'independent_report.json',report)
    I.save(data/'report.json',report)
    print('SHAPE COMPLETE',label,report['status'],report.get('volume_cm3'),flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object',nargs='?',default='B')
    parser.add_argument('--poses',nargs='+',required=True)
    args=parser.parse_args()
    run(args.object,args.poses)
