"""Optimize the accepted Step3.2; publish exactly one flat overview PNG."""
import argparse
from pathlib import Path
import json
import shutil
import time
import numpy as np
import trimesh

from whole_task_2.step1.common import ROOT,DEFAULT_OUTPUT,sha256,write_json
from whole_task_2.step3 import independent_geometry as G
from whole_task_2.operations.construction import CompactConstructionModel, overlaps
from .optimize import Optimizer
from .validate_optimized import validate
from whole_task_2.operations.installations import floor_angles,standing_pairs

POLICY = json.loads(Path(__file__).with_name('policy.json').read_text())


def material_reuse(candidate,model):
    support=G.solid(candidate['support'])
    tasks=candidate['tasks']
    rows=[]
    bounds=[G.mesh(t['stock']).bounds for t in tasks]
    for i in range(len(tasks)):
        for j in range(i):
            if overlaps(bounds[i],bounds[j]):
                # Shared casting backing must be above BOTH installed feet;
                # there is no universal material-z plate after a side turn.
                lo,hi=candidate['support'].bounds
                above=G.box(lo-.001,hi+.001)
                try:
                    for k in (i,j):
                        M=tasks[k]['material'];n=M[:3,2]
                        above=above.trim_by_plane(n,float(n@M[:3,3]+model.policy['plate_thickness_m']+1e-8)/G.SCALE)
                    shared=support^tasks[i]['stock']^tasks[j]['stock']^above
                    volume=G.volume(shared)*1e6
                except RuntimeError:
                    # This optional measurement is not a legality criterion.
                    # Preserve an unresolved result rather than inventing a
                    # shared-volume number or aborting a certified export.
                    rows.append(dict(tasks=[model.ids[j],model.ids[i]],shared_above_plate_cm3=None,
                        numerical_status='unresolved optional intersection; not used for acceptance'))
                    continue
                if volume>1.:
                    rows.append(dict(tasks=[model.ids[j],model.ids[i]],shared_above_plate_cm3=float(volume)))
    return rows


def run_legacy(output_root=DEFAULT_OUTPUT,object_name='B',workdir=None,resume=False,
        phases=('sharing',),merge_trials=24,publish=True, *, rounds=32, samples=32,
        repair_rounds=1, seed=42, workers=4):
    began=time.monotonic()
    output_root=Path(output_root).resolve()
    data_root=ROOT/'whole_task_2/data'/object_name
    active=data_root/'step4_active.json'
    if workdir is None and resume and active.exists() and 'sharing' not in phases:
        workdir=json.loads(active.read_text())['workdir']
    workdir=Path(workdir or data_root/'step4_group_merges').resolve()
    workdir.mkdir(parents=True,exist_ok=True)
    model=CompactConstructionModel(output_root,object_name)
    # Publication without new proposals needs the fresh final disk replay
    # below; avoid running that same expensive source-prism proof twice.
    # Persistent populations already have certified parent checkpoints and
    # advisory contact signatures. New commits and the final replay still
    # undergo independent complete audits.
    audit_resume=bool(phases) and not ('sharing' in phases and
        (workdir/'population_seed_contacts.npz').exists())
    optimizer=Optimizer(model,workdir,resume,audit_resume=audit_resume)
    optimizer.solve(phases,merge_trials,rounds=rounds,samples=samples,
                    repair_rounds=repair_rounds,seed=seed,workers=workers)
    sharing_used='sharing' in phases or (workdir/'sharing_search.json').exists()
    sides=standing_pairs(optimizer.layout)
    if not sharing_used and phases and not sides:
        raise ValueError('Final design has no actual 90 degree whole-fixture installation')
    replay_model,candidate,validation=validate(workdir,output_root,object_name)
    support=candidate['support']
    mesh_path=workdir/'support.obj'
    mesh_path.write_text(trimesh.exchange.obj.export_obj(support,digits=17,include_normals=False))
    reloaded=trimesh.load(mesh_path,force='mesh',process=False)
    np.testing.assert_array_equal(reloaded.faces,support.faces)
    np.testing.assert_allclose(reloaded.vertices,support.vertices,atol=1e-15,rtol=0)
    execution=[]
    angles=floor_angles(optimizer.layout)
    seats=optimizer.layout.material@optimizer.layout.poses
    shared_seats=[dict(tasks=[model.ids[j],model.ids[i]],relative_floor_angle_deg=float(angles[i,j]))
                  for i in range(len(model.ids))for j in range(i)
                  if angles[i,j]>1e-5 and np.allclose(seats[i],seats[j],atol=1e-8,rtol=0)]
    for i,name in enumerate(model.ids):
        execution.append(dict(task=name,world_from_support=np.linalg.inv(optimizer.layout.material[i]).tolist(),
            fixture_from_object=(optimizer.layout.material[i]@optimizer.layout.poses[i]).tolist(),
            world_from_object=optimizer.layout.poses[i].tolist(),head_world_m=model.head.tolist(),
            original_working_face_ids=model.records[i]['working_face_ids'],
            exit_direction_world=optimizer.layout.exits[i].tolist(),exit_length_m=candidate['tasks'][i]['length']))
    source_paths=[*replay_model.inputs,Path(__file__).with_name('policy.json'),*Path(__file__).parent.glob('*.py'),
                  *(ROOT/'whole_task_2/operations').glob('*.py')]
    report=dict(status='passed',method=('random_family_union_from_step3' if sharing_used
                                       else 'fully_certified_monotone_volume_operations'),
        initial_volume_cm3=optimizer.initial_volume,final_volume_cm3=optimizer.volume,
        volume_reduction_fraction=1-optimizer.volume/optimizer.initial_volume,
        accepted_operations=optimizer.accepted,task_count=len(model.ids),execution=execution,
        selected_surface_union_fraction=model.report['selected_surface_union_fraction'],
        actual_material_reuse=material_reuse(candidate,replay_model),
        unchanged_original_task_regions=True,head_position_and_nominal_world_direction_fixed=True,
        forces_recomputed_in_current_world_pose=True,world_gravity_fixed=[0.,0.,-9.81],
        actual_contact_force_and_moment_pairs=True,continuous_whole_object_exits=True,
        max_relative_installation_floor_angle_deg=float(angles.max()),
        final_uses_nonparallel_installation_floors=bool(angles.max()>1e-5),
        final_uses_90_degree_side_installation=bool(sides),
        same_complete_fixture_for_all_tasks=True,shared_object_seat_task_pairs=shared_seats,
        side_installation_task_pairs=[[model.ids[i],model.ids[j]]for i,j in sides],
        design_requirements=optimizer.requirements,
        all_accepted_states_fully_certified=True,global_minimum_claimed=False,
        native_robot_motion_certified=False,fixture_stress_or_strength_certified=False,
        installation_search='SO(3) whole-fixture installations, including side faces, interleaved with task-level juxtaposition',
        base_design='8mm pressure patches, narrow side posts and base strips, clipped by every installed floor',
        final_construction=json.loads((workdir/f'state_{optimizer.accepted:03}.json').read_text()).get('construction',{}),
        baseline_reconstruction_in_first_translation=not sharing_used,
        geometry_tolerance_m3=model.tolerance,force_tolerance=2e-9,
        source_sha256={str(p):sha256(p)for p in source_paths},
        checkpoint_manifest_sha256=sha256(workdir/'current.json'),validation_sha256=sha256(workdir/'validation.json'),
        support_obj_sha256=sha256(mesh_path),seconds=time.monotonic()-began,
        timing_scope='this invocation, including resume replay; earlier invocations retained in the checkpoint trace')
    if (workdir/'requirements.json').exists():
        report['design_requirements_sha256']=sha256(workdir/'requirements.json')
    if (workdir/'sharing_search.json').exists():
        report['sharing_search']=json.loads((workdir/'sharing_search.json').read_text())
    if (workdir/'selection.json').exists():
        report['selection']=json.loads((workdir/'selection.json').read_text())
        report['selection_sha256']=sha256(workdir/'selection.json')
    if (workdir/'warm_start.json').exists():
        report['warm_start']=json.loads((workdir/'warm_start.json').read_text())
        report['reduction_from_warm_start_fraction']=1-optimizer.volume/report['warm_start']['volume_cm3']
    previous=data_root/'step4_spatial_best/report.json'
    if previous.exists() and previous.parent!=workdir:
        volume=json.loads(previous.read_text())['final_volume_cm3']
        report['previous_flat_result_volume_cm3']=volume
        report['reduction_from_previous_flat_result_fraction']=1-optimizer.volume/volume
    # Preserve a smaller published result while a cold-start experiment runs.
    if publish and active.exists():
        incumbent=json.loads(active.read_text())
        if incumbent['volume_cm3'] <= optimizer.volume+1e-5 and Path(incumbent['workdir']) != workdir:
            report['publication']=dict(status='retained_existing_smaller_design',
                published_volume_cm3=incumbent['volume_cm3'],workdir=incumbent['workdir'])
            publish=False
    report['published']=bool(publish)
    write_json(workdir/'report.json',report)
    if publish:
        from .overview import render
        image=render(replay_model,candidate,validation,workdir/'overview.png')
        out=output_root/object_name/'step4'
        if out.exists() and any(p.name!='overview.png'for p in out.iterdir()):
            archive=workdir/'legacy_output'
            if archive.exists():
                archive=workdir/f'previous_output_{time.time_ns()}'
            shutil.move(str(out),str(archive))
        out.mkdir(parents=True,exist_ok=True)
        shutil.copy2(image,out/'overview.png')
        assert sorted(p.name for p in out.iterdir())==['overview.png']
        report['overview_sha256']=sha256(out/'overview.png')
        report['overview']=str(out/'overview.png')
        write_json(workdir/'report.json',report)
        write_json(active,dict(workdir=str(workdir),state=optimizer.accepted,
            volume_cm3=optimizer.volume,report_sha256=sha256(workdir/'report.json'),
            overview_sha256=report['overview_sha256']))
    print('STEP4 PASS',json.dumps({k:report[k]for k in ('initial_volume_cm3','final_volume_cm3','volume_reduction_fraction','accepted_operations')},ensure_ascii=False),flush=True)
    return report


def run(output_root=DEFAULT_OUTPUT,object_name='B',workdir=None,resume=False,
        *,rounds=32,samples=32,repair_rounds=1,seed=42,workers=4,publish=True,target_max_groups=1):
    """Default Step4: 25% saving per merge and certified subgroup recombination."""
    from .membrane_search import run as run_membranes
    return run_membranes(output_root,object_name,workdir,rounds=rounds,samples=samples,
        workers=workers,repair_rounds=repair_rounds,seed=seed,resume=resume,publish=publish,target_max_groups=target_max_groups)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--object',default='B')
    p.add_argument('--workdir',type=Path)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--phases',nargs='+',choices=['sharing','poses','standing','task_turns','quarters','turns','merges','exits','translations'],default=['sharing'])
    p.add_argument('--merge-trials',type=int,default=24)
    p.add_argument('--rounds',type=int,default=POLICY['rounds'])
    p.add_argument('--samples',type=int,default=POLICY['samples_per_operation'])
    p.add_argument('--repair-rounds',type=int,default=POLICY['repair_rounds'])
    p.add_argument('--seed',type=int,default=POLICY['random_seed'])
    p.add_argument('--workers',type=int,default=POLICY.get('workers',4))
    p.add_argument('--no-publish',action='store_true')
    p.add_argument('--legacy-connected',action='store_true',help='Replay the historical common-floor model')
    p.add_argument('--target-groups',type=int,choices=[1,2],default=1)
    a=p.parse_args()
    if not a.legacy_connected:
        from .membrane_search import run as run_membranes
        return run_membranes(a.output_root,a.object,a.workdir,rounds=a.rounds,samples=a.samples,
            workers=a.workers,repair_rounds=a.repair_rounds,seed=a.seed,resume=a.resume,publish=not a.no_publish,
            target_max_groups=a.target_groups)
    run_legacy(a.output_root,a.object,a.workdir,a.resume,a.phases,a.merge_trials,not a.no_publish,
        rounds=a.rounds,samples=a.samples,repair_rounds=a.repair_rounds,seed=a.seed,workers=a.workers)


if __name__=='__main__':
    main()
