"""Run computed-task Step4.1 and station-constrained Co-optimize Step4.2."""
import argparse
import json
from pathlib import Path
import time
import subprocess
import sys
import numpy as np

from whole_task.step1.common import DEFAULT_OUTPUT, write_json, sha256
from whole_task.step3.legacy import kernel as K, loaded_sources
from .adapter import TaskModel
from .search import StationSearch, context
from whole_search.model import Model
from whole_search.exact_worker import read_result
from whole_search.common import result_mesh, transform_mesh
from whole_pipeline import initialization_trials
from whole_step4_render import render_initialization, sheet, raster_tile, layers_for_pose
from material_force import certify_material_contacts
from whole_task.step2.wrenches import endpoint_demands


POLICY=Path(__file__).with_name('policy.json')


def save_geometry(model,result,folder):
    folder.mkdir(parents=True,exist_ok=True)
    support=result_mesh(result)
    K.D.export_exact_obj(support,folder/'support.obj')
    np.savez_compressed(folder/'layout.npz',**result['layout'].arrays(model.native))
    for k in result['layout'].active:
        np.savez_compressed(folder/f'{model.poses[k]}_force.npz',mask=result['masks'][k],supply_7d=result['supplies'][k])
    return support


def run(output_root=DEFAULT_OUTPUT,object_name='B',rounds=None,population=None,nominal_diagnostic=False):
    began=time.monotonic();policy=json.loads(POLICY.read_text())
    if rounds is not None:policy['maximum_rounds']=rounds
    if population is not None:policy['population_size']=population
    policy['nominal_diagnostic_trial']=bool(nominal_diagnostic)
    if policy['maximum_rounds']<0 or policy['population_size']<3:raise ValueError('Invalid search budget')
    out=Path(output_root).resolve()/object_name/'step4'
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'run_status.json',dict(status='running',policy=policy))
    print('Step4: loading computed poses, loads and current Step3.1',flush=True)
    model=TaskModel(name=object_name,output_root=output_root)
    q,metadata=model.initial();failures=[]
    first=out/'step4.1';first.mkdir(exist_ok=True);(first/'data').mkdir(exist_ok=True)
    print('Step4.1: constructing complete insertion/withdrawal paths',flush=True)
    trials=[] if nominal_diagnostic else initialization_trials(model,q)
    trial=-1
    for trial,layout in enumerate(trials):
        try:
            worker_layout=first/'data'/f'trial_{trial}_layout.npz'
            np.savez_compressed(worker_layout,**layout.arrays(model.native))
            prefix=first/'data'/f'trial_{trial}_result'
            with (first/'data'/f'trial_{trial}.log').open('w')as log:
                try:
                    process=subprocess.run([sys.executable,'-B','-m','whole_task.step4.exact_worker',
                        '--output-root',str(Path(output_root).resolve()),'--object',object_name,
                        '--layout',str(worker_layout),'--out',str(prefix)],stdout=log,stderr=subprocess.STDOUT,timeout=180)
                except subprocess.TimeoutExpired as error:
                    raise RuntimeError('Exact geometry candidate exceeded the reference 180-second limit')from error
            if process.returncode:
                raise RuntimeError(f'Exact worker failed; see {first / "data" / f"trial_{trial}.log"}')
            actual=read_result(prefix,trial+1);q=layout;break
        except (RuntimeError,ValueError,AssertionError) as error:
            failures.append(dict(trial=trial,error=str(error)))
            print('Step4.1 numerical recovery',trial,str(error),flush=True)
    else:
        if not nominal_diagnostic:raise RuntimeError('Step4.1 construction unresolved: '+str(failures))
        from whole_step4_render import NominalBuilder
        from whole_search.common import unpack_solid, transform_solid
        solid=NominalBuilder(model).solid(q);mesh=unpack_solid(solid)
        proof=certify_material_contacts(model,q,mesh)
        overlaps=[]
        for k in q.active:
            d=q.placements[k,:3,:3].T@q.directions[k]
            overlaps.append(K.material_volume(solid ^ transform_solid(model.clearance.sweep(model.length*d,padded=False),q.placements[k])))
        if max(overlaps)>1e-10:raise RuntimeError('Nominal diagnostic has an exit collision')
        actual=dict(proof,layout=q,remaining_mesh=mesh,remaining=solid,
                    volume_cm3=K.material_volume(solid)*1e6,
                    diagnostics=dict(nominal_sweep_overlap_m3=max(overlaps),additional_clearance_certified=False))
    save_geometry(model,actual,first)
    render_initialization(model,actual,first)
    if nominal_diagnostic:
        path=first/'data/render.json';visual=json.loads(path.read_text())
        visual['final_support_policy']='nominal complete-path geometry; additional 1% clearance unresolved'
        write_json(path,visual)
    write_json(first/'report.json',dict(status='nominal_diagnostic' if nominal_diagnostic else 'initialized',initialization=metadata,
        additional_clearance_certified=not nominal_diagnostic,
        numerical_recovery_trial=trial,construction_failures=failures,counts=actual['counts'],
        geometry=actual['diagnostics'],volume_cm3=actual['volume_cm3'],
        directions_fixture=q.directions.tolist(),fixed_task_poses=True,full_fixture_accepted=False))
    print('Step4.1 actual-contact counts',actual['counts'],flush=True)
    second=out/'step4.2';second.mkdir(exist_ok=True)
    search=StationSearch(context(model,second/'search',iterations=policy['maximum_rounds'],
                                population=policy['population_size'],seed=policy['random_seed']))
    print('Step4.2: evaluating connected contact core and searching coupled layouts',flush=True)
    initial=model.evaluate(q)
    selected=search.solve(initial)
    core=save_geometry(model,selected.result,second)
    core=K.trimesh.load(second/'support.obj',force='mesh',process=False)
    # Final records always reflect polygons of the actual saved core, even
    # when the search ended without a complete solid.
    proof=certify_material_contacts(model,selected.layout,core)
    endpoint_counts={}
    for k in selected.layout.active:
        endpoints=endpoint_demands(model.tasks[k].domain)['endpoint_need_wrench']*model.tasks[k].scale
        mask,_=K.J.classify(proof['supplies'][k],endpoints)
        endpoint_counts[model.poses[k]]=dict(passed=int(mask.sum()),total=len(mask))
        np.savez_compressed(second/f'{model.poses[k]}_force.npz',mask=proof['masks'][k],
            supply_7d=proof['supplies'][k],endpoint_mask=mask)
    tiles=[raster_tile(layers_for_pose(model,selected.layout,core,k)) for k in selected.layout.active]
    sheet(tiles,second/'overview.png')
    if selected.solid is not None:
        K.D.export_exact_obj(selected.solid,second/'complete_support.obj')
        if selected.completed:
            np.savez_compressed(second/'complete_force_certificate.npz',
                **{f'mask_{k}':v for k,v in selected.full_certificate['masks'].items()},
                **{f'supply_{k}':v for k,v in selected.full_certificate['supplies'].items()})
        sheet([raster_tile(layers_for_pose(model,selected.layout,selected.solid,k))
               for k in selected.layout.active],second/'complete_overview.png')
    if selected.witness is not None:K.D.export_exact_obj(selected.witness,second/'legal_component.obj')
    write_json(second/'oracle.json',selected.oracle)
    write_json(second/'construction.json',selected.construction or dict(passed=False,attempted=False))
    source=model.upstream_report
    inputs=dict(source['input_sha256']);inputs[str(model.initialization_report)]=sha256(model.initialization_report)
    for name,digest in source['artifact_sha256'].items():inputs[str(model.initialization_report.parent/name)]=digest
    for path in [POLICY,*loaded_sources(),*Path(__file__).parent.glob("*.py")]:inputs[str(path)]=sha256(path)
    report=dict(schema='whole_task_fixed_station_step4_v1',object=object_name,output_root=str(Path(output_root).resolve()),
        status=('nominal_candidate_only' if nominal_diagnostic else 'pass') if selected.completed else 'search_budget_exhausted',
        full_fixture_accepted=bool(selected.completed and not nominal_diagnostic),
        nominal_complete_solid_accepted=selected.completed,additional_initialization_clearance_certified=not nominal_diagnostic,
        policy=policy,poses=model.poses,step41_counts=actual['counts'],
        final_core_counts={model.poses[k]:int(v.sum())for k,v in proof['masks'].items()},
        continuous_endpoint_counts=endpoint_counts,core_volume_cm3=abs(float(core.volume))*1e6,
        core_connected=selected.result.get('connected_contact_core'),
        base_oracle_status=selected.oracle['status'],construction=selected.construction,
        evaluation=selected.evaluation,iterations=len(search.audit),
        adaptations=['computed Step2 paired loads and poses','finite Step3 surface-to-head access',
            'coupled seat/installation moves preserving every robot world task',
            'bounded station-preserving local direction and translation proposals'],
        global_infeasibility_claimed=False,source_geometry_review_required=source['source_geometry_review_required'],
        seconds=time.monotonic()-began,input_sha256=inputs,
        artifact_sha256={str(p.relative_to(out)):sha256(p)for p in out.rglob('*')
                         if p.is_file()and p not in (out/'report.json',out/'run_status.json',out/'validation.json')})
    write_json(out/'report.json',report)
    write_json(out/'run_status.json',dict(status=report['status'],seconds=report['seconds']))
    print('Step4 result',report['status'],report['final_core_counts'],flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT);p.add_argument('--object',default='B')
    p.add_argument('--rounds',type=int);p.add_argument('--population',type=int)
    p.add_argument('--nominal-diagnostic',action='store_true',help='Explicit experiment without certified extra 1% clearance; cannot produce full PASS')
    a=p.parse_args()
    try:run(a.output_root,a.object,a.rounds,a.population,a.nominal_diagnostic)
    except Exception as error:
        folder=a.output_root/a.object/'step4';folder.mkdir(parents=True,exist_ok=True)
        write_json(folder/'run_status.json',dict(status='unresolved',error=f'{type(error).__name__}: {error}'))
        raise


if __name__=='__main__':main()
