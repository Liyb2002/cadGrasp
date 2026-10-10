"""An explicit small world-up Direction margin for numerical initialization.

All poses remain registered; no change to original loads, material domain,
work exclusions or geometric tolerances. Preview never replaces an active
case. Publish is a separately recorded continuation after that run ends.
"""
import _bootstrap
import argparse,time
from co_common import *
from run_all import saved_groups,output_root
from whole_pipeline import optimize,stage_record,SCHEMA
from whole_search.fast_search import FastModel
from whole_search.model import Model
from whole_search.exact_worker import write_result,read_result
from whole_search.search import failed_count
from whole_search.common import legal_direction
from whole_step4_render import render_initialization


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object');parser.add_argument('group')
    parser.add_argument('--degrees',type=float,default=.25)
    parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    assert 0<args.degrees<=5.
    group=next(g for g in saved_groups(args.object) if g['id']==args.group)
    base=output_root(args.object)/group['id'];out=base/'step4/step4.1'
    folder=out/'data/upward_margin_recovery'/f'{args.degrees:g}deg'
    folder.mkdir(parents=True,exist_ok=True);began=time.monotonic()
    program=folder/'executed_helper.py'
    if not program.exists():program.write_text(Path(__file__).read_text())
    source=base/'step3/step3.1/data/report.json';I.check_report(source)
    model=FastModel(group['poses'],args.object,initialization_report=source)
    layout,metadata=model.initial()
    for k in layout.active:
        normal=model.floor_normal(layout,k)
        layout.directions[k]=legal_direction(layout.directions[k]+np.tan(np.radians(args.degrees))*normal,normal)
    os.environ['COOPT_WHOLE_RUN_TOKEN']=json.loads((output_root(args.object)/'data/whole_step4_active_run.json').read_text())['run_token']
    if (folder/'result.json').exists():actual=read_result(folder/'result',1)
    else:
        model.checkpoint_dir=folder/'exact_states';model.worker_program=HERE/'helper_func/whole_search/exact_worker.py'
        actual=model.exact(layout);write_result(actual,folder/'result')
    metadata.update(numerical_geometry_recovery=True,
        recovery_policy='tiny per-pose world-up Direction margin; all registered placements unchanged',
        recovery_degrees=args.degrees,recovery_provenance=provenance([source,folder/'result.npz',folder/'result.json'],[program]))
    save(folder/'report.json',dict(constructed=True,published=args.publish,counts=actual['counts'],
        original_model_unchanged=True,metadata=metadata,seconds=time.monotonic()-began))
    print('INITIALIZATION RECOVERY',group['id'],'CONSTRUCTED',actual['counts'],flush=True)
    if not args.publish:return
    # The caller must wait for the original case worker to stop before publish.
    failure=base/'step4/data/failure.json'
    assert failure.exists(),'Publish only after the original case is recorded unresolved'
    (folder/'original_case_failure.json').write_text(failure.read_text())
    write_result(actual,out/'data/initial_geometry')
    initial=Model.save(model,actual,out,dict(initialization=metadata,status='initialized',passed=True,
        construction_completed=True,force_and_path_initialization_gate=failed_count(actual)==0,
        initial_counts={model.poses[k]:int(actual['masks'][k].sum()) for k in layout.active},
        step4_ready=True,optimized=False,diagnostics_are_acceptance_gates=False))
    render_initialization(model,actual,out);initial=stage_record(out,initial,group,'step4.1')
    (out/'README.md').write_text('# Whole Step4.1：数值构造接续\n\n'
        '[方向](exit_directions.png) · [退出sweep](exit_sweeps.png) · [共享支撑](final_result.png)\n\n'
        f'原初始化未完成；给原方向加入每个pose自己的{args.degrees:g}°小离地分量。注册位置、载荷、工作禁区和容差不变；完整接续记录在data/upward_margin_recovery。\n')
    options=dict(iterations=8,finalists=3,branch_rounds=3,seed=42,volume_rounds=3,screen_budget=96)
    final=optimize(model,layout,metadata,actual,group,base,options)
    row=dict(complete=True,schema=SCHEMA,id=group['id'],poses=group['poses'],status=final['status'],
        passed=final['passed'],step41_force_passed=initial['force_and_path_initialization_gate'],
        step41_volume_cm3=initial['volume_cm3'],final_volume_cm3=final['volume_cm3'],
        rotating_poses=final['rotating_reuse_pose_count'],juxtaposed_poses=final['juxtaposed_pose_count'],
        final_counts=final['counts'],sampled_evaluations=final['sample_evaluations'],
        seconds=time.monotonic()-began,continued_numerical_initialization=True,full_fixture_accepted=False)
    save(base/'step4/data/report.json',row);failure.unlink(missing_ok=True)
    print('RECOVERED WHOLE',group['id'],row['status'],flush=True)


if __name__=='__main__':main()
