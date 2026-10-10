"""Give an OWN saved Step4 candidate more time for unchanged final checks.

No search, changed layout, load subsampling or tolerance change. Keep the
original timed-out attempts and record this separate continuation budget.
"""
import _bootstrap
import argparse,subprocess,sys,time
from co_common import *
from run_all import saved_groups,output_root
from whole_pipeline import load_layout,stage_record,SCHEMA
from whole_search.fast_search import FastModel
from whole_search.model import Model
from whole_search.search import failed_count
from whole_search.exact_worker import read_result
from whole_search.common import tangent_frame,legal_direction
from whole_search.reuse_first import registered,juxtaposed_count
from whole_step4_render import render_search


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object');parser.add_argument('group')
    parser.add_argument('--timeout',type=float,default=600.)
    parser.add_argument('--direction-nudge',type=float,default=0.,help='Additional coherent Direction step in degrees, fully checked before export')
    parser.add_argument('--axis',type=int,choices=[0,1],default=0)
    args=parser.parse_args();began=time.monotonic()
    group=next(g for g in saved_groups(args.object) if g['id']==args.group)
    base=output_root(args.object)/group['id'];out=base/'step4/step4.2'
    initial=I.check_report(base/'step4/step4.1/data/report.json')
    sampled=json.loads((out/'search_report.json').read_text())
    assert sampled['poses']==group['poses'] and sampled['sampled_force_passed']
    layout=load_layout(out/'sampled_layout.npz')
    assert layout.active==tuple(range(len(group['poses'])))
    stamp=time.strftime('%Y%m%d_%H%M%S')
    folder=out/'data/validation_continuations'/stamp;folder.mkdir(parents=True)
    source=out/'sampled_layout.npz'
    model=FastModel(group['poses'],args.object,initialization_report=base/'step3/step3.1/data/report.json')
    original_layout=layout.copy()
    if args.direction_nudge:
        for k in layout.active:
            layout.directions[k]=legal_direction(layout.directions[k]+np.tan(np.radians(args.direction_nudge))*tangent_frame(layout.directions[k])[:,args.axis],model.floor_normal(layout,k))
        checked_sample=model.evaluate(layout)
        assert failed_count(checked_sample)==0,'Direction nudge did not preserve all original sampled loads'
        source=folder/'direction_nudge_layout.npz'
        np.savez_compressed(source,placements=layout.placements,directions=layout.directions,hosts=layout.hosts,active=np.array(layout.active))
    env=os.environ.copy();env['COOPT_REPO_ROOT']=str(ROOT)
    env['PYTHONPATH']=str(HERE/'helper_func')+os.pathsep+env.get('PYTHONPATH','')
    generation=output_root(args.object)/'data/whole_step4_active_run.json'
    env['COOPT_WHOLE_RUN_TOKEN']=json.loads(generation.read_text())['run_token']
    command=[sys.executable,'-u','-m','whole_search.exact_worker','--object',args.object,
        '--poses',','.join(group['poses']),'--initialization-report',str(base/'step3/step3.1/data/report.json'),
        '--layout',str(source),'--out',str(folder/'result')]
    record=dict(timeout_seconds=args.timeout,new_search_run=False,changed_layout=bool(args.direction_nudge),
        additional_direction_operation_degrees=args.direction_nudge,direction_axis=args.axis,
        original_failed_attempts_preserved=True,
        provenance=provenance([source,out/'search_report.json',out/'final_validation_attempts.json'],[Path(__file__)]))
    save(folder/'request.json',record)
    try:
        with (folder/'worker.log').open('w') as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout,check=True)
    except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as error:
        save(folder/'failure.json',dict(passed=False,error=str(error),seconds=time.monotonic()-began,request=record))
        raise
    actual=read_result(folder/'result',1)
    assert failed_count(actual)==0 and all(r['passed'] for r in actual['actual_work_surface_checks'])
    if initial['force_and_path_initialization_gate']:
        assert actual['volume_cm3']<=initial['volume_cm3']+1e-4
    if args.direction_nudge:
        process_path=out/'process.json';process=json.loads(process_path.read_text())
        (folder/'original_process.json').write_text(process_path.read_text())
        index=len(process);saved=out/f'process_states/{index:03d}.npz'
        np.savez_compressed(saved,placements=layout.placements,directions=layout.directions,hosts=layout.hosts,active=np.array(layout.active))
        changes=[dict(pose=group['poses'][k],direction_change_degrees=float(np.degrees(np.arccos(np.clip(original_layout.directions[k]@layout.directions[k],-1,1)))),translation_change_mm=0.,old_host=group['poses'][layout.hosts[k]],host=group['poses'][layout.hosts[k]],rehosted=False) for k in layout.active]
        process.append(dict(index=index,phase='verified_direction_nudge',layout=str(saved.relative_to(out)),serial=actual['serial'],active_poses=group['poses'],counts={group['poses'][k]:32768 for k in layout.active},failed_load_count=0,estimated_volume_cm3=actual['volume_cm3'],volume_epoch=-1,changes=changes,rotating_reuse_pose_count=sum(registered(layout,k) for k in layout.active),juxtaposed_pose_count=juxtaposed_count(layout),geometry_verified=True,elapsed_seconds=time.monotonic()-began))
        save(process_path,process)
    fields=['initialization','initial_counts','events','sample_evaluations','timing','search_seconds',
        'fixture_placement_count_is_not_cost','all_pose_juxtapose_disabled','separated_fallback_used']
    extra={key:sampled[key] for key in fields if key in sampled}
    extra.update(strategy='whole',policy='rotate-first-delta-search-final-exact-validation',
        final_acceptance_run=True,exact_evaluations_during_search=0,
        rotating_reuse_pose_count=sum(registered(layout,k) for k in layout.active),
        juxtaposed_pose_count=juxtaposed_count(layout),
        selected_earlier_sampled_state=False,status='pass',passed=True,
        step41_volume_cm3=initial['volume_cm3'],step41_counts=initial['counts'],
        validation_continuation=record|dict(seconds=time.monotonic()-began,
            result_files_provenance=provenance([folder/'result.npz',folder/'result.json'],[])),
        initial_validation_attempts=json.loads((out/'final_validation_attempts.json').read_text()))
    report=Model.save(model,actual,out,extra)
    render_search(model,out,final_result=report)
    report=stage_record(out,report,group,'step4.2')
    row=dict(complete=True,schema=SCHEMA,id=group['id'],poses=group['poses'],status='pass',passed=True,
        step41_force_passed=initial['force_and_path_initialization_gate'],step41_volume_cm3=initial['volume_cm3'],
        final_volume_cm3=report['volume_cm3'],rotating_poses=report['rotating_reuse_pose_count'],
        juxtaposed_poses=report['juxtaposed_pose_count'],final_counts=report['counts'],
        sampled_evaluations=report['sample_evaluations'],seconds=time.monotonic()-began,
        seconds_are_validation_continuation_only=True,full_fixture_accepted=False)
    save(base/'step4/data/report.json',row);(base/'step4/data/failure.json').unlink(missing_ok=True)
    (out/'README.md').write_text('# Whole：原候选继续几何检查\n\n'
        '[过程图](process.png) · [最终各pose](final_result.png)\n\n'
        f"原搜索已完成；原三个180s超时检查保留。本次预算{args.timeout:g}s；额外Direction步幅{args.direction_nudge:g}°，不改变载荷或容差。"
        f"最终原载荷全部通过，实际支撑{report['volume_cm3']:.3f}cm³。\n\n"
        '所有pose从初始化共同参与。过程图是保存的名义布局，最后一格及最终图为真实净空支撑。'
        '继续检查记录见data/validation_continuations；整件接地、连通和强度尚未验收。\n')
    print('VALIDATION CONTINUED',group['id'],'PASS',report['volume_cm3'],flush=True)


if __name__=='__main__':main()
