"""Run outside-in Step5 on saved layouts, with explicit thickness references."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from types import SimpleNamespace
import argparse,json,os,shutil,subprocess,sys,time
from feasible_pipeline import ROOT,COOPT,read,save,digest,relative,operation,deliver


def calibrate(path):
    sys.path[:0]=[str(COOPT/name)for name in ['helper_func','step5.2','step5.3','step4.2']]
    from periphery import structural_profile
    import trimesh
    refs=[]
    for label in ['pose1+2+3+4+5+6+7+8','pose1+2+3+4+6']:
        source=COOPT/'data/results/B'/label/'step5/step5.3/step5_show_all52_20261014_02/support_with_base.obj'
        profile,_,_=structural_profile(trimesh.load(source,force='mesh',process=False))
        refs.append(dict(set=label,source=relative(source),sha256=digest(source),**profile))
    # Select the largest measured sphere radius retaining at least 99% of the
    # structural core in one piece in BOTH references. All disconnected tips
    # remain in the report; this is a connector metric, not uniform wall gauge.
    indices=[i for i in range(len(refs[0]['profiles']))
        if all(r['profiles'][i]['separated_core_fraction']<=.01 for r in refs)]
    if not indices:raise ValueError('References have no common resolved connector radius')
    index=max(indices)
    save(path,dict(complete=True,pitch_m=refs[0]['pitch_m'],references=refs,
        guard_radius_m=refs[0]['profiles'][index]['radius_m'],
        maximum_separated_core_fraction=max(r['profiles'][index]['separated_core_fraction']for r in refs),
        criterion='sphere erosion of load-bearing connections; terminal fragments are reported separately',
        strength_certificate=False,uniform_wall_thickness_claimed=False))


def run_case(case,args,base):
    root=base/case['id']/'step5/feasible'/args.experiment
    root.mkdir(parents=True,exist_ok=True)
    common=dict(object=args.object,selection_summary=str(args.selection_summary),
        step51_experiment=args.step51_experiment,case=case)
    reference=args.complete_reference_experiment
    if reference is None:
        candidate=base/case['id']/'step5/step5.3/step5_show_all52_20261014_02/data/report.json'
        if candidate.exists()and read(candidate)['selected_step4_experiment']==case['experiment']:
            reference='step5_show_all52_20261014_02'
        else:reference=case.get('complete_reference_experiment')
    if reference:common['complete_reference_experiment']=reference
    baseline=root/'baseline'
    if (baseline/'report.json').exists():
        record=read(baseline/'report.json')
        assert record['passed']and digest(baseline/'support.obj')==record['support_sha256']
        first=dict(passed=True,output=relative(baseline),record=record)
    else:first=operation(dict(common,operation='baseline'),baseline)
    if not first['passed']:raise ValueError('Complete baseline failed: '+str(first))
    finished=root/'outside_in'
    payload=dict(common,checkpoint=str(baseline),output=str(finished),
        calibration=str(args.calibration),peel_resolution=args.peel_resolution)
    save(root/'outside_in.payload.json',payload)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    recovered=root/'outside_in_retained_interfaces'
    if (recovered/'report.json').exists():
        saved=read(recovered/'report.json')
        assert saved['passed']and digest(recovered/'support.obj')==saved['support_sha256']
        finished=recovered
    def invoke(worker,parameters,name):
        payload_path=root/(name+'.payload.json');save(payload_path,parameters)
        with (root/(name+'.log')).open('w')as log:
            return subprocess.run([sys.executable,str(COOPT/'step5'/worker),str(payload_path)],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if not (finished/'report.json').exists():
        worker='tube_worker.py'if args.geometry=='rounded-tubes'else'periphery_worker.py'
        status=invoke(worker,payload,'outside_in')
        if status.returncode:
            # Preserve the failed proposal and all accepted checkpoints. Repair
            # only actual finite interfaces, then repeat the unchanged checker.
            failed=finished/'candidate.obj'
            if args.geometry=='rounded-tubes'and failed.exists():
                recovery=dict(payload,output=str(recovered),failed_candidate=str(failed))
                if not(recovered/'report.json').exists():
                    status=invoke('recover_interfaces.py',recovery,'recover_interfaces')
                else:status=SimpleNamespace(returncode=0)
                if not status.returncode:finished=recovered
            if status.returncode:
                failure=read(finished/'failure.json')if(finished/'failure.json').exists()else{}
                raise ValueError('Outside-in worker failed; accepted checkpoints retained: '+str(failure or status.returncode))
    if args.geometry=='rounded-tubes':
        connected=root/'connected_width'
        parameters=dict(payload,parent_directory=str(finished),output=str(connected),
            parent_source_snapshot=str(base/'data'/('step5_feasible_'+args.experiment)/'executed_sources'))
        if not(connected/'report.json').exists():
            status=invoke('junction_finish.py',parameters,'connected_width')
            if status.returncode:
                raise ValueError('Connected-width finishing failed: '+str(read(connected/'failure.json')))
        finished=connected
    final_record=read(finished/'report.json')
    assert final_record['passed']and digest(finished/'support.obj')==final_record['support_sha256']
    stages=[dict(stage='baseline',**first),dict(stage='outside_in',passed=True,committed=True,
        output=relative(finished),record=final_record)]
    row=deliver(case,args,finished,stages,base)
    row.update(thickness_target_passed=final_record['thickness_target_passed'],
        input_contact_triangles=final_record['input_contact_triangles'],
        final_contact_triangles=final_record['final_contact_triangles'])
    report_path=ROOT/row['output']/'data/report.json';report=read(report_path)
    report.update(pipeline='outside_in_round_support_with_continuous_native_width_backbone'if args.geometry=='rounded-tubes'else'certified_contact_ports_then_round_bands_then_rounding',
        contact_pruning=next(r for r in final_record['history']if r['operation']=='outside_in_contact_pruning'),
        thickness_target_passed=final_record['thickness_target_passed'],
        thickness_calibration=relative(args.calibration),
        all_original_contacts_frozen=False,grid_pocket_cutting_used=False,
        finishing_wall_clock_cutoff_used=False,
        before_contact_metrics=final_record['before_contact_metrics'],
        after_contact_metrics=final_record['after_contact_metrics'],
        connector_quality=final_record['connector_quality'])
    if args.geometry=='rounded-tubes':
        report.update(unified_core_and_ground_construction=True,
            minimum_round_connector_radius_m=final_record['minimum_round_connector_radius_m'],
            thickness_certificate=final_record['thickness_certificate'],
            parent_execution_source_snapshots=final_record['parent_source_snapshots'])
        intermediate=base/case['id']/'step5/step5.2'/args.experiment/'data/report.json'
        record52=read(intermediate);record52.update(pipeline=report['pipeline'],
            thickness_certificate=final_record['thickness_certificate'],minimum_connector_core_connected=True)
        save(intermediate,record52)
    row['cropped_core_and_feet_committed']=False
    save(report_path,report)
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object',nargs='?',default='B');parser.add_argument('--experiment',required=True)
    parser.add_argument('--selection-summary',type=Path,default=COOPT/'data/results/B/data/step5_show_all52_20261014_02_inputs/selection.json')
    parser.add_argument('--step51-experiment',default='pso_all52_20261013_01_floor')
    parser.add_argument('--complete-reference-experiment',help='Explicit same-layout complete reference; otherwise auto-select or certify own Step4 full solid')
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--sets',nargs='+');parser.add_argument('--jobs',type=int,default=1)
    parser.add_argument('--peel-resolution',type=float,default=.0005)
    parser.add_argument('--geometry',choices=['rounded-tubes','skin'],default='rounded-tubes')
    parser.add_argument('--skip-images',action='store_true');parser.add_argument('--skip-evaluation',action='store_true')
    parser.add_argument('--calibrate-only',action='store_true')
    args=parser.parse_args()
    if Path(args.experiment).name!=args.experiment or args.jobs<1 or args.peel_resolution<=0:parser.error('Invalid experiment/jobs/resolution')
    base=COOPT/'data/results'/args.object;directory=base/'data'/('step5_feasible_'+args.experiment)
    directory.mkdir(parents=True,exist_ok=True)
    args.selection_summary=args.selection_summary.resolve()
    args.calibration=(args.calibration or directory/'thickness_calibration.json').resolve()
    if not args.calibration.exists():calibrate(args.calibration)
    if args.calibrate_only:return
    selection=read(args.selection_summary);assert selection['complete']
    cases=selection.get('results',selection.get('selected',[]))
    if args.sets:
        if set(args.sets)-{c['id']for c in cases}:parser.error('Unknown pose set')
        cases=[c for c in cases if c['id']in args.sets]
    archive=directory/'executed_sources'
    # Freeze sources before workers start, including lazily imported helpers.
    sources={}
    for folder in ['step5','step5.2','step5.3','step4.2','helper_func','evaluation','vis_func']:
        for source in (COOPT/folder).rglob('*.py'):
            target=archive/source.relative_to(ROOT)
            if target.exists():
                if digest(target)!=digest(source):raise ValueError('Source changed in a resumable experiment: '+str(source))
            else:target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
            sources[relative(source)]=digest(source)
    results=[];began=time.monotonic()
    def record():
        save(directory/'summary.json',dict(complete=len(results)==len(cases),group_count=len(cases),
            passed=sum(r['passed']for r in results),results=results,
            thickness_passed=sum(r.get('thickness_target_passed',False)for r in results),
            appearance_passed=all(r.get('thickness_target_passed',False)for r in results if r['passed']),
            selection=relative(args.selection_summary),seconds=time.monotonic()-began,
            stage_order=['certified_complete_baseline','outside_in_contact_pruning','actual_rounding',
                'continuous_full_width_backbone','fresh_full_physical_proof']))
    eligible=[]
    for case in cases:
        out=base/case['id']/'step5/step5.3'/args.experiment/'data/report.json'
        if out.exists():
            report=read(out);r=dict(id=case['id'],pose_count=case['pose_count'],experiment=case['experiment'],
                passed=True,status='PASS',output=relative(out.parent.parent),protected=report['provenance']['inputs'],
                thickness_target_passed=report['thickness_target_passed'])
            results.append(r)
        elif not case['passed']:results.append(dict(id=case['id'],pose_count=case['pose_count'],passed=False,status='STEP4_FAIL'))
        else:eligible.append(case)
    record()
    with ThreadPoolExecutor(max_workers=args.jobs)as pool:
        pending={pool.submit(run_case,c,args,base):c for c in sorted(eligible,key=lambda r:(r['pose_count'],r['id']))}
        for future in as_completed(pending):
            case=pending[future]
            try:row=future.result()
            except Exception as error:row=dict(id=case['id'],pose_count=case['pose_count'],passed=False,status='STEP5_ERROR',error=str(error))
            results.append(row);record();print('OUTSIDE_IN_RESULT',len(results),'/',len(cases),row,flush=True)
    protected={p:h for r in results for p,h in r.get('protected',{}).items()}
    changed=[p for p,h in protected.items()if digest(ROOT/p)!=h]
    code={}
    for row in results:
        if row['passed']:
            report=read(ROOT/row['output']/'data/report.json')
            for stage in report['optional_operations']:
                for p,h in stage.get('record',{}).get('executed_code',{}).items():
                    code[p]=h;target=archive/p
                    if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,target)
                    if digest(target)!=h:raise ValueError('Actual worker differs from archived source: '+p)
    source_changed=[p for p,h in code.items()if digest(ROOT/p)!=h]
    save(directory/'verification.json',dict(passed=not changed,protected_inputs=protected,
        changed=changed,executed_code=code,changed_sources=source_changed,
        source_and_execution_snapshot_hashes_match=not source_changed))
    if changed or source_changed or any(r['status']=='STEP5_ERROR'for r in results):raise SystemExit(1)
    valid=[c for c in cases if c['passed']]
    saved_selection=directory/'selection.json';save(saved_selection,dict(complete=True,results=valid))
    if not args.skip_images or not args.skip_evaluation:
        command=[sys.executable,str(COOPT/'step5/feasible_views.py'),args.object,'--experiment',args.experiment,
            '--selection-summary',str(saved_selection),'--step51-experiment',args.step51_experiment]
        if args.skip_images:command.append('--skip-images')
        if args.skip_evaluation:command.append('--skip-evaluation')
        subprocess.run(command,cwd=ROOT,check=True)


if __name__=='__main__':main()
