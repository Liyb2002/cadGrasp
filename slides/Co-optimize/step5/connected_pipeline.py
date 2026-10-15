"""Complete a saved all52 round-support campaign with native width proofs."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from types import SimpleNamespace
import argparse,json,os,shutil,subprocess,sys,time
from feasible_pipeline import ROOT,COOPT,read,save,digest,relative,deliver


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-experiment',required=True);p.add_argument('--experiment',required=True)
    p.add_argument('--jobs',type=int,default=4);p.add_argument('--sets',nargs='+')
    p.add_argument('--skip-images',action='store_true')
    args=p.parse_args();base=COOPT/'data/results/B'
    selection=base/'data/step5_show_all52_20261014_02_inputs/selection.json'
    calibration=base/'data/step5_feasible_outside_in_calibration_20261015_03/thickness_calibration.json'
    cases=read(selection)['results']
    if args.sets:cases=[c for c in cases if c['id']in args.sets]
    directory=base/'data'/('step5_feasible_'+args.experiment);directory.mkdir(parents=True,exist_ok=True)
    archive=directory/'executed_sources';launch={}
    for folder in ['step5','step5.2','step5.3','step4.2','helper_func','evaluation','vis_func']:
        for source in(COOPT/folder).rglob('*.py'):
            target=archive/source.relative_to(ROOT)
            if target.exists():assert digest(target)==digest(source),str(source)
            else:target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
            launch[relative(source)]=digest(source)
    save(directory/'launch_sources.json',launch)
    results=[];began=time.monotonic();common_args=SimpleNamespace(object='B',experiment=args.experiment,
        step51_experiment='pso_all52_20261013_01_floor')
    def update():
        save(directory/'summary.json',dict(complete=len(results)==len(cases),group_count=len(cases),
            passed=sum(r['passed']for r in results),thickness_passed=sum(r.get('thickness_target_passed',False)for r in results),
            results=results,source_experiment=args.source_experiment,selection=relative(selection),
            stage_order=['saved_certified_round_support','continuous_full_width_backbone','fresh_full_physical_proof'],
            seconds=time.monotonic()-began))
    def one(case):
        old=base/case['id']/'step5/feasible'/args.source_experiment
        retained=old/'outside_in_retained_interfaces'
        parent=retained if(retained/'report.json').exists()and read(retained/'report.json')['passed']else old/'outside_in'
        assert(parent/'report.json').exists()or(parent/'proposal_interruption.json').exists(),case['id']
        output=base/case['id']/'step5/feasible'/args.experiment/'connected_width'
        payload=read(old/'outside_in.payload.json')
        payload.update(parent_directory=str(parent),output=str(output),
            parent_source_snapshot=str(base/'data'/('step5_feasible_'+args.source_experiment)/'executed_sources'))
        payload_path=directory/'payloads'/(case['id'].replace('/','__')+'.json');save(payload_path,payload)
        if not(output/'report.json').exists():
            log=output.parent/'connected_width.log';log.parent.mkdir(parents=True,exist_ok=True)
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
            with log.open('w')as f:
                result=subprocess.run([sys.executable,str(COOPT/'step5/junction_finish.py'),str(payload_path)],
                    cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            if result.returncode:raise ValueError(str(read(output/'failure.json'))if(output/'failure.json').exists()else str(result.returncode))
        record=read(output/'report.json');assert record['passed']and record['minimum_connector_core_connected']
        s53=base/case['id']/'step5/step5.3'/args.experiment
        if not(s53/'data/report.json').exists():
            baseline=old/'baseline';baseline_record=read(baseline/'report.json')
            stages=[dict(stage='certified_original_complete_baseline',passed=True,output=relative(baseline),record=baseline_record),
                dict(stage='connected_width',passed=True,committed=True,output=relative(output),record=record)]
            row=deliver(case,common_args,output,stages,base)
            report=read(s53/'data/report.json')
            report.update(pipeline='outside_in_round_support_with_continuous_native_width_backbone',
                contact_pruning=next(h for h in record['history']if h['operation']=='outside_in_contact_pruning'),
                thickness_target_passed=True,thickness_calibration=relative(calibration),
                before_contact_metrics=record['before_contact_metrics'],after_contact_metrics=record['after_contact_metrics'],
                connector_quality=record['connector_quality'],thickness_certificate=record['thickness_certificate'],
                minimum_round_connector_radius_m=record['minimum_round_connector_radius_m'],
                all_original_contacts_frozen=False,grid_pocket_cutting_used=False,
                finishing_wall_clock_cutoff_used=False,unresolved_proposals_conservatively_rejected=True,
                unified_core_and_ground_construction=True,parent_execution_source_snapshots=record['parent_source_snapshots'])
            save(s53/'data/report.json',report)
            s52=base/case['id']/'step5/step5.2'/args.experiment
            report52=read(s52/'data/report.json');report52.update(pipeline=report['pipeline'],
                thickness_certificate=record['thickness_certificate'],minimum_connector_core_connected=True)
            save(s52/'data/report.json',report52)
        else:
            report=read(s53/'data/report.json');row=dict(id=case['id'],pose_count=case['pose_count'],
                experiment=case['experiment'],passed=True,status='PASS',output=relative(s53),protected=record['protected'])
        row.update(thickness_target_passed=True,minimum_backbone_radius_m=record['minimum_round_connector_radius_m'])
        return row
    eligible=[]
    for case in cases:
        if case['passed']:eligible.append(case)
        else:results.append(dict(id=case['id'],pose_count=case['pose_count'],passed=False,status='STEP4_FAIL'))
    update()
    with ThreadPoolExecutor(max_workers=args.jobs)as pool:
        futures={pool.submit(one,c):c for c in eligible}
        for future in as_completed(futures):
            case=futures[future]
            try:row=future.result()
            except Exception as error:row=dict(id=case['id'],pose_count=case['pose_count'],passed=False,status='STEP5_ERROR',error=str(error))
            results.append(row);update();print('CONNECTED_RESULT',len(results),'/',len(cases),row['id'],row['status'],flush=True)
    protected={p:h for r in results for p,h in r.get('protected',{}).items()};code={};parent_sources={}
    for row in results:
        if not row['passed']:continue
        report=read(ROOT/row['output']/'data/report.json');worker=read(ROOT/report['checkpoint']/'report.json')
        parent_sources[row['id']]=worker['parent_source_snapshots']
        for name,h in worker['executed_code'].items():
            if name in code:assert code[name]==h,name
            code[name]=h;target=archive/name
            if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
            assert digest(target)==h,name
    changed=[name for name,h in protected.items()if digest(ROOT/name)!=h]
    changed_code=[name for name,h in code.items()if digest(ROOT/name)!=h]
    save(directory/'verification.json',dict(passed=not changed and not changed_code,
        protected_inputs=protected,executed_code=code,changed=changed,changed_sources=changed_code,
        source_and_execution_snapshot_hashes_match=not changed_code,parent_execution_source_snapshots=parent_sources,
        parent_sources_verified_by_each_worker=True))
    if changed or changed_code or any(r['status']=='STEP5_ERROR'for r in results):raise SystemExit(1)
    selected=directory/'selection.json';save(selected,dict(complete=True,results=eligible))
    if not args.skip_images:
        subprocess.run([sys.executable,str(COOPT/'step5/feasible_views.py'),'B','--experiment',args.experiment,
            '--selection-summary',str(selected),'--step51-experiment',common_args.step51_experiment],cwd=ROOT,check=True)


if __name__=='__main__':main()
