"""Default consolidated Step5: connect/wrap, then certify progressive deletion."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

COOPT=Path(__file__).resolve().parents[1];ROOT=COOPT.parents[1]
DEFAULT_EXPERIMENT='wrap_balanced_all52_20261015_02'
DEFAULT_SELECTION=COOPT/'data/results/B/data/step5_show_all52_20261014_02_inputs/selection.json'

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.writing')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)

def freeze(directory):
    index={}
    for parent in [ROOT/'codes',ROOT/'slides']:
        for folder,dirs,files in os.walk(parent):
            dirs[:]=[d for d in dirs if d not in {'data','output','assets','history','__pycache__','.git','.search_runtime'}]
            for name in files:
                path=Path(folder)/name
                if path.suffix not in {'.py','.cpp','.h','.so'}:continue
                relative=str(path.relative_to(ROOT));target=directory/relative
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
                index[relative]=sha(target)
                if index[relative]!=sha(path):raise ValueError('Source changed during snapshot: '+relative)
    save(directory/'index.json',index)
    return index

def baseline(case,base):
    # The original pipeline identifies its complete input explicitly. Avoid
    # confusing that input with its later, separately reconstructed thin solid.
    report_path=base/case['id']/'step5/step5.3/outside_in_all52_20261015_03/data/report.json'
    if not report_path.exists():return None
    report=read(report_path)
    if report['selected_step4_experiment']!=case['experiment']:raise ValueError('Layout mismatch')
    path=ROOT/report['optional_operations'][0]['output']/'support.obj'
    if not path.exists():
        path=COOPT/'data/results'/path.relative_to(COOPT/'output')
    return path

def protect_originals(cases,base):
    paths=set()
    for case in cases:
        root=base/case['id'];visual=COOPT/'output'/base.name/case['id']
        for path in [root/'evaluation.json',visual/'evaluation.png']:
            if path.is_file():paths.add(path)
        for stage in ['step5.1','step5.2','step5.3']:
            tree=visual/stage
            if tree.exists():paths.update(p for p in tree.rglob('*') if p.is_file())
        previous=root/'step5/wrap_then_prune_all52_20261015_02'
        if previous.exists():paths.update(p for p in previous.rglob('*') if p.is_file())
        if case['passed']:
            folder=root/'step5/step5.3/outside_in_all52_20261015_03'
            path=folder/'data/report.json'
            if not path.exists():continue
            paths.add(path)
            record=read(path)
            paths.update(folder/name for name in record['artifacts'] if (folder/name).is_file())
    paths.update((COOPT/'idea').glob('*.mp4'))
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}

def run_case(case,args,base,directory,slot):
    out=base/case['id']/'step5'/args.experiment
    report_path=out/'data/report.json'
    width_review_failed=(out/'data/width_audit_failure.json').exists()
    if report_path.exists() and not width_review_failed:
        report=read(report_path)
        from wrap_dimensions import dimensions
        if report.get('dimensions')!=dimensions():
            raise ValueError('Saved result predates the current 8 mm width requirements; use a new experiment')
        if not report['passed'] or sha(out/'support.obj')!=report['support_sha256']:
            raise ValueError('Saved result mismatch: '+case['id'])
        return dict(id=case['id'],experiment=case['experiment'],pose_count=case['pose_count'],status='PASS',passed=True,output=str(out.relative_to(ROOT)),report=report,reused=True)
    if out.exists() and any(out.iterdir()):
        if not args.resume_failed:
            raise ValueError('Unfinished output retained; use --resume-failed or a new experiment: '+str(out))
        interrupted=out/'interrupted_attempts'/str(time.time_ns());interrupted.mkdir(parents=True)
        for path in list(out.iterdir()):
            if path.name!='interrupted_attempts':shutil.move(str(path),interrupted/path.name)
        label=case['id'].replace('/','__')
        for path,name in [(directory/'logs'/(label+'.log'),'worker.log'),
                          (directory/'payloads'/(label+'.json'),'payload.json')]:
            if path.exists():shutil.copy2(path,interrupted/name)
    cpus=sorted(os.sched_getaffinity(0))
    assigned=cpus[max(0,len(cpus)-2*args.jobs)+2*slot: max(0,len(cpus)-2*args.jobs)+2*slot+2]
    reference=baseline(case,base)
    payload=dict(object=args.object,case=case,experiment=args.experiment,
        selection_summary=str(args.selection_summary),floor_experiment=args.floor_experiment,
        output=str(out),baseline=str(reference) if reference else None,grids=args.grids,
        repository_root=str(ROOT),source_snapshot=str(args.source_snapshot),cpus=assigned or cpus[-2:])
    payload.update(ground_margin_scale=getattr(args,'ground_margin_scale',1.),
        terminal_foot_width_m=getattr(args,'terminal_foot_width_mm',8.)/1000,
        connector_radius_m=getattr(args,'connector_radius_mm',8.)/1000)
    if args.initial_source_experiment:
        source=base/case['id']/'step5'/args.initial_source_experiment
        if (source/'construction.json').exists():payload['initial_source']=str(source)
    p=directory/'payloads'/(case['id'].replace('/','__')+'.json');save(p,payload)
    log=directory/'logs'/(case['id'].replace('/','__')+'.log');log.parent.mkdir(exist_ok=True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',TBB_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    began=time.monotonic()
    with log.open('w') as handle:
        result=subprocess.run([sys.executable,str(COOPT/'step5/wrap_frozen_worker.py'),str(p)],
                              cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
    if result.returncode or not report_path.exists():
        return dict(id=case['id'],status='STEP5_FAILED',passed=False,exit_code=result.returncode,
            output=str(out.relative_to(ROOT)),seconds=time.monotonic()-began,
            failure=read(out/'failure.json') if (out/'failure.json').exists() else {'error':'Worker did not save a report'})
    report=read(report_path)
    return dict(id=case['id'],experiment=case['experiment'],pose_count=case['pose_count'],status='PASS',passed=True,output=str(out.relative_to(ROOT)),report=report,
                seconds=time.monotonic()-began)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object',nargs='?',default='B')
    parser.add_argument('--experiment',default=DEFAULT_EXPERIMENT)
    parser.add_argument('--selection-summary',type=Path,default=DEFAULT_SELECTION)
    parser.add_argument('--floor-experiment',default='pso_all52_20261013_01_floor',help=argparse.SUPPRESS)
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--sets',nargs='+')
    parser.add_argument('--grids',nargs='+',type=int,default=[3,5])
    parser.add_argument('--numeric-only',action='store_true',help='Save numerical results without independent audit and previews')
    parser.add_argument('--source-snapshot',type=Path,help='Use a separate immutable source snapshot for this execution segment')
    parser.add_argument('--resume-failed',action='store_true',help='Preserve interrupted attempts and recompute only missing results')
    parser.add_argument('--initial-source-experiment',help='Reuse and revalidate exact completed same-core construction stock; always rerun pruning')
    parser.add_argument('--ground-margin-scale',type=float,default=1.)
    parser.add_argument('--terminal-foot-width-mm',type=float,default=8.)
    parser.add_argument('--connector-radius-mm',type=float,default=8.)
    args=parser.parse_args();args.selection_summary=args.selection_summary.resolve()
    if args.jobs<1 or any(g<2 for g in args.grids):parser.error('Positive jobs and grids >=2 required')
    base=COOPT/'data/results'/args.object
    cases=read(args.selection_summary)['results']
    if args.sets:
        missing=set(args.sets)-{c['id'] for c in cases}
        if missing:parser.error('Unknown sets: '+str(missing))
        cases=[c for c in cases if c['id'] in args.sets]
    directory=base/'data'/('step5_'+args.experiment);directory.mkdir(parents=True,exist_ok=True)
    protected_path=directory/'protected_originals.json'
    prior=read(protected_path) if protected_path.exists() else {}
    current=protect_originals(cases,base)
    if any(current[p]!=h for p,h in prior.items() if p in current):
        raise ValueError('Previously protected original changed before this execution segment')
    save(protected_path,dict(current,**prior))
    archive=(args.source_snapshot or directory/'executed_sources').resolve();args.source_snapshot=archive
    if not (archive/'index.json').exists():freeze(archive)
    began=time.monotonic();results=[]
    for case in cases:
        if not case['passed']:results.append(dict(id=case['id'],passed=False,status='UPSTREAM_STEP4_FAIL',source=case))
    # Fixed queues keep CPU ownership disjoint; no unrelated process is changed.
    eligible=[c for c in cases if c['passed']]
    def queue(slot):
        rows=[]
        for case in eligible[slot::args.jobs]:
            try:row=run_case(case,args,base,directory,slot)
            except Exception as error:row=dict(id=case['id'],passed=False,status='STEP5_FAILED',error=str(error))
            rows.append(row)
            save(directory/'progress'/(case['id'].replace('/','__')+'.json'),row)
            print(row['status'],case['id'],flush=True)
        return rows
    execution=dict(complete=False,experiment=args.experiment,started_unix=time.time(),
        groups=[c['id'] for c in cases],selection=str(args.selection_summary.relative_to(ROOT)),
        selection_sha256=sha(args.selection_summary),step4_rerun=False,source_snapshot=str(archive.relative_to(ROOT)),
        orchestrator_sha256=sha(Path(__file__)))
    segments=read(directory/'execution_segments.json') if (directory/'execution_segments.json').exists() else []
    if not segments and (directory/'execution.json').exists():segments.append(read(directory/'execution.json'))
    segments.append(execution);save(directory/'execution_segments.json',segments)
    save(directory/'execution.json',execution)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures=[pool.submit(queue,k) for k in range(args.jobs)]
        for future in as_completed(futures):results.extend(future.result())
    order={c['id']:k for k,c in enumerate(cases)};results.sort(key=lambda r:order[r['id']])
    summary=dict(complete=True,experiment=args.experiment,pipeline='connected_hull_wrap_then_progressive_deletion',
        group_count=len(cases),eligible=len(eligible),passed=sum(r['passed'] for r in results),
        upstream_step4_fail=sum(r['status']=='UPSTREAM_STEP4_FAIL' for r in results),
        step5_failed=sum(r['status']=='STEP5_FAILED' for r in results),results=results,seconds=time.monotonic()-began,
        step4_rerun=False,ring_perimeter_protected=False)
    save(directory/'summary.json',summary)
    execution.update(complete=True,finished_unix=time.time())
    save(directory/'execution.json',execution)
    segments[-1]=execution;save(directory/'execution_segments.json',segments)
    print(json.dumps({k:v for k,v in summary.items() if k!='results'}),flush=True)
    if not args.numeric_only:
        subprocess.run([sys.executable,str(COOPT/'step5/wrap_finish.py'),
            '--experiment',args.experiment,'--jobs',str(args.jobs)],cwd=ROOT,check=True)

if __name__=='__main__':main()
