"""Publish live case statuses, then audit all saved Step4 bodies once complete."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'slides/Co-optimize/data/results/B/data'


def read(path):return json.loads(path.read_text())


def save(path,value):
    temporary=path.with_suffix('.monitor-writing.json')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    temporary.replace(path)


def cleanup_finished_caches(directory,system,rows):
    """Retire only disposable state of saved, normally completed searches."""
    manifest=system/'completed_cache_cleanup.json'
    cleaned=read(manifest) if manifest.exists() else {}
    changed=False
    for label,row in rows.items():
        if label in cleaned or row.get('error') or row.get('status') in ('runtime_failure','unresolved'):
            continue
        folder=directory.parent.parent/label/'step4/step4.2'/directory.name
        report_path=folder/'report.json'
        if not report_path.exists():continue
        report=read(report_path)
        if report.get('passed') is not row['passed']:continue
        final=set(report.get('artifacts',{}))
        candidates=['pso_memory.npz','pso_refinement_memory.npz','.search_runtime',
            'process_states','mesh_states','sampled_states','exact_states',
            'data/initial_exact_states','data/worker_logs']
        removed=[]
        for name in candidates:
            if any(p==name or p.startswith(name+'/') for p in final):
                raise RuntimeError('Disposable path declared as final artifact: '+name)
            path=folder/name
            if not path.exists():continue
            files=list(path.rglob('*')) if path.is_dir() else [path]
            for item in files:
                if item.is_file():removed.append(dict(path=str(item.relative_to(ROOT)),
                    bytes=item.stat().st_size,sha256=hashlib.sha256(item.read_bytes()).hexdigest()))
            if path.is_dir():shutil.rmtree(path)
            else:path.unlink()
        cleaned[label]=dict(normal_result_passed=row['passed'],removed=removed,
            cleaned_utc=datetime.now(timezone.utc).isoformat())
        changed=True
    if changed:save(manifest,cleaned)


def manage_handoff(directory,system,complete):
    path=system/'operational_concurrency.json'
    if not path.exists():return
    data=read(path);groups=[]
    if data.get('midrange_parent_paused') and not data.get('midrange_parent_resumed'):
        groups.append(dict(parent=data['midrange_parent_paused'],legacy=True,
            labels=data['handoff_waiting_for'],children=data['midrange_children_continue']))
    groups.extend(group for group in data.get('additional_paused_queues',[]) if not group.get('resumed'))
    changed=False
    for group in groups:
        for label,pid in zip(group['labels'],group['children']):
            result=directory/'workers'/(label.replace('/','__')+'.json')
            if result.exists():continue
            stat=Path('/proc')/str(pid)/'stat'
            try:fields=stat.read_text().rsplit(')',1)[1].split()
            except FileNotFoundError:continue
            if fields[0]!='Z':continue
            code=os.waitstatus_to_exitcode(int(fields[49]))
            uptime=float(Path('/proc/uptime').read_text().split()[0])
            upper_wall=max(0.,uptime-int(fields[19])/os.sysconf('SC_CLK_TCK'))
            save(result,dict(id=label,passed=False,status='runtime_failure',
                error='Worker exit '+str(code),worker_exit_code=code,process_wall_seconds=upper_wall,
                process_wall_time_is_detection_upper_bound=True,child_exit_status_source=str(stat),
                checkpoint_retained=True,physical_infeasibility_established=False))
            data.setdefault('handoff_child_failures',[]).append(dict(id=label,pid=pid,exit_code=code));changed=True
        if complete:
            try:os.kill(group['parent'],signal.SIGCONT)
            except ProcessLookupError:pass
            if group.get('legacy'):data['midrange_parent_resumed']=True
            else:group['resumed']=True
            data.update(handoff_resolution='all52_results_complete',
                handoff_resolution_utc=datetime.now(timezone.utc).isoformat());changed=True
    if changed:save(path,data)


def reconcile_checkpoint_recoveries(directory,system):
    path=system/'checkpoint_recovery_registry.json'
    if not path.exists():return
    registry=read(path)
    if not registry.get('active') or registry.get('phase')!='RESULT_READY':return
    if not all(case.get('result_ready') for case in registry['cases']):return
    if registry.get('retry_review_pending'):return
    batch=read(directory/'batch.json')
    if not batch['complete']:return
    rows={row['id']:row for p in (directory/'workers').glob('*.json') for row in [read(p)]}
    if set(rows)!=set(batch['sets']):return
    archive=system/'checkpoint_recovery/batch_before_recovered_results.json'
    if not archive.exists():save(archive,batch)
    batch['results']=[rows[label] for label in batch['sets']]
    batch['passed']=sum(row['passed'] for row in rows.values())
    if registry.get('batch_started_boottime') is not None:
        batch['seconds']=float(Path('/proc/uptime').read_text().split()[0])-registry['batch_started_boottime']
    batch['checkpoint_continuations_explicit']=registry['cases']
    save(directory/'batch.json',batch)
    registry.update(active=False,phase='COMPLETE',completed_utc=datetime.now(timezone.utc).isoformat())
    save(path,registry)


def progress(directory,system):
    reconcile_checkpoint_recoveries(directory,system)
    batch=read(directory/'batch.json')
    rows={}
    for path in (directory/'workers').glob('*.json'):
        try:value=read(path)
        except json.JSONDecodeError:continue
        rows[value['id']]=value
    cleanup_finished_caches(directory,system,rows)
    cases=[]
    for label in batch['sets']:
        result=rows.get(label)
        log=directory/'logs'/(label.replace('/','__')+'.log')
        for supplemental in system.glob('supplemental*/logs/'+label.replace('/','__')+'.log'):
            if not log.exists() or supplemental.stat().st_size>log.stat().st_size:
                log=supplemental
        status=('PASS' if result['passed'] else result.get('status','SEARCH_FAIL')) if result else 'QUEUED'
        latest=None
        if result is None and log.exists():
            status='RUNNING'
            with log.open('rb') as stream:
                stream.seek(max(0,log.stat().st_size-12000))
                lines=stream.read().decode(errors='replace').splitlines()
            events=[line for line in lines if 'REFINEMENT' in line or 'SOFT32 ROUND' in line or 'SOFT SLOTS' in line]
            latest=events[-1] if events else (lines[-1] if lines else 'INITIALIZING')
            if not events and latest.startswith('WAITING_FOR_NATIVE_MEMORY'):
                status='WAITING_MEMORY'
        cases.append(dict(id=label,status=status,latest=latest,
            seconds=result.get('actual_execution_wall_seconds',result.get('seconds')) if result else None))
    recovery_path=system/'checkpoint_recovery_registry.json'
    recoveries=read(recovery_path) if recovery_path.exists() else {}
    state=dict(complete=batch['complete'] and not recoveries.get('active',False),
        checkpoint_recovery=recoveries,updated_utc=datetime.now(timezone.utc).isoformat(),
        planned=len(batch['sets']),completed=len(rows),passed=sum(r['passed'] for r in rows.values()),
        failed=sum(not r['passed'] for r in rows.values()),
        running=sum(r['status']=='RUNNING' for r in cases),cases=cases)
    save(system/'progress.json',state)
    lines=['# 加速版PSO：52组Step4进度','',f"更新时间：{state['updated_utc']}",'',
        f"实际完成{state['completed']}/52，通过{state['passed']}，失败{state['failed']}。最终保存实体审计尚未完成。",'',
        '| 组 | 状态 | 本次进程耗时s |','| --- | --- | ---: |']
    lines.extend(f"| {r['id']} | {r['status']} | {r['seconds']:.1f} |" if r['seconds'] is not None
                 else f"| {r['id']} | {r['status']} | — |" for r in cases)
    (system/'PROGRESS.txt').write_text('\n'.join(lines)+'\n')
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',default='pso_accelerated_all52_step4_20261014_01')
    args=parser.parse_args()
    directory=BASE/args.name;system=BASE/('system_'+args.name)
    save(system/'finalizer_state.json',dict(complete=False,phase='WAITING_FOR_ALL52',pid=os.getpid()))
    while True:
        state=progress(directory,system)
        manage_handoff(directory,system,state['complete'])
        if state['complete']:break
        time.sleep(20)
    evaluator=Path(__file__).with_name('evaluate_pso_refinement_all52.py')
    archive=system/'actual_evaluator_source.py';shutil.copy2(evaluator,archive)
    with (system/'final_evaluation.log').open('w') as log:
        code=subprocess.call([str(ROOT/'.venv/bin/python'),str(evaluator),'--name',args.name],
            cwd=ROOT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT)
    save(system/'finalizer_state.json',dict(complete=code==0,phase='AUDIT_PASS' if code==0 else 'AUDIT_FAILURE',
        returncode=code,evaluator_sha256=hashlib.sha256(archive.read_bytes()).hexdigest()))
    print('ALL52 FINAL AUDIT',code,flush=True)
    if code==0:
        publisher=Path(__file__).with_name('publish_accelerated_step4.py')
        shutil.copy2(publisher,system/'actual_publisher_source.py')
        with (system/'final_publication.log').open('w') as log:
            result=subprocess.call([str(ROOT/'.venv/bin/python'),str(publisher),'--name',args.name],
                cwd=ROOT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT)
        save(system/'publication_state.json',dict(complete=result==0,returncode=result))
        print('ALL52 PUBLICATION',result,flush=True)


if __name__=='__main__':main()
