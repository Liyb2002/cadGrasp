"""Give an unresolved native geometry check the otherwise idle machine."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]


def read(path):return json.loads(path.read_text())


def save(path,value):
    temporary=path.with_suffix('.serial-writing.json')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def live_claim_owners(system):
    owners=[]
    for path in (system/'claims').glob('*/pid'):
        try:
            pid=int(path.read_text())
            proc=Path('/proc')/str(pid)
            fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
            command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        except (FileNotFoundError,ProcessLookupError):continue
        if fields[0]!='Z' and str(system/'execution_scripts/step4_worker.py') in command:
            owners.append(dict(pid=pid,case=path.parent.name))
    return owners


def record_serial_failure(payload,registry_path,recovery,code,seconds,pid,reason=None):
    result_path=Path(payload['result']);value=read(result_path)
    (recovery/'before_serial_failure_worker.json').write_text(json.dumps(value,indent=2)+'\n')
    previous=(read(Path(payload['continuation_failure_archive']))
              if payload.get('continuation_failure_archive') else value)
    prior=previous.get('actual_execution_wall_seconds',previous.get('seconds',previous.get('process_wall_seconds',0.)))
    value.update(passed=False,status='runtime_failure',worker_exit_code=code,
        error=reason or ('Otherwise-idle same-checkpoint continuation exited '+str(code)+'; resource/runtime unresolved'),
        checkpoint_retained=True,physical_infeasibility_established=False,
        seconds=prior+seconds,actual_execution_wall_seconds=prior+seconds,
        duration_is_detection_upper_bound=True,
        serial_continuation=dict(pid=pid,worker_exit_code=code,seconds=seconds,
            same_owned_checkpoint=True,extra_search_rounds=0,
            log=str(recovery/'serial01.log'),admission=str(recovery/'serial01_admission.json')))
    save(result_path,value)
    registry=read(registry_path)
    for case in registry['cases']:
        if case['id']==payload['id']:
            case.update(result_ready=True,passed=False,serial_continuation_exit_code=code,
                runtime_unresolved_after_idle_continuation=True,checkpoint_retained=True,
                physical_infeasibility_established=False,
                continuation_result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest())
    registry['phase']=('RESULT_READY' if all(c.get('result_ready') for c in registry['cases'])
                       else 'WAITING_FOR_SERIAL_QUIET_MACHINE')
    save(registry_path,registry)


def wait_for_idle(directory,system,previous,previous_system):
    while True:
        current=read(directory/'batch.json');old=read(previous)
        old_live=live_claim_owners(previous_system)
        current_live=live_claim_owners(system)
        memory={x.split()[0].rstrip(':'):int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines()
                if x.split()[0] in ('MemTotal:','MemAvailable:')}
        if (current['complete'] and not old_live and not current_live
                and memory['MemAvailable']>=.85*memory['MemTotal']):
            return dict(previous_batch_complete=old['complete'],previous_active_claim_owners=old_live,
                current_active_claim_owners=current_live,
                previous_completed_worker_records=len(list((previous.parent/'workers').glob('*.json'))),
                memory_kib=memory,required_available_fraction=.85,
                admission='Current batch ended; no live numerical owners; available memory threshold met')
        time.sleep(20)


def continue_case(system,case,admission):
    label=case['id'];recovery=system/'checkpoint_recovery'/label
    if (recovery/'serial01_admission.json').exists():
        raise RuntimeError('Serial continuation already started; reconcile its result before retrying: '+label)
    (recovery/'serial01_admission.json').write_text(json.dumps(admission,indent=2)+'\n')
    payload=read(recovery/'continuation01_payload.json')
    failure=case.get('failure_archive')
    if not failure:
        failure=recovery/('attempt04_native_oom_result.json'
            if (recovery/'attempt04_native_oom_result.json').exists() else 'attempt01_worker_failure.json')
    payload.update(continuation_failure_archive=str(failure),
                   python_trace_file=str(recovery/'serial01_python_stacks.log'))
    (recovery/'serial01_payload.json').write_text(json.dumps(payload,indent=2)+'\n')
    claim=system/'claims'/label.replace('/','__')
    if claim.exists():
        pid=int((claim/'pid').read_text());status=Path('/proc')/str(pid)/'stat'
        assert not status.exists() or status.read_text().rsplit(')',1)[1].split()[0]=='Z'
        claim.rename(recovery/'claim_before_serial01')
    registry_path=system/'checkpoint_recovery_registry.json';registry=read(registry_path)
    registry.update(phase='SERIAL_CONTINUATION_RUNNING',serial_current_case=label)
    save(registry_path,registry)
    began=time.monotonic()
    with (recovery/'serial01.log').open('w') as log:
        child=subprocess.Popen([str(ROOT/'.venv/bin/python'),str(system/'execution_scripts/step4_worker.py'),
                              str(recovery/'serial01_payload.json')],cwd=ROOT,
            env=dict(os.environ,COOPT_REPO_ROOT=str(ROOT),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'),
            stdout=log,stderr=subprocess.STDOUT)
        code=child.wait()
    print('SERIAL CHECKPOINT CONTINUATION',label,code,flush=True)
    result=read(Path(payload['result']))
    if code or result.get('error') or result.get('status') in ('runtime_failure','unresolved'):
        reason=None if code else 'Otherwise-idle continuation returned unresolved: '+str(result.get('error',result.get('status')))
        record_serial_failure(payload,registry_path,recovery,code,time.monotonic()-began,child.pid,reason)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',default='pso_accelerated_all52_step4_20261014_01')
    parser.add_argument('--case',help='Otherwise process every pending registered recovery, one at a time')
    args=parser.parse_args();base=ROOT/'slides/Co-optimize/data/results/B/data'
    directory=base/args.name;system=base/('system_'+args.name)
    previous=base/'pso_refine_all52_step4_20261014_01/batch.json'
    previous_system=base/'system_pso_refine_all52_step4_20261014_01'
    registry_path=system/'checkpoint_recovery_registry.json'
    lock_directory=system/'checkpoint_recovery';lock_directory.mkdir(exist_ok=True)
    with (lock_directory/'serial_driver.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another serial recovery driver owns the queue')
        while True:
            registry=read(registry_path)
            pending=[c for c in registry['cases'] if not c.get('result_ready')
                     and (args.case is None or c['id']==args.case)]
            if not registry.get('active') or not pending:return
            admission=wait_for_idle(directory,system,previous,previous_system)
            # Cases may be registered while the normal queue is still running.
            registry=read(registry_path)
            pending=[c for c in registry['cases'] if not c.get('result_ready')
                     and (args.case is None or c['id']==args.case)]
            if pending:continue_case(system,pending[0],admission)


if __name__=='__main__':main()
