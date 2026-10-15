"""Queue another owned-checkpoint attempt after a confirmed quiet-machine OOM.

Keep the healthy serial supervisor. Archive its completed attempt and give its
fixed serial01 filenames generation-specific targets for the next attempt.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import pickletools
import shutil

ROOT=Path(__file__).resolve().parents[3]


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path);temporary=path.with_suffix('.retry-writing.json')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(1024*1024):digest.update(block)
    return digest.hexdigest()


def duration(row):
    return row.get('actual_execution_wall_seconds',row.get('seconds',row.get('process_wall_seconds',0.)))


def checkpoint_record(folder):
    path,=list((folder/'.search_runtime/checkpoint').glob('*.bin'))
    magic=b'COOPT-RUNTIME-1\n';digest=hashlib.sha256()
    with path.open('rb') as stream:
        header=stream.read(len(magic)+32);assert header.startswith(magic)
        while block:=stream.read(1024*1024):digest.update(block)
        assert digest.digest()==header[len(magic):],'Checkpoint checksum failed'
        stream.seek(len(magic)+32);metadata={};pending=None
        for op,arg,pos in pickletools.genops(stream):
            if op.name=='SHORT_BINUNICODE' and arg in ('phase','next_iteration'):pending=arg
            elif pending=='phase' and op.name=='SHORT_BINUNICODE':metadata[pending]=arg;pending=None
            elif pending=='next_iteration' and op.name.startswith('BININT'):metadata[pending]=arg;break
    assert metadata['phase'] in ('cold','cold_done','refine')
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path),
        payload_checksum_verified=True,metadata=metadata,owner=read(folder/'.search_runtime/owner.json'))


def prepare_retry(system,directory,label,native_cpus,kernel_log=None,*,cause='kernel_oom',proc_root=Path('/proc')):
    system=Path(system);directory=Path(directory)
    recovery=system/'checkpoint_recovery'/label
    registry_path=system/'checkpoint_recovery_registry.json';registry=read(registry_path)
    assert registry['active'],'Recovery registry is already closed'
    case,=[c for c in registry['cases'] if c['id']==label]
    assert case.get('result_ready') and not case['passed'],'Only a completed failed attempt may retry'
    result_path=directory/'workers'/(label.replace('/','__')+'.json');row=read(result_path)
    assert not row['passed'] and row.get('status')=='runtime_failure'
    fragment=row['serial_continuation'];pid=fragment['pid']
    status=Path(proc_root)/str(pid)/'stat'
    assert not status.exists() or status.read_text().rsplit(')',1)[1].split()[0]=='Z','Attempt is still alive'
    if cause=='kernel_oom':
        assert fragment['worker_exit_code']==-9
        kernel_log=Path(kernel_log);kernel=kernel_log.read_text()
        assert 'global_oom' in kernel and ('Killed process '+str(pid)+' ') in kernel,'Kernel OOM evidence missing'
        failure_cause='confirmed_kernel_global_oom_on_otherwise_idle_machine'
    elif cause=='observer_segv':
        assert fragment['worker_exit_code']==-11
        stack=(recovery/'serial01_python_stacks.log').read_text()
        assert 'Timeout (0:00:30)!' in stack and not stack.endswith('\n')
        assert stack.rsplit('Timeout (0:00:30)!',1)[1].count('File "')==1
        failure_cause='SIGSEGV_during_periodic_traceback_observer_suspected'
    else:
        assert cause=='startup_import' and fragment['worker_exit_code']==0
        startup=read(recovery/'before_serial_failure_worker.json')
        assert startup['error']=="No module named 'deterministic_base'"
        assert not startup['runtime_cache']['continued_from_checkpoint']
        assert not any(stat['writes'] for stat in startup['runtime_cache']['cache'].values())
        failure_cause='adapter_startup_import_error_before_search'
    admission=read(recovery/'serial01_admission.json')
    assert not admission['current_active_claim_owners'] and not admission['previous_active_claim_owners']
    assert admission['memory_kib']['MemAvailable']>=admission['required_available_fraction']*admission['memory_kib']['MemTotal']
    payload=read(recovery/'serial01_payload.json');options=dict(payload['options'])
    assert payload['id']==label and payload['continue_owned_checkpoint'] and options['continue_checkpoint']
    assert type(native_cpus) is int and native_cpus>0
    checkpoint=checkpoint_record(directory.parent.parent/label/'step4/step4.2'/directory.name)
    prior=read(payload['continuation_failure_archive'])
    elapsed=duration(prior)+fragment['seconds']
    assert abs(duration(row)-elapsed)<1e-6,'Prior and fragment durations disagree'
    old_generation=case.get('serial_generation','serial01')
    number=case.get('serial_attempt_number',1)+1
    generation=f'serial{number:02d}_cpu{native_cpus}'
    archive=recovery/('archive_'+old_generation)
    assert not archive.exists(),'Previous attempt was already archived'
    names=['serial01_admission.json','serial01_payload.json','serial01.log',
           'serial01_python_stacks.log','before_serial_failure_worker.json','claim_before_serial01']
    for name in names:
        path=recovery/name
        if path.is_symlink():
            target=path.resolve(strict=True)
            assert target.parent==recovery.resolve() and target.name.startswith(old_generation)
    for suffix in ['_admission.json','_payload.json','.log','_python_stacks.log','_before_failure_worker.json']:
        assert not (recovery/(generation+suffix)).exists()
    archive.mkdir()
    shutil.copy2(result_path,archive/'worker_failure_raw.json')
    shutil.copy2(recovery/'continuation01_payload.json',archive/'continuation_payload_before_retry.json')
    if cause=='kernel_oom':shutil.copy2(kernel_log,archive/'kernel_oom.log')
    moved={}
    for name in names:
        path=recovery/name
        if not path.exists():continue
        if path.is_symlink():
            target=path.resolve();target.rename(archive/target.name);path.unlink()
            (archive/name).symlink_to(target.name)
        else:path.rename(archive/name)
        moved[str(path)]=str(archive/name)
    normalized=dict(row,process_wall_seconds=elapsed,seconds=elapsed,
        actual_execution_wall_seconds=elapsed,worker_exit_code=fragment['worker_exit_code'],
        original_reused_process_wall_seconds=row.get('process_wall_seconds'),
        serial_continuation=dict(fragment,log=str(archive/'serial01.log'),
            admission=str(archive/'serial01_admission.json')))
    failure=archive/'worker_failure_normalized.json';save(failure,normalized)
    aliases={'serial01_admission.json':generation+'_admission.json',
        'serial01_payload.json':generation+'_payload.json','serial01.log':generation+'.log',
        'serial01_python_stacks.log':generation+'_python_stacks.log',
        'before_serial_failure_worker.json':generation+'_before_failure_worker.json'}
    for name,target in aliases.items():(recovery/name).symlink_to(target)
    assert not (recovery/'serial01_admission.json').exists(),'New admission must remain unstarted'
    payload.update(native_cpu_count=native_cpus,serial_generation=generation,
        continuation_failure_archive=str(failure))
    if cause=='observer_segv':payload['periodic_python_trace']=False
    assert payload['options']==options,'Search configuration changed'
    save(recovery/'continuation01_payload.json',payload)
    history=dict(generation=old_generation,pid=pid,worker_exit_code=fragment['worker_exit_code'],
        cause=failure_cause,
        archive=str(archive),fragment_seconds=fragment['seconds'],aggregate_seconds=elapsed,
        normalized_failure=str(failure),
        kernel_log_sha256=sha(archive/'kernel_oom.log') if cause=='kernel_oom' else None)
    record=dict(prepared_utc=datetime.now(timezone.utc).isoformat(),case=label,
        generation=generation,native_cpu_count=native_cpus,previous_attempt=history,
        checkpoint=checkpoint,aliases=aliases,archived_paths=moved,
        extra_search_rounds=0,search_options_unchanged=True,healthy_supervisor_untouched=True)
    save(recovery/(generation+'_queue_manifest.json'),record)
    current=read(registry_path);assert current['active']
    current_case,=[c for c in current['cases'] if c['id']==label]
    assert current_case.get('result_ready') and not current_case['passed']
    current_case.setdefault('serial_attempt_history',[]).append(history)
    current_case.update(result_ready=False,passed=False,failure_archive=str(failure),
        serial_attempt_number=number,serial_generation=generation,
        pending_native_cpu_count=native_cpus,runtime_unresolved_after_idle_continuation=False,
        checkpoint_retained=True,checkpoint_sha256=checkpoint['sha256'],
        checkpoint_metadata=checkpoint['metadata'],physical_infeasibility_established=False)
    if current['phase']=='RESULT_READY':current['phase']='WAITING_FOR_SERIAL_QUIET_MACHINE'
    save(registry_path,current)
    save(result_path,dict(normalized,checkpoint_retry_queued=True,queued_generation=generation))
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',default='pso_accelerated_all52_step4_20261014_01')
    parser.add_argument('--case',required=True)
    parser.add_argument('--native-cpus',type=int,default=8)
    parser.add_argument('--kernel-log',type=Path)
    parser.add_argument('--cause',choices=['kernel_oom','startup_import','observer_segv'],default='kernel_oom')
    args=parser.parse_args();base=ROOT/'slides/Co-optimize/data/results/B/data'
    print(json.dumps(prepare_retry(base/('system_'+args.name),base/args.name,args.case,
        args.native_cpus,args.kernel_log,cause=args.cause),indent=2))


if __name__=='__main__':main()
