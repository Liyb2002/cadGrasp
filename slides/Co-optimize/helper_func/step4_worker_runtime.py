from pathlib import Path
import json, sys, resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
payload=json.loads(Path(sys.argv[1]).read_text())
import os,time
os.environ['COOPT_REPO_ROOT']=payload['root']
result_path=Path(payload['result'])
claim=Path(payload['private_output']).parent/'claims'/payload['id'].replace('/','__')
claim.parent.mkdir(parents=True,exist_ok=True)
while True:
    if result_path.exists() and not payload.get('continue_owned_checkpoint',False):
        try:saved=json.loads(result_path.read_text())
        except json.JSONDecodeError:
            time.sleep(.2);continue
        if saved.get('id')==payload['id']:
            print('REUSED same-experiment result',payload['id'],flush=True)
            sys.exit(0)
    try:
        claim.mkdir();(claim/'pid').write_text(str(os.getpid()));break
    except FileExistsError:time.sleep(.5)
def large_native_peer(proc_root=Path('/proc'),threshold_kib=32*1024**2,worker_script=None):
    script=str(worker_script or Path(__file__).resolve()).encode()
    for proc in proc_root.iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            rows=(proc/'status').read_text().splitlines()
            rss=next((int(row.split()[1]) for row in rows if row.startswith('VmRSS:')),0)
            if rss<threshold_kib:continue
            command=(proc/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if script in command:return dict(pid=int(proc.name),rss_kib=rss)
    return None
def native_admission_ready(available,total,peer):
    return available>=total*(.35 if peer is not None else .15)
waiting_printed=False
while True:
    memory={x.split()[0].rstrip(':'):int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.split()[0] in ('MemTotal:','MemAvailable:')}
    peer=large_native_peer()
    if native_admission_ready(memory['MemAvailable'],memory['MemTotal'],peer):break
    if not waiting_printed:
        print('WAITING_FOR_NATIVE_MEMORY',dict(large_peer=peer,available_bytes=memory['MemAvailable']),flush=True)
        waiting_printed=True
    time.sleep(5)
native_cpu_policy=None
if payload.get('native_cpu_count') is not None:
    available=sorted(os.sched_getaffinity(0));count=payload['native_cpu_count']
    if type(count) is not int or not 1<=count<=len(available):
        raise ValueError('native_cpu_count must fit the inherited CPU affinity')
    selected=available[:count];os.sched_setaffinity(0,selected)
    native_cpu_policy=dict(inherited_cpus=available,selected_cpus=selected,
        applied_before_numerical_imports=True,per_process_memory_limit=None)
    print('NATIVE_CPU_AFFINITY',native_cpu_policy,flush=True)
tree=Path(payload['frozen'])
sys.path[:0]=[str(tree),str(tree/'slides/tools'),str(tree/'slides/obj_supp'),
             str(tree/'slides/Co-optimize/helper_func'),str(tree/'slides/Co-optimize/step4.2')]
import _bootstrap
import artifact_store as A
original=A._visual_target
def private_visual_target(path):
    target=original(path)
    return Path(payload['private_output'])/target.relative_to(A.VISUALS) if target.is_relative_to(A.VISUALS) else target
A._visual_target=private_visual_target
from run_oracle import worker
import importlib.util,hashlib
adapter_directory=Path(payload['private_output']).parent/'reporting_adapter_sources'
reporting_adapter=None
cleanup_statistics_adapter=None
additional_sources={}
if (adapter_directory/'manifest.json').exists():
    import run_oracle as _oracle_module
    _original_solve_frontier=_oracle_module.OracleSearch.solve_frontier
    def _solve_with_reporting(self,current,*args,**kwargs):
        global reporting_adapter,additional_sources,cleanup_statistics_adapter
        spec=importlib.util.spec_from_file_location('coopt_volume_reporting_adapter',adapter_directory/'closed_volume_reporting_adapter.py')
        adapter=importlib.util.module_from_spec(spec);sys.modules[spec.name]=adapter;spec.loader.exec_module(adapter)
        reporting_adapter=adapter.install(adapter_directory)
        additional_sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in adapter_directory.iterdir() if path.is_file()}
        export_source=adapter_directory.parent/'artifact_export_adapter_sources/exact_witness_export.py'
        if export_source.exists():
            export_spec=importlib.util.spec_from_file_location('coopt_exact_witness_export',export_source)
            export_adapter=importlib.util.module_from_spec(export_spec)
            sys.modules[export_spec.name]=export_adapter;export_spec.loader.exec_module(export_adapter)
            export_adapter.install()
            additional_sources[str(export_source)]=hashlib.sha256(export_source.read_bytes()).hexdigest()
        cleanup_source=adapter_directory.parent/'artifact_export_adapter_sources/cleanup_statistics_adapter.py'
        if cleanup_source.exists():
            for source in [cleanup_source,cleanup_source.with_name('cleanup_without_unused_statistics.py')]:
                additional_sources[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
            cleanup_spec=importlib.util.spec_from_file_location('coopt_cleanup_statistics_adapter',cleanup_source)
            cleanup_adapter=importlib.util.module_from_spec(cleanup_spec)
            sys.modules[cleanup_spec.name]=cleanup_adapter;cleanup_spec.loader.exec_module(cleanup_adapter)
            cleanup_statistics_adapter=cleanup_adapter.install(cleanup_source.parent)
            print('UNUSED_CLEANUP_STATISTICS_ADAPTER',cleanup_statistics_adapter,flush=True)
        _oracle_module.OracleSearch.solve_frontier=_original_solve_frontier
        return _original_solve_frontier(self,current,*args,**kwargs)
    _oracle_module.OracleSearch.solve_frontier=_solve_with_reporting
stack_file=None
if payload.get('python_trace_file'):
    import faulthandler
    stack_file=Path(payload['python_trace_file']).open('w')
    faulthandler.enable(file=stack_file)
    # Older CPython can crash while its watchdog walks a trampoline frame.
    # Fatal-signal reporting has no periodic background stack walker.
    if payload.get('periodic_python_trace',False):
        faulthandler.dump_traceback_later(30,repeat=True,file=stack_file)
result=worker((payload['id'],payload['options']))
if native_cpu_policy is not None:result['native_cpu_policy']=native_cpu_policy
if payload.get('serial_generation'):result['serial_generation']=payload['serial_generation']
if stack_file is not None:
    faulthandler.cancel_dump_traceback_later();stack_file.close()
if reporting_adapter:
    result['reporting_adapter']=reporting_adapter
    result['additional_executed_sources']=additional_sources
if cleanup_statistics_adapter:
    result['unused_cleanup_statistics_adapter']=cleanup_statistics_adapter
if payload.get('continuation_failure_archive'):
    prior=json.loads(Path(payload['continuation_failure_archive']).read_text())
    fragment=result.get('seconds',0.)
    result['checkpoint_recovery']=dict(own_checkpoint=True,original_worker_exit_code=prior['worker_exit_code'],
        original_failure_archive=payload['continuation_failure_archive'],
        original_fragment_seconds_detection_upper_bound=prior['process_wall_seconds'],
        continuation_fragment_seconds=fragment,extra_search_rounds=0)
    result['seconds']=prior['process_wall_seconds']+fragment
    result['actual_execution_wall_seconds']=result['seconds']
    result['duration_is_detection_upper_bound']=True
imports=[]
for name, module in list(sys.modules.items()):
    location=getattr(module,'__file__',None)
    if location and Path(location).is_relative_to(Path(payload['root'])):
        imports.append(dict(module=name,path=location,frozen=Path(location).is_relative_to(tree)))
result['execution_imports']=imports
p=Path(payload['result']);p.parent.mkdir(parents=True,exist_ok=True)
temporary=p.with_suffix('.worker-writing.json')
temporary.write_text(json.dumps(result,indent=2)+'\n')
temporary.replace(p)
if payload.get('checkpoint_recovery_registry'):
    registry_path=Path(payload['checkpoint_recovery_registry']);registry=json.loads(registry_path.read_text())
    for case in registry['cases']:
        if case['id']==payload['id']:
            case.update(result_ready=not bool(result.get('error')),passed=bool(result.get('passed')),
                checkpoint_retained=bool(result.get('error') or result.get('status') in ('runtime_failure','unresolved')),
                runtime_unresolved_after_idle_continuation=bool(result.get('error') or result.get('status') in ('runtime_failure','unresolved')),
                continuation_result_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    registry['phase']='RESULT_READY' if all(c.get('result_ready') for c in registry['cases']) else 'CONTINUATION_RUNTIME_FAILURE'
    temp=registry_path.with_suffix('.worker-writing.json');temp.write_text(json.dumps(registry,indent=2)+'\n');temp.replace(registry_path)
