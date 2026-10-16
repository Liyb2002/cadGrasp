"""Run archived Python bytes while retaining original data lookup paths."""
from pathlib import Path
import hashlib
import importlib.machinery
import json
import os
import sys

payload_path=Path(sys.argv[1]);payload=json.loads(payload_path.read_text())
launcher_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
# Apply an explicitly recorded immutable source revision only at the boundary
# of a NEW numerical case. Already running workers retain their frozen bytes;
# audits always retain the numerical case's own archived source selection.
control=payload_path.parent.parent/'worker_source_override.json'
if not payload.get('action') and control.exists():
    revision=json.loads(control.read_text())
    if revision['experiment']!=payload['experiment']:raise ValueError('Source revision experiment mismatch')
    snapshot=Path(revision['source_snapshot'])
    if hashlib.sha256((snapshot/'index.json').read_bytes()).hexdigest()!=revision['index_sha256']:
        raise ValueError('Source revision index changed')
    payload.update(previous_source_snapshot=payload['source_snapshot'],source_snapshot=str(snapshot),
                   source_revision_record=str(control))
payload['execution_launcher_sha256']=launcher_sha
temporary=payload_path.with_suffix('.launching');temporary.write_text(json.dumps(payload,indent=2)+'\n');temporary.replace(payload_path)
root=Path(payload['repository_root']);archive=Path(payload['source_snapshot'])
index=json.loads((archive/'index.json').read_text())
allowed=sorted(os.sched_getaffinity(0))
requested=payload.get('cpus',allowed[-2:])
os.sched_setaffinity(0,set(requested)&set(allowed))
sys.wrap_executed_sources={}
original=importlib.machinery.SourceFileLoader.get_code

def frozen_code(self,fullname):
    path=Path(self.path).resolve()
    if path.is_relative_to(root) and not path.is_relative_to(root/'.venv'):
        relative=str(path.relative_to(root))
        if relative not in index:
            raise ImportError('Local source absent from frozen execution: '+relative)
        content=(archive/relative).read_bytes()
        if hashlib.sha256(content).hexdigest()!=index[relative]:
            raise ImportError('Frozen source changed: '+relative)
        sys.wrap_executed_sources[relative]=index[relative]
        return compile(content,str(path),'exec',dont_inherit=True)
    return original(self,fullname)

importlib.machinery.SourceFileLoader.get_code=frozen_code
worker=root/'slides/Co-optimize/step5/wrap_worker.py'
relative=str(worker.relative_to(root));source=(archive/relative).read_bytes()
assert hashlib.sha256(source).hexdigest()==index[relative]
sys.wrap_executed_sources[relative]=index[relative]
sys.path.insert(0,str(worker.parent));sys.argv=[str(worker),str(payload_path)]
exec(compile(source,str(worker),'exec'),dict(__name__='__main__',__file__=str(worker)))
