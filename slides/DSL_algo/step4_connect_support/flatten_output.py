"""Move the current Step4 one level up without rebuilding any physical data."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I


def flatten(baseline_path):
    baseline=json.loads(Path(baseline_path).read_text());base=I.OUTPUTS/'B'
    old_roots=sorted(base.glob('pose*/step4/convex_landing'))
    if len(old_roots)!=8:raise ValueError('Expected exactly the eight current result folders')
    for old in old_roots:
        parent=old.parent
        for child in old.iterdir():
            target=parent/child.name
            if target.exists():
                if child.name!='placements.png' or I.sha256(child)!=I.sha256(target):
                    raise ValueError(f'Conflicting destination: {target}')
                target.unlink()
            child.rename(target)
        old.rmdir()
    def relocated(value):
        return value.replace('/step4/convex_landing','/step4')
    def rewrite(value):
        if isinstance(value,str):return relocated(value)
        if isinstance(value,list):return [rewrite(x) for x in value]
        if isinstance(value,dict):return {relocated(k):rewrite(v) for k,v in value.items()}
        return value
    # These derived presentations are regenerated, not carried as stale audits.
    for group in base.glob('pose*/step4'):
        (group/'data/visualization.json').unlink(missing_ok=True)
    for filename in ('convex_landing_results.json','convex_landing_review.json'):
        (base/filename).unlink(missing_ok=True)
    if (base/'convex_landing_batch.json').exists():
        (base/'convex_landing_batch.json').rename(base/'step4_batch.json')
    paths=[p.resolve() for group in base.glob('pose*/step4') for p in group.rglob('*.json')]
    documents={p:rewrite(json.loads(p.read_text())) for p in paths}
    changed_code={str((I.ROOT/'slides/DSL_algo/step4_connect_support'/f).relative_to(I.ROOT))
        for f in ('reseating.py','run_pocketed_feet.py')}
    completed=set();pending=set()
    def update(path):
        path=path.resolve()
        if path not in documents or path in completed:return
        if path in pending:raise ValueError(f'Cyclic report dependency: {path}')
        pending.add(path);data=documents[path]
        if isinstance(data,dict):
            if 'construction' in data and 'body_directory' in data:
                child=path.parent.parent/data['body_directory']/'report.json'
                update(child);data['construction']=documents[child.resolve()]
            provenance=data.get('provenance',{})
            for kind in ('inputs','code'):
                records=provenance.get(kind,{})
                for name,expected in list(records.items()):
                    dependency=I.ROOT/name
                    update(dependency)
                    current=I.sha256(dependency)
                    if current!=expected and dependency.resolve() not in documents:
                        if kind!='code' or name not in changed_code:
                            raise ValueError(f'Unexpected changed dependency: {name}')
                        provenance.setdefault('generation_code_before_output_relocation',{})[name]=expected
                    records[name]=current
            for name in data.get('artifacts',{}):
                dependency=path.parent/name;update(dependency)
                data['artifacts'][name]=I.sha256(dependency)
        I.save(path,data);pending.remove(path);completed.add(path)
    for path in paths:update(path)
    checks=[]
    for name,record in baseline.items():
        current=I.ROOT/relocated(name)
        if current.suffix in ('.obj','.npz'):
            assert I.sha256(current)==record['sha256'],name
        else:
            r=I.check_report(current)
            checks.append(dict(original_path=name,current_path=str(current.relative_to(I.ROOT)),
                original_sha256=record['sha256'],current_sha256=I.sha256(current),
                original_provenance=record['provenance']))
    I.save(base/'step4_layout_migration.json',dict(complete=True,
        operation='Move current Step4 output to step4 itself; rewrite references only',
        physical_geometry_and_certificates_byte_identical=True,
        physical_binary_count=sum(Path(p).suffix in ('.obj','.npz') for p in baseline),
        original_physical_validation_reused=True,regenerated_physical_validation=False,
        reports=checks,provenance=dict(inputs=I.hashes([I.ROOT/relocated(p) for p in baseline
            if Path(p).suffix in ('.obj','.npz')]),code=I.hashes([Path(__file__)]))))
    print('FLATTENED',len(old_roots),'groups; unchanged physical binaries',
        sum(Path(p).suffix in ('.obj','.npz') for p in baseline),flush=True)


if __name__=='__main__':flatten(sys.argv[1])
