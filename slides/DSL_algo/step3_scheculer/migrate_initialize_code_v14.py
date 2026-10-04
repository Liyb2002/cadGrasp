"""Retain exact source bytes after unrelated external dataset migration edits."""
from pathlib import Path
import json
from step3_scheculer import contacts as I

def main():
    history=Path(__file__).with_name('history')
    sources={
        'slides/DSL_algo/step3_scheculer/initialize_gpu_v12.py':history/'initialize_gpu_v12_fixed20.py',
        'slides/DSL_algo/step3_scheculer/pair_tasks.py':history/'pair_tasks_before_dataset_migration.py'}
    updated=[]
    for p in (I.OUTPUTS/'B'/'independent_poses_gpu_v12').glob('pose_*/step3_scheculer/schedule.json'):
        d=json.loads(p.read_text());code=d['provenance']['code'];changes=[]
        for key,source in sources.items():
            digest=I.sha256(source)
            if code.get(key)!=digest:continue
            newkey=str(source.resolve().relative_to(I.ROOT));code.pop(key);code[newkey]=digest
            changes.append(dict(original_path=key,preserved_path=newkey,sha256=digest))
        # Finish the first partially written metadata record too.
        if not changes and not d.get('generator_source_path_migration'):continue
        code.update(I.hashes([Path(__file__)]))
        d['generator_source_path_migration']=dict(sources=changes,source_bytes_unchanged=True,geometry_changed=False,
            reason='External dataset migration edited CLI defaults and task-folder lookup; identical original sources retained')
        I.save(p,d);I.check_report(p);updated.append(p.parents[1].name)
    print('Identical generator sources preserved for',updated)

if __name__=='__main__':main()
