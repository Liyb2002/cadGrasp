"""Own explicit warm inputs before source trials clean their process files."""
from pathlib import Path
import hashlib,json
import numpy as np
from whole_search.common import C
from whole_pipeline import load_layout
from whole_search.reuse_first import registered
from material_floor import project_floor_layout


def attach_material_seeds(search):
    experiment=getattr(search,'warm_checkpoint_experiment',None)
    manifest=getattr(search,'initialization_seeds',None)
    if not experiment and not manifest:return
    base=C.A.result_root('B');label=search.out.relative_to(base).parts[:-3]
    target=base/'data'/search.out.name/'warm_inputs'/Path(*label)
    target.mkdir(parents=True,exist_ok=True);sources=[]
    if experiment:
        old=search.out.parent/experiment
        process=old/'process.json'
        if process.exists():
            rows=json.loads(process.read_text())
            rows.sort(key=lambda row:(row['failed_load_count'],-row['index']))
            sources.extend((old/r['layout'],dict(source_checkpoint=r)) for r in rows[:3])
        elif (old/'layout.npz').exists():
            sources.append((old/'layout.npz',dict(source_finished_experiment=experiment)))
    if manifest:
        path=Path(manifest)
        payload=path.read_bytes();copy=target/'explicit_seed_manifest.json';copy.write_bytes(payload)
        search.model.inputs.append(copy)
        record=json.loads(payload)
        for item in record.get('cases',record).get('/'.join(label),[])[:8]:
            path=Path(item);path=path if path.is_absolute() else C.ROOT/path
            sources.append((path,dict(explicit_seed_manifest=str(manifest))))
    copied=[]
    for position,(path,metadata) in enumerate(sources):
        try:data=path.read_bytes()
        except FileNotFoundError:continue
        owned=target/f'{position:02d}_{path.name}';owned.write_bytes(data)
        info=owned.with_suffix('.json')
        info.write_text(json.dumps(dict(metadata,source=str(path),
            source_sha256=hashlib.sha256(data).hexdigest(),seed_only=True),indent=2)+'\n')
        q=load_layout(owned)
        if len(q.placements)!=len(search.model.poses):raise ValueError('Warm seed pose count differs')
        for k in q.active:search.model.validate_placement(q,k)
        search.material_warm_layouts.append((str(owned.relative_to(C.ROOT)),q))
        search.model.inputs.extend([owned,info]);copied.append(q)
    if copied:
        seed=copied[0]
        for k in seed.active:
            if not registered(seed,k):continue
            q=project_floor_layout(search.model,seed,fixed_ground=(k,))
            if q is None:continue
            path=target/f'affine_ground_{k}.npz'
            np.savez_compressed(path,**q.arrays(search.model.native))
            search.material_warm_layouts.append((str(path.relative_to(C.ROOT)),q))
            search.model.inputs.append(path)
