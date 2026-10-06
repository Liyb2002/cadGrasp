"""Portable checked per-pose head cache; input changes invalidate it."""
import copy,gzip,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
AREAS=(.01,.02,.005)
SCHEMA='native_pose_heads_v1'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def sources():
    files=[ROOT/'codes/precompute_objects'/p for p in ('head_geometry.py','head_directions.py','head_cache.py','precompute_heads.py')]
    files+=sorted((ROOT/'codes/precompute_objects/heads').glob('*.py'))
    files+=[ROOT/'slides/obj_supp/insert_trajectory/angles.py']
    files+=[ROOT/'slides/baseline_algo/step3_scheculer'/p for p in ('contacts.py','floor_support.py','passive_support.py')]
    return {str(p.relative_to(ROOT)):digest(p) for p in files}
def inputs(folder):
    folder=Path(folder)
    return {str(p.relative_to(ROOT)):digest(p) for p in [folder/'setup.npz',folder/'needs.json',folder.parent.parent/'mesh.stl',folder.parent.parent/'poses.json']}
def json_value(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)
def load(folder,count=200,strict=True):
    folder=Path(folder);target=folder/'step2';path=target/'report.json'
    if not path.exists():return None
    report=json.loads(path.read_text())
    if report['schema']!=SCHEMA or report['count']!=count:return None
    if report['inputs']!=inputs(folder) or report['sources']!=sources():
        if strict:raise ValueError(f'Stale precomputed heads: {target}')
        return None
    for name,value in report['artifacts'].items():
        if digest(target/name)!=value:raise ValueError(f'Corrupt head cache: {target/name}')
    with gzip.open(target/'entries.json.gz','rt') as f:data=json.load(f)
    with np.load(target/'geometry.npz') as z:
        arrays={k:z[k].copy() for k in z.files}
    pools={}
    for fraction in AREAS:
        key=f'{fraction:g}';entries=data['pools'][key];prefix='a'+key
        offsets=arrays[prefix+'_offsets'];force_offsets=arrays[prefix+'_force_offsets']
        for i,e in enumerate(entries):
            c=e['contact'];lo,hi=offsets[i:i+2];fl,fh=force_offsets[i:i+2]
            c.update(center_m=arrays['centers'][i],triangles_m=arrays[prefix+'_triangles'][lo:hi],
                source_faces=arrays[prefix+'_faces'][lo:hi],triangle_areas_m2=arrays[prefix+'_areas'][lo:hi],
                wrench_generators=arrays[prefix+'_forces'][fl:fh],wrench_com_m=arrays['com'])
            if 'root_m' in e:e['root_m']=np.asarray(e['root_m'])
        pools[fraction]=entries
    return dict(report=report,pools=pools,arrays=arrays,catalogue=data['catalogue'],roadmap=data['roadmap'],sampling=data['sampling'])
def pool(cache,fraction,catalogue=None):
    rows=copy.deepcopy(cache['pools'][fraction])
    if catalogue is None:return rows
    old=np.asarray(cache['catalogue']['vectors']);new=np.asarray(catalogue['vectors'])
    matching=np.linalg.norm(new[:,None,:]-old[None,:,:],axis=2)
    ids=matching.argmin(axis=1)
    if np.any(matching[np.arange(len(new)),ids]>1e-7):raise ValueError('Direction menu is not a cache subset')
    for e in rows:
        if 'directions' not in e:continue
        allowed=set(e['directions'][0]);e['directions']=[[i for i,j in enumerate(ids) if int(j) in allowed]]
        e.pop('direction_records',None)
        e['valid']=bool(e['directions'][0]);e['reason']='valid' if e['valid'] else 'no_requested_exit_witness'
    return rows
