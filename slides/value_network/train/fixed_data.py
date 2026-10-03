"""Canonical dense targets for fitting the fixed task universe."""
import json
from pathlib import Path
import numpy as np
from train import DATA,digest

def signature(poses,selected):return tuple((p,tuple(sorted(g))) for p,g in zip(poses,selected))

def features(poses,selected,universe):
    indices={p:k for k,p in enumerate(universe)};state=np.zeros(len(universe)*200,np.float32);membership=np.zeros(len(universe),np.float32);counts=np.zeros(len(universe),np.float32)
    for p,g in zip(poses,selected):
        k=indices[p];membership[k]=1;counts[k]=len(g)
        for i in g:state[k*200+i]=1
    return state,membership,counts

def collect(extra=None,config_path=None):
    cfg=json.loads(Path(config_path or DATA/'pilot20_shared/config.json').read_text());state_root=Path(cfg.get('state_root',DATA));universe=sorted({p for _,ps in cfg['groups'] for p in ps},key=lambda p:int(p.split('_')[1]));pindex={p:k for k,p in enumerate(universe)}
    records={};sources={}
    paths=[state_root/name/'pilot20'/f'state_{sid:03d}.json' for name,_ in cfg['groups'] for sid in range(cfg.get('states_per_group',20))]
    allowed={tuple(ps) for _,ps in cfg['groups']}
    if extra and Path(extra).exists():paths.extend(sorted(Path(extra).glob('*.json')))
    for path in paths:
        d=json.loads(path.read_text())
        if tuple(d['poses']) not in allowed:continue
        key=signature(d['poses'],d['selected_indices']);sources[str(path)]=digest(path)
        if key not in records:
            state,member,counts=features(d['poses'],d['selected_indices'],universe);n=len(universe)*200
            records[key]=dict(state=state,membership=member,counts=counts,target=np.zeros(n,np.float32),known=np.zeros(n,bool),observed=np.zeros(n,bool),poses=d['poses'],selected=d['selected_indices'])
        r=records[key]
        for row in d['rows']:
            i=pindex[row['pose']]*200+row['index'];r['observed'][i]=True
            if row['value'] is not None:
                # Extra certified-bank labels may improve the original upper bound.
                if not r['known'][i] or row['value']<r['target'][i]:r['target'][i]=row['value']
                r['known'][i]=True
    keys=list(records);arrays={k:np.array([records[key][k] for key in keys]) for k in ['state','membership','counts','target','known','observed']}
    return universe,arrays,keys,records,sources
