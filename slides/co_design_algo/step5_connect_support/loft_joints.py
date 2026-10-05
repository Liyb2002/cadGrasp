"""Short reference-style joints offered inside the same physical candidate graph."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pickle
import time

import numpy as np
from scipy.spatial import cKDTree

from step5_connect_support import build_coupled_saddle as S,material_graph as M
from step5_connect_support.loft_candidates import volume,carve


def augment(graph,context,work):
    key=dict(base=graph['key'],short_joints_source=S.I.sha256(Path(__file__)))
    cache=Path(work)/'loft_and_joints.pickle'
    if cache.exists():
        with cache.open('rb') as stream:saved=pickle.load(stream)
        if saved['key']==key:
            saved['solids']=[S.solid(S.trimesh.Trimesh(v,f,process=False)) for v,f in zip(saved['vertices'],saved['faces'])]
            print('Reusing loft and short-joint library',saved['stats'],flush=True)
            return saved
    began=time.monotonic();solids=list(graph['solids']);records=list(graph['records'])
    anchors=[i for i,r in enumerate(records) if r['kind']=='immutable_head' or
        (r['kind']=='head_floor_loft' and r['terminal_family']=='projected_head')]
    points={}
    for i in anchors:
        vertices=graph['vertices'][i];valid=np.ones(len(vertices),bool)
        for b,o in zip(context['bases'],context['offsets']):
            valid&=(vertices-o)@b[2]>.0045
        points[i]=vertices[valid]
    up=context['bases'][0][2]+context['bases'][1][2]
    up=up/np.linalg.norm(up) if np.linalg.norm(up)>1e-9 else context['bases'][0][0]
    specs=[]
    for a in anchors:
        va=points[a]
        if not len(va):continue
        for b in anchors:
            if b>=a or records[a]['head']==records[b]['head']:continue
            vb=points[b]
            if not len(vb) or volume(solids[a]^solids[b])>.0001:continue
            distances,nearest=cKDTree(vb).query(va);used=[]
            for q in np.argsort(distances):
                if distances[q]>.030:break
                x,y=va[q],vb[nearest[q]]
                if any(np.linalg.norm(x-u)+np.linalg.norm(y-v)<.008 for u,v in used):continue
                used.append((x,y))
                for lift in (0.,.008,.016):
                    path=np.array([x,y] if lift==0 else [x,(x+y)/2+lift*up,y])
                    if np.linalg.norm(np.diff(path,axis=0),axis=1).sum()<=.040:
                        specs.append(dict(kind='short_body_joint',anchor_nodes=[a,b],path_m=path.tolist(),radius_m=.004))
                if len(used)==12:break
    bead=S.trimesh.creation.icosphere(subdivisions=1,radius=.004).vertices
    def make(spec):
        path=np.asarray(spec['path_m']);a,b=spec['anchor_nodes']
        raw=M.union(S.solid(S.G.hull_mesh(np.vstack([x+bead,y+bead]))) for x,y in zip(path[:-1],path[1:]))
        trimmed=carve(raw,context)
        for part in sorted(trimmed.decompose(),key=lambda p:-p.volume()):
            if volume(part^solids[a])<=.0001 or volume(part^solids[b])<=.0001:continue
            retained=volume(part)/volume(raw)
            if retained>=.5:return part,dict(spec,retained_fraction=retained,volume_cm3=volume(part))
        return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        for made in pool.map(make,specs):
            if made is not None:
                solid,record=made;solids.append(solid);records.append(record)
    old_count=len(graph['solids']);meshes=[S.unpack(s) for s in solids]
    bounds=np.array([m.bounds for m in meshes]);pairs=[]
    for i in range(old_count,len(solids)):
        for j in range(i):
            if np.all(np.minimum(bounds[i,1],bounds[j,1])-np.maximum(bounds[i,0],bounds[j,0])>1e-7):pairs.append((i,j))
    def edge(pair):
        i,j=pair
        return pair if volume(solids[i]^solids[j])>.0001 else None
    with ThreadPoolExecutor(max_workers=4) as pool:
        added_edges=[e for e in pool.map(edge,pairs) if e is not None]
    fixed=M.union(context['heads'])
    saved=dict(key=key,vertices=[m.vertices for m in meshes],faces=[m.faces for m in meshes],
        edges=np.vstack([graph['edges'],np.asarray(added_edges,int).reshape(-1,2)]),
        floors=graph['floors']+[M.floor_vertices(m,context['bases'],context['offsets']) for m in meshes[old_count:]],
        cost=np.r_[graph['cost'],[volume(s-fixed) for s in solids[old_count:]]],
        heads=graph['heads'],records=records,stats=dict(graph['stats'],
            nodes=len(solids),edges=len(graph['edges'])+len(added_edges),short_joint_specs=len(specs),
            legal_short_joints=len(solids)-old_count,maximum_joint_length_m=.040,
            joint_radius_m=.004,joint_retained_fraction_minimum=.5,joint_build_seconds=time.monotonic()-began))
    with cache.open('wb') as stream:pickle.dump(saved,stream,protocol=5)
    saved['solids']=solids
    print('Loft and short-joint library ready',saved['stats'],flush=True)
    return saved
