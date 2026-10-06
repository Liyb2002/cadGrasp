"""Check aggregate contact extraction against positive components in construction."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import _bootstrap
from physics_guided import *

base=HERE/'data/experiments/history/diagnostics/physics_guided_verified/B/pose1+4+7+12+21+27'
group=next(g for g in json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets'] if g['id']=='pose1+4+7+12+21+27')
search=PhysicsSearch('B',group,out=HERE/'data/experiments/history/diagnostics/physics_guided_contact_review',geometry='normal',directions=base/'directions.npz')
result=search.exact(search.warm())
physical=search.allowed[np.max(search.mesh.face_normals[search.allowed]@result['directions'].T,axis=1)<=1e-9]
triangles=[];sources=[];discarded=[]
for part in result['components']:
    volume=material_volume(part)
    if volume<1e-12:
        discarded.append(volume);continue
    tri,src=contact_boundary(search.mesh,S.unpack(part),physical)
    triangles.append(tri);sources.append(src)
tri=np.concatenate(triangles);src=np.concatenate(sources)
masks,infos,supplies=search.classify(tri,src)
report=dict(complete=True,aggregate_counts=result['counts'],positive_component_counts=[int(m.sum()) for m in masks],
    discarded_component_volumes_m3=discarded,policy='Checks within one fresh construction; no exported mesh replay',
    provenance=provenance([search.seed_path,search.contact_path,base/'directions.npz']+[p for task,T in search.states for p in task.inputs],
        [Path(__file__),HERE/'step4.2/physics_guided.py']))
save(search.out/'review.json',report)
print('CONTACT REVIEW',report['aggregate_counts'],report['positive_component_counts'],'discarded',len(discarded),flush=True)
