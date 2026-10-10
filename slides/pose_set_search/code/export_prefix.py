"""Export an actually accepted incremental prefix; no future-task material."""
import argparse
import json
import hashlib
from common import *
from model import Model,Layout
from classify import classify


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--size',type=int,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    base=args.run/'pose1-10'
    stages=json.loads((base/'incremental/insertion_stages.json').read_text())[:args.size]
    assert len(stages)==args.size and all(row['passed'] for row in stages)
    checkpoints=[]
    for path in (base/'exact_states').glob('feasible_*.npz'):
        with np.load(path) as data:
            if len(data['active'])==args.size:
                checkpoints.append(path)
    path=max(checkpoints,key=lambda p:int(p.stem.split('_')[1]))
    data=np.load(path)
    poses=[row['added_pose'] for row in stages]
    model=Model(poses)
    active=tuple(range(args.size))
    layout=Layout(data['placements'][:args.size],data['directions'][:args.size],data['hosts'][:args.size],active)
    assert all(host<args.size for host in layout.hosts)
    actual=C.trimesh.Trimesh(data['vertices'],data['faces'],process=False)
    material=C.S.solid(actual)
    masks={};supplies={};contacts={};classifiers={};footprints=[]
    for k in active:
        q=layout.placements[k];direction=q[:3,:3].T @ layout.directions[k]
        allowed=model.allowed[k][model.mesh.face_normals[model.allowed[k]] @ direction<=1e-9]
        tri,src=C.contact_boundary(transform_mesh(model.mesh,q),actual,allowed)
        supply=C.supply(model.tasks[k],model.native[k] @ np.linalg.inv(q),tri,src)
        mask,info=classify(supply,model.tasks[k].targets)
        assert mask.all(),'Exported prefix lost an original load'
        masks[k]=mask;supplies[k]=supply;classifiers[k]=info
        contacts[k]=dict(triangles=tri,sources=src,triangle_count=len(tri),
                        area_m2=float(C.trimesh.triangles.area(tri).sum()))
        xy=C.transform_points(actual.vertices,model.native[layout.hosts[k]])[:,:2]
        footprints.append(float(ConvexHull(xy).volume))
    receipt=json.loads(path.with_suffix('.json').read_text())
    result=dict(layout=layout,serial=int(path.stem.split('_')[1]),remaining=material,
        masks=masks,supplies=supplies,contacts=contacts,classifiers=classifiers,
        counts={str(k):int(mask.sum()) for k,mask in masks.items()},diagnostics=receipt['diagnostics'],
        volume_cm3=C.material_volume(material)*1e6,maximum_projected_footprint_m2=max(footprints))
    result['actual_work_surface_checks']=model.verify_work(result)
    assert all(row['passed'] for row in result['actual_work_surface_checks'])
    args.out.mkdir(parents=True,exist_ok=False)
    model.save(result,args.out,dict(strategy='incremental',insertion_stages=stages,
        source_run=str(args.run),source_checkpoint=str(path),
        source_checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        prefix_has_no_future_pose_material=True,exact_search_evaluations_through_prefix=result['serial'],
        exported_prefix_of_same_incremental_search=True,
        search_runtime_manifest=str(args.run/'runtime_sources/manifest.json')))
    print('PREFIX EXPORTED',args.size,result['volume_cm3'],result['maximum_projected_footprint_m2'],flush=True)


if __name__=='__main__':
    main()
