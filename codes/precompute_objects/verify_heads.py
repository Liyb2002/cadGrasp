"""Audit every saved candidate, its force generators and finite exit certificates."""
import json,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'slides/DSL_algo'))
import numpy as np
from PIL import Image
from codes.precompute_objects import head_cache as C
from step3_scheculer.pair_tasks import read_task

def verify(item):
    name,pose=item;folder=ROOT/'objects'/name/'poses'/pose;cache=C.load(folder)
    if cache is None:raise ValueError(f'Missing Step2: {name}/{pose}')
    assert set(p.name for p in (folder/'step2').iterdir())=={'geometry.npz','entries.json.gz','report.json','candidates.png'}
    task=read_task(name,pose);mesh=task.domain.mesh;valid=0;count=0
    centers=cache['arrays']['centers'];faces=cache['arrays']['center_faces'];assert centers.shape==(200,3) and faces.shape==(200,)
    assert np.isfinite(centers).all()
    assert not np.isin(faces,task.domain.work_ids).any()
    assert centers[:,2].min()>=.0015-1e-9
    vectors=np.array(cache['catalogue']['vectors']);np.testing.assert_allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-12)
    assert vectors[:,2].max()<=1e-10
    for fraction,entries in cache['pools'].items():
        assert len(entries)==200
        for i,e in enumerate(entries):
            count+=1;c=e['contact'];assert c['candidate_index']==i
            np.testing.assert_array_equal(c['center_m'],centers[i]);assert c['center_face']==faces[i]
            triangles=c['triangles_m'];source=c['source_faces'];areas=c['triangle_areas_m2']
            assert len(triangles)==len(source)==len(areas)
            assert not np.isin(source,task.domain.work_ids).any()
            assert np.isfinite(triangles).all() and np.isfinite(c['wrench_generators']).all()
            np.testing.assert_allclose(areas,np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)/2,atol=1e-18)
            normals=np.repeat(-mesh.face_normals[source],3,axis=0)
            expected=np.c_[normals,np.cross(triangles.reshape(-1,3)-task.domain.com,normals)]
            np.testing.assert_array_equal(c['wrench_generators'],expected)
            if 'direction_records' in e:
                records=e['direction_records'];assert len(records)==1
                r=records[0];assert r['analysis']['full_translation_sweeps']
                ids=[int(row['direction_id']) for row in r['analysis']['checks'] if row['clear']]
                assert ids==e['directions'][0]==r['certified_directions']['ids']
                assert all(0<=j<len(vectors) for j in ids)
                assert e['valid']==bool(ids)
            if e['valid']:
                valid+=1;assert e['path_components'];assert e['local_clearance']['valid']
                assert e['relative_area_error']<=1e-4
                assert abs(areas.sum()/mesh.area-fraction)<=fraction*1e-4+1e-15
    with Image.open(folder/'step2/candidates.png') as im:assert im.size==(1380,1040);im.verify()
    return dict(object=name,pose=pose,candidates=count,accepted=valid,passed=True)

def main():
    began=time.monotonic();names=json.loads((ROOT/'objects/cases.json').read_text())['active_objects']
    jobs=[(name,f'pose_{i}') for name in names for i in range(1,31)]
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(verify,jobs))
    sets_images=[]
    for name in names:
        sets_images.append(dict(object=name,path=f'objects/{name}/sets.png',sha256=C.digest(ROOT/'objects'/name/'sets.png'),mesh_sha256=C.digest(ROOT/'objects'/name/'mesh.stl'),poses_sha256=C.digest(ROOT/'objects'/name/'poses.json')))
        with Image.open(ROOT/'objects'/name/'sets.png') as im:assert im.size==(2736,3456);im.verify()
    report=dict(passed=True,objects=len(names),poses=len(rows),candidates=sum(r['candidates'] for r in rows),
        accepted=sum(r['accepted'] for r in rows),updated_images=len(rows)+len(names),seconds=round(time.monotonic()-began,3),sets_images=sets_images,cases=rows)
    (ROOT/'codes/precompute_objects/heads_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases','sets_images')}))
if __name__=='__main__':main()
