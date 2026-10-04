"""Independent full-load audit of the published dataset and cleaned layout."""
import json
import itertools
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import trimesh
from codes.precompute_objects.work_regions import digest, write, components
from codes.precompute_objects.dataset import read_sets, verify_files
from codes.precompute_objects.run import compatible_sets


def verify(name):
    folder=ROOT/'objects'/name; data=read_sets(name)
    manifest=json.loads((folder/'poses.json').read_text())
    assert data['set_count']==20 and manifest['pose_count']==30
    assert set(p.name for p in folder.iterdir())-{'sets.png'}=={'mesh.stl','meta.json','poses','poses.json','pose_sets.json'}
    names=[f'pose_{i}' for i in range(1,31)]
    assert [r['pose_id'] for r in manifest['poses']]==names
    assert set(p.name for p in (folder/'poses').iterdir())==set(names)
    raw=trimesh.load(folder/'mesh.stl',force='mesh'); meshes={0:raw}; frames=[]; clouds=[]
    assert digest(folder/'mesh.stl')==manifest['mesh_sha256']
    for pose,row in zip(names,manifest['poses']):
        target=verify_files(name,pose)
        assert set(p.name for p in target.iterdir())-{'step2'}=={'setup.npz','setup.json','needs.json','samples.json','floor_contact.npz'}
        meta=json.loads((target/'setup.json').read_text()); domain=json.loads((target/'needs.json').read_text())
        samples=json.loads((target/'samples.json').read_text())
        assert meta['source_snapshot_sha256']==digest(target/'setup.npz')
        assert domain['provenance']['setup_snapshot_sha256']==digest(target/'setup.npz')
        assert samples['provenance']['physical_domain_sha256']==digest(target/'needs.json')
        with np.load(target/'setup.npz') as setup:
            T=setup['T_world_mesh'].copy();mask=setup['work_faces'].copy();com=setup['com_m'].copy();pivot=setup['floor_contact_m'].copy()
            assert str(setup['poses_sha256'])==digest(folder/'poses.json')
            assert str(setup['mesh_sha256'])==digest(folder/'mesh.stl')
            assert float(setup['K'])==.5 and float(setup['cone_half_deg'])==30.
        np.testing.assert_array_equal(T,row['T_world_mesh'])
        np.testing.assert_allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-12,rtol=0)
        assert abs(np.linalg.det(T[:3,:3])-1)<1e-12
        vertices=raw.vertices@T[:3,:3].T+T[:3,3]
        assert abs(vertices[:,2].min())<1e-9 and (vertices[:,2]<1e-8).sum()==1
        np.testing.assert_allclose(com,raw.center_mass@T[:3,:3].T+T[:3,3],atol=1e-12,rtol=0)
        np.testing.assert_allclose(pivot,vertices[np.argmin(vertices[:,2])],atol=1e-9,rtol=0)
        assert np.linalg.norm(com[:2]-pivot[:2])>=.001
        rounds=meta['uniform_subdivision_rounds']
        for level in range(1,rounds+1):
            if level not in meshes:meshes[level]=meshes[level-1].subdivide()
        mesh=meshes[rounds]
        assert mask.dtype==bool and len(mask)==len(mesh.faces) and components(mesh,mask)==1
        assert .06<=mesh.area_faces[mask].sum()/mesh.area<=.10
        world=mesh.vertices@T[:3,:3].T+T[:3,3]
        assert world[mesh.faces[mask],2].min()>.0015
        assert (mesh.face_normals[mask]@T[2,:3]).min()>.35
        forces=np.asarray(samples['force_push_mg']);points=np.asarray(samples['pt_m'])
        assert len(forces)==32768 and np.max(np.linalg.norm(forces,axis=1))<=.5+1e-12
        expected=np.c_[np.array([0,0,1])-forces,-np.cross(points-com,forces)]
        np.testing.assert_allclose(expected,samples['need_wrench'],atol=1e-13,rtol=0)
        # World-origin external balance independent of Step0 pressure_centers.
        moments=np.cross(com,[0,0,1])-np.cross(points,forces)
        xy=np.c_[-moments[:,1]/expected[:,2],moments[:,0]/expected[:,2]]
        with np.load(target/'floor_contact.npz') as saved:
            np.testing.assert_allclose(saved['floor_demands_xy_m'],xy,atol=1e-12,rtol=0)
            np.testing.assert_array_equal(saved['load_wrenches'],samples['need_wrench'])
        frames.append(T);clouds.append(xy)
    # Check every raw sample in all 900 directed pose pairs.
    local=[(np.c_[p,np.zeros(len(p))]-T[:3,3])@T[:3,:3] for p,T in zip(clouds,frames)]
    counts=np.array([[(p@T[2,:3]+T[2,3]<-1e-9).sum() for T in frames] for p in local])
    np.testing.assert_array_equal(counts,data['directed_violating_counts'])
    seen=set();sizes={size:0 for size in range(2,7)}
    for group in data['sets']:
        ids=[names.index(p) for p in group['poses']]
        assert group['size']==len(ids) and len(set(ids))==len(ids) and tuple(ids) not in seen
        seen.add(tuple(ids));sizes[len(ids)]+=1
        assert not counts[np.ix_(ids,ids)].any()
    assert set(sizes.values())=={4}
    return dict(object=name,passed=True,poses=30,sets=20,sampled_loads=30*32768,ordered_pairs=900,clean_layout=True)


def main():
    names=json.loads((ROOT/'objects/cases.json').read_text())['active_objects']; began=time.monotonic()
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(verify,names))
    report=dict(passed=True,objects=len(rows),pose_count=sum(r['poses'] for r in rows),set_count=sum(r['sets'] for r in rows),
        sampled_loads=sum(r['sampled_loads'] for r in rows),ordered_pairs=sum(r['ordered_pairs'] for r in rows),
        seconds=round(time.monotonic()-began,3),cases=rows)
    write(ROOT/'codes/precompute_objects/verification.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)

if __name__=='__main__':main()
